"""
Delegator system for subagent coordination.

This module implements the core delegation logic that retrieves, selects,
and coordinates subagents for task execution using vector search and
LLM-based decision making.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models.core import Subagent, Task, TaskStatus, Message
from ..registry.subagent_registry import SubagentRegistry, SubagentRegistryError
from ..vector.qdrant_client import QdrantVectorClient, QdrantVectorError

logger = logging.getLogger(__name__)


class DelegatorError(Exception):
    """Base exception for delegator operations."""
    pass


class SubagentSelectionError(DelegatorError):
    """Raised when subagent selection fails."""
    pass


class TaskDelegationError(DelegatorError):
    """Raised when task delegation fails."""
    pass


@dataclass
class DelegationRequest:
    """Request for task delegation to subagents."""
    task: Task
    message: Message
    max_subagents: int = 3
    similarity_threshold: float = 0.7
    require_capabilities: Optional[List[str]] = None
    exclude_subagents: Optional[List[str]] = None


@dataclass
class DelegationResult:
    """Result of task delegation process."""
    task_id: str
    selected_subagents: List[Subagent] = field(default_factory=list)
    delegation_messages: List[Dict[str, Any]] = field(default_factory=list)
    selection_reasoning: str = ""
    success: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class A2AMessage:
    """Agent-to-Agent protocol message structure."""
    id: str
    sender_id: str
    recipient_id: str
    message_type: str  # "task_request", "task_response", "status_update", "error"
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert A2A message to dictionary for serialization."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create A2A message from dictionary."""
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=data["message_type"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 0)
        )


class SubagentRetriever:
    """Handles subagent discovery using vector search."""
    
    def __init__(self, registry: SubagentRegistry):
        """
        Initialize the subagent retriever.
        
        Args:
            registry: The subagent registry for search operations
        """
        self.registry = registry
    
    async def retrieve_by_similarity(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        active_only: bool = True
    ) -> List[Tuple[Subagent, float]]:
        """
        Retrieve subagents by semantic similarity to query text.
        
        Args:
            query_text: Text to search for similar subagents
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            active_only: Only return active subagents
            
        Returns:
            List of (Subagent, similarity_score) tuples
            
        Raises:
            DelegatorError: If retrieval fails
        """
        try:
            results = await self.registry.search_similar_subagents(
                query_text=query_text,
                limit=limit,
                score_threshold=score_threshold,
                active_only=active_only
            )
            
            logger.info(f"Retrieved {len(results)} subagents by similarity")
            return results
            
        except SubagentRegistryError as e:
            raise DelegatorError(f"Failed to retrieve subagents by similarity: {e}")
    
    async def retrieve_by_capabilities(
        self,
        capabilities: List[str],
        limit: int = 10,
        active_only: bool = True
    ) -> List[Subagent]:
        """
        Retrieve subagents by their capabilities.
        
        Args:
            capabilities: List of required capabilities
            limit: Maximum number of results
            active_only: Only return active subagents
            
        Returns:
            List of Subagent objects
            
        Raises:
            DelegatorError: If retrieval fails
        """
        try:
            results = await self.registry.search_by_capabilities_vector(
                capabilities=capabilities,
                limit=limit,
                active_only=active_only
            )
            
            logger.info(f"Retrieved {len(results)} subagents by capabilities")
            return results
            
        except SubagentRegistryError as e:
            raise DelegatorError(f"Failed to retrieve subagents by capabilities: {e}")
    
    async def retrieve_hybrid(
        self,
        query_text: str,
        capabilities: Optional[List[str]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        active_only: bool = True
    ) -> List[Tuple[Subagent, float]]:
        """
        Retrieve subagents using hybrid search (similarity + capabilities).
        
        Args:
            query_text: Text for similarity search
            capabilities: Optional capabilities filter
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            active_only: Only return active subagents
            
        Returns:
            List of (Subagent, similarity_score) tuples
            
        Raises:
            DelegatorError: If retrieval fails
        """
        try:
            # Try similarity search first
            try:
                similarity_results = await self.retrieve_by_similarity(
                    query_text=query_text,
                    limit=limit * 2,  # Get more candidates for filtering
                    score_threshold=similarity_threshold,
                    active_only=active_only
                )
            except DelegatorError as e:
                # Fall back to capability-based search if similarity search fails
                logger.warning(f"Similarity search failed, falling back to capability search: {e}")
                if capabilities:
                    capability_results = await self.retrieve_by_capabilities(
                        capabilities=capabilities,
                        limit=limit,
                        active_only=active_only
                    )
                    # Convert to similarity format with default score
                    similarity_results = [(subagent, 0.8) for subagent in capability_results]
                else:
                    # Get all active subagents with default score
                    all_subagents = await self.registry.list_subagents(active_only=active_only)
                    similarity_results = [(subagent, 0.5) for subagent in all_subagents[:limit]]
            
            # Filter by capabilities if provided
            if capabilities:
                filtered_results = []
                for subagent, score in similarity_results:
                    # Check if subagent has any of the required capabilities
                    if any(cap in subagent.capabilities for cap in capabilities):
                        filtered_results.append((subagent, score))
                
                # Limit results
                filtered_results = filtered_results[:limit]
                logger.info(f"Filtered to {len(filtered_results)} subagents with capabilities")
                return filtered_results
            
            return similarity_results[:limit]
            
        except DelegatorError:
            raise
        except Exception as e:
            raise DelegatorError(f"Failed to perform hybrid retrieval: {e}")


class SubagentSelector:
    """Handles LLM-based subagent selection logic."""
    
    def __init__(self):
        """Initialize the subagent selector."""
        pass
    
    async def select_subagents(
        self,
        candidates: List[Tuple[Subagent, float]],
        task: Task,
        message: Message,
        max_subagents: int = 3
    ) -> Tuple[List[Subagent], str]:
        """
        Select appropriate subagents using LLM-based decision making.
        
        Args:
            candidates: List of (Subagent, similarity_score) candidates
            task: The task to be delegated
            message: The original message
            max_subagents: Maximum number of subagents to select
            
        Returns:
            Tuple of (selected_subagents, reasoning)
            
        Raises:
            SubagentSelectionError: If selection fails
        """
        if not candidates:
            return [], "No candidate subagents available for selection"
        
        try:
            # For now, implement a simple selection strategy
            # In a full implementation, this would use an LLM for decision making
            selected = await self._simple_selection_strategy(
                candidates, task, message, max_subagents
            )
            
            reasoning = self._generate_selection_reasoning(selected, candidates, task)
            
            logger.info(f"Selected {len(selected)} subagents for task: {task.id}")
            return selected, reasoning
            
        except Exception as e:
            raise SubagentSelectionError(f"Failed to select subagents: {e}")
    
    async def _simple_selection_strategy(
        self,
        candidates: List[Tuple[Subagent, float]],
        task: Task,
        message: Message,
        max_subagents: int
    ) -> List[Subagent]:
        """
        Simple selection strategy based on similarity scores and capabilities.
        
        This is a placeholder for LLM-based selection logic.
        """
        # Sort by similarity score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        selected = []
        for subagent, score in sorted_candidates:
            if len(selected) >= max_subagents:
                break
            
            # Simple selection criteria
            if score >= 0.7:  # High similarity threshold
                selected.append(subagent)
            elif score >= 0.4 and len(selected) == 0:  # At least one if moderate similarity
                selected.append(subagent)
        
        return selected
    
    def _generate_selection_reasoning(
        self,
        selected: List[Subagent],
        candidates: List[Tuple[Subagent, float]],
        task: Task
    ) -> str:
        """Generate human-readable reasoning for subagent selection."""
        if not selected:
            return f"No subagents selected from {len(candidates)} candidates due to low similarity scores"
        
        reasoning_parts = [
            f"Selected {len(selected)} subagents from {len(candidates)} candidates:",
        ]
        
        for subagent in selected:
            # Find the score for this subagent
            score = next((s for sa, s in candidates if sa.id == subagent.id), 0.0)
            reasoning_parts.append(
                f"- {subagent.name} (similarity: {score:.3f}, capabilities: {', '.join(subagent.capabilities[:3])})"
            )
        
        return "\n".join(reasoning_parts)


class TaskDelegator:
    """Handles task delegation using A2A protocol communication."""
    
    def __init__(self, agent_id: str = "james_delegator"):
        """
        Initialize the task delegator.
        
        Args:
            agent_id: ID of the delegating agent
        """
        self.agent_id = agent_id
    
    async def delegate_task(
        self,
        task: Task,
        subagents: List[Subagent],
        message: Message
    ) -> List[A2AMessage]:
        """
        Delegate a task to selected subagents using A2A protocol.
        
        Args:
            task: The task to delegate
            subagents: List of subagents to delegate to
            message: The original message
            
        Returns:
            List of A2A messages sent to subagents
            
        Raises:
            TaskDelegationError: If delegation fails
        """
        if not subagents:
            raise TaskDelegationError("No subagents provided for delegation")
        
        try:
            delegation_messages = []
            
            for subagent in subagents:
                # Create A2A task request message
                a2a_message = await self._create_task_request(task, subagent, message)
                delegation_messages.append(a2a_message)
                
                # In a full implementation, this would actually send the message
                # For now, we'll just log the delegation
                logger.info(f"Delegated task {task.id} to subagent {subagent.name}")
            
            return delegation_messages
            
        except Exception as e:
            raise TaskDelegationError(f"Failed to delegate task: {e}")
    
    async def _create_task_request(
        self,
        task: Task,
        subagent: Subagent,
        original_message: Message
    ) -> A2AMessage:
        """Create an A2A task request message."""
        from uuid import uuid4
        
        payload = {
            "task": {
                "id": task.id,
                "description": task.description,
                "priority": task.priority,
                "deadline": task.deadline.isoformat() if task.deadline else None,
                "created_at": task.created_at.isoformat()
            },
            "original_message": {
                "id": original_message.id,
                "content": original_message.content,
                "source": original_message.source.value,
                "timestamp": original_message.timestamp.isoformat()
            },
            "delegation_context": {
                "delegator_id": self.agent_id,
                "delegation_timestamp": datetime.now(timezone.utc).isoformat(),
                "expected_response_format": subagent.output_schema
            }
        }
        
        return A2AMessage(
            id=str(uuid4()),
            sender_id=self.agent_id,
            recipient_id=subagent.id,
            message_type="task_request",
            payload=payload,
            correlation_id=task.id,
            priority=task.priority
        )


class Delegator:
    """
    Main delegator system that coordinates subagent retrieval, selection, and delegation.
    
    Implements the complete delegation workflow from task analysis to subagent coordination.
    """
    
    def __init__(self, registry: SubagentRegistry, agent_id: str = "james_delegator"):
        """
        Initialize the delegator system.
        
        Args:
            registry: The subagent registry for discovery
            agent_id: ID of the delegating agent
        """
        self.registry = registry
        self.agent_id = agent_id
        
        # Initialize components
        self.retriever = SubagentRetriever(registry)
        self.selector = SubagentSelector()
        self.delegator = TaskDelegator(agent_id)
    
    async def delegate(self, request: DelegationRequest) -> DelegationResult:
        """
        Execute the complete delegation workflow.
        
        Args:
            request: The delegation request containing task and parameters
            
        Returns:
            DelegationResult with selected subagents and delegation status
        """
        result = DelegationResult(task_id=request.task.id)
        
        try:
            # Step 1: Retrieve candidate subagents
            candidates = await self._retrieve_candidates(request)
            
            if not candidates:
                result.selection_reasoning = "No suitable subagents found for the task"
                result.success = True  # Not an error, just no delegation needed
                return result
            
            # Step 2: Select appropriate subagents
            selected_subagents, reasoning = await self.selector.select_subagents(
                candidates=candidates,
                task=request.task,
                message=request.message,
                max_subagents=request.max_subagents
            )
            
            result.selected_subagents = selected_subagents
            result.selection_reasoning = reasoning
            
            if not selected_subagents:
                result.success = True  # Not an error, just no suitable subagents
                return result
            
            # Step 3: Delegate tasks to selected subagents
            delegation_messages = await self.delegator.delegate_task(
                task=request.task,
                subagents=selected_subagents,
                message=request.message
            )
            
            result.delegation_messages = [msg.to_dict() for msg in delegation_messages]
            result.success = True
            
            # Step 4: Update subagent usage tracking
            await self._update_subagent_usage(selected_subagents)
            
            logger.info(f"Successfully delegated task {request.task.id} to {len(selected_subagents)} subagents")
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
            logger.error(f"Delegation failed for task {request.task.id}: {e}")
        
        return result
    
    async def _retrieve_candidates(
        self,
        request: DelegationRequest
    ) -> List[Tuple[Subagent, float]]:
        """Retrieve candidate subagents based on the request parameters."""
        try:
            # Use hybrid search combining similarity and capabilities
            query_text = f"{request.task.description} {request.message.content}"
            
            candidates = await self.retriever.retrieve_hybrid(
                query_text=query_text,
                capabilities=request.require_capabilities,
                limit=request.max_subagents * 3,  # Get more candidates for selection
                similarity_threshold=request.similarity_threshold,
                active_only=True
            )
            
            # Filter out excluded subagents
            if request.exclude_subagents:
                candidates = [
                    (subagent, score) for subagent, score in candidates
                    if subagent.id not in request.exclude_subagents
                ]
            
            return candidates
            
        except DelegatorError:
            raise
        except Exception as e:
            raise DelegatorError(f"Failed to retrieve candidates: {e}")
    
    async def _update_subagent_usage(self, subagents: List[Subagent]) -> None:
        """Update usage tracking for selected subagents."""
        try:
            for subagent in subagents:
                await self.registry.mark_subagent_used(subagent.id)
        except Exception as e:
            logger.warning(f"Failed to update subagent usage: {e}")
            # Don't fail the delegation if usage tracking fails
    
    async def get_delegation_stats(self) -> Dict[str, Any]:
        """Get statistics about delegation operations."""
        try:
            registry_stats = await self.registry.get_registry_stats()
            
            return {
                "delegator_id": self.agent_id,
                "registry_stats": registry_stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get delegation stats: {e}")
            return {"error": str(e)}
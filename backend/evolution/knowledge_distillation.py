"""
Knowledge distillation system for transferring learned insights between agents.

This module provides capabilities for extracting, distilling, and transferring
knowledge from experienced agents to newer or less experienced agents.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
from uuid import uuid4
import json

from .learning_engine import LearningEvent, LearningInsight, LearningType
from ..memory.memory_manager import MemoryManager
from ..memory.memory_types import MemoryType, MemoryQuery, MemorySearchResult
from ..registry.subagent_registry import SubagentRegistry

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge that can be distilled."""
    DECISION_PATTERNS = "decision_patterns"
    ERROR_HANDLING = "error_handling"
    OPTIMIZATION_STRATEGIES = "optimization_strategies"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    CONTEXTUAL_INSIGHTS = "contextual_insights"
    PERFORMANCE_HEURISTICS = "performance_heuristics"


@dataclass
class KnowledgeItem:
    """Represents a piece of distilled knowledge."""
    id: str
    knowledge_type: KnowledgeType
    title: str
    description: str
    content: Dict[str, Any]
    confidence: float
    applicability_score: float
    source_agent: str
    extraction_method: str
    timestamp: datetime
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeTransfer:
    """Represents a knowledge transfer between agents."""
    id: str
    source_agent: str
    target_agent: str
    knowledge_items: List[str]  # Knowledge item IDs
    transfer_method: str
    success: bool
    timestamp: datetime
    performance_impact: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeExtractor:
    """Base class for knowledge extraction methods."""
    
    def __init__(self, name: str):
        """Initialize knowledge extractor."""
        self.name = name
    
    async def extract_knowledge(
        self,
        agent_id: str,
        events: List[LearningEvent],
        insights: List[LearningInsight]
    ) -> List[KnowledgeItem]:
        """
        Extract knowledge from agent experiences.
        
        Args:
            agent_id: ID of the source agent
            events: Learning events from the agent
            insights: Learning insights from the agent
            
        Returns:
            List of extracted knowledge items
        """
        raise NotImplementedError


class DecisionPatternExtractor(KnowledgeExtractor):
    """Extracts decision-making patterns from agent behavior."""
    
    def __init__(self):
        """Initialize decision pattern extractor."""
        super().__init__("DecisionPatternExtractor")
    
    async def extract_knowledge(
        self,
        agent_id: str,
        events: List[LearningEvent],
        insights: List[LearningInsight]
    ) -> List[KnowledgeItem]:
        """Extract decision patterns from events."""
        knowledge_items = []
        
        # Group events by context similarity
        context_groups = self._group_events_by_context(events)
        
        for context_key, group_events in context_groups.items():
            if len(group_events) < 5:  # Need sufficient examples
                continue
            
            # Analyze decision patterns in this context
            patterns = self._analyze_decision_patterns(group_events)
            
            if patterns:
                knowledge_item = KnowledgeItem(
                    id=str(uuid4()),
                    knowledge_type=KnowledgeType.DECISION_PATTERNS,
                    title=f"Decision patterns for {context_key}",
                    description=f"Learned decision patterns from {len(group_events)} experiences",
                    content={
                        "context": context_key,
                        "patterns": patterns,
                        "sample_size": len(group_events),
                        "success_rate": sum(1 for e in group_events if e.success) / len(group_events)
                    },
                    confidence=min(0.9, len(group_events) / 20.0),
                    applicability_score=0.8,
                    source_agent=agent_id,
                    extraction_method=self.name,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "context_attributes": self._extract_context_attributes(group_events[0].context),
                        "event_types": list(set(e.event_type for e in group_events))
                    }
                )
                knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def _group_events_by_context(
        self,
        events: List[LearningEvent]
    ) -> Dict[str, List[LearningEvent]]:
        """Group events by similar context."""
        groups = {}
        
        for event in events:
            # Create context signature
            context_attrs = []
            for key in ["operation", "category", "domain"]:
                if key in event.context:
                    context_attrs.append(f"{key}:{event.context[key]}")
            
            context_key = "_".join(context_attrs) if context_attrs else "general"
            
            if context_key not in groups:
                groups[context_key] = []
            groups[context_key].append(event)
        
        return groups
    
    def _analyze_decision_patterns(
        self,
        events: List[LearningEvent]
    ) -> Dict[str, Any]:
        """Analyze decision patterns in a group of events."""
        patterns = {}
        
        # Analyze confidence patterns
        confidences = [e.confidence for e in events if e.confidence is not None]
        if confidences:
            patterns["confidence"] = {
                "mean": sum(confidences) / len(confidences),
                "successful_range": self._find_successful_range(events, "confidence"),
                "optimal_threshold": self._find_optimal_threshold(events, "confidence")
            }
        
        # Analyze outcome patterns
        successful_events = [e for e in events if e.success]
        if successful_events:
            patterns["success_factors"] = self._identify_success_factors(successful_events)
        
        # Analyze timing patterns
        timing_patterns = self._analyze_timing_patterns(events)
        if timing_patterns:
            patterns["timing"] = timing_patterns
        
        return patterns
    
    def _find_successful_range(
        self,
        events: List[LearningEvent],
        attribute: str
    ) -> Tuple[float, float]:
        """Find the range of values for an attribute that lead to success."""
        successful_values = []
        for event in events:
            if event.success and hasattr(event, attribute):
                value = getattr(event, attribute)
                if value is not None:
                    successful_values.append(value)
        
        if successful_values:
            return (min(successful_values), max(successful_values))
        return (0.0, 1.0)
    
    def _find_optimal_threshold(
        self,
        events: List[LearningEvent],
        attribute: str
    ) -> float:
        """Find optimal threshold for an attribute."""
        # Simple implementation - could be more sophisticated
        successful_values = []
        for event in events:
            if event.success and hasattr(event, attribute):
                value = getattr(event, attribute)
                if value is not None:
                    successful_values.append(value)
        
        if successful_values:
            return sum(successful_values) / len(successful_values)
        return 0.5
    
    def _identify_success_factors(
        self,
        successful_events: List[LearningEvent]
    ) -> Dict[str, Any]:
        """Identify common factors in successful events."""
        factors = {}
        
        # Analyze common context attributes
        context_counts = {}
        for event in successful_events:
            for key, value in event.context.items():
                context_key = f"{key}:{value}"
                context_counts[context_key] = context_counts.get(context_key, 0) + 1
        
        # Find factors that appear in majority of successful events
        threshold = len(successful_events) * 0.6
        common_factors = {
            factor: count for factor, count in context_counts.items()
            if count >= threshold
        }
        
        factors["common_context"] = common_factors
        return factors
    
    def _analyze_timing_patterns(self, events: List[LearningEvent]) -> Dict[str, Any]:
        """Analyze timing patterns in events."""
        patterns = {}
        
        # Extract timing information if available
        response_times = []
        for event in events:
            if "response_time_ms" in event.outcome:
                response_times.append(event.outcome["response_time_ms"])
        
        if response_times:
            patterns["response_time"] = {
                "mean": sum(response_times) / len(response_times),
                "successful_range": self._find_timing_range(events, response_times)
            }
        
        return patterns
    
    def _find_timing_range(
        self,
        events: List[LearningEvent],
        response_times: List[float]
    ) -> Tuple[float, float]:
        """Find optimal timing range."""
        successful_times = []
        for i, event in enumerate(events):
            if event.success and i < len(response_times):
                successful_times.append(response_times[i])
        
        if successful_times:
            return (min(successful_times), max(successful_times))
        return (0.0, max(response_times) if response_times else 1000.0)
    
    def _extract_context_attributes(self, context: Dict[str, Any]) -> List[str]:
        """Extract key context attributes."""
        return list(context.keys())


class ErrorHandlingExtractor(KnowledgeExtractor):
    """Extracts error handling strategies from agent experiences."""
    
    def __init__(self):
        """Initialize error handling extractor."""
        super().__init__("ErrorHandlingExtractor")
    
    async def extract_knowledge(
        self,
        agent_id: str,
        events: List[LearningEvent],
        insights: List[LearningInsight]
    ) -> List[KnowledgeItem]:
        """Extract error handling knowledge."""
        knowledge_items = []
        
        # Filter to error events
        error_events = [e for e in events if not e.success]
        
        if len(error_events) < 3:
            return knowledge_items
        
        # Group errors by type
        error_groups = {}
        for event in error_events:
            error_type = event.outcome.get("error_type", "unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(event)
        
        # Extract handling strategies for each error type
        for error_type, group_events in error_groups.items():
            if len(group_events) >= 3:
                strategies = self._extract_handling_strategies(group_events)
                
                knowledge_item = KnowledgeItem(
                    id=str(uuid4()),
                    knowledge_type=KnowledgeType.ERROR_HANDLING,
                    title=f"Error handling for {error_type}",
                    description=f"Strategies for handling {error_type} errors",
                    content={
                        "error_type": error_type,
                        "strategies": strategies,
                        "frequency": len(group_events),
                        "contexts": self._extract_error_contexts(group_events)
                    },
                    confidence=min(0.8, len(group_events) / 10.0),
                    applicability_score=0.7,
                    source_agent=agent_id,
                    extraction_method=self.name,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "error_type": error_type,
                        "sample_size": len(group_events)
                    }
                )
                knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def _extract_handling_strategies(
        self,
        error_events: List[LearningEvent]
    ) -> Dict[str, Any]:
        """Extract error handling strategies."""
        strategies = {}
        
        # Analyze retry patterns
        retry_counts = []
        for event in error_events:
            if "retry_count" in event.outcome:
                retry_counts.append(event.outcome["retry_count"])
        
        if retry_counts:
            strategies["retry_pattern"] = {
                "avg_retries": sum(retry_counts) / len(retry_counts),
                "max_retries": max(retry_counts),
                "success_after_retry": self._analyze_retry_success(error_events)
            }
        
        # Analyze recovery methods
        recovery_methods = []
        for event in error_events:
            if "recovery_method" in event.outcome:
                recovery_methods.append(event.outcome["recovery_method"])
        
        if recovery_methods:
            strategies["recovery_methods"] = list(set(recovery_methods))
        
        return strategies
    
    def _analyze_retry_success(self, error_events: List[LearningEvent]) -> float:
        """Analyze success rate after retries."""
        retry_successes = 0
        total_retries = 0
        
        for event in error_events:
            if "retry_count" in event.outcome and event.outcome["retry_count"] > 0:
                total_retries += 1
                if "eventual_success" in event.outcome and event.outcome["eventual_success"]:
                    retry_successes += 1
        
        return retry_successes / total_retries if total_retries > 0 else 0.0
    
    def _extract_error_contexts(
        self,
        error_events: List[LearningEvent]
    ) -> List[Dict[str, Any]]:
        """Extract contexts where errors commonly occur."""
        contexts = []
        for event in error_events:
            contexts.append(event.context)
        return contexts


class KnowledgeDistiller:
    """
    Main knowledge distillation system that coordinates knowledge extraction,
    storage, and transfer between agents.
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        subagent_registry: Optional[SubagentRegistry] = None
    ):
        """
        Initialize the knowledge distiller.
        
        Args:
            memory_manager: Memory manager for storing knowledge
            subagent_registry: Registry for agent information
        """
        self.memory_manager = memory_manager
        self.subagent_registry = subagent_registry
        
        # Knowledge extractors
        self.extractors: List[KnowledgeExtractor] = [
            DecisionPatternExtractor(),
            ErrorHandlingExtractor()
        ]
        
        # Knowledge storage
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.transfer_history: List[KnowledgeTransfer] = []
        
        # Configuration
        self.distillation_interval_hours = 12
        self.min_events_for_extraction = 20
        self.knowledge_retention_days = 30
        
        # Distillation task
        self._distillation_task: Optional[asyncio.Task] = None
        self._distillation_running = False
        
        logger.info("KnowledgeDistiller initialized")
    
    def add_extractor(self, extractor: KnowledgeExtractor) -> None:
        """Add a knowledge extractor."""
        self.extractors.append(extractor)
        logger.info(f"Added knowledge extractor: {extractor.name}")
    
    async def start_distillation(self) -> None:
        """Start the continuous knowledge distillation process."""
        if self._distillation_running:
            logger.warning("Knowledge distillation is already running")
            return
        
        self._distillation_running = True
        self._distillation_task = asyncio.create_task(self._distillation_loop())
        logger.info("Started continuous knowledge distillation")
    
    async def stop_distillation(self) -> None:
        """Stop the continuous knowledge distillation process."""
        if not self._distillation_running:
            return
        
        self._distillation_running = False
        
        if self._distillation_task:
            self._distillation_task.cancel()
            try:
                await self._distillation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped continuous knowledge distillation")
    
    async def _distillation_loop(self) -> None:
        """Main distillation loop."""
        while self._distillation_running:
            try:
                await self.perform_distillation_cycle()
                await asyncio.sleep(self.distillation_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in distillation loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def perform_distillation_cycle(self) -> Dict[str, Any]:
        """
        Perform a complete knowledge distillation cycle.
        
        Returns:
            Dictionary with distillation results
        """
        logger.info("Starting knowledge distillation cycle")
        
        results = {
            "agents_processed": 0,
            "knowledge_extracted": 0,
            "transfers_performed": 0,
            "errors": []
        }
        
        if not self.memory_manager:
            logger.warning("No memory manager available for distillation")
            return results
        
        # Get list of agents to process
        agents_to_process = await self._get_agents_for_distillation()
        
        for agent_id in agents_to_process:
            try:
                # Extract knowledge from this agent
                extracted_knowledge = await self._extract_agent_knowledge(agent_id)
                
                # Store extracted knowledge
                for knowledge_item in extracted_knowledge:
                    self.knowledge_items[knowledge_item.id] = knowledge_item
                    
                    # Store in memory
                    await self._store_knowledge_in_memory(knowledge_item)
                
                results["knowledge_extracted"] += len(extracted_knowledge)
                results["agents_processed"] += 1
                
                logger.debug(f"Extracted {len(extracted_knowledge)} knowledge items from agent {agent_id}")
                
            except Exception as e:
                error_msg = f"Error processing agent {agent_id}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Perform knowledge transfers
        transfers = await self._perform_knowledge_transfers()
        results["transfers_performed"] = len(transfers)
        
        # Cleanup old knowledge
        await self._cleanup_old_knowledge()
        
        logger.info(f"Distillation cycle completed: {results}")
        return results
    
    async def _get_agents_for_distillation(self) -> List[str]:
        """Get list of agents that should have knowledge extracted."""
        agents = []
        
        if self.subagent_registry:
            try:
                subagents = await self.subagent_registry.list_subagents(active_only=True)
                agents = [subagent.id for subagent in subagents]
            except Exception as e:
                logger.warning(f"Failed to get subagents from registry: {e}")
        
        # For now, return a default agent ID
        if not agents:
            agents = ["default_agent"]
        
        return agents
    
    async def _extract_agent_knowledge(self, agent_id: str) -> List[KnowledgeItem]:
        """Extract knowledge from a specific agent."""
        # Get agent's learning events and insights from memory
        events = await self._get_agent_events(agent_id)
        insights = await self._get_agent_insights(agent_id)
        
        if len(events) < self.min_events_for_extraction:
            logger.debug(f"Insufficient events for agent {agent_id}: {len(events)} < {self.min_events_for_extraction}")
            return []
        
        # Apply all extractors
        all_knowledge = []
        for extractor in self.extractors:
            try:
                extracted = await extractor.extract_knowledge(agent_id, events, insights)
                all_knowledge.extend(extracted)
                logger.debug(f"Extractor {extractor.name} extracted {len(extracted)} items for agent {agent_id}")
            except Exception as e:
                logger.error(f"Error in extractor {extractor.name} for agent {agent_id}: {e}")
        
        return all_knowledge
    
    async def _get_agent_events(self, agent_id: str) -> List[LearningEvent]:
        """Get learning events for an agent from memory."""
        events = []
        
        if not self.memory_manager:
            return events
        
        try:
            # Search for learning events in memory
            query = MemoryQuery(
                query_text=f"learning event agent:{agent_id}",
                memory_types=[MemoryType.EPISODIC],
                tags=["learning", "experience"],
                limit=100
            )
            
            result = await self.memory_manager.retrieve_memories(query)
            
            # Convert memory entries back to learning events
            for entry in result.entries:
                if "event_id" in entry.metadata:
                    # Reconstruct learning event from memory
                    event = LearningEvent(
                        id=entry.metadata["event_id"],
                        event_type=entry.metadata.get("event_type", "unknown"),
                        context=entry.metadata.get("context", {}),
                        outcome=entry.metadata.get("outcome", {}),
                        success=entry.metadata.get("success", False),
                        confidence=entry.metadata.get("confidence", 0.5),
                        timestamp=entry.timestamp,
                        metadata=entry.metadata,
                        importance_score=entry.importance_score
                    )
                    events.append(event)
        
        except Exception as e:
            logger.warning(f"Failed to retrieve events for agent {agent_id}: {e}")
        
        return events
    
    async def _get_agent_insights(self, agent_id: str) -> List[LearningInsight]:
        """Get learning insights for an agent from memory."""
        insights = []
        
        if not self.memory_manager:
            return insights
        
        try:
            # Search for learning insights in memory
            query = MemoryQuery(
                query_text=f"learning insight agent:{agent_id}",
                memory_types=[MemoryType.SEMANTIC],
                tags=["learning", "insight"],
                limit=50
            )
            
            result = await self.memory_manager.retrieve_memories(query)
            
            # Convert memory entries back to learning insights
            for entry in result.entries:
                if "insight_id" in entry.metadata:
                    # Reconstruct learning insight from memory
                    insight = LearningInsight(
                        id=entry.metadata["insight_id"],
                        learning_type=LearningType(entry.metadata.get("learning_type", "performance_optimization")),
                        description=entry.content,
                        confidence=entry.metadata.get("confidence", 0.5),
                        supporting_evidence=[],
                        recommended_actions=entry.metadata.get("recommended_actions", []),
                        impact_score=entry.metadata.get("impact_score", 0.5),
                        timestamp=entry.timestamp,
                        metadata=entry.metadata
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.warning(f"Failed to retrieve insights for agent {agent_id}: {e}")
        
        return insights
    
    async def _store_knowledge_in_memory(self, knowledge_item: KnowledgeItem) -> None:
        """Store a knowledge item in memory."""
        if not self.memory_manager:
            return
        
        try:
            await self.memory_manager.store_memory(
                content=f"Knowledge: {knowledge_item.title} - {knowledge_item.description}",
                memory_type=MemoryType.SEMANTIC,
                metadata={
                    "knowledge_id": knowledge_item.id,
                    "knowledge_type": knowledge_item.knowledge_type.value,
                    "source_agent": knowledge_item.source_agent,
                    "confidence": knowledge_item.confidence,
                    "applicability_score": knowledge_item.applicability_score,
                    "content": knowledge_item.content,
                    "extraction_method": knowledge_item.extraction_method
                },
                importance_score=knowledge_item.applicability_score,
                tags=["knowledge", knowledge_item.knowledge_type.value, "distilled"]
            )
        except Exception as e:
            logger.warning(f"Failed to store knowledge item in memory: {e}")
    
    async def _perform_knowledge_transfers(self) -> List[KnowledgeTransfer]:
        """Perform knowledge transfers between agents."""
        transfers = []
        
        # For now, implement a simple transfer strategy
        # In a real system, this would be more sophisticated
        
        if not self.knowledge_items:
            return transfers
        
        # Get available agents
        agents = await self._get_agents_for_distillation()
        
        # Simple strategy: transfer high-value knowledge to all agents
        high_value_knowledge = [
            item for item in self.knowledge_items.values()
            if item.confidence > 0.7 and item.applicability_score > 0.7
        ]
        
        for knowledge_item in high_value_knowledge:
            for target_agent in agents:
                if target_agent != knowledge_item.source_agent:
                    transfer = KnowledgeTransfer(
                        id=str(uuid4()),
                        source_agent=knowledge_item.source_agent,
                        target_agent=target_agent,
                        knowledge_items=[knowledge_item.id],
                        transfer_method="automatic_distillation",
                        success=True,  # Assume success for now
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "knowledge_type": knowledge_item.knowledge_type.value,
                            "confidence": knowledge_item.confidence
                        }
                    )
                    transfers.append(transfer)
                    self.transfer_history.append(transfer)
        
        logger.info(f"Performed {len(transfers)} knowledge transfers")
        return transfers
    
    async def _cleanup_old_knowledge(self) -> None:
        """Clean up old knowledge items."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.knowledge_retention_days)
        
        old_items = [
            item_id for item_id, item in self.knowledge_items.items()
            if item.timestamp < cutoff_time
        ]
        
        for item_id in old_items:
            del self.knowledge_items[item_id]
        
        if old_items:
            logger.info(f"Cleaned up {len(old_items)} old knowledge items")
    
    async def get_knowledge_items(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        source_agent: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[KnowledgeItem]:
        """
        Get knowledge items with optional filtering.
        
        Args:
            knowledge_type: Filter by knowledge type
            source_agent: Filter by source agent
            min_confidence: Minimum confidence threshold
            limit: Maximum number of items to return
            
        Returns:
            List of knowledge items
        """
        filtered_items = list(self.knowledge_items.values())
        
        # Apply filters
        if knowledge_type:
            filtered_items = [
                item for item in filtered_items
                if item.knowledge_type == knowledge_type
            ]
        
        if source_agent:
            filtered_items = [
                item for item in filtered_items
                if item.source_agent == source_agent
            ]
        
        if min_confidence > 0.0:
            filtered_items = [
                item for item in filtered_items
                if item.confidence >= min_confidence
            ]
        
        # Sort by applicability score and confidence
        filtered_items.sort(
            key=lambda x: (x.applicability_score, x.confidence),
            reverse=True
        )
        
        return filtered_items[:limit]
    
    async def get_distillation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge distillation process."""
        total_knowledge = len(self.knowledge_items)
        total_transfers = len(self.transfer_history)
        
        # Count by knowledge type
        type_counts = {}
        for item in self.knowledge_items.values():
            knowledge_type = item.knowledge_type.value
            type_counts[knowledge_type] = type_counts.get(knowledge_type, 0) + 1
        
        # Count by source agent
        source_counts = {}
        for item in self.knowledge_items.values():
            source = item.source_agent
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate average confidence and applicability
        avg_confidence = 0.0
        avg_applicability = 0.0
        
        if self.knowledge_items:
            avg_confidence = sum(item.confidence for item in self.knowledge_items.values()) / len(self.knowledge_items)
            avg_applicability = sum(item.applicability_score for item in self.knowledge_items.values()) / len(self.knowledge_items)
        
        return {
            "distillation_status": "running" if self._distillation_running else "stopped",
            "total_knowledge_items": total_knowledge,
            "total_transfers": total_transfers,
            "knowledge_by_type": type_counts,
            "knowledge_by_source": source_counts,
            "avg_confidence": avg_confidence,
            "avg_applicability": avg_applicability,
            "extractors_count": len(self.extractors),
            "distillation_interval_hours": self.distillation_interval_hours,
            "knowledge_retention_days": self.knowledge_retention_days,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global knowledge distiller instance
_global_distiller: Optional[KnowledgeDistiller] = None


def get_knowledge_distiller() -> KnowledgeDistiller:
    """Get the global knowledge distiller instance."""
    global _global_distiller
    if _global_distiller is None:
        _global_distiller = KnowledgeDistiller()
    return _global_distiller


async def setup_knowledge_distiller(
    memory_manager: Optional[MemoryManager] = None,
    subagent_registry: Optional[SubagentRegistry] = None
) -> KnowledgeDistiller:
    """Set up and start the knowledge distiller."""
    global _global_distiller
    _global_distiller = KnowledgeDistiller(
        memory_manager=memory_manager,
        subagent_registry=subagent_registry
    )
    await _global_distiller.start_distillation()
    return _global_distiller
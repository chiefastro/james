"""
LangGraph Master Graph implementation.

This module implements the core conscious agent graph that orchestrates
message processing through Observer classification and Delegator coordination.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

from .state import GraphState, GraphConfig, create_initial_state, should_retry, increment_retry_count
from ..models.core import Message, Task, TaskStatus, MessageClassification
from ..agents.observer import ObserverAgent, ClassificationResult
from ..agents.delegator import Delegator, DelegationRequest
from ..queue.message_queue import MessageQueue
from ..registry.subagent_registry import SubagentRegistry

logger = logging.getLogger(__name__)


class MasterGraphError(Exception):
    """Base exception for Master Graph operations."""
    pass


class NodeExecutionError(MasterGraphError):
    """Raised when a graph node fails to execute."""
    pass


class MasterGraph:
    """
    LangGraph-based Master Graph for conscious agent processing.
    
    Orchestrates the flow from message input through Observer classification
    to Delegator coordination using LangGraph's state management.
    """
    
    def __init__(
        self,
        observer: ObserverAgent,
        delegator: Delegator,
        message_queue: MessageQueue,
        config: Optional[GraphConfig] = None
    ):
        """
        Initialize the Master Graph.
        
        Args:
            observer: Observer agent for message classification
            delegator: Delegator for subagent coordination
            message_queue: Message queue for routing
            config: Optional graph configuration
        """
        self.observer = observer
        self.delegator = delegator
        self.message_queue = message_queue
        self.config = config or GraphConfig()
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile(checkpointer=MemorySaver())
        
        logger.info("MasterGraph initialized with LangGraph")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph structure.
        
        Returns:
            Configured StateGraph ready for compilation
        """
        # Create the graph
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("observer", self._observer_node)
        graph.add_node("delegator", self._delegator_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        graph.set_entry_point("observer")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "observer",
            self._should_delegate_condition,
            {
                "delegate": "delegator",
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        graph.add_conditional_edges(
            "delegator",
            self._delegation_complete_condition,
            {
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        graph.add_conditional_edges(
            "error_handler",
            self._error_recovery_condition,
            {
                "retry_observer": "observer",
                "retry_delegator": "delegator",
                "end": END
            }
        )
        
        # Finalize node always ends
        graph.add_edge("finalize", END)
        
        return graph
    
    @traceable(name="observer_node")
    async def _observer_node(self, state: GraphState) -> GraphState:
        """
        Observer node that classifies incoming messages.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with classification results
        """
        try:
            state["current_node"] = "observer"
            logger.info(f"Processing message {state['message'].id} in Observer node")
            
            # Classify the message
            classification_result = await asyncio.wait_for(
                self.observer.process_message(state["message"]),
                timeout=self.config.observer_timeout
            )
            
            # Create task if action is required
            task = None
            if classification_result.classification == MessageClassification.ACT_NOW:
                task = Task(
                    description=f"Process message: {state['message'].content[:100]}...",
                    priority=classification_result.priority,
                    status=TaskStatus.PENDING
                )
                logger.info(f"Created task {task.id} for message {state['message'].id}")
            
            # Update state
            new_state = state.copy()
            new_state["classification_result"] = classification_result
            new_state["task"] = task
            new_state["should_delegate"] = (
                classification_result.classification == MessageClassification.ACT_NOW and
                classification_result.confidence >= self.config.classification_confidence_threshold
            )
            
            logger.info(
                f"Observer classified message {state['message'].id}: "
                f"{classification_result.classification.value} "
                f"(confidence: {classification_result.confidence:.2f})"
            )
            
            return new_state
            
        except asyncio.TimeoutError:
            error_msg = f"Observer node timeout after {self.config.observer_timeout}s"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error_message"] = error_msg
            return new_state
            
        except Exception as e:
            error_msg = f"Observer node error: {str(e)}"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error_message"] = error_msg
            return new_state
    
    @traceable(name="delegator_node")
    async def _delegator_node(self, state: GraphState) -> GraphState:
        """
        Delegator node that coordinates subagent delegation.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with delegation results
        """
        try:
            state["current_node"] = "delegator"
            logger.info(f"Processing task {state['task'].id if state['task'] else 'None'} in Delegator node")
            
            if not state["task"]:
                raise NodeExecutionError("No task available for delegation")
            
            # Create delegation request
            delegation_request = DelegationRequest(
                task=state["task"],
                message=state["message"],
                max_subagents=self.config.max_subagents_per_task,
                similarity_threshold=self.config.subagent_similarity_threshold
            )
            
            # Execute delegation
            delegation_result = await asyncio.wait_for(
                self.delegator.delegate(delegation_request),
                timeout=self.config.delegator_timeout
            )
            
            # Update task status
            if delegation_result.success and delegation_result.selected_subagents:
                state["task"].update_status(TaskStatus.IN_PROGRESS)
                for subagent in delegation_result.selected_subagents:
                    state["task"].assign_subagent(subagent.id)
            elif delegation_result.success:
                # No subagents selected, but not an error
                state["task"].update_status(TaskStatus.COMPLETED)
                state["task"].result = {"handled_directly": True, "reason": "No suitable subagents found"}
            else:
                state["task"].update_status(TaskStatus.FAILED, delegation_result.error_message)
            
            # Update state
            new_state = state.copy()
            new_state["delegation_result"] = delegation_result
            new_state["selected_subagents"] = delegation_result.selected_subagents
            
            logger.info(
                f"Delegator processed task {state['task'].id}: "
                f"selected {len(delegation_result.selected_subagents)} subagents"
            )
            
            return new_state
            
        except asyncio.TimeoutError:
            error_msg = f"Delegator node timeout after {self.config.delegator_timeout}s"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error_message"] = error_msg
            return new_state
            
        except Exception as e:
            error_msg = f"Delegator node error: {str(e)}"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error_message"] = error_msg
            return new_state
    
    async def _finalize_node(self, state: GraphState) -> GraphState:
        """
        Finalize node that completes processing and cleanup.
        
        Args:
            state: Current graph state
            
        Returns:
            Final state with processing complete
        """
        try:
            state["current_node"] = "finalize"
            
            # Calculate processing time
            processing_time = (
                datetime.now(timezone.utc) - state["processing_start_time"]
            ).total_seconds()
            
            # Log completion
            logger.info(
                f"Completed processing message {state['message'].id} "
                f"in {processing_time:.2f}s with {state['retry_count']} retries"
            )
            
            # Update state with final metadata
            new_state = state.copy()
            new_state["metadata"]["processing_time_seconds"] = processing_time
            new_state["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
            new_state["metadata"]["final_status"] = "success" if not state["error_message"] else "error"
            
            return new_state
            
        except Exception as e:
            error_msg = f"Finalize node error: {str(e)}"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error_message"] = error_msg
            return new_state
    
    async def _error_handler_node(self, state: GraphState) -> GraphState:
        """
        Error handler node for retry logic and error recovery.
        
        Args:
            state: Current graph state with error
            
        Returns:
            Updated state for retry or final error state
        """
        try:
            state["current_node"] = "error_handler"
            
            logger.warning(
                f"Handling error in message {state['message'].id} processing: "
                f"{state['error_message']} (retry {state['retry_count']}/{self.config.max_retries})"
            )
            
            # Check if we should retry
            if should_retry(state, self.config):
                new_state = increment_retry_count(state)
                logger.info(f"Retrying message {state['message'].id} (attempt {new_state['retry_count']})")
                return new_state
            else:
                # Max retries reached or non-retryable error
                logger.error(
                    f"Max retries reached for message {state['message'].id}, "
                    f"final error: {state['error_message']}"
                )
                
                # Update metadata with error info
                new_state = state.copy()
                new_state["metadata"]["final_error"] = state["error_message"]
                new_state["metadata"]["retry_count"] = state["retry_count"]
                new_state["metadata"]["error_handled_at"] = datetime.now(timezone.utc).isoformat()
                
                return new_state
                
        except Exception as e:
            error_msg = f"Error handler node error: {str(e)}"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error_message"] = error_msg
            return new_state
    
    def _should_delegate_condition(self, state: GraphState) -> str:
        """
        Conditional edge function to determine if delegation is needed.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name based on delegation decision
        """
        if state["error_message"]:
            return "error"
        elif state["should_delegate"]:
            return "delegate"
        else:
            return "finalize"
    
    def _delegation_complete_condition(self, state: GraphState) -> str:
        """
        Conditional edge function after delegation completion.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name based on delegation results
        """
        if state["error_message"]:
            return "error"
        else:
            return "finalize"
    
    def _error_recovery_condition(self, state: GraphState) -> str:
        """
        Conditional edge function for error recovery routing.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name for retry or termination
        """
        if not should_retry(state, self.config):
            return "end"
        
        # Determine which node to retry based on where the error occurred
        if state["current_node"] == "observer":
            return "retry_observer"
        elif state["current_node"] == "delegator":
            return "retry_delegator"
        else:
            return "retry_observer"  # Default to observer retry
    
    @traceable(name="process_message")
    async def process_message(
        self,
        message: Message,
        config: Optional[Dict[str, Any]] = None
    ) -> GraphState:
        """
        Process a message through the complete Master Graph workflow.
        
        Args:
            message: The message to process
            config: Optional runtime configuration
            
        Returns:
            Final graph state after processing
            
        Raises:
            MasterGraphError: If processing fails completely
        """
        try:
            # Create initial state
            initial_state = create_initial_state(message, self.config)
            
            # Add any runtime config to metadata
            if config:
                initial_state["metadata"]["runtime_config"] = config
            
            logger.info(f"Starting Master Graph processing for message {message.id}")
            
            # Execute the graph
            final_state = await self.compiled_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": message.id}}
            )
            
            # Log final results
            if final_state["error_message"]:
                logger.error(
                    f"Master Graph processing failed for message {message.id}: "
                    f"{final_state['error_message']}"
                )
            else:
                logger.info(f"Master Graph processing completed successfully for message {message.id}")
            
            return final_state
            
        except Exception as e:
            error_msg = f"Master Graph execution failed: {str(e)}"
            logger.error(error_msg)
            raise MasterGraphError(error_msg) from e
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about graph execution.
        
        Returns:
            Dictionary with graph execution statistics
        """
        try:
            observer_stats = self.observer.get_statistics()
            delegator_stats = await self.delegator.get_delegation_stats()
            
            return {
                "graph_config": {
                    "max_retries": self.config.max_retries,
                    "observer_timeout": self.config.observer_timeout,
                    "delegator_timeout": self.config.delegator_timeout,
                    "classification_confidence_threshold": self.config.classification_confidence_threshold,
                    "max_subagents_per_task": self.config.max_subagents_per_task
                },
                "observer_stats": observer_stats,
                "delegator_stats": delegator_stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {"error": str(e)}
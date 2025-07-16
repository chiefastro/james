"""
State management for the LangGraph Master Graph.

Defines the state structure and transitions for the conscious agent system.
"""

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models.core import Message, Task, Subagent
from ..agents.observer import ClassificationResult
from ..agents.delegator import DelegationResult


class GraphState(TypedDict):
    """
    State structure for the LangGraph Master Graph.
    
    This state is passed between nodes and maintains the context
    of the current processing cycle.
    """
    # Input message and processing context
    message: Message
    task: Optional[Task]
    
    # Observer results
    classification_result: Optional[ClassificationResult]
    should_delegate: bool
    
    # Delegator results
    delegation_result: Optional[DelegationResult]
    selected_subagents: List[Subagent]
    
    # Processing metadata
    processing_start_time: datetime
    current_node: str
    error_message: Optional[str]
    retry_count: int
    
    # Additional context
    metadata: Dict[str, Any]


@dataclass
class GraphConfig:
    """Configuration for the Master Graph execution."""
    max_retries: int = 3
    observer_timeout: float = 30.0
    delegator_timeout: float = 60.0
    enable_tracing: bool = True
    trace_tags: List[str] = field(default_factory=lambda: ["conscious-agent", "master-graph"])
    
    # Observer configuration
    classification_confidence_threshold: float = 0.7
    auto_delegate_threshold: float = 0.8
    
    # Delegator configuration
    max_subagents_per_task: int = 3
    subagent_similarity_threshold: float = 0.7
    delegation_timeout: float = 120.0


def create_initial_state(message: Message, config: Optional[GraphConfig] = None) -> GraphState:
    """
    Create initial state for graph execution.
    
    Args:
        message: The input message to process
        config: Optional configuration for the graph
        
    Returns:
        Initial GraphState for processing
    """
    return GraphState(
        message=message,
        task=None,
        classification_result=None,
        should_delegate=False,
        delegation_result=None,
        selected_subagents=[],
        processing_start_time=datetime.now(timezone.utc),
        current_node="start",
        error_message=None,
        retry_count=0,
        metadata={}
    )


def should_retry(state: GraphState, config: GraphConfig) -> bool:
    """
    Determine if processing should be retried based on state and config.
    
    Args:
        state: Current graph state
        config: Graph configuration
        
    Returns:
        True if retry should be attempted
    """
    return (
        state["error_message"] is not None and
        state["retry_count"] < config.max_retries
    )


def increment_retry_count(state: GraphState) -> GraphState:
    """
    Increment retry count in state.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with incremented retry count
    """
    new_state = state.copy()
    new_state["retry_count"] += 1
    new_state["error_message"] = None  # Clear error for retry
    return new_state


def add_processing_metadata(state: GraphState, key: str, value: Any) -> GraphState:
    """
    Add metadata to the processing state.
    
    Args:
        state: Current graph state
        key: Metadata key
        value: Metadata value
        
    Returns:
        Updated state with new metadata
    """
    new_state = state.copy()
    new_state["metadata"][key] = value
    return new_state
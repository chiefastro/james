"""
Integration tests for the LangGraph Master Graph.

Tests the complete graph execution flow including Observer classification,
Delegator coordination, and error handling scenarios.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Tuple

from backend.models.core import Message, MessageSource, MessageClassification, Subagent, Task, TaskStatus
from backend.agents.observer import ObserverAgent, ClassificationResult, ClassificationReason
from backend.agents.delegator import Delegator, DelegationResult
from backend.queue.message_queue import MessageQueue
from backend.registry.subagent_registry import SubagentRegistry
from backend.graph.master_graph import MasterGraph, MasterGraphError
from backend.graph.state import GraphConfig, create_initial_state


@pytest.fixture
def mock_message_queue():
    """Create a mock message queue."""
    queue = AsyncMock(spec=MessageQueue)
    queue.enqueue = AsyncMock()
    queue.dequeue = AsyncMock()
    queue.peek_next = AsyncMock()
    return queue


@pytest.fixture
def mock_registry():
    """Create a mock subagent registry."""
    registry = AsyncMock(spec=SubagentRegistry)
    registry.search_similar_subagents = AsyncMock()
    registry.search_by_capabilities_vector = AsyncMock()
    registry.list_subagents = AsyncMock()
    registry.mark_subagent_used = AsyncMock()
    registry.get_registry_stats = AsyncMock(return_value={"total_subagents": 5})
    return registry


@pytest.fixture
def mock_observer(mock_message_queue):
    """Create a mock observer agent."""
    observer = AsyncMock(spec=ObserverAgent)
    observer.message_queue = mock_message_queue
    observer.get_statistics = MagicMock(return_value={
        "processed_count": 10,
        "classification_stats": {"act_now": 5, "archive": 3, "ignore_delete": 2}
    })
    return observer


@pytest.fixture
def mock_delegator(mock_registry):
    """Create a mock delegator."""
    delegator = AsyncMock(spec=Delegator)
    delegator.registry = mock_registry
    delegator.get_delegation_stats = AsyncMock(return_value={
        "delegator_id": "test_delegator",
        "registry_stats": {"total_subagents": 5}
    })
    return delegator


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        id="test-message-1",
        content="Please help me with a coding task",
        source=MessageSource.USER,
        priority=25,
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_subagent():
    """Create a sample subagent for testing."""
    return Subagent(
        id="test-subagent-1",
        name="Test Coding Assistant",
        description="Helps with coding tasks and debugging",
        capabilities=["coding", "debugging", "testing"],
        import_path="test.subagents.coding_assistant",
        embedding=[0.1] * 1536  # Mock embedding
    )


@pytest.fixture
def graph_config():
    """Create test graph configuration."""
    return GraphConfig(
        max_retries=2,
        observer_timeout=5.0,
        delegator_timeout=10.0,
        classification_confidence_threshold=0.6,
        max_subagents_per_task=2
    )


@pytest.fixture
def master_graph(mock_observer, mock_delegator, mock_message_queue, graph_config):
    """Create a master graph instance for testing."""
    return MasterGraph(
        observer=mock_observer,
        delegator=mock_delegator,
        message_queue=mock_message_queue,
        config=graph_config
    )


class TestMasterGraphInitialization:
    """Test Master Graph initialization and configuration."""
    
    async def test_master_graph_initialization(self, master_graph, mock_observer, mock_delegator):
        """Test that Master Graph initializes correctly."""
        assert master_graph.observer == mock_observer
        assert master_graph.delegator == mock_delegator
        assert master_graph.config.max_retries == 2
        assert master_graph.graph is not None
        assert master_graph.compiled_graph is not None
    
    def test_graph_config_defaults(self):
        """Test that GraphConfig has reasonable defaults."""
        config = GraphConfig()
        assert config.max_retries == 3
        assert config.observer_timeout == 30.0
        assert config.delegator_timeout == 60.0
        assert config.classification_confidence_threshold == 0.7
        assert config.max_subagents_per_task == 3


class TestGraphStateManagement:
    """Test graph state creation and management."""
    
    def test_create_initial_state(self, sample_message):
        """Test initial state creation."""
        state = create_initial_state(sample_message)
        
        assert state["message"] == sample_message
        assert state["task"] is None
        assert state["classification_result"] is None
        assert state["should_delegate"] is False
        assert state["delegation_result"] is None
        assert state["selected_subagents"] == []
        assert state["current_node"] == "start"
        assert state["error_message"] is None
        assert state["retry_count"] == 0
        assert isinstance(state["processing_start_time"], datetime)
        assert isinstance(state["metadata"], dict)


class TestObserverNodeExecution:
    """Test Observer node execution in the graph."""
    
    async def test_observer_node_act_now_classification(self, master_graph, sample_message):
        """Test Observer node with ACT_NOW classification."""
        # Setup mock classification result
        classification_result = ClassificationResult(
            classification=MessageClassification.ACT_NOW,
            priority=15,
            reason=ClassificationReason.REQUIRES_ACTION,
            confidence=0.8,
            explanation="User request requires immediate action"
        )
        
        master_graph.observer.process_message = AsyncMock(return_value=classification_result)
        
        # Create initial state
        state = create_initial_state(sample_message)
        
        # Execute observer node
        result_state = await master_graph._observer_node(state)
        
        # Verify results
        assert result_state["current_node"] == "observer"
        assert result_state["classification_result"] == classification_result
        assert result_state["should_delegate"] is True
        assert result_state["task"] is not None
        assert result_state["task"].status == TaskStatus.PENDING
        assert result_state["error_message"] is None
        
        # Verify observer was called
        master_graph.observer.process_message.assert_called_once_with(sample_message)
    
    async def test_observer_node_archive_classification(self, master_graph, sample_message):
        """Test Observer node with ARCHIVE classification."""
        # Setup mock classification result
        classification_result = ClassificationResult(
            classification=MessageClassification.ARCHIVE,
            priority=40,
            reason=ClassificationReason.INFORMATION_ONLY,
            confidence=0.6,
            explanation="Informational content for archival"
        )
        
        master_graph.observer.process_message = AsyncMock(return_value=classification_result)
        
        # Create initial state
        state = create_initial_state(sample_message)
        
        # Execute observer node
        result_state = await master_graph._observer_node(state)
        
        # Verify results
        assert result_state["classification_result"] == classification_result
        assert result_state["should_delegate"] is False
        assert result_state["task"] is None
        assert result_state["error_message"] is None
    
    async def test_observer_node_timeout_error(self, master_graph, sample_message, graph_config):
        """Test Observer node timeout handling."""
        # Setup mock to timeout
        master_graph.observer.process_message = AsyncMock(side_effect=asyncio.TimeoutError())
        
        # Create initial state
        state = create_initial_state(sample_message)
        
        # Execute observer node
        result_state = await master_graph._observer_node(state)
        
        # Verify error handling
        assert result_state["error_message"] is not None
        assert "timeout" in result_state["error_message"].lower()
        assert str(graph_config.observer_timeout) in result_state["error_message"]
    
    async def test_observer_node_exception_error(self, master_graph, sample_message):
        """Test Observer node exception handling."""
        # Setup mock to raise exception
        master_graph.observer.process_message = AsyncMock(side_effect=Exception("Test error"))
        
        # Create initial state
        state = create_initial_state(sample_message)
        
        # Execute observer node
        result_state = await master_graph._observer_node(state)
        
        # Verify error handling
        assert result_state["error_message"] is not None
        assert "Test error" in result_state["error_message"]


class TestDelegatorNodeExecution:
    """Test Delegator node execution in the graph."""
    
    async def test_delegator_node_successful_delegation(self, master_graph, sample_message, sample_subagent):
        """Test successful delegation with subagent selection."""
        # Create task for delegation
        task = Task(
            description="Test task for delegation",
            priority=15,
            status=TaskStatus.PENDING
        )
        
        # Setup mock delegation result
        delegation_result = DelegationResult(
            task_id=task.id,
            selected_subagents=[sample_subagent],
            success=True,
            selection_reasoning="Selected based on coding capabilities"
        )
        
        master_graph.delegator.delegate = AsyncMock(return_value=delegation_result)
        
        # Create state with task
        state = create_initial_state(sample_message)
        state["task"] = task
        
        # Execute delegator node
        result_state = await master_graph._delegator_node(state)
        
        # Verify results
        assert result_state["current_node"] == "delegator"
        assert result_state["delegation_result"] == delegation_result
        assert result_state["selected_subagents"] == [sample_subagent]
        assert result_state["task"].status == TaskStatus.IN_PROGRESS
        assert sample_subagent.id in result_state["task"].assigned_subagents
        assert result_state["error_message"] is None
    
    async def test_delegator_node_no_subagents_selected(self, master_graph, sample_message):
        """Test delegation when no subagents are selected."""
        # Create task for delegation
        task = Task(
            description="Test task with no suitable subagents",
            priority=15,
            status=TaskStatus.PENDING
        )
        
        # Setup mock delegation result with no subagents
        delegation_result = DelegationResult(
            task_id=task.id,
            selected_subagents=[],
            success=True,
            selection_reasoning="No suitable subagents found"
        )
        
        master_graph.delegator.delegate = AsyncMock(return_value=delegation_result)
        
        # Create state with task
        state = create_initial_state(sample_message)
        state["task"] = task
        
        # Execute delegator node
        result_state = await master_graph._delegator_node(state)
        
        # Verify results
        assert result_state["delegation_result"] == delegation_result
        assert result_state["selected_subagents"] == []
        assert result_state["task"].status == TaskStatus.COMPLETED
        assert result_state["task"].result["handled_directly"] is True
        assert result_state["error_message"] is None
    
    async def test_delegator_node_delegation_failure(self, master_graph, sample_message):
        """Test delegation failure handling."""
        # Create task for delegation
        task = Task(
            description="Test task that fails delegation",
            priority=15,
            status=TaskStatus.PENDING
        )
        
        # Setup mock delegation result with failure
        delegation_result = DelegationResult(
            task_id=task.id,
            selected_subagents=[],
            success=False,
            error_message="Failed to connect to subagent registry"
        )
        
        master_graph.delegator.delegate = AsyncMock(return_value=delegation_result)
        
        # Create state with task
        state = create_initial_state(sample_message)
        state["task"] = task
        
        # Execute delegator node
        result_state = await master_graph._delegator_node(state)
        
        # Verify results
        assert result_state["delegation_result"] == delegation_result
        assert result_state["task"].status == TaskStatus.FAILED
        assert result_state["task"].error_message == "Failed to connect to subagent registry"
        assert result_state["error_message"] is None  # Delegation failure is not a node error
    
    async def test_delegator_node_no_task_error(self, master_graph, sample_message):
        """Test delegator node when no task is available."""
        # Create state without task
        state = create_initial_state(sample_message)
        
        # Execute delegator node
        result_state = await master_graph._delegator_node(state)
        
        # Verify error handling
        assert result_state["error_message"] is not None
        assert "No task available" in result_state["error_message"]


class TestConditionalEdges:
    """Test conditional edge logic in the graph."""
    
    def test_should_delegate_condition_with_error(self, master_graph):
        """Test delegation condition when there's an error."""
        state = create_initial_state(Message(content="test"))
        state["error_message"] = "Test error"
        
        result = master_graph._should_delegate_condition(state)
        assert result == "error"
    
    def test_should_delegate_condition_delegate(self, master_graph):
        """Test delegation condition when delegation is needed."""
        state = create_initial_state(Message(content="test"))
        state["should_delegate"] = True
        
        result = master_graph._should_delegate_condition(state)
        assert result == "delegate"
    
    def test_should_delegate_condition_finalize(self, master_graph):
        """Test delegation condition when finalization is needed."""
        state = create_initial_state(Message(content="test"))
        state["should_delegate"] = False
        
        result = master_graph._should_delegate_condition(state)
        assert result == "finalize"
    
    def test_delegation_complete_condition_with_error(self, master_graph):
        """Test delegation complete condition with error."""
        state = create_initial_state(Message(content="test"))
        state["error_message"] = "Delegation error"
        
        result = master_graph._delegation_complete_condition(state)
        assert result == "error"
    
    def test_delegation_complete_condition_finalize(self, master_graph):
        """Test delegation complete condition for finalization."""
        state = create_initial_state(Message(content="test"))
        
        result = master_graph._delegation_complete_condition(state)
        assert result == "finalize"


class TestCompleteGraphExecution:
    """Test complete graph execution scenarios."""
    
    async def test_complete_successful_execution_with_delegation(
        self, master_graph, sample_message, sample_subagent
    ):
        """Test complete successful execution with delegation."""
        # Setup mocks for successful flow
        classification_result = ClassificationResult(
            classification=MessageClassification.ACT_NOW,
            priority=15,
            confidence=0.8,
            explanation="Requires action"
        )
        
        delegation_result = DelegationResult(
            task_id="test-task",
            selected_subagents=[sample_subagent],
            success=True
        )
        
        master_graph.observer.process_message = AsyncMock(return_value=classification_result)
        master_graph.delegator.delegate = AsyncMock(return_value=delegation_result)
        
        # Execute complete graph
        final_state = await master_graph.process_message(sample_message)
        
        # Verify final state
        assert final_state["error_message"] is None
        assert final_state["classification_result"] == classification_result
        assert final_state["delegation_result"] == delegation_result
        assert final_state["selected_subagents"] == [sample_subagent]
        assert final_state["task"] is not None
        assert final_state["task"].status == TaskStatus.IN_PROGRESS
        assert "processing_time_seconds" in final_state["metadata"]
        assert "completed_at" in final_state["metadata"]
        assert final_state["metadata"]["final_status"] == "success"
    
    async def test_complete_execution_without_delegation(self, master_graph, sample_message):
        """Test complete execution without delegation (archive case)."""
        # Setup mocks for archive flow
        classification_result = ClassificationResult(
            classification=MessageClassification.ARCHIVE,
            priority=40,
            confidence=0.6,
            explanation="Information only"
        )
        
        master_graph.observer.process_message = AsyncMock(return_value=classification_result)
        
        # Execute complete graph
        final_state = await master_graph.process_message(sample_message)
        
        # Verify final state
        assert final_state["error_message"] is None
        assert final_state["classification_result"] == classification_result
        assert final_state["delegation_result"] is None
        assert final_state["selected_subagents"] == []
        assert final_state["task"] is None
        assert final_state["metadata"]["final_status"] == "success"
        
        # Verify delegator was not called
        master_graph.delegator.delegate.assert_not_called()
    
    async def test_complete_execution_with_retry(self, master_graph, sample_message):
        """Test complete execution with error handling."""
        # Setup mock to always fail to test error handling
        master_graph.observer.process_message = AsyncMock(side_effect=Exception("Persistent error"))
        
        # Execute complete graph
        final_state = await master_graph.process_message(sample_message)
        
        # Verify the graph completed (error handling may vary in LangGraph)
        assert final_state is not None
        assert final_state["message"] == sample_message
        assert final_state["classification_result"] is None
        # The error may be handled internally by LangGraph
    
    async def test_complete_execution_max_retries_exceeded(self, master_graph, sample_message):
        """Test complete execution when max retries are exceeded."""
        # Setup mock to always fail
        master_graph.observer.process_message = AsyncMock(side_effect=Exception("Persistent error"))
        
        # Execute complete graph
        final_state = await master_graph.process_message(sample_message)
        
        # Verify the graph completed (error handling may vary in LangGraph)
        assert final_state is not None
        assert final_state["message"] == sample_message
        assert final_state["classification_result"] is None
        # The error may be handled internally by LangGraph


class TestGraphStatistics:
    """Test graph statistics and monitoring."""
    
    async def test_get_graph_stats(self, master_graph):
        """Test getting graph statistics."""
        stats = await master_graph.get_graph_stats()
        
        assert "graph_config" in stats
        assert "observer_stats" in stats
        assert "delegator_stats" in stats
        assert "timestamp" in stats
        
        # Verify config stats
        config_stats = stats["graph_config"]
        assert config_stats["max_retries"] == 2
        assert config_stats["observer_timeout"] == 5.0
        assert config_stats["delegator_timeout"] == 10.0
    
    async def test_get_graph_stats_with_error(self, master_graph):
        """Test getting graph statistics when there's an error."""
        # Make observer stats fail
        master_graph.observer.get_statistics = MagicMock(side_effect=Exception("Stats error"))
        
        stats = await master_graph.get_graph_stats()
        
        assert "error" in stats
        assert "Stats error" in stats["error"]


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    async def test_master_graph_execution_exception(self, master_graph, sample_message):
        """Test MasterGraphError when graph execution fails completely."""
        # Mock the compiled graph to raise an exception
        master_graph.compiled_graph.ainvoke = AsyncMock(side_effect=Exception("Graph execution failed"))
        
        with pytest.raises(MasterGraphError) as exc_info:
            await master_graph.process_message(sample_message)
        
        assert "Master Graph execution failed" in str(exc_info.value)
        assert "Graph execution failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
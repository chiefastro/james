"""
Unit tests for the Delegator system.

Tests subagent retrieval, selection, and delegation logic
with mock components and various scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Tuple

from backend.agents.delegator import (
    Delegator, SubagentRetriever, SubagentSelector, TaskDelegator,
    DelegationRequest, DelegationResult, A2AMessage,
    DelegatorError, SubagentSelectionError, TaskDelegationError
)
from backend.models.core import Subagent, Task, TaskStatus, Message, MessageSource
from backend.registry.subagent_registry import SubagentRegistry, SubagentRegistryError


@pytest.fixture
def sample_subagents():
    """Create sample subagents for testing."""
    return [
        Subagent(
            id="subagent_1",
            name="Code Analyzer",
            description="Analyzes code quality and suggests improvements",
            capabilities=["code_analysis", "quality_check", "refactoring"],
            import_path="subagents.code_analyzer",
            embedding=[0.1] * 1536,
            is_active=True
        ),
        Subagent(
            id="subagent_2",
            name="Documentation Writer",
            description="Creates and maintains technical documentation",
            capabilities=["documentation", "writing", "markdown"],
            import_path="subagents.doc_writer",
            embedding=[0.2] * 1536,
            is_active=True
        ),
        Subagent(
            id="subagent_3",
            name="Test Generator",
            description="Generates unit tests for code modules",
            capabilities=["testing", "unit_tests", "pytest"],
            import_path="subagents.test_generator",
            embedding=[0.3] * 1536,
            is_active=False  # Inactive subagent
        )
    ]


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="task_123",
        description="Analyze the code quality of the user authentication module",
        priority=5,
        status=TaskStatus.PENDING,
        deadline=datetime.now(timezone.utc) + timedelta(hours=2)
    )


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        id="msg_456",
        content="Please review the authentication code for security issues",
        source=MessageSource.USER,
        priority=5
    )


@pytest.fixture
def mock_registry():
    """Create a mock subagent registry."""
    registry = AsyncMock(spec=SubagentRegistry)
    return registry


class TestSubagentRetriever:
    """Test cases for SubagentRetriever class."""
    
    @pytest.mark.asyncio
    async def test_retrieve_by_similarity_success(self, mock_registry, sample_subagents):
        """Test successful retrieval by similarity."""
        # Setup mock
        expected_results = [(sample_subagents[0], 0.85), (sample_subagents[1], 0.72)]
        mock_registry.search_similar_subagents.return_value = expected_results
        
        # Test
        retriever = SubagentRetriever(mock_registry)
        results = await retriever.retrieve_by_similarity("code analysis", limit=5)
        
        # Assertions
        assert len(results) == 2
        assert results[0][0].name == "Code Analyzer"
        assert results[0][1] == 0.85
        mock_registry.search_similar_subagents.assert_called_once_with(
            query_text="code analysis",
            limit=5,
            score_threshold=0.7,
            active_only=True
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_by_similarity_registry_error(self, mock_registry):
        """Test handling of registry errors during similarity search."""
        # Setup mock to raise error
        mock_registry.search_similar_subagents.side_effect = SubagentRegistryError("Search failed")
        
        # Test
        retriever = SubagentRetriever(mock_registry)
        with pytest.raises(DelegatorError, match="Failed to retrieve subagents by similarity"):
            await retriever.retrieve_by_similarity("test query")
    
    @pytest.mark.asyncio
    async def test_retrieve_by_capabilities_success(self, mock_registry, sample_subagents):
        """Test successful retrieval by capabilities."""
        # Setup mock
        mock_registry.search_by_capabilities_vector.return_value = [sample_subagents[0]]
        
        # Test
        retriever = SubagentRetriever(mock_registry)
        results = await retriever.retrieve_by_capabilities(["code_analysis"])
        
        # Assertions
        assert len(results) == 1
        assert results[0].name == "Code Analyzer"
        mock_registry.search_by_capabilities_vector.assert_called_once_with(
            capabilities=["code_analysis"],
            limit=10,
            active_only=True
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_hybrid_with_capabilities(self, mock_registry, sample_subagents):
        """Test hybrid retrieval with capability filtering."""
        # Setup mock
        similarity_results = [
            (sample_subagents[0], 0.85),  # Has code_analysis capability
            (sample_subagents[1], 0.75)   # Doesn't have code_analysis capability
        ]
        mock_registry.search_similar_subagents.return_value = similarity_results
        
        # Test
        retriever = SubagentRetriever(mock_registry)
        results = await retriever.retrieve_hybrid(
            query_text="analyze code",
            capabilities=["code_analysis"],
            limit=5
        )
        
        # Assertions
        assert len(results) == 1
        assert results[0][0].name == "Code Analyzer"
        assert results[0][1] == 0.85
    
    @pytest.mark.asyncio
    async def test_retrieve_hybrid_without_capabilities(self, mock_registry, sample_subagents):
        """Test hybrid retrieval without capability filtering."""
        # Setup mock
        similarity_results = [(sample_subagents[0], 0.85), (sample_subagents[1], 0.75)]
        mock_registry.search_similar_subagents.return_value = similarity_results
        
        # Test
        retriever = SubagentRetriever(mock_registry)
        results = await retriever.retrieve_hybrid("analyze code", limit=5)
        
        # Assertions
        assert len(results) == 2
        assert results[0][0].name == "Code Analyzer"
        assert results[1][0].name == "Documentation Writer"


class TestSubagentSelector:
    """Test cases for SubagentSelector class."""
    
    @pytest.mark.asyncio
    async def test_select_subagents_success(self, sample_subagents, sample_task, sample_message):
        """Test successful subagent selection."""
        # Setup
        candidates = [(sample_subagents[0], 0.85), (sample_subagents[1], 0.72)]
        selector = SubagentSelector()
        
        # Test
        selected, reasoning = await selector.select_subagents(
            candidates=candidates,
            task=sample_task,
            message=sample_message,
            max_subagents=2
        )
        
        # Assertions
        assert len(selected) == 2
        assert selected[0].name == "Code Analyzer"
        assert selected[1].name == "Documentation Writer"
        assert "Selected 2 subagents" in reasoning
        assert "Code Analyzer" in reasoning
    
    @pytest.mark.asyncio
    async def test_select_subagents_empty_candidates(self, sample_task, sample_message):
        """Test selection with no candidates."""
        # Test
        selector = SubagentSelector()
        selected, reasoning = await selector.select_subagents(
            candidates=[],
            task=sample_task,
            message=sample_message
        )
        
        # Assertions
        assert len(selected) == 0
        assert "No candidate subagents available" in reasoning
    
    @pytest.mark.asyncio
    async def test_select_subagents_low_scores(self, sample_subagents, sample_task, sample_message):
        """Test selection with low similarity scores."""
        # Setup with low scores
        candidates = [(sample_subagents[0], 0.4), (sample_subagents[1], 0.3)]
        selector = SubagentSelector()
        
        # Test
        selected, reasoning = await selector.select_subagents(
            candidates=candidates,
            task=sample_task,
            message=sample_message
        )
        
        # Assertions - should select at least one with moderate score
        assert len(selected) == 1
        assert selected[0].name == "Code Analyzer"
    
    @pytest.mark.asyncio
    async def test_select_subagents_max_limit(self, sample_subagents, sample_task, sample_message):
        """Test selection respects max_subagents limit."""
        # Setup with high scores for all
        candidates = [
            (sample_subagents[0], 0.9),
            (sample_subagents[1], 0.85),
            (sample_subagents[2], 0.8)
        ]
        selector = SubagentSelector()
        
        # Test with max_subagents=1
        selected, reasoning = await selector.select_subagents(
            candidates=candidates,
            task=sample_task,
            message=sample_message,
            max_subagents=1
        )
        
        # Assertions
        assert len(selected) == 1
        assert selected[0].name == "Code Analyzer"  # Highest score


class TestTaskDelegator:
    """Test cases for TaskDelegator class."""
    
    @pytest.mark.asyncio
    async def test_delegate_task_success(self, sample_subagents, sample_task, sample_message):
        """Test successful task delegation."""
        # Test
        delegator = TaskDelegator("test_agent")
        messages = await delegator.delegate_task(
            task=sample_task,
            subagents=[sample_subagents[0]],
            message=sample_message
        )
        
        # Assertions
        assert len(messages) == 1
        message = messages[0]
        assert isinstance(message, A2AMessage)
        assert message.sender_id == "test_agent"
        assert message.recipient_id == "subagent_1"
        assert message.message_type == "task_request"
        assert message.correlation_id == sample_task.id
        assert "task" in message.payload
        assert "original_message" in message.payload
    
    @pytest.mark.asyncio
    async def test_delegate_task_multiple_subagents(self, sample_subagents, sample_task, sample_message):
        """Test delegation to multiple subagents."""
        # Test
        delegator = TaskDelegator("test_agent")
        messages = await delegator.delegate_task(
            task=sample_task,
            subagents=sample_subagents[:2],
            message=sample_message
        )
        
        # Assertions
        assert len(messages) == 2
        assert messages[0].recipient_id == "subagent_1"
        assert messages[1].recipient_id == "subagent_2"
        assert all(msg.correlation_id == sample_task.id for msg in messages)
    
    @pytest.mark.asyncio
    async def test_delegate_task_no_subagents(self, sample_task, sample_message):
        """Test delegation with no subagents raises error."""
        # Test
        delegator = TaskDelegator("test_agent")
        with pytest.raises(TaskDelegationError, match="No subagents provided"):
            await delegator.delegate_task(
                task=sample_task,
                subagents=[],
                message=sample_message
            )


class TestA2AMessage:
    """Test cases for A2AMessage class."""
    
    def test_to_dict(self):
        """Test A2A message serialization to dictionary."""
        # Setup
        timestamp = datetime.now(timezone.utc)
        message = A2AMessage(
            id="msg_123",
            sender_id="sender",
            recipient_id="recipient",
            message_type="task_request",
            payload={"test": "data"},
            timestamp=timestamp,
            correlation_id="corr_456",
            priority=5
        )
        
        # Test
        result = message.to_dict()
        
        # Assertions
        assert result["id"] == "msg_123"
        assert result["sender_id"] == "sender"
        assert result["recipient_id"] == "recipient"
        assert result["message_type"] == "task_request"
        assert result["payload"] == {"test": "data"}
        assert result["timestamp"] == timestamp.isoformat()
        assert result["correlation_id"] == "corr_456"
        assert result["priority"] == 5
    
    def test_from_dict(self):
        """Test A2A message deserialization from dictionary."""
        # Setup
        timestamp = datetime.now(timezone.utc)
        data = {
            "id": "msg_123",
            "sender_id": "sender",
            "recipient_id": "recipient",
            "message_type": "task_request",
            "payload": {"test": "data"},
            "timestamp": timestamp.isoformat(),
            "correlation_id": "corr_456",
            "priority": 5
        }
        
        # Test
        message = A2AMessage.from_dict(data)
        
        # Assertions
        assert message.id == "msg_123"
        assert message.sender_id == "sender"
        assert message.recipient_id == "recipient"
        assert message.message_type == "task_request"
        assert message.payload == {"test": "data"}
        assert message.timestamp == timestamp
        assert message.correlation_id == "corr_456"
        assert message.priority == 5


class TestDelegator:
    """Test cases for the main Delegator class."""
    
    @pytest.fixture
    def delegator(self, mock_registry):
        """Create a delegator instance with mock registry."""
        return Delegator(mock_registry, "test_delegator")
    
    @pytest.mark.asyncio
    async def test_delegate_success(self, delegator, sample_subagents, sample_task, sample_message):
        """Test successful complete delegation workflow."""
        # Setup mocks
        candidates = [(sample_subagents[0], 0.85)]
        delegator.retriever.retrieve_hybrid = AsyncMock(return_value=candidates)
        delegator.selector.select_subagents = AsyncMock(
            return_value=([sample_subagents[0]], "Selected Code Analyzer")
        )
        delegator.delegator.delegate_task = AsyncMock(
            return_value=[A2AMessage(
                id="msg_1", sender_id="test", recipient_id="subagent_1",
                message_type="task_request", payload={}
            )]
        )
        delegator.registry.mark_subagent_used = AsyncMock()
        
        # Create request
        request = DelegationRequest(
            task=sample_task,
            message=sample_message,
            max_subagents=2
        )
        
        # Test
        result = await delegator.delegate(request)
        
        # Assertions
        assert result.success is True
        assert len(result.selected_subagents) == 1
        assert result.selected_subagents[0].name == "Code Analyzer"
        assert "Selected Code Analyzer" in result.selection_reasoning
        assert len(result.delegation_messages) == 1
        assert result.error_message is None
        
        # Verify method calls
        delegator.retriever.retrieve_hybrid.assert_called_once()
        delegator.selector.select_subagents.assert_called_once()
        delegator.delegator.delegate_task.assert_called_once()
        delegator.registry.mark_subagent_used.assert_called_once_with("subagent_1")
    
    @pytest.mark.asyncio
    async def test_delegate_no_candidates(self, delegator, sample_task, sample_message):
        """Test delegation when no candidates are found."""
        # Setup mocks
        delegator.retriever.retrieve_hybrid = AsyncMock(return_value=[])
        
        # Create request
        request = DelegationRequest(task=sample_task, message=sample_message)
        
        # Test
        result = await delegator.delegate(request)
        
        # Assertions
        assert result.success is True
        assert len(result.selected_subagents) == 0
        assert "No suitable subagents found" in result.selection_reasoning
        assert len(result.delegation_messages) == 0
    
    @pytest.mark.asyncio
    async def test_delegate_no_selection(self, delegator, sample_subagents, sample_task, sample_message):
        """Test delegation when no subagents are selected."""
        # Setup mocks
        candidates = [(sample_subagents[0], 0.3)]  # Low score
        delegator.retriever.retrieve_hybrid = AsyncMock(return_value=candidates)
        delegator.selector.select_subagents = AsyncMock(
            return_value=([], "No suitable subagents with high enough scores")
        )
        
        # Create request
        request = DelegationRequest(task=sample_task, message=sample_message)
        
        # Test
        result = await delegator.delegate(request)
        
        # Assertions
        assert result.success is True
        assert len(result.selected_subagents) == 0
        assert "No suitable subagents with high enough scores" in result.selection_reasoning
        assert len(result.delegation_messages) == 0
    
    @pytest.mark.asyncio
    async def test_delegate_retrieval_error(self, delegator, sample_task, sample_message):
        """Test delegation when retrieval fails."""
        # Setup mocks
        delegator.retriever.retrieve_hybrid = AsyncMock(
            side_effect=DelegatorError("Retrieval failed")
        )
        
        # Create request
        request = DelegationRequest(task=sample_task, message=sample_message)
        
        # Test
        result = await delegator.delegate(request)
        
        # Assertions
        assert result.success is False
        assert "Retrieval failed" in result.error_message
        assert len(result.selected_subagents) == 0
    
    @pytest.mark.asyncio
    async def test_delegate_selection_error(self, delegator, sample_subagents, sample_task, sample_message):
        """Test delegation when selection fails."""
        # Setup mocks
        candidates = [(sample_subagents[0], 0.85)]
        delegator.retriever.retrieve_hybrid = AsyncMock(return_value=candidates)
        delegator.selector.select_subagents = AsyncMock(
            side_effect=SubagentSelectionError("Selection failed")
        )
        
        # Create request
        request = DelegationRequest(task=sample_task, message=sample_message)
        
        # Test
        result = await delegator.delegate(request)
        
        # Assertions
        assert result.success is False
        assert "Selection failed" in result.error_message
        assert len(result.selected_subagents) == 0
    
    @pytest.mark.asyncio
    async def test_delegate_delegation_error(self, delegator, sample_subagents, sample_task, sample_message):
        """Test delegation when task delegation fails."""
        # Setup mocks
        candidates = [(sample_subagents[0], 0.85)]
        delegator.retriever.retrieve_hybrid = AsyncMock(return_value=candidates)
        delegator.selector.select_subagents = AsyncMock(
            return_value=([sample_subagents[0]], "Selected subagent")
        )
        delegator.delegator.delegate_task = AsyncMock(
            side_effect=TaskDelegationError("Delegation failed")
        )
        
        # Create request
        request = DelegationRequest(task=sample_task, message=sample_message)
        
        # Test
        result = await delegator.delegate(request)
        
        # Assertions
        assert result.success is False
        assert "Delegation failed" in result.error_message
        assert len(result.selected_subagents) == 1  # Selection succeeded
    
    @pytest.mark.asyncio
    async def test_delegate_with_exclusions(self, delegator, sample_subagents, sample_task, sample_message):
        """Test delegation with excluded subagents."""
        # Setup mocks
        candidates = [(sample_subagents[0], 0.85), (sample_subagents[1], 0.75)]
        delegator.retriever.retrieve_hybrid = AsyncMock(return_value=candidates)
        
        # Create request with exclusions
        request = DelegationRequest(
            task=sample_task,
            message=sample_message,
            exclude_subagents=["subagent_1"]
        )
        
        # Test
        await delegator.delegate(request)
        
        # Verify retrieval was called
        delegator.retriever.retrieve_hybrid.assert_called_once()
        call_args = delegator.retriever.retrieve_hybrid.call_args
        
        # The exclusion filtering happens in _retrieve_candidates
        # We can't easily test the filtering without mocking _retrieve_candidates
        # But we can verify the method was called with correct parameters
        assert call_args[1]["query_text"] == f"{sample_task.description} {sample_message.content}"
    
    @pytest.mark.asyncio
    async def test_get_delegation_stats(self, delegator):
        """Test getting delegation statistics."""
        # Setup mock
        delegator.registry.get_registry_stats = AsyncMock(
            return_value={"total_subagents": 5, "active_subagents": 3}
        )
        
        # Test
        stats = await delegator.get_delegation_stats()
        
        # Assertions
        assert stats["delegator_id"] == "test_delegator"
        assert stats["registry_stats"]["total_subagents"] == 5
        assert stats["registry_stats"]["active_subagents"] == 3
        assert "timestamp" in stats
    
    @pytest.mark.asyncio
    async def test_get_delegation_stats_error(self, delegator):
        """Test getting delegation statistics when registry fails."""
        # Setup mock to raise error
        delegator.registry.get_registry_stats = AsyncMock(
            side_effect=Exception("Registry error")
        )
        
        # Test
        stats = await delegator.get_delegation_stats()
        
        # Assertions
        assert "error" in stats
        assert "Registry error" in stats["error"]


class TestDelegationRequest:
    """Test cases for DelegationRequest dataclass."""
    
    def test_delegation_request_defaults(self, sample_task, sample_message):
        """Test delegation request with default values."""
        request = DelegationRequest(task=sample_task, message=sample_message)
        
        assert request.task == sample_task
        assert request.message == sample_message
        assert request.max_subagents == 3
        assert request.similarity_threshold == 0.7
        assert request.require_capabilities is None
        assert request.exclude_subagents is None
    
    def test_delegation_request_custom_values(self, sample_task, sample_message):
        """Test delegation request with custom values."""
        request = DelegationRequest(
            task=sample_task,
            message=sample_message,
            max_subagents=5,
            similarity_threshold=0.8,
            require_capabilities=["testing"],
            exclude_subagents=["subagent_1"]
        )
        
        assert request.max_subagents == 5
        assert request.similarity_threshold == 0.8
        assert request.require_capabilities == ["testing"]
        assert request.exclude_subagents == ["subagent_1"]


class TestDelegationResult:
    """Test cases for DelegationResult dataclass."""
    
    def test_delegation_result_defaults(self):
        """Test delegation result with default values."""
        result = DelegationResult(task_id="test_task")
        
        assert result.task_id == "test_task"
        assert result.selected_subagents == []
        assert result.delegation_messages == []
        assert result.selection_reasoning == ""
        assert result.success is False
        assert result.error_message is None
        assert isinstance(result.timestamp, datetime)
    
    def test_delegation_result_custom_values(self, sample_subagents):
        """Test delegation result with custom values."""
        timestamp = datetime.now(timezone.utc)
        result = DelegationResult(
            task_id="test_task",
            selected_subagents=[sample_subagents[0]],
            delegation_messages=[{"test": "message"}],
            selection_reasoning="Test reasoning",
            success=True,
            error_message="Test error",
            timestamp=timestamp
        )
        
        assert len(result.selected_subagents) == 1
        assert result.delegation_messages == [{"test": "message"}]
        assert result.selection_reasoning == "Test reasoning"
        assert result.success is True
        assert result.error_message == "Test error"
        assert result.timestamp == timestamp


if __name__ == "__main__":
    pytest.main([__file__])
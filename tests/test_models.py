"""
Unit tests for core data models and API validation.

Tests cover data model validation, serialization, and edge cases
for Message, Subagent, and Task models.
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from backend.models.core import (
    Message, 
    Subagent, 
    Task, 
    MessageSource, 
    MessageClassification, 
    TaskStatus
)
from backend.models.api import (
    MessageRequest,
    MessageResponse,
    SubagentRequest,
    SubagentResponse,
    TaskRequest,
    TaskResponse,
    AgentStatusResponse,
    ErrorResponse,
    HealthResponse,
)
from pydantic import ValidationError


class TestMessage:
    """Test cases for Message dataclass."""
    
    def test_message_creation_with_defaults(self):
        """Test creating a message with default values."""
        message = Message(content="Hello world")
        
        assert message.content == "Hello world"
        assert message.source == MessageSource.USER
        assert message.priority == 0
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
        assert message.classification is None
        assert message.delay_seconds is None
        assert len(message.id) > 0
    
    def test_message_creation_with_all_fields(self):
        """Test creating a message with all fields specified."""
        timestamp = datetime.now(timezone.utc)
        metadata = {"key": "value"}
        
        message = Message(
            id="test-id",
            content="Test message",
            source=MessageSource.SUBAGENT,
            priority=5,
            timestamp=timestamp,
            metadata=metadata,
            classification=MessageClassification.ACT_NOW,
            delay_seconds=30
        )
        
        assert message.id == "test-id"
        assert message.content == "Test message"
        assert message.source == MessageSource.SUBAGENT
        assert message.priority == 5
        assert message.timestamp == timestamp
        assert message.metadata == metadata
        assert message.classification == MessageClassification.ACT_NOW
        assert message.delay_seconds == 30
    
    def test_message_validation_empty_content(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(content="")
        
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(content="   ")
    
    def test_message_validation_negative_priority(self):
        """Test that negative priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be non-negative"):
            Message(content="Test", priority=-1)
    
    def test_message_validation_negative_delay(self):
        """Test that negative delay_seconds raises ValueError."""
        with pytest.raises(ValueError, match="Delay seconds must be non-negative"):
            Message(content="Test", delay_seconds=-1)


class TestSubagent:
    """Test cases for Subagent dataclass."""
    
    def test_subagent_creation_with_defaults(self):
        """Test creating a subagent with required fields."""
        subagent = Subagent(
            name="Test Agent",
            description="A test subagent",
            import_path="test.module"
        )
        
        assert subagent.name == "Test Agent"
        assert subagent.description == "A test subagent"
        assert subagent.import_path == "test.module"
        assert subagent.input_schema == {}
        assert subagent.output_schema == {}
        assert subagent.embedding == []
        assert subagent.capabilities == []
        assert isinstance(subagent.created_at, datetime)
        assert subagent.last_used is None
        assert subagent.is_active is True
        assert len(subagent.id) > 0
    
    def test_subagent_creation_with_all_fields(self):
        """Test creating a subagent with all fields specified."""
        created_at = datetime.now(timezone.utc)
        last_used = datetime.now(timezone.utc)
        embedding = [0.1] * 1536  # Valid OpenAI embedding
        
        subagent = Subagent(
            id="test-id",
            name="Full Agent",
            description="A complete subagent",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            import_path="full.module",
            embedding=embedding,
            capabilities=["chat", "analysis"],
            created_at=created_at,
            last_used=last_used,
            is_active=False
        )
        
        assert subagent.id == "test-id"
        assert subagent.name == "Full Agent"
        assert subagent.description == "A complete subagent"
        assert subagent.input_schema == {"type": "object"}
        assert subagent.output_schema == {"type": "string"}
        assert subagent.import_path == "full.module"
        assert subagent.embedding == embedding
        assert subagent.capabilities == ["chat", "analysis"]
        assert subagent.created_at == created_at
        assert subagent.last_used == last_used
        assert subagent.is_active is False
    
    def test_subagent_validation_empty_fields(self):
        """Test that empty required fields raise ValueError."""
        with pytest.raises(ValueError, match="Subagent name cannot be empty"):
            Subagent(name="", description="Test", import_path="test.module")
        
        with pytest.raises(ValueError, match="Subagent description cannot be empty"):
            Subagent(name="Test", description="", import_path="test.module")
        
        with pytest.raises(ValueError, match="Import path cannot be empty"):
            Subagent(name="Test", description="Test", import_path="")
    
    def test_subagent_validation_invalid_embedding(self):
        """Test that invalid embedding dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Embedding must be 1536 dimensions"):
            Subagent(
                name="Test",
                description="Test",
                import_path="test.module",
                embedding=[0.1] * 100  # Wrong dimension
            )
    
    def test_subagent_mark_used(self):
        """Test marking subagent as used updates timestamp."""
        subagent = Subagent(
            name="Test",
            description="Test",
            import_path="test.module"
        )
        
        assert subagent.last_used is None
        
        subagent.mark_used()
        
        assert subagent.last_used is not None
        assert isinstance(subagent.last_used, datetime)


class TestTask:
    """Test cases for Task dataclass."""
    
    def test_task_creation_with_defaults(self):
        """Test creating a task with default values."""
        task = Task(description="Test task")
        
        assert task.description == "Test task"
        assert task.priority == 0
        assert task.status == TaskStatus.PENDING
        assert task.assigned_subagents == []
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
        assert task.deadline is None
        assert task.parent_task_id is None
        assert task.result is None
        assert task.error_message is None
        assert len(task.id) > 0
    
    def test_task_creation_with_all_fields(self):
        """Test creating a task with all fields specified."""
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        deadline = datetime.now(timezone.utc) + timedelta(days=1)
        result = {"output": "success"}
        
        task = Task(
            id="test-id",
            description="Complete task",
            priority=10,
            status=TaskStatus.COMPLETED,
            assigned_subagents=["agent1", "agent2"],
            created_at=created_at,
            updated_at=updated_at,
            deadline=deadline,
            parent_task_id="parent-id",
            result=result,
            error_message="No errors"
        )
        
        assert task.id == "test-id"
        assert task.description == "Complete task"
        assert task.priority == 10
        assert task.status == TaskStatus.COMPLETED
        assert task.assigned_subagents == ["agent1", "agent2"]
        assert task.created_at == created_at
        assert task.updated_at == updated_at
        assert task.deadline == deadline
        assert task.parent_task_id == "parent-id"
        assert task.result == result
        assert task.error_message == "No errors"
    
    def test_task_validation_empty_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="Task description cannot be empty"):
            Task(description="")
        
        with pytest.raises(ValueError, match="Task description cannot be empty"):
            Task(description="   ")
    
    def test_task_validation_negative_priority(self):
        """Test that negative priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be non-negative"):
            Task(description="Test", priority=-1)
    
    def test_task_validation_past_deadline(self):
        """Test that past deadline raises ValueError."""
        past_deadline = datetime.now(timezone.utc) - timedelta(days=1)
        with pytest.raises(ValueError, match="Deadline must be in the future"):
            Task(description="Test", deadline=past_deadline)
    
    def test_task_update_status(self):
        """Test updating task status."""
        task = Task(description="Test task")
        original_updated_at = task.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        task.update_status(TaskStatus.IN_PROGRESS)
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.updated_at > original_updated_at
        assert task.error_message is None
        
        task.update_status(TaskStatus.FAILED, "Test error")
        
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Test error"
    
    def test_task_assign_subagent(self):
        """Test assigning subagents to task."""
        task = Task(description="Test task")
        original_updated_at = task.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        task.assign_subagent("agent1")
        
        assert "agent1" in task.assigned_subagents
        assert task.updated_at > original_updated_at
        
        # Assigning same agent again should not duplicate
        task.assign_subagent("agent1")
        assert task.assigned_subagents.count("agent1") == 1
        
        task.assign_subagent("agent2")
        assert "agent2" in task.assigned_subagents
        assert len(task.assigned_subagents) == 2
    
    def test_task_is_overdue(self):
        """Test checking if task is overdue."""
        # Task without deadline is not overdue
        task = Task(description="Test task")
        assert not task.is_overdue()
        
        # Task with future deadline is not overdue
        future_deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        task.deadline = future_deadline
        assert not task.is_overdue()
        
        # Task with past deadline is overdue
        past_deadline = datetime.now(timezone.utc) - timedelta(hours=1)
        task.deadline = past_deadline
        assert task.is_overdue()


class TestPydanticModels:
    """Test cases for Pydantic API models."""
    
    def test_message_request_validation(self):
        """Test MessageRequest validation."""
        # Valid request
        request = MessageRequest(content="Hello")
        assert request.content == "Hello"
        assert request.source == MessageSource.USER
        assert request.priority == 0
        assert request.metadata == {}
        
        # Invalid empty content
        with pytest.raises(ValidationError):
            MessageRequest(content="")
        
        with pytest.raises(ValidationError):
            MessageRequest(content="   ")
        
        # Invalid negative priority
        with pytest.raises(ValidationError):
            MessageRequest(content="Hello", priority=-1)
    
    def test_subagent_request_validation(self):
        """Test SubagentRequest validation."""
        # Valid request
        request = SubagentRequest(
            name="Test Agent",
            description="A test agent",
            import_path="test.module"
        )
        assert request.name == "Test Agent"
        assert request.description == "A test agent"
        assert request.import_path == "test.module"
        assert request.is_active is True
        
        # Invalid empty fields
        with pytest.raises(ValidationError):
            SubagentRequest(name="", description="Test", import_path="test.module")
        
        with pytest.raises(ValidationError):
            SubagentRequest(name="Test", description="", import_path="test.module")
        
        with pytest.raises(ValidationError):
            SubagentRequest(name="Test", description="Test", import_path="")
    
    def test_task_request_validation(self):
        """Test TaskRequest validation."""
        # Valid request
        request = TaskRequest(description="Test task")
        assert request.description == "Test task"
        assert request.priority == 0
        assert request.deadline is None
        assert request.parent_task_id is None
        
        # Invalid empty description
        with pytest.raises(ValidationError):
            TaskRequest(description="")
        
        # Invalid negative priority
        with pytest.raises(ValidationError):
            TaskRequest(description="Test", priority=-1)
        
        # Invalid past deadline
        past_deadline = datetime.now(timezone.utc) - timedelta(days=1)
        with pytest.raises(ValidationError):
            TaskRequest(description="Test", deadline=past_deadline)
    
    def test_response_model_serialization(self):
        """Test that response models serialize correctly."""
        # Create core models
        message = Message(content="Test message")
        subagent = Subagent(
            name="Test Agent",
            description="Test description",
            import_path="test.module"
        )
        task = Task(description="Test task")
        
        # Create response models
        message_response = MessageResponse(
            id=message.id,
            content=message.content,
            source=message.source,
            priority=message.priority,
            timestamp=message.timestamp,
            metadata=message.metadata,
            classification=message.classification,
            delay_seconds=message.delay_seconds
        )
        
        subagent_response = SubagentResponse(
            id=subagent.id,
            name=subagent.name,
            description=subagent.description,
            input_schema=subagent.input_schema,
            output_schema=subagent.output_schema,
            import_path=subagent.import_path,
            capabilities=subagent.capabilities,
            created_at=subagent.created_at,
            last_used=subagent.last_used,
            is_active=subagent.is_active
        )
        
        task_response = TaskResponse(
            id=task.id,
            description=task.description,
            priority=task.priority,
            status=task.status,
            assigned_subagents=task.assigned_subagents,
            created_at=task.created_at,
            updated_at=task.updated_at,
            deadline=task.deadline,
            parent_task_id=task.parent_task_id,
            result=task.result,
            error_message=task.error_message
        )
        
        # Test JSON serialization
        message_json = message_response.model_dump_json()
        subagent_json = subagent_response.model_dump_json()
        task_json = task_response.model_dump_json()
        
        # Verify JSON is valid
        assert json.loads(message_json)
        assert json.loads(subagent_json)
        assert json.loads(task_json)
        
        # Verify enum values are serialized as strings
        message_data = json.loads(message_json)
        assert message_data["source"] == "user"
        
        task_data = json.loads(task_json)
        assert task_data["status"] == "pending"
    
    def test_agent_status_response(self):
        """Test AgentStatusResponse model."""
        status_response = AgentStatusResponse(
            is_active=True,
            current_tasks=[],
            message_queue_size=5,
            active_subagents=3,
            memory_usage={"ram": "256MB"},
            uptime_seconds=3600,
            last_activity=datetime.now(timezone.utc)
        )
        
        assert status_response.is_active is True
        assert status_response.current_tasks == []
        assert status_response.message_queue_size == 5
        assert status_response.active_subagents == 3
        assert status_response.memory_usage == {"ram": "256MB"}
        assert status_response.uptime_seconds == 3600
        assert isinstance(status_response.last_activity, datetime)
    
    def test_error_response(self):
        """Test ErrorResponse model."""
        error_response = ErrorResponse(
            error="ValidationError",
            message="Invalid input data",
            details={"field": "content", "issue": "empty"}
        )
        
        assert error_response.error == "ValidationError"
        assert error_response.message == "Invalid input data"
        assert error_response.details == {"field": "content", "issue": "empty"}
        assert isinstance(error_response.timestamp, datetime)
    
    def test_health_response(self):
        """Test HealthResponse model."""
        health_response = HealthResponse(
            status="healthy",
            version="0.1.0",
            components={"database": "ok", "queue": "ok"}
        )
        
        assert health_response.status == "healthy"
        assert health_response.version == "0.1.0"
        assert health_response.components == {"database": "ok", "queue": "ok"}
        assert isinstance(health_response.timestamp, datetime)
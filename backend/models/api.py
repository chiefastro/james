"""
Pydantic models for API request/response validation.

This module contains Pydantic models that provide validation and serialization
for API endpoints, ensuring type safety and proper data handling.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from .core import MessageClassification, MessageSource, TaskStatus


class MessageRequest(BaseModel):
    """Request model for creating new messages."""
    content: str = Field(..., min_length=1, description="Message content")
    source: MessageSource = Field(default=MessageSource.USER, description="Message source")
    priority: int = Field(default=0, ge=0, description="Message priority (0 or higher)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class MessageResponse(BaseModel):
    """Response model for message data."""
    id: str
    content: str
    source: MessageSource
    priority: int
    timestamp: datetime
    metadata: Dict[str, Any]
    classification: Optional[MessageClassification] = None
    delay_seconds: Optional[int] = None
    
    model_config = ConfigDict(use_enum_values=True)


class SubagentRequest(BaseModel):
    """Request model for creating/updating subagents."""
    name: str = Field(..., min_length=1, description="Subagent name")
    description: str = Field(..., min_length=1, description="Subagent description")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    import_path: str = Field(..., min_length=1, description="Python import path")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities")
    is_active: bool = Field(default=True, description="Whether subagent is active")
    
    @field_validator('name', 'description', 'import_path')
    @classmethod
    def strings_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Field cannot be empty or whitespace only')
        return v.strip()


class SubagentResponse(BaseModel):
    """Response model for subagent data."""
    id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    import_path: str
    capabilities: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool
    
    model_config = ConfigDict(use_enum_values=True)


class TaskRequest(BaseModel):
    """Request model for creating new tasks."""
    description: str = Field(..., min_length=1, description="Task description")
    priority: int = Field(default=0, ge=0, description="Task priority (0 or higher)")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    parent_task_id: Optional[str] = Field(None, description="Parent task ID for subtasks")
    
    @field_validator('description')
    @classmethod
    def description_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Description cannot be empty or whitespace only')
        return v.strip()
    
    @field_validator('deadline')
    @classmethod
    def deadline_must_be_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and v <= datetime.now(timezone.utc):
            raise ValueError('Deadline must be in the future')
        return v


class TaskResponse(BaseModel):
    """Response model for task data."""
    id: str
    description: str
    priority: int
    status: TaskStatus
    assigned_subagents: List[str]
    created_at: datetime
    updated_at: datetime
    deadline: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    model_config = ConfigDict(use_enum_values=True)


class AgentStatusResponse(BaseModel):
    """Response model for overall agent status."""
    is_active: bool = Field(description="Whether the agent is currently active")
    current_tasks: List[TaskResponse] = Field(description="Currently active tasks")
    message_queue_size: int = Field(description="Number of messages in queue")
    active_subagents: int = Field(description="Number of active subagents")
    memory_usage: Dict[str, Any] = Field(description="Memory usage statistics")
    uptime_seconds: int = Field(description="System uptime in seconds")
    last_activity: Optional[datetime] = Field(description="Last activity timestamp")
    
    model_config = ConfigDict(use_enum_values=True)


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Check timestamp")
    version: str = Field(description="API version")
    components: Dict[str, str] = Field(description="Component health status")


class AgentCreationRequest(BaseModel):
    """Request model for creating new agent instances."""
    agent_type: str = Field(..., description="Type of agent to create (observer, delegator, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    
    @field_validator('agent_type')
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        valid_types = ["observer", "delegator"]
        if v not in valid_types:
            raise ValueError(f"Agent type must be one of: {', '.join(valid_types)}")
        return v


class AgentCreationResponse(BaseModel):
    """Response model for agent creation."""
    agent_id: str = Field(..., description="Unique ID of the created agent")
    agent_type: str = Field(..., description="Type of agent created")
    status: str = Field(..., description="Status of agent creation")
"""
Core data models for the Conscious Agent System.

This module contains the fundamental data structures used throughout the system,
including Message, Subagent, and Task models with proper type hints.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MessageSource(Enum):
    """Source of incoming messages."""
    USER = "user"
    SUBAGENT = "subagent"
    SYSTEM = "system"
    EXTERNAL = "external"


class MessageClassification(Enum):
    """Classification of messages by the Observer agent."""
    IGNORE_DELETE = "ignore_delete"
    DELAY = "delay"
    ARCHIVE = "archive"
    ACT_NOW = "act_now"


class TaskStatus(Enum):
    """Status of tasks in the system."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Message:
    """
    Core message model for all communications in the system.
    
    Represents messages from users, subagents, and system components
    with priority-based processing capabilities.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    source: MessageSource = MessageSource.USER
    priority: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    classification: Optional[MessageClassification] = None
    delay_seconds: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate message data after initialization."""
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.delay_seconds is not None and self.delay_seconds < 0:
            raise ValueError("Delay seconds must be non-negative")


@dataclass
class Subagent:
    """
    Subagent model representing specialized agents in the system.
    
    Contains metadata about capabilities, schemas, and usage patterns
    for dynamic discovery and delegation.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    import_path: str = ""
    embedding: List[float] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self) -> None:
        """Validate subagent data after initialization."""
        if not self.name.strip():
            raise ValueError("Subagent name cannot be empty")
        if not self.description.strip():
            raise ValueError("Subagent description cannot be empty")
        if not self.import_path.strip():
            raise ValueError("Import path cannot be empty")
        if self.embedding and len(self.embedding) != 1536:  # OpenAI embedding dimension
            raise ValueError("Embedding must be 1536 dimensions for OpenAI compatibility")
    
    def mark_used(self) -> None:
        """Mark the subagent as recently used."""
        self.last_used = datetime.now(timezone.utc)


@dataclass
class Task:
    """
    Task model for tracking work items in the system.
    
    Supports hierarchical task structures with priority-based scheduling
    and subagent assignment tracking.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_subagents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate task data after initialization."""
        if not self.description.strip():
            raise ValueError("Task description cannot be empty")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.deadline and self.deadline <= self.created_at:
            raise ValueError("Deadline must be in the future")
    
    def update_status(self, status: TaskStatus, error_message: Optional[str] = None) -> None:
        """Update task status with timestamp."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)
        if error_message:
            self.error_message = error_message
    
    def assign_subagent(self, subagent_id: str) -> None:
        """Assign a subagent to this task."""
        if subagent_id not in self.assigned_subagents:
            self.assigned_subagents.append(subagent_id)
            self.updated_at = datetime.now(timezone.utc)
    
    def is_overdue(self) -> bool:
        """Check if the task is past its deadline."""
        return self.deadline is not None and datetime.now(timezone.utc) > self.deadline
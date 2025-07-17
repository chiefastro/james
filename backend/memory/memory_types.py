"""Memory types and data structures for the Mem0 integration."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MemoryType(Enum):
    """Types of memory supported by the system."""
    EPISODIC = "episodic"      # Specific experiences and interactions
    SEMANTIC = "semantic"      # General knowledge and learned concepts  
    PROCEDURAL = "procedural"  # Skills and capabilities
    WORKING = "working"        # Current context and active information


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the system."""
    id: str
    content: str
    memory_type: MemoryType
    metadata: Dict[str, Any]
    timestamp: datetime
    importance_score: float = 0.5  # 0.0 to 1.0, higher = more important
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass 
class MemoryQuery:
    """Query structure for memory retrieval."""
    query_text: str
    memory_types: List[MemoryType] = None
    limit: int = 10
    min_importance: float = 0.0
    tags: List[str] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    
    def __post_init__(self):
        if self.memory_types is None:
            self.memory_types = list(MemoryType)
        if self.tags is None:
            self.tags = []


@dataclass
class MemorySearchResult:
    """Result from memory search operations."""
    entries: List[MemoryEntry]
    total_count: int
    search_time_ms: float
    query: MemoryQuery
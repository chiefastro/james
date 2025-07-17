"""Memory cleanup strategies for managing memory storage limits."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Protocol
from .memory_types import MemoryEntry, MemoryType
import logging

logger = logging.getLogger(__name__)


class MemoryCleanupStrategy(ABC):
    """Abstract base class for memory cleanup strategies."""
    
    @abstractmethod
    async def should_cleanup(self, total_memories: int, memory_size_mb: float) -> bool:
        """Determine if cleanup should be triggered."""
        pass
    
    @abstractmethod
    async def select_memories_for_cleanup(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Select which memories should be cleaned up."""
        pass


class LRUCleanupStrategy(MemoryCleanupStrategy):
    """Least Recently Used cleanup strategy."""
    
    def __init__(self, max_memories: int = 10000, max_size_mb: float = 500.0):
        self.max_memories = max_memories
        self.max_size_mb = max_size_mb
    
    async def should_cleanup(self, total_memories: int, memory_size_mb: float) -> bool:
        """Trigger cleanup if limits are exceeded."""
        return total_memories > self.max_memories or memory_size_mb > self.max_size_mb
    
    async def select_memories_for_cleanup(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Select least recently used memories for cleanup."""
        # Sort by last_accessed (None values go first, then oldest first)
        sorted_memories = sorted(
            memories,
            key=lambda m: m.last_accessed or datetime.min
        )
        
        # Remove 20% of memories to avoid frequent cleanup
        cleanup_count = max(1, len(memories) // 5)
        return sorted_memories[:cleanup_count]


class ImportanceBasedCleanupStrategy(MemoryCleanupStrategy):
    """Cleanup strategy based on importance scores."""
    
    def __init__(self, max_memories: int = 10000, max_size_mb: float = 500.0, min_importance_threshold: float = 0.3):
        self.max_memories = max_memories
        self.max_size_mb = max_size_mb
        self.min_importance_threshold = min_importance_threshold
    
    async def should_cleanup(self, total_memories: int, memory_size_mb: float) -> bool:
        """Trigger cleanup if limits are exceeded."""
        return total_memories > self.max_memories or memory_size_mb > self.max_size_mb
    
    async def select_memories_for_cleanup(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Select low-importance memories for cleanup."""
        # First, select memories below importance threshold
        low_importance = [m for m in memories if m.importance_score < self.min_importance_threshold]
        
        if len(low_importance) > 0:
            return low_importance
        
        # If no low-importance memories, fall back to LRU for lowest importance
        sorted_memories = sorted(memories, key=lambda m: m.importance_score)
        cleanup_count = max(1, len(memories) // 5)
        return sorted_memories[:cleanup_count]


class HybridCleanupStrategy(MemoryCleanupStrategy):
    """Hybrid strategy combining age, importance, and access patterns."""
    
    def __init__(self, max_memories: int = 10000, max_size_mb: float = 500.0):
        self.max_memories = max_memories
        self.max_size_mb = max_size_mb
    
    async def should_cleanup(self, total_memories: int, memory_size_mb: float) -> bool:
        """Trigger cleanup if limits are exceeded."""
        return total_memories > self.max_memories or memory_size_mb > self.max_size_mb
    
    async def select_memories_for_cleanup(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Select memories using hybrid scoring."""
        now = datetime.now()
        
        def calculate_cleanup_score(memory: MemoryEntry) -> float:
            """Calculate cleanup score (higher = more likely to be cleaned up)."""
            score = 0.0
            
            # Age factor (older = higher cleanup score)
            age_days = (now - memory.timestamp).days
            age_score = min(age_days / 365.0, 1.0)  # Normalize to 0-1 over a year
            score += age_score * 0.4
            
            # Importance factor (lower importance = higher cleanup score)
            importance_score = 1.0 - memory.importance_score
            score += importance_score * 0.4
            
            # Access pattern factor (less accessed = higher cleanup score)
            if memory.last_accessed:
                days_since_access = (now - memory.last_accessed).days
                access_score = min(days_since_access / 90.0, 1.0)  # Normalize over 3 months
            else:
                access_score = 1.0  # Never accessed
            score += access_score * 0.2
            
            return score
        
        # Calculate cleanup scores and sort
        memory_scores = [(m, calculate_cleanup_score(m)) for m in memories]
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove 20% of memories
        cleanup_count = max(1, len(memories) // 5)
        return [m for m, _ in memory_scores[:cleanup_count]]


class TypeBasedCleanupStrategy(MemoryCleanupStrategy):
    """Cleanup strategy that preserves certain memory types."""
    
    def __init__(self, max_memories: int = 10000, max_size_mb: float = 500.0):
        self.max_memories = max_memories
        self.max_size_mb = max_size_mb
        # Priority order for cleanup (higher index = more likely to be cleaned)
        self.cleanup_priority = [
            MemoryType.WORKING,     # Most expendable
            MemoryType.EPISODIC,    # Specific experiences
            MemoryType.SEMANTIC,    # General knowledge
            MemoryType.PROCEDURAL   # Skills - least expendable
        ]
    
    async def should_cleanup(self, total_memories: int, memory_size_mb: float) -> bool:
        """Trigger cleanup if limits are exceeded."""
        return total_memories > self.max_memories or memory_size_mb > self.max_size_mb
    
    async def select_memories_for_cleanup(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Select memories prioritizing by type."""
        cleanup_candidates = []
        target_cleanup = max(1, len(memories) // 5)
        
        # Go through types in cleanup priority order
        for memory_type in self.cleanup_priority:
            type_memories = [m for m in memories if m.memory_type == memory_type]
            
            if len(cleanup_candidates) >= target_cleanup:
                break
                
            # Within each type, use hybrid scoring
            hybrid_strategy = HybridCleanupStrategy()
            type_cleanup = await hybrid_strategy.select_memories_for_cleanup(type_memories)
            
            remaining_needed = target_cleanup - len(cleanup_candidates)
            cleanup_candidates.extend(type_cleanup[:remaining_needed])
        
        return cleanup_candidates
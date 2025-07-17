"""Main memory manager using Mem0 for intelligent memory storage and retrieval."""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from mem0 import Memory

from .cleanup_strategies import MemoryCleanupStrategy, HybridCleanupStrategy
from .memory_types import MemoryEntry, MemoryQuery, MemorySearchResult, MemoryType

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages episodic, semantic, and procedural memory using Mem0."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cleanup_strategy: Optional[MemoryCleanupStrategy] = None
    ):
        """Initialize the memory manager.
        
        Args:
            config: Mem0 configuration dictionary
            cleanup_strategy: Strategy for memory cleanup
        """
        self.config = config or self._get_default_config()
        self.cleanup_strategy = cleanup_strategy or HybridCleanupStrategy()
        self._memory_client = None
        self._user_id = "james_conscious_agent"
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Mem0 configuration."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                    "collection_name": "james_memories"
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            },
            "custom_fact_extraction_prompt": None,
            "fact_retrieval_config": {
                "limit": 5
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the Mem0 client and setup collections."""
        try:
            # Initialize Mem0 with the correct approach
            from mem0 import Memory
            self._memory_client = Memory.from_config(self.config)
            
            logger.info("Memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> MemoryEntry:
        """Store a new memory entry.
        
        Args:
            content: The memory content to store
            memory_type: Type of memory (episodic, semantic, procedural, working)
            metadata: Additional metadata for the memory
            importance_score: Importance score (0.0 to 1.0)
            tags: List of tags for categorization
            
        Returns:
            MemoryEntry: The stored memory entry
        """
        if not self._memory_client:
            await self.initialize()
        
        # Create memory entry
        memory_entry = MemoryEntry(
            id="",  # Will be set by Mem0
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            timestamp=datetime.now(),
            importance_score=importance_score,
            tags=tags or []
        )
        
        # Prepare metadata for Mem0
        mem0_metadata = {
            "memory_type": memory_type.value,
            "importance_score": importance_score,
            "tags": tags or [],
            "timestamp": memory_entry.timestamp.isoformat(),
            **memory_entry.metadata
        }
        
        try:
            # Store in Mem0
            result = self._memory_client.add(
                messages=[{"role": "user", "content": content}],
                user_id=self._user_id,
                metadata=mem0_metadata
            )
            
            # Update memory entry with Mem0 ID
            if result and len(result) > 0:
                memory_entry.id = result[0].get("id", memory_entry.id)
            
            logger.info(f"Stored {memory_type.value} memory: {memory_entry.id}")
            
            # Check if cleanup is needed
            await self._check_and_cleanup()
            
            return memory_entry
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Retrieve memories based on query.
        
        Args:
            query: Memory query parameters
            
        Returns:
            MemorySearchResult: Search results with matching memories
        """
        if not self._memory_client:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Build search filters
            filters = {}
            if query.memory_types and len(query.memory_types) < len(MemoryType):
                filters["memory_type"] = [mt.value for mt in query.memory_types]
            
            if query.min_importance > 0.0:
                filters["importance_score"] = {"$gte": query.min_importance}
            
            if query.tags:
                filters["tags"] = {"$in": query.tags}
            
            # Search in Mem0
            results = self._memory_client.search(
                query=query.query_text,
                user_id=self._user_id,
                limit=query.limit,
                filters=filters if filters else None
            )
            
            # Convert results to MemoryEntry objects
            entries = []
            for result in results:
                memory_data = result.get("memory", {})
                metadata = result.get("metadata", {})
                
                entry = MemoryEntry(
                    id=result.get("id", ""),
                    content=memory_data.get("data", ""),
                    memory_type=MemoryType(metadata.get("memory_type", "working")),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["memory_type", "importance_score", "tags", "timestamp"]},
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                    importance_score=metadata.get("importance_score", 0.5),
                    tags=metadata.get("tags", [])
                )
                entries.append(entry)
            
            search_time = (time.time() - start_time) * 1000
            
            return MemorySearchResult(
                entries=entries,
                total_count=len(entries),
                search_time_ms=search_time,
                query=query
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update an existing memory entry.
        
        Args:
            memory_id: ID of the memory to update
            content: New content (if provided)
            metadata: New metadata (if provided)
            importance_score: New importance score (if provided)
            tags: New tags (if provided)
            
        Returns:
            bool: True if update was successful
        """
        if not self._memory_client:
            await self.initialize()
        
        try:
            # Get existing memory
            existing = self._memory_client.get(memory_id, user_id=self._user_id)
            if not existing:
                logger.warning(f"Memory {memory_id} not found for update")
                return False
            
            # Prepare update data
            update_data = {}
            if content is not None:
                update_data["data"] = content
            
            if any([metadata, importance_score is not None, tags is not None]):
                existing_metadata = existing.get("metadata", {})
                if metadata:
                    existing_metadata.update(metadata)
                if importance_score is not None:
                    existing_metadata["importance_score"] = importance_score
                if tags is not None:
                    existing_metadata["tags"] = tags
                update_data["metadata"] = existing_metadata
            
            # Update in Mem0
            self._memory_client.update(memory_id, data=update_data, user_id=self._user_id)
            logger.info(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if deletion was successful
        """
        if not self._memory_client:
            await self.initialize()
        
        try:
            self._memory_client.delete(memory_id, user_id=self._user_id)
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories.
        
        Returns:
            Dict with memory statistics
        """
        if not self._memory_client:
            await self.initialize()
        
        try:
            # Get all memories for stats
            all_memories = self._memory_client.get_all(user_id=self._user_id)
            
            stats = {
                "total_memories": len(all_memories),
                "memory_types": {},
                "avg_importance": 0.0,
                "oldest_memory": None,
                "newest_memory": None
            }
            
            if all_memories:
                importance_scores = []
                timestamps = []
                
                for memory in all_memories:
                    metadata = memory.get("metadata", {})
                    memory_type = metadata.get("memory_type", "unknown")
                    
                    # Count by type
                    stats["memory_types"][memory_type] = stats["memory_types"].get(memory_type, 0) + 1
                    
                    # Collect importance scores
                    importance = metadata.get("importance_score", 0.5)
                    importance_scores.append(importance)
                    
                    # Collect timestamps
                    timestamp_str = metadata.get("timestamp")
                    if timestamp_str:
                        timestamps.append(datetime.fromisoformat(timestamp_str))
                
                # Calculate averages and extremes
                if importance_scores:
                    stats["avg_importance"] = sum(importance_scores) / len(importance_scores)
                
                if timestamps:
                    stats["oldest_memory"] = min(timestamps).isoformat()
                    stats["newest_memory"] = max(timestamps).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def _check_and_cleanup(self) -> None:
        """Check if cleanup is needed and perform it."""
        try:
            stats = await self.get_memory_stats()
            total_memories = stats.get("total_memories", 0)
            
            # Estimate memory size (rough calculation)
            estimated_size_mb = total_memories * 0.01  # Rough estimate
            
            if await self.cleanup_strategy.should_cleanup(total_memories, estimated_size_mb):
                await self._perform_cleanup()
                
        except Exception as e:
            logger.error(f"Error during cleanup check: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform memory cleanup based on the configured strategy."""
        try:
            logger.info("Starting memory cleanup")
            
            # Get all memories
            all_memories_raw = self._memory_client.get_all(user_id=self._user_id)
            
            # Convert to MemoryEntry objects
            all_memories = []
            for memory in all_memories_raw:
                metadata = memory.get("metadata", {})
                entry = MemoryEntry(
                    id=memory.get("id", ""),
                    content=memory.get("memory", {}).get("data", ""),
                    memory_type=MemoryType(metadata.get("memory_type", "working")),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["memory_type", "importance_score", "tags", "timestamp"]},
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                    importance_score=metadata.get("importance_score", 0.5),
                    tags=metadata.get("tags", [])
                )
                all_memories.append(entry)
            
            # Select memories for cleanup
            cleanup_memories = await self.cleanup_strategy.select_memories_for_cleanup(all_memories)
            
            # Delete selected memories
            deleted_count = 0
            for memory in cleanup_memories:
                if await self.delete_memory(memory.id):
                    deleted_count += 1
            
            logger.info(f"Memory cleanup completed: deleted {deleted_count} memories")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    async def close(self) -> None:
        """Close the memory manager and cleanup resources."""
        # Mem0 doesn't require explicit cleanup, but we can log
        logger.info("Memory manager closed")
        self._memory_client = None
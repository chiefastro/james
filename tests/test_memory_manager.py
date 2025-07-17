"""Tests for the memory management system."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_types import MemoryType, MemoryEntry, MemoryQuery, MemorySearchResult
from backend.memory.cleanup_strategies import (
    LRUCleanupStrategy, 
    ImportanceBasedCleanupStrategy,
    HybridCleanupStrategy,
    TypeBasedCleanupStrategy
)


class TestMemoryTypes:
    """Test memory type classes and data structures."""
    
    def test_memory_entry_creation(self):
        """Test MemoryEntry creation and defaults."""
        entry = MemoryEntry(
            id="test-id",
            content="Test memory content",
            memory_type=MemoryType.EPISODIC,
            metadata={"source": "test"},
            timestamp=datetime.now()
        )
        
        assert entry.id == "test-id"
        assert entry.content == "Test memory content"
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.metadata["source"] == "test"
        assert entry.importance_score == 0.5  # default
        assert entry.access_count == 0  # default
        assert entry.tags == []  # default
    
    def test_memory_entry_auto_id(self):
        """Test automatic ID generation when not provided."""
        entry = MemoryEntry(
            id="",
            content="Test content",
            memory_type=MemoryType.SEMANTIC,
            metadata={},
            timestamp=datetime.now()
        )
        
        assert len(entry.id) > 0  # UUID should be generated
    
    def test_memory_query_defaults(self):
        """Test MemoryQuery default values."""
        query = MemoryQuery(query_text="test query")
        
        assert query.query_text == "test query"
        assert query.memory_types == list(MemoryType)  # all types
        assert query.limit == 10
        assert query.min_importance == 0.0
        assert query.tags == []
        assert query.time_range is None


class TestCleanupStrategies:
    """Test memory cleanup strategies."""
    
    @pytest.fixture
    def sample_memories(self) -> List[MemoryEntry]:
        """Create sample memories for testing."""
        now = datetime.now()
        memories = []
        
        # Old, low importance, never accessed
        memories.append(MemoryEntry(
            id="old-low",
            content="Old unimportant memory",
            memory_type=MemoryType.WORKING,
            metadata={},
            timestamp=now - timedelta(days=100),
            importance_score=0.1,
            access_count=0,
            last_accessed=None
        ))
        
        # Recent, high importance, frequently accessed
        memories.append(MemoryEntry(
            id="recent-high",
            content="Recent important memory",
            memory_type=MemoryType.PROCEDURAL,
            metadata={},
            timestamp=now - timedelta(days=1),
            importance_score=0.9,
            access_count=10,
            last_accessed=now - timedelta(hours=1)
        ))
        
        # Medium age, medium importance
        memories.append(MemoryEntry(
            id="medium",
            content="Medium memory",
            memory_type=MemoryType.EPISODIC,
            metadata={},
            timestamp=now - timedelta(days=30),
            importance_score=0.5,
            access_count=3,
            last_accessed=now - timedelta(days=5)
        ))
        
        return memories
    
    @pytest.mark.asyncio
    async def test_lru_cleanup_strategy(self, sample_memories):
        """Test LRU cleanup strategy."""
        strategy = LRUCleanupStrategy(max_memories=2)
        
        # Should trigger cleanup
        assert await strategy.should_cleanup(3, 100.0)
        
        # Should select least recently used
        cleanup_candidates = await strategy.select_memories_for_cleanup(sample_memories)
        
        # Should select the one never accessed first
        assert len(cleanup_candidates) >= 1
        assert cleanup_candidates[0].id == "old-low"
    
    @pytest.mark.asyncio
    async def test_importance_based_cleanup(self, sample_memories):
        """Test importance-based cleanup strategy."""
        strategy = ImportanceBasedCleanupStrategy(max_memories=2, min_importance_threshold=0.3)
        
        # Should trigger cleanup
        assert await strategy.should_cleanup(3, 100.0)
        
        # Should select low importance memories
        cleanup_candidates = await strategy.select_memories_for_cleanup(sample_memories)
        
        # Should select the low importance memory
        assert len(cleanup_candidates) >= 1
        assert any(m.id == "old-low" for m in cleanup_candidates)
    
    @pytest.mark.asyncio
    async def test_hybrid_cleanup_strategy(self, sample_memories):
        """Test hybrid cleanup strategy."""
        strategy = HybridCleanupStrategy(max_memories=2)
        
        # Should trigger cleanup
        assert await strategy.should_cleanup(3, 100.0)
        
        # Should select based on combined factors
        cleanup_candidates = await strategy.select_memories_for_cleanup(sample_memories)
        
        assert len(cleanup_candidates) >= 1
        # Old, low importance, never accessed should score highest for cleanup
        assert cleanup_candidates[0].id == "old-low"
    
    @pytest.mark.asyncio
    async def test_type_based_cleanup(self, sample_memories):
        """Test type-based cleanup strategy."""
        strategy = TypeBasedCleanupStrategy(max_memories=2)
        
        # Should trigger cleanup
        assert await strategy.should_cleanup(3, 100.0)
        
        # Should prioritize working memory for cleanup
        cleanup_candidates = await strategy.select_memories_for_cleanup(sample_memories)
        
        assert len(cleanup_candidates) >= 1
        # Working memory should be cleaned up first
        working_memories = [m for m in cleanup_candidates if m.memory_type == MemoryType.WORKING]
        assert len(working_memories) > 0


class TestMemoryManager:
    """Test the main MemoryManager class."""
    
    @pytest.fixture
    def mock_mem0_client(self):
        """Create a mock Mem0 client."""
        mock_client = MagicMock()
        mock_client.add.return_value = [{"id": "test-memory-id"}]
        mock_client.search.return_value = [
            {
                "id": "search-result-1",
                "memory": {"data": "Test memory content"},
                "metadata": {
                    "memory_type": "episodic",
                    "importance_score": 0.7,
                    "tags": ["test"],
                    "timestamp": datetime.now().isoformat()
                }
            }
        ]
        mock_client.get.return_value = {
            "id": "test-memory-id",
            "memory": {"data": "Test content"},
            "metadata": {"memory_type": "semantic"}
        }
        mock_client.get_all.return_value = [
            {
                "id": "memory-1",
                "memory": {"data": "Memory 1"},
                "metadata": {
                    "memory_type": "episodic",
                    "importance_score": 0.5,
                    "timestamp": datetime.now().isoformat()
                }
            }
        ]
        return mock_client
    
    @pytest.fixture
    def memory_manager(self):
        """Create a MemoryManager instance for testing."""
        config = {
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}}
        }
        return MemoryManager(config=config)
    
    @pytest.mark.asyncio
    async def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initialization."""
        with patch('backend.memory.memory_manager.Memory') as mock_memory:
            mock_memory.return_value = MagicMock()
            
            await memory_manager.initialize()
            
            assert memory_manager._memory_client is not None
            mock_memory.assert_called_once_with(config=memory_manager.config)
    
    @pytest.mark.asyncio
    async def test_store_memory(self, memory_manager, mock_mem0_client):
        """Test storing a memory."""
        memory_manager._memory_client = mock_mem0_client
        
        entry = await memory_manager.store_memory(
            content="Test memory content",
            memory_type=MemoryType.EPISODIC,
            metadata={"source": "test"},
            importance_score=0.8,
            tags=["test", "memory"]
        )
        
        assert entry.id == "test-memory-id"
        assert entry.content == "Test memory content"
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.importance_score == 0.8
        assert "test" in entry.tags
        
        # Verify Mem0 client was called correctly
        mock_mem0_client.add.assert_called_once()
        call_args = mock_mem0_client.add.call_args
        assert call_args[1]["user_id"] == "james_conscious_agent"
        assert call_args[1]["metadata"]["memory_type"] == "episodic"
        assert call_args[1]["metadata"]["importance_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_retrieve_memories(self, memory_manager, mock_mem0_client):
        """Test retrieving memories."""
        memory_manager._memory_client = mock_mem0_client
        
        query = MemoryQuery(
            query_text="test query",
            memory_types=[MemoryType.EPISODIC],
            limit=5
        )
        
        result = await memory_manager.retrieve_memories(query)
        
        assert isinstance(result, MemorySearchResult)
        assert len(result.entries) == 1
        assert result.entries[0].id == "search-result-1"
        assert result.entries[0].memory_type == MemoryType.EPISODIC
        assert result.total_count == 1
        assert result.search_time_ms > 0
        
        # Verify search was called correctly
        mock_mem0_client.search.assert_called_once()
        call_args = mock_mem0_client.search.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_update_memory(self, memory_manager, mock_mem0_client):
        """Test updating a memory."""
        memory_manager._memory_client = mock_mem0_client
        
        success = await memory_manager.update_memory(
            memory_id="test-memory-id",
            content="Updated content",
            importance_score=0.9
        )
        
        assert success is True
        mock_mem0_client.get.assert_called_once_with("test-memory-id", user_id="james_conscious_agent")
        mock_mem0_client.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_manager, mock_mem0_client):
        """Test deleting a memory."""
        memory_manager._memory_client = mock_mem0_client
        
        success = await memory_manager.delete_memory("test-memory-id")
        
        assert success is True
        mock_mem0_client.delete.assert_called_once_with("test-memory-id", user_id="james_conscious_agent")
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_manager, mock_mem0_client):
        """Test getting memory statistics."""
        memory_manager._memory_client = mock_mem0_client
        
        stats = await memory_manager.get_memory_stats()
        
        assert stats["total_memories"] == 1
        assert "episodic" in stats["memory_types"]
        assert stats["memory_types"]["episodic"] == 1
        assert stats["avg_importance"] == 0.5
        assert "oldest_memory" in stats
        assert "newest_memory" in stats
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_trigger(self, memory_manager, mock_mem0_client):
        """Test that cleanup is triggered when limits are exceeded."""
        memory_manager._memory_client = mock_mem0_client
        
        # Mock cleanup strategy to always trigger cleanup
        mock_strategy = AsyncMock()
        mock_strategy.should_cleanup.return_value = True
        mock_strategy.select_memories_for_cleanup.return_value = []
        memory_manager.cleanup_strategy = mock_strategy
        
        # Store a memory (which triggers cleanup check)
        await memory_manager.store_memory(
            content="Test content",
            memory_type=MemoryType.WORKING
        )
        
        # Verify cleanup was checked
        mock_strategy.should_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_store_memory(self, memory_manager):
        """Test error handling when storing memory fails."""
        # Mock client that raises exception
        mock_client = MagicMock()
        mock_client.add.side_effect = Exception("Storage failed")
        memory_manager._memory_client = mock_client
        
        with pytest.raises(Exception, match="Storage failed"):
            await memory_manager.store_memory(
                content="Test content",
                memory_type=MemoryType.EPISODIC
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_retrieve_memories(self, memory_manager):
        """Test error handling when retrieving memories fails."""
        # Mock client that raises exception
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Search failed")
        memory_manager._memory_client = mock_client
        
        query = MemoryQuery(query_text="test")
        
        with pytest.raises(Exception, match="Search failed"):
            await memory_manager.retrieve_memories(query)
    
    def test_default_config(self, memory_manager):
        """Test default configuration generation."""
        config = memory_manager._get_default_config()
        
        assert "vector_store" in config
        assert config["vector_store"]["provider"] == "qdrant"
        assert "llm" in config
        assert config["llm"]["provider"] == "openai"
        assert "embedder" in config
        assert config["embedder"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_integration_memory_workflow():
    """Integration test for complete memory workflow."""
    # This test would require actual Mem0 setup, so we'll mock it
    with patch('backend.memory.memory_manager.Memory') as mock_memory_class:
        mock_client = MagicMock()
        mock_client.add.return_value = [{"id": "integration-test-id"}]
        mock_client.search.return_value = [
            {
                "id": "integration-test-id",
                "memory": {"data": "Integration test memory"},
                "metadata": {
                    "memory_type": "semantic",
                    "importance_score": 0.8,
                    "tags": ["integration", "test"],
                    "timestamp": datetime.now().isoformat()
                }
            }
        ]
        mock_memory_class.return_value = mock_client
        
        # Create memory manager
        manager = MemoryManager()
        await manager.initialize()
        
        # Store a memory
        stored_entry = await manager.store_memory(
            content="Integration test memory",
            memory_type=MemoryType.SEMANTIC,
            importance_score=0.8,
            tags=["integration", "test"]
        )
        
        assert stored_entry.id == "integration-test-id"
        assert stored_entry.memory_type == MemoryType.SEMANTIC
        
        # Retrieve the memory
        query = MemoryQuery(query_text="integration test")
        results = await manager.retrieve_memories(query)
        
        assert len(results.entries) == 1
        assert results.entries[0].id == "integration-test-id"
        assert results.entries[0].content == "Integration test memory"
        
        # Clean up
        await manager.close()
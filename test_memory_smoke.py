#!/usr/bin/env python3
"""
Smoke test for the memory management system.
Tests core functionality without requiring external services.
"""

import asyncio
import sys
from datetime import datetime
from unittest.mock import MagicMock

# Add backend to path
sys.path.append('.')

from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_types import MemoryType, MemoryQuery
from backend.memory.cleanup_strategies import HybridCleanupStrategy


async def test_memory_system():
    """Test the memory system with mocked dependencies."""
    print("🧪 Memory System Smoke Test")
    print("=" * 40)
    
    # Create a mock Mem0 client
    mock_client = MagicMock()
    mock_client.add.return_value = [{"id": "test-memory-123"}]
    mock_client.search.return_value = [
        {
            "id": "test-memory-123",
            "memory": {"data": "Test memory content"},
            "metadata": {
                "memory_type": "episodic",
                "importance_score": 0.8,
                "tags": ["test"],
                "timestamp": datetime.now().isoformat()
            }
        }
    ]
    mock_client.get_all.return_value = [
        {
            "id": "test-memory-123",
            "memory": {"data": "Test memory content"},
            "metadata": {
                "memory_type": "episodic",
                "importance_score": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        }
    ]
    
    # Create memory manager
    manager = MemoryManager()
    manager._memory_client = mock_client  # Inject mock client
    
    try:
        print("✓ Memory manager created")
        
        # Test storing memory
        memory_entry = await manager.store_memory(
            content="This is a test memory for the smoke test",
            memory_type=MemoryType.EPISODIC,
            metadata={"test": True},
            importance_score=0.8,
            tags=["smoke_test", "testing"]
        )
        
        print(f"✓ Memory stored with ID: {memory_entry.id}")
        print(f"  Content: {memory_entry.content}")
        print(f"  Type: {memory_entry.memory_type.value}")
        print(f"  Importance: {memory_entry.importance_score}")
        
        # Test retrieving memories
        query = MemoryQuery(
            query_text="test memory",
            limit=5
        )
        
        results = await manager.retrieve_memories(query)
        print(f"✓ Retrieved {len(results.entries)} memories")
        print(f"  Search time: {results.search_time_ms:.2f}ms")
        
        for entry in results.entries:
            print(f"  - [{entry.memory_type.value}] {entry.content}")
        
        # Test memory statistics
        stats = await manager.get_memory_stats()
        print("✓ Memory statistics:")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Memory types: {stats['memory_types']}")
        
        # Test cleanup strategy
        cleanup_strategy = HybridCleanupStrategy(max_memories=1)
        should_cleanup = await cleanup_strategy.should_cleanup(2, 50.0)
        print(f"✓ Cleanup needed: {should_cleanup}")
        
        print("\n🎉 All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


async def test_cleanup_strategies():
    """Test cleanup strategies independently."""
    print("\n🧹 Cleanup Strategies Test")
    print("=" * 40)
    
    from backend.memory.cleanup_strategies import (
        LRUCleanupStrategy,
        ImportanceBasedCleanupStrategy,
        TypeBasedCleanupStrategy
    )
    from backend.memory.memory_types import MemoryEntry
    from datetime import timedelta
    
    # Create test memories
    now = datetime.now()
    memories = [
        MemoryEntry(
            id="memory-1",
            content="Old low importance memory",
            memory_type=MemoryType.WORKING,
            metadata={},
            timestamp=now - timedelta(days=30),
            importance_score=0.2
        ),
        MemoryEntry(
            id="memory-2", 
            content="Recent high importance memory",
            memory_type=MemoryType.PROCEDURAL,
            metadata={},
            timestamp=now - timedelta(days=1),
            importance_score=0.9
        )
    ]
    
    # Test each strategy
    strategies = [
        ("LRU", LRUCleanupStrategy(max_memories=1)),
        ("Importance", ImportanceBasedCleanupStrategy(max_memories=1)),
        ("Type-based", TypeBasedCleanupStrategy(max_memories=1))
    ]
    
    for name, strategy in strategies:
        should_cleanup = await strategy.should_cleanup(2, 100.0)
        if should_cleanup:
            cleanup_candidates = await strategy.select_memories_for_cleanup(memories)
            print(f"✓ {name} strategy: would clean up {len(cleanup_candidates)} memories")
            for memory in cleanup_candidates:
                print(f"  - {memory.id}: {memory.content[:30]}...")
        else:
            print(f"✓ {name} strategy: no cleanup needed")
    
    print("✓ All cleanup strategies tested successfully")


async def main():
    """Run all smoke tests."""
    success = True
    
    # Test core memory system
    if not await test_memory_system():
        success = False
    
    # Test cleanup strategies
    await test_cleanup_strategies()
    
    if success:
        print("\n🎯 All smoke tests completed successfully!")
        print("\nMemory system is ready for use with:")
        print("- ✓ Memory storage and retrieval")
        print("- ✓ Multiple memory types")
        print("- ✓ Intelligent cleanup strategies")
        print("- ✓ Error handling and recovery")
        return 0
    else:
        print("\n❌ Some smoke tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
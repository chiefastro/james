#!/usr/bin/env python3
"""
Demo script showing the Mem0 memory management system capabilities.

This script demonstrates:
1. Storing different types of memories
2. Retrieving memories with queries
3. Memory cleanup strategies
4. Error handling and recovery
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_types import MemoryType, MemoryQuery
from backend.memory.cleanup_strategies import HybridCleanupStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_memory_storage():
    """Demonstrate storing different types of memories."""
    print("\n=== Memory Storage Demo ===")
    
    # Create memory manager with demo config (using default config for now)
    config = None  # Use default configuration
    
    manager = MemoryManager(config=config)
    
    try:
        await manager.initialize()
        print("✓ Memory manager initialized")
        
        # Store episodic memory (specific experience)
        episodic_memory = await manager.store_memory(
            content="Had a productive conversation with a user about implementing a memory system. They were particularly interested in the cleanup strategies and how different memory types are handled.",
            memory_type=MemoryType.EPISODIC,
            metadata={
                "conversation_id": "conv_001",
                "user_satisfaction": "high",
                "topics": ["memory", "cleanup", "architecture"]
            },
            importance_score=0.8,
            tags=["conversation", "memory", "productive"]
        )
        print(f"✓ Stored episodic memory: {episodic_memory.id}")
        
        # Store semantic memory (general knowledge)
        semantic_memory = await manager.store_memory(
            content="Mem0 is a memory layer for AI applications that provides intelligent memory management with vector storage, automatic relevance scoring, and cleanup strategies.",
            memory_type=MemoryType.SEMANTIC,
            metadata={
                "knowledge_domain": "ai_tools",
                "confidence": 0.9
            },
            importance_score=0.9,
            tags=["mem0", "ai", "knowledge", "tools"]
        )
        print(f"✓ Stored semantic memory: {semantic_memory.id}")
        
        # Store procedural memory (skill/capability)
        procedural_memory = await manager.store_memory(
            content="To implement memory cleanup: 1) Check memory limits, 2) Select cleanup strategy, 3) Identify candidates based on age/importance/access patterns, 4) Delete selected memories, 5) Log cleanup results.",
            memory_type=MemoryType.PROCEDURAL,
            metadata={
                "skill_category": "memory_management",
                "complexity": "intermediate"
            },
            importance_score=0.95,
            tags=["cleanup", "procedure", "memory", "skill"]
        )
        print(f"✓ Stored procedural memory: {procedural_memory.id}")
        
        # Store working memory (current context)
        working_memory = await manager.store_memory(
            content="Currently demonstrating the memory system. User is interested in seeing practical examples of how memories are stored and retrieved.",
            memory_type=MemoryType.WORKING,
            metadata={
                "context": "demo_session",
                "active": True
            },
            importance_score=0.3,
            tags=["demo", "current", "context"]
        )
        print(f"✓ Stored working memory: {working_memory.id}")
        
        return manager
        
    except Exception as e:
        logger.error(f"Error in memory storage demo: {e}")
        print(f"✗ Error: {e}")
        return None


async def demo_memory_retrieval(manager: MemoryManager):
    """Demonstrate memory retrieval with different queries."""
    print("\n=== Memory Retrieval Demo ===")
    
    try:
        # Query 1: Search for memories about memory management
        query1 = MemoryQuery(
            query_text="memory management and cleanup",
            limit=5
        )
        results1 = await manager.retrieve_memories(query1)
        print(f"✓ Found {len(results1.entries)} memories about memory management:")
        for entry in results1.entries:
            print(f"  - [{entry.memory_type.value}] {entry.content[:80]}...")
        
        # Query 2: Search only procedural memories
        query2 = MemoryQuery(
            query_text="how to implement",
            memory_types=[MemoryType.PROCEDURAL],
            limit=3
        )
        results2 = await manager.retrieve_memories(query2)
        print(f"\n✓ Found {len(results2.entries)} procedural memories:")
        for entry in results2.entries:
            print(f"  - {entry.content[:80]}...")
        
        # Query 3: Search high-importance memories
        query3 = MemoryQuery(
            query_text="important knowledge",
            min_importance=0.8,
            limit=5
        )
        results3 = await manager.retrieve_memories(query3)
        print(f"\n✓ Found {len(results3.entries)} high-importance memories:")
        for entry in results3.entries:
            print(f"  - [Score: {entry.importance_score}] {entry.content[:60]}...")
        
        # Query 4: Search by tags
        query4 = MemoryQuery(
            query_text="conversation",
            tags=["conversation"],
            limit=3
        )
        results4 = await manager.retrieve_memories(query4)
        print(f"\n✓ Found {len(results4.entries)} memories tagged 'conversation':")
        for entry in results4.entries:
            print(f"  - {entry.content[:80]}...")
            
    except Exception as e:
        logger.error(f"Error in memory retrieval demo: {e}")
        print(f"✗ Error: {e}")


async def demo_memory_stats(manager: MemoryManager):
    """Demonstrate memory statistics."""
    print("\n=== Memory Statistics Demo ===")
    
    try:
        stats = await manager.get_memory_stats()
        print("✓ Memory Statistics:")
        print(f"  - Total memories: {stats.get('total_memories', 0)}")
        print(f"  - Average importance: {stats.get('avg_importance', 0):.2f}")
        print("  - Memory types:")
        for mem_type, count in stats.get('memory_types', {}).items():
            print(f"    * {mem_type}: {count}")
        
        if stats.get('oldest_memory'):
            print(f"  - Oldest memory: {stats['oldest_memory']}")
        if stats.get('newest_memory'):
            print(f"  - Newest memory: {stats['newest_memory']}")
            
    except Exception as e:
        logger.error(f"Error in memory stats demo: {e}")
        print(f"✗ Error: {e}")


async def demo_cleanup_strategies():
    """Demonstrate different cleanup strategies."""
    print("\n=== Cleanup Strategies Demo ===")
    
    from backend.memory.cleanup_strategies import (
        LRUCleanupStrategy,
        ImportanceBasedCleanupStrategy,
        TypeBasedCleanupStrategy
    )
    from backend.memory.memory_types import MemoryEntry
    from datetime import timedelta
    
    # Create sample memories for cleanup demo
    now = datetime.now()
    sample_memories = [
        MemoryEntry(
            id="old-working",
            content="Old working memory",
            memory_type=MemoryType.WORKING,
            metadata={},
            timestamp=now - timedelta(days=30),
            importance_score=0.2,
            access_count=1,
            last_accessed=now - timedelta(days=20)
        ),
        MemoryEntry(
            id="important-skill",
            content="Important procedural skill",
            memory_type=MemoryType.PROCEDURAL,
            metadata={},
            timestamp=now - timedelta(days=10),
            importance_score=0.9,
            access_count=15,
            last_accessed=now - timedelta(hours=2)
        ),
        MemoryEntry(
            id="recent-episode",
            content="Recent episodic memory",
            memory_type=MemoryType.EPISODIC,
            metadata={},
            timestamp=now - timedelta(days=1),
            importance_score=0.6,
            access_count=3,
            last_accessed=now - timedelta(hours=5)
        )
    ]
    
    # Test LRU strategy
    lru_strategy = LRUCleanupStrategy(max_memories=2)
    should_cleanup = await lru_strategy.should_cleanup(3, 100.0)
    if should_cleanup:
        lru_cleanup = await lru_strategy.select_memories_for_cleanup(sample_memories)
        print(f"✓ LRU Strategy would clean up: {[m.id for m in lru_cleanup]}")
    
    # Test importance-based strategy
    importance_strategy = ImportanceBasedCleanupStrategy(max_memories=2)
    importance_cleanup = await importance_strategy.select_memories_for_cleanup(sample_memories)
    print(f"✓ Importance Strategy would clean up: {[m.id for m in importance_cleanup]}")
    
    # Test type-based strategy
    type_strategy = TypeBasedCleanupStrategy(max_memories=2)
    type_cleanup = await type_strategy.select_memories_for_cleanup(sample_memories)
    print(f"✓ Type-based Strategy would clean up: {[m.id for m in type_cleanup]}")


async def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Error Handling Demo ===")
    
    # Create manager with invalid config to test error handling
    invalid_config = {
        "vector_store": {
            "provider": "invalid_provider",
            "config": {"host": "nonexistent"}
        }
    }
    
    manager = MemoryManager(config=invalid_config)
    
    try:
        # This should handle the error gracefully
        await manager.store_memory(
            content="Test memory",
            memory_type=MemoryType.WORKING
        )
        print("✓ Error handling worked - no crash occurred")
    except Exception as e:
        print(f"✓ Error properly caught and handled: {type(e).__name__}")


async def main():
    """Run the complete memory management demo."""
    print("🧠 Mem0 Memory Management System Demo")
    print("=" * 50)
    
    # Check if required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
    
    # Run storage demo
    manager = await demo_memory_storage()
    
    if manager:
        # Run retrieval demo
        await demo_memory_retrieval(manager)
        
        # Show statistics
        await demo_memory_stats(manager)
        
        # Close manager
        await manager.close()
    
    # Demo cleanup strategies (doesn't require live connection)
    await demo_cleanup_strategies()
    
    # Demo error handling
    await demo_error_handling()
    
    print("\n✅ Demo completed!")
    print("\nKey Features Demonstrated:")
    print("- ✓ Multiple memory types (episodic, semantic, procedural, working)")
    print("- ✓ Intelligent storage with metadata and importance scoring")
    print("- ✓ Flexible retrieval with queries, filters, and limits")
    print("- ✓ Multiple cleanup strategies for memory management")
    print("- ✓ Comprehensive error handling and recovery")
    print("- ✓ Memory statistics and monitoring")


if __name__ == "__main__":
    asyncio.run(main())
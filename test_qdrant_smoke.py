#!/usr/bin/env python3
"""
Smoke test for Qdrant integration without requiring running Qdrant service.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from backend.models.core import Subagent
from backend.vector.qdrant_client import QdrantVectorClient, QdrantConnectionError
from backend.registry.subagent_registry import SubagentRegistry


async def test_basic_functionality():
    """Test basic functionality without requiring Qdrant service."""
    print("🧪 Qdrant Integration Smoke Test")
    print("=" * 40)
    
    # Test 1: Vector client initialization
    print("\n1. Testing vector client initialization...")
    try:
        # This should fail gracefully when Qdrant is not available
        client = QdrantVectorClient(host="localhost", port=6333)
        print("✅ Vector client created (connection not tested)")
    except QdrantConnectionError as e:
        print(f"✅ Expected connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 2: Subagent model with embeddings
    print("\n2. Testing subagent model with embeddings...")
    try:
        subagent = Subagent(
            id="test-subagent",
            name="Test Subagent",
            description="A test subagent for smoke testing",
            capabilities=["testing", "validation"],
            import_path="test.subagent",
            embedding=[0.1] * 1536,  # Valid embedding size
            is_active=True
        )
        print(f"✅ Subagent created: {subagent.name}")
        print(f"   Embedding size: {len(subagent.embedding)}")
        print(f"   Capabilities: {', '.join(subagent.capabilities)}")
    except Exception as e:
        print(f"❌ Subagent creation failed: {e}")
        return False
    
    # Test 3: Registry with vector database disabled
    print("\n3. Testing registry with vector database disabled...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "test_subagents.csv"
            registry = SubagentRegistry(registry_path=str(registry_path), enable_vector_db=False)
            await registry.register_subagent(subagent)
            
            # Test CSV-based operations
            retrieved = await registry.get_subagent_by_id(subagent.id)
            if retrieved and retrieved.id == subagent.id:
                print("✅ Registry operations work without vector DB")
            else:
                print("❌ Registry operations failed")
                return False
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False
    
    # Test 4: Registry with vector database enabled but not connected
    print("\n4. Testing registry with vector database enabled (no connection)...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "test_subagents_vector.csv"
            registry_with_vector = SubagentRegistry(registry_path=str(registry_path), enable_vector_db=True)
            
            # This should work even if vector DB is not available
            # because the registry falls back gracefully
            test_subagent = Subagent(
                id="test-subagent-2",
                name="Test Subagent 2",
                description="Another test subagent",
                capabilities=["testing"],
                import_path="test.subagent2",
                is_active=True
            )
            
            await registry_with_vector.register_subagent(test_subagent)
            print("✅ Registry with vector DB enabled works (graceful fallback)")
            
            # Test vector database stats
            stats = await registry_with_vector.get_vector_database_stats()
            print(f"   Vector DB enabled: {stats.get('enabled', False)}")
            print(f"   Vector DB healthy: {stats.get('healthy', False)}")
        
    except Exception as e:
        print(f"❌ Registry with vector DB test failed: {e}")
        return False
    
    print("\n✅ All smoke tests passed!")
    return True


async def main():
    """Main smoke test function."""
    success = await test_basic_functionality()
    
    if success:
        print("\n🎉 Smoke test completed successfully!")
        print("\nTo test with actual Qdrant service:")
        print("1. Start Qdrant: docker-compose up qdrant")
        print("2. Run: python examples/qdrant_demo.py")
        return 0
    else:
        print("\n❌ Smoke test failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
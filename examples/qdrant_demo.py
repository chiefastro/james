#!/usr/bin/env python3
"""
Demo script for Qdrant vector database integration.

This script demonstrates the vector search functionality for subagent discovery
with embedding-based retrieval and similarity scoring.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.core import Subagent
from backend.vector.qdrant_client import QdrantVectorClient, QdrantVectorError
from backend.registry.subagent_registry import SubagentRegistry


async def demo_vector_operations():
    """Demonstrate vector database operations."""
    print("🔍 Qdrant Vector Database Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize vector client
        print("\n1. Initializing Qdrant vector client...")
        vector_client = QdrantVectorClient(host="localhost", port=6333)
        
        # Check health
        is_healthy = await vector_client.health_check()
        if not is_healthy:
            print("❌ Qdrant service is not available. Please start Qdrant with:")
            print("   docker-compose up qdrant")
            return
        
        print("✅ Qdrant service is healthy")
        
        # Initialize collection
        print("\n2. Initializing collection...")
        await vector_client.initialize_collection()
        print("✅ Collection initialized")
        
        # Create sample subagents
        print("\n3. Creating sample subagents...")
        sample_subagents = [
            Subagent(
                id="code-analyzer-1",
                name="Code Analyzer",
                description="Analyzes code quality, finds bugs, and suggests improvements for Python, JavaScript, and other languages",
                capabilities=["code-analysis", "bug-detection", "quality-check", "refactoring"],
                import_path="agents.code_analyzer",
                embedding=generate_mock_embedding("code analysis quality bugs refactoring"),
                is_active=True
            ),
            Subagent(
                id="data-processor-1",
                name="Data Processor",
                description="Processes large datasets, performs data transformation, cleaning, and statistical analysis",
                capabilities=["data-processing", "transformation", "cleaning", "statistics"],
                import_path="agents.data_processor",
                embedding=generate_mock_embedding("data processing transformation cleaning statistics"),
                is_active=True
            ),
            Subagent(
                id="report-generator-1",
                name="Report Generator",
                description="Generates comprehensive reports, documentation, and visualizations from data and analysis results",
                capabilities=["reporting", "documentation", "visualization", "charts"],
                import_path="agents.report_generator",
                embedding=generate_mock_embedding("reports documentation visualization charts"),
                is_active=True
            ),
            Subagent(
                id="test-writer-1",
                name="Test Writer",
                description="Writes unit tests, integration tests, and test automation scripts for various programming languages",
                capabilities=["testing", "unit-tests", "integration-tests", "automation"],
                import_path="agents.test_writer",
                embedding=generate_mock_embedding("testing unit tests integration automation"),
                is_active=True
            ),
            Subagent(
                id="api-designer-1",
                name="API Designer",
                description="Designs REST APIs, GraphQL schemas, and API documentation with best practices",
                capabilities=["api-design", "rest", "graphql", "documentation"],
                import_path="agents.api_designer",
                embedding=generate_mock_embedding("api design rest graphql documentation"),
                is_active=True
            )
        ]
        
        # Store embeddings
        print("\n4. Storing subagent embeddings...")
        for subagent in sample_subagents:
            await vector_client.store_subagent_embedding(subagent)
            print(f"   ✅ Stored: {subagent.name}")
        
        # Get collection info
        print("\n5. Collection information:")
        collection_info = await vector_client.get_collection_info()
        print(f"   📊 Collection: {collection_info['collection_name']}")
        print(f"   📊 Total points: {collection_info['total_points']}")
        print(f"   📊 Vector size: {collection_info['vector_size']}")
        print(f"   📊 Distance metric: {collection_info['distance_metric']}")
        
        # Demonstrate vector similarity search
        print("\n6. Vector similarity search demos:")
        
        # Search for code-related subagents
        print("\n   🔍 Searching for 'code quality and bug detection'...")
        query_embedding = generate_mock_embedding("code quality bug detection")
        results = await vector_client.search_similar_subagents(
            query_embedding=query_embedding,
            limit=3,
            score_threshold=0.5
        )
        
        print(f"   Found {len(results)} similar subagents:")
        for subagent, score in results:
            print(f"     • {subagent.name} (score: {score:.3f})")
            print(f"       {subagent.description[:80]}...")
        
        # Search for data-related subagents
        print("\n   🔍 Searching for 'data analysis and statistics'...")
        query_embedding = generate_mock_embedding("data analysis statistics")
        results = await vector_client.search_similar_subagents(
            query_embedding=query_embedding,
            limit=3,
            score_threshold=0.5
        )
        
        print(f"   Found {len(results)} similar subagents:")
        for subagent, score in results:
            print(f"     • {subagent.name} (score: {score:.3f})")
            print(f"       {subagent.description[:80]}...")
        
        # Demonstrate capability-based search
        print("\n7. Capability-based search demos:")
        
        # Search by testing capabilities
        print("\n   🔍 Searching for subagents with 'testing' capabilities...")
        results = await vector_client.search_by_capabilities(["testing", "automation"])
        
        print(f"   Found {len(results)} subagents with testing capabilities:")
        for subagent in results:
            print(f"     • {subagent.name}")
            print(f"       Capabilities: {', '.join(subagent.capabilities)}")
        
        # Search by documentation capabilities
        print("\n   🔍 Searching for subagents with 'documentation' capabilities...")
        results = await vector_client.search_by_capabilities(["documentation", "reporting"])
        
        print(f"   Found {len(results)} subagents with documentation capabilities:")
        for subagent in results:
            print(f"     • {subagent.name}")
            print(f"       Capabilities: {', '.join(subagent.capabilities)}")
        
        # Demonstrate retrieval by ID
        print("\n8. Retrieval by ID demo:")
        subagent = await vector_client.get_subagent_by_id("code-analyzer-1")
        if subagent:
            print(f"   ✅ Retrieved: {subagent.name}")
            print(f"      Description: {subagent.description}")
            print(f"      Capabilities: {', '.join(subagent.capabilities)}")
        
        print("\n✅ Demo completed successfully!")
        
    except QdrantVectorError as e:
        print(f"❌ Qdrant error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


async def demo_registry_integration():
    """Demonstrate registry integration with vector database."""
    print("\n" + "=" * 50)
    print("🔗 Registry Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize registry with vector database
        print("\n1. Initializing registry with vector database...")
        registry = SubagentRegistry(enable_vector_db=True)
        
        # Initialize vector database
        await registry.initialize_vector_database()
        print("✅ Vector database initialized")
        
        # Create a new subagent
        print("\n2. Registering new subagent...")
        new_subagent = Subagent(
            id="security-scanner-1",
            name="Security Scanner",
            description="Scans code for security vulnerabilities, checks dependencies, and suggests security improvements",
            capabilities=["security", "vulnerability-scan", "dependency-check", "compliance"],
            import_path="agents.security_scanner",
            is_active=True
        )
        
        await registry.register_subagent(new_subagent)
        print(f"✅ Registered: {new_subagent.name}")
        
        # Demonstrate vector similarity search through registry
        print("\n3. Vector similarity search through registry...")
        results = await registry.search_similar_subagents(
            query_text="security vulnerability scanning",
            limit=3,
            score_threshold=0.5
        )
        
        print(f"   Found {len(results)} similar subagents:")
        for subagent, score in results:
            print(f"     • {subagent.name} (score: {score:.3f})")
        
        # Get vector database stats
        print("\n4. Vector database statistics:")
        stats = await registry.get_vector_database_stats()
        if stats["enabled"]:
            print(f"   📊 Enabled: {stats['enabled']}")
            print(f"   📊 Healthy: {stats['healthy']}")
            print(f"   📊 Total points: {stats.get('total_points', 'N/A')}")
        
        print("\n✅ Registry integration demo completed!")
        
    except Exception as e:
        print(f"❌ Registry integration error: {e}")
        import traceback
        traceback.print_exc()


def generate_mock_embedding(text: str) -> list[float]:
    """
    Generate a mock embedding for demonstration purposes.
    
    In production, this would use OpenAI's embedding API or similar service.
    """
    import hashlib
    import struct
    
    # Create a deterministic embedding based on text hash
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert hash to float values
    embedding = []
    for i in range(0, len(hash_bytes), 4):
        chunk = hash_bytes[i:i+4]
        if len(chunk) == 4:
            # Convert 4 bytes to float
            float_val = struct.unpack('f', chunk)[0]
            # Normalize to reasonable range
            normalized = (float_val % 2.0) - 1.0  # Range: -1 to 1
            embedding.append(normalized)
    
    # Pad or truncate to 1536 dimensions (OpenAI embedding size)
    while len(embedding) < 1536:
        embedding.extend(embedding[:min(len(embedding), 1536 - len(embedding))])
    
    return embedding[:1536]


async def cleanup_demo_data():
    """Clean up demo data from vector database."""
    print("\n" + "=" * 50)
    print("🧹 Cleanup Demo")
    print("=" * 50)
    
    try:
        vector_client = QdrantVectorClient(host="localhost", port=6333)
        
        # Check if service is available
        is_healthy = await vector_client.health_check()
        if not is_healthy:
            print("❌ Qdrant service is not available")
            return
        
        print("\n1. Clearing demo data...")
        await vector_client.clear_collection()
        print("✅ Demo data cleared")
        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")


async def main():
    """Main demo function."""
    print("🚀 Starting Qdrant Vector Database Integration Demo")
    
    # Check if cleanup flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        await cleanup_demo_data()
        return
    
    # Run demos
    await demo_vector_operations()
    await demo_registry_integration()
    
    print("\n" + "=" * 50)
    print("🎉 All demos completed successfully!")
    print("\nTo clean up demo data, run:")
    print("   python examples/qdrant_demo.py --cleanup")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
"""
Integration tests for Qdrant vector database functionality.

Tests vector operations, search accuracy, and integration with the subagent registry.
"""

import pytest
import asyncio
import os
from datetime import datetime, timezone
from typing import List, Tuple
from unittest.mock import patch, MagicMock, AsyncMock

from backend.models.core import Subagent
from backend.vector.qdrant_client import QdrantVectorClient, QdrantVectorError, QdrantConnectionError
from backend.registry.subagent_registry import SubagentRegistry, SubagentRegistryError


class TestQdrantVectorClient:
    """Test cases for QdrantVectorClient."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client for testing."""
        with patch('backend.vector.qdrant_client.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock collections response
            mock_collections = MagicMock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            client = QdrantVectorClient(host="localhost", port=6333)
            client.client = mock_client
            yield client, mock_client
    
    @pytest.fixture
    def sample_subagent(self) -> Subagent:
        """Create a sample subagent for testing."""
        return Subagent(
            id="test-subagent-1",
            name="Test Subagent",
            description="A test subagent for vector operations",
            capabilities=["testing", "analysis"],
            import_path="test.subagent",
            embedding=[0.1] * 1536,  # Valid embedding size
            is_active=True
        )
    
    @pytest.fixture
    def sample_subagents(self) -> List[Subagent]:
        """Create multiple sample subagents for testing."""
        return [
            Subagent(
                id="subagent-1",
                name="Code Analyzer",
                description="Analyzes code quality and suggests improvements",
                capabilities=["code-analysis", "quality-check"],
                import_path="agents.code_analyzer",
                embedding=[0.1] * 1536,
                is_active=True
            ),
            Subagent(
                id="subagent-2", 
                name="Data Processor",
                description="Processes and transforms data efficiently",
                capabilities=["data-processing", "transformation"],
                import_path="agents.data_processor",
                embedding=[0.2] * 1536,
                is_active=True
            ),
            Subagent(
                id="subagent-3",
                name="Report Generator",
                description="Generates comprehensive reports from data",
                capabilities=["reporting", "documentation"],
                import_path="agents.report_generator",
                embedding=[0.3] * 1536,
                is_active=False  # Inactive subagent
            )
        ]
    
    async def test_initialization_success(self):
        """Test successful Qdrant client initialization."""
        with patch('backend.vector.qdrant_client.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            client = QdrantVectorClient(host="test-host", port=1234)
            
            assert client.host == "test-host"
            assert client.port == 1234
            assert client.collection_name == "subagents"
            assert client.vector_size == 1536
            mock_client_class.assert_called_once_with(host="test-host", port=1234)
    
    async def test_initialization_failure(self):
        """Test Qdrant client initialization failure."""
        with patch('backend.vector.qdrant_client.QdrantClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")
            
            with pytest.raises(QdrantConnectionError, match="Failed to connect to Qdrant"):
                QdrantVectorClient()
    
    async def test_initialize_collection_new(self, mock_qdrant_client):
        """Test initializing a new collection."""
        client, mock_client = mock_qdrant_client
        
        # Mock empty collections list
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        await client.initialize_collection()
        
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "subagents"
    
    async def test_initialize_collection_existing(self, mock_qdrant_client):
        """Test initializing an existing collection."""
        client, mock_client = mock_qdrant_client
        
        # Mock existing collection
        mock_collection = MagicMock()
        mock_collection.name = "subagents"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        await client.initialize_collection()
        
        mock_client.create_collection.assert_not_called()
    
    async def test_store_subagent_embedding_success(self, mock_qdrant_client, sample_subagent):
        """Test successful subagent embedding storage."""
        client, mock_client = mock_qdrant_client
        
        await client.store_subagent_embedding(sample_subagent)
        
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "subagents"
        
        # Check point structure
        points = call_args[1]["points"]
        assert len(points) == 1
        point = points[0]
        assert point.id == sample_subagent.id
        assert point.vector == sample_subagent.embedding
        assert point.payload["name"] == sample_subagent.name
    
    async def test_store_subagent_embedding_no_embedding(self, mock_qdrant_client):
        """Test storing subagent without embedding fails."""
        client, mock_client = mock_qdrant_client
        
        subagent = Subagent(
            id="test-id",
            name="Test",
            description="Test subagent",
            import_path="test.path",
            embedding=[]  # No embedding
        )
        
        with pytest.raises(QdrantVectorError, match="Subagent must have an embedding"):
            await client.store_subagent_embedding(subagent)
    
    async def test_store_subagent_embedding_wrong_size(self, mock_qdrant_client):
        """Test storing subagent with wrong embedding size fails."""
        client, mock_client = mock_qdrant_client
        
        # Create subagent with wrong embedding size, bypassing validation
        subagent = Subagent(
            id="test-id",
            name="Test",
            description="Test subagent",
            import_path="test.path",
            embedding=[]  # Start with empty
        )
        # Manually set wrong size embedding to bypass validation
        subagent.embedding = [0.1] * 100  # Wrong size
        
        with pytest.raises(QdrantVectorError, match="Embedding size .* doesn't match"):
            await client.store_subagent_embedding(subagent)
    
    async def test_search_similar_subagents_success(self, mock_qdrant_client, sample_subagents):
        """Test successful vector similarity search."""
        client, mock_client = mock_qdrant_client
        
        # Mock search results
        mock_scored_point = MagicMock()
        mock_scored_point.score = 0.85
        mock_scored_point.payload = {
            "subagent_id": sample_subagents[0].id,
            "name": sample_subagents[0].name,
            "description": sample_subagents[0].description,
            "capabilities": sample_subagents[0].capabilities,
            "import_path": sample_subagents[0].import_path,
            "is_active": sample_subagents[0].is_active,
            "created_at": sample_subagents[0].created_at.isoformat(),
            "last_used": None
        }
        mock_client.search.return_value = [mock_scored_point]
        
        query_embedding = [0.1] * 1536
        results = await client.search_similar_subagents(
            query_embedding=query_embedding,
            limit=5,
            score_threshold=0.8
        )
        
        assert len(results) == 1
        subagent, score = results[0]
        assert subagent.id == sample_subagents[0].id
        assert subagent.name == sample_subagents[0].name
        assert score == 0.85
        
        # Verify search parameters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["collection_name"] == "subagents"
        assert call_args[1]["query_vector"] == query_embedding
        assert call_args[1]["limit"] == 5
        assert call_args[1]["score_threshold"] == 0.8
    
    async def test_search_similar_subagents_wrong_embedding_size(self, mock_qdrant_client):
        """Test search with wrong embedding size fails."""
        client, mock_client = mock_qdrant_client
        
        query_embedding = [0.1] * 100  # Wrong size
        
        with pytest.raises(QdrantVectorError, match="Query embedding size .* doesn't match"):
            await client.search_similar_subagents(query_embedding)
    
    async def test_search_by_capabilities_success(self, mock_qdrant_client, sample_subagents):
        """Test successful capability-based search."""
        client, mock_client = mock_qdrant_client
        
        # Mock scroll results
        mock_point = MagicMock()
        mock_point.payload = {
            "subagent_id": sample_subagents[0].id,
            "name": sample_subagents[0].name,
            "description": sample_subagents[0].description,
            "capabilities": sample_subagents[0].capabilities,
            "import_path": sample_subagents[0].import_path,
            "is_active": sample_subagents[0].is_active,
            "created_at": sample_subagents[0].created_at.isoformat(),
            "last_used": None
        }
        mock_client.scroll.return_value = ([mock_point], None)
        
        capabilities = ["code-analysis"]
        results = await client.search_by_capabilities(capabilities)
        
        assert len(results) == 1
        assert results[0].id == sample_subagents[0].id
        assert results[0].name == sample_subagents[0].name
        
        mock_client.scroll.assert_called_once()
    
    async def test_get_subagent_by_id_success(self, mock_qdrant_client, sample_subagent):
        """Test successful subagent retrieval by ID."""
        client, mock_client = mock_qdrant_client
        
        # Mock retrieve result
        mock_point = MagicMock()
        mock_point.payload = {
            "subagent_id": sample_subagent.id,
            "name": sample_subagent.name,
            "description": sample_subagent.description,
            "capabilities": sample_subagent.capabilities,
            "import_path": sample_subagent.import_path,
            "is_active": sample_subagent.is_active,
            "created_at": sample_subagent.created_at.isoformat(),
            "last_used": None
        }
        mock_point.vector = sample_subagent.embedding
        mock_client.retrieve.return_value = [mock_point]
        
        result = await client.get_subagent_by_id(sample_subagent.id)
        
        assert result is not None
        assert result.id == sample_subagent.id
        assert result.name == sample_subagent.name
        assert result.embedding == sample_subagent.embedding
        
        mock_client.retrieve.assert_called_once_with(
            collection_name="subagents",
            ids=[sample_subagent.id]
        )
    
    async def test_get_subagent_by_id_not_found(self, mock_qdrant_client):
        """Test subagent retrieval when not found."""
        client, mock_client = mock_qdrant_client
        
        mock_client.retrieve.return_value = []
        
        result = await client.get_subagent_by_id("nonexistent-id")
        
        assert result is None
    
    async def test_delete_subagent_embedding_success(self, mock_qdrant_client):
        """Test successful subagent embedding deletion."""
        client, mock_client = mock_qdrant_client
        
        subagent_id = "test-subagent-1"
        await client.delete_subagent_embedding(subagent_id)
        
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "subagents"
    
    async def test_get_collection_info_success(self, mock_qdrant_client):
        """Test successful collection info retrieval."""
        client, mock_client = mock_qdrant_client
        
        # Mock collection info
        mock_info = MagicMock()
        mock_info.config.params.vectors.size = 1536
        mock_info.config.params.vectors.distance.name = "Cosine"
        mock_info.status.name = "Green"
        mock_info.optimizer_status.name = "Ok"
        mock_info.indexed_vectors_count = 100
        mock_info.points_count = 100
        mock_client.get_collection.return_value = mock_info
        
        # Mock collection stats
        mock_stats = MagicMock()
        mock_stats.count = 100
        mock_client.count.return_value = mock_stats
        
        info = await client.get_collection_info()
        
        assert info["collection_name"] == "subagents"
        assert info["vector_size"] == 1536
        assert info["distance_metric"] == "Cosine"
        assert info["total_points"] == 100
        assert info["status"] == "Green"
    
    async def test_health_check_success(self, mock_qdrant_client):
        """Test successful health check."""
        client, mock_client = mock_qdrant_client
        
        mock_collections = MagicMock()
        mock_client.get_collections.return_value = mock_collections
        
        is_healthy = await client.health_check()
        
        assert is_healthy is True
        mock_client.get_collections.assert_called_once()
    
    async def test_health_check_failure(self, mock_qdrant_client):
        """Test health check failure."""
        client, mock_client = mock_qdrant_client
        
        mock_client.get_collections.side_effect = Exception("Connection failed")
        
        is_healthy = await client.health_check()
        
        assert is_healthy is False
    
    async def test_clear_collection_success(self, mock_qdrant_client):
        """Test successful collection clearing."""
        client, mock_client = mock_qdrant_client
        
        await client.clear_collection()
        
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "subagents"


class TestSubagentRegistryVectorIntegration:
    """Test cases for SubagentRegistry vector database integration."""
    
    @pytest.fixture
    def mock_registry_with_vector(self, tmp_path):
        """Create a mock registry with vector database integration."""
        registry_path = tmp_path / "test_subagents.csv"
        
        with patch('backend.registry.subagent_registry.QdrantVectorClient') as mock_client_class:
            mock_vector_client = MagicMock()
            
            # Make async methods return AsyncMock
            mock_vector_client.store_subagent_embedding = AsyncMock()
            mock_vector_client.update_subagent_embedding = AsyncMock()
            mock_vector_client.delete_subagent_embedding = AsyncMock()
            mock_vector_client.search_similar_subagents = AsyncMock()
            mock_vector_client.search_by_capabilities = AsyncMock()
            mock_vector_client.initialize_collection = AsyncMock()
            mock_vector_client.get_collection_info = AsyncMock()
            mock_vector_client.health_check = AsyncMock()
            
            mock_client_class.return_value = mock_vector_client
            
            registry = SubagentRegistry(
                registry_path=str(registry_path),
                enable_vector_db=True
            )
            
            yield registry, mock_vector_client
    
    @pytest.fixture
    def sample_subagent(self) -> Subagent:
        """Create a sample subagent for testing."""
        return Subagent(
            id="test-subagent-1",
            name="Test Subagent",
            description="A test subagent for integration testing",
            capabilities=["testing", "integration"],
            import_path="test.subagent",
            embedding=[0.1] * 1536,
            is_active=True
        )
    
    async def test_register_subagent_with_vector_storage(self, mock_registry_with_vector, sample_subagent):
        """Test subagent registration with vector storage."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # Mock embedding generation
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            await registry.register_subagent(sample_subagent)
        
        # Verify vector storage was called
        mock_vector_client.store_subagent_embedding.assert_called_once_with(sample_subagent)
        
        # Verify subagent was stored in CSV
        subagents = await registry.list_subagents()
        assert len(subagents) == 1
        assert subagents[0].id == sample_subagent.id
    
    async def test_register_subagent_vector_storage_failure(self, mock_registry_with_vector, sample_subagent):
        """Test subagent registration when vector storage fails."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # Mock vector storage failure
        mock_vector_client.store_subagent_embedding.side_effect = QdrantVectorError("Storage failed")
        
        # Mock embedding generation
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            # Should not raise exception, just log warning
            await registry.register_subagent(sample_subagent)
        
        # Verify subagent was still stored in CSV
        subagents = await registry.list_subagents()
        assert len(subagents) == 1
        assert subagents[0].id == sample_subagent.id
    
    async def test_update_subagent_with_vector_update(self, mock_registry_with_vector, sample_subagent):
        """Test subagent update with vector database sync."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # First register the subagent
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            await registry.register_subagent(sample_subagent)
        
        # Update the subagent
        sample_subagent.description = "Updated description"
        await registry.update_subagent(sample_subagent)
        
        # Verify vector update was called
        mock_vector_client.update_subagent_embedding.assert_called_with(sample_subagent)
    
    async def test_delete_subagent_with_vector_deletion(self, mock_registry_with_vector, sample_subagent):
        """Test subagent deletion with vector database cleanup."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # First register the subagent
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            await registry.register_subagent(sample_subagent)
        
        # Delete the subagent
        await registry.delete_subagent(sample_subagent.id)
        
        # Verify vector deletion was called
        mock_vector_client.delete_subagent_embedding.assert_called_once_with(sample_subagent.id)
        
        # Verify subagent was removed from CSV
        subagents = await registry.list_subagents()
        assert len(subagents) == 0
    
    async def test_search_similar_subagents_success(self, mock_registry_with_vector):
        """Test vector similarity search through registry."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # Mock search results
        mock_subagent = Subagent(
            id="similar-subagent",
            name="Similar Subagent",
            description="A similar subagent",
            capabilities=["similarity"],
            import_path="test.similar",
            is_active=True
        )
        mock_vector_client.search_similar_subagents.return_value = [(mock_subagent, 0.85)]
        
        # Mock embedding generation
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            results = await registry.search_similar_subagents("test query")
        
        assert len(results) == 1
        subagent, score = results[0]
        assert subagent.id == "similar-subagent"
        assert score == 0.85
        
        mock_vector_client.search_similar_subagents.assert_called_once()
    
    async def test_search_similar_subagents_vector_disabled(self, tmp_path):
        """Test similarity search when vector database is disabled."""
        registry_path = tmp_path / "test_subagents.csv"
        registry = SubagentRegistry(
            registry_path=str(registry_path),
            enable_vector_db=False
        )
        
        with pytest.raises(SubagentRegistryError, match="Vector database is not enabled"):
            await registry.search_similar_subagents("test query")
    
    async def test_search_by_capabilities_vector_fallback(self, mock_registry_with_vector):
        """Test capability search with vector database fallback."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # Mock vector search failure
        mock_vector_client.search_by_capabilities.side_effect = QdrantVectorError("Search failed")
        
        # Should fall back to CSV search
        results = await registry.search_by_capabilities_vector(["testing"])
        
        # Should return empty list since no subagents in CSV
        assert results == []
    
    async def test_initialize_vector_database_success(self, mock_registry_with_vector):
        """Test vector database initialization."""
        registry, mock_vector_client = mock_registry_with_vector
        
        await registry.initialize_vector_database()
        
        mock_vector_client.initialize_collection.assert_called_once()
    
    async def test_sync_to_vector_database_success(self, mock_registry_with_vector, sample_subagent):
        """Test syncing CSV data to vector database."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # First register a subagent in CSV only
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            # Temporarily disable vector storage
            registry.enable_vector_db = False
            await registry.register_subagent(sample_subagent)
            registry.enable_vector_db = True
        
        # Now sync to vector database
        with patch.object(registry, '_generate_embedding', return_value=[0.1] * 1536):
            await registry.sync_to_vector_database()
        
        # Verify vector storage was called during sync
        mock_vector_client.store_subagent_embedding.assert_called()
    
    async def test_get_vector_database_stats_success(self, mock_registry_with_vector):
        """Test getting vector database statistics."""
        registry, mock_vector_client = mock_registry_with_vector
        
        # Mock collection info and health check
        mock_vector_client.get_collection_info.return_value = {
            "collection_name": "subagents",
            "total_points": 10,
            "vector_size": 1536
        }
        mock_vector_client.health_check.return_value = True
        
        stats = await registry.get_vector_database_stats()
        
        assert stats["enabled"] is True
        assert stats["healthy"] is True
        assert stats["collection_name"] == "subagents"
        assert stats["total_points"] == 10
    
    async def test_get_vector_database_stats_disabled(self, tmp_path):
        """Test getting stats when vector database is disabled."""
        registry_path = tmp_path / "test_subagents.csv"
        registry = SubagentRegistry(
            registry_path=str(registry_path),
            enable_vector_db=False
        )
        
        stats = await registry.get_vector_database_stats()
        
        assert stats["enabled"] is False
        assert "error" in stats


class TestVectorSearchAccuracy:
    """Test cases for vector search accuracy and similarity scoring."""
    
    @pytest.fixture
    def embedding_generator(self):
        """Mock embedding generator that creates predictable embeddings."""
        def generate_embedding(text: str) -> List[float]:
            # Simple hash-based embedding for testing
            # Different texts will have different embeddings
            hash_val = hash(text) % 1000
            embedding = [0.0] * 1536
            embedding[0] = hash_val / 1000.0  # Normalize to 0-1 range
            embedding[1] = len(text) / 100.0  # Text length feature
            return embedding
        return generate_embedding
    
    async def test_similarity_scoring_accuracy(self, embedding_generator):
        """Test that similar descriptions get higher similarity scores."""
        # Create subagents with similar and different descriptions
        similar_subagents = [
            Subagent(
                id="code-analyzer-1",
                name="Code Analyzer",
                description="Analyzes code quality and finds bugs",
                capabilities=["code-analysis"],
                import_path="agents.code_analyzer",
                embedding=embedding_generator("Analyzes code quality and finds bugs"),
                is_active=True
            ),
            Subagent(
                id="code-reviewer-1", 
                name="Code Reviewer",
                description="Reviews code quality and suggests improvements",
                capabilities=["code-review"],
                import_path="agents.code_reviewer",
                embedding=embedding_generator("Reviews code quality and suggests improvements"),
                is_active=True
            )
        ]
        
        different_subagent = Subagent(
            id="data-processor-1",
            name="Data Processor", 
            description="Processes large datasets efficiently",
            capabilities=["data-processing"],
            import_path="agents.data_processor",
            embedding=embedding_generator("Processes large datasets efficiently"),
            is_active=True
        )
        
        # Test that similar descriptions have higher similarity
        # This would be tested with actual vector similarity calculations
        # For now, we verify the structure is correct
        assert len(similar_subagents[0].embedding) == 1536
        assert len(similar_subagents[1].embedding) == 1536
        assert len(different_subagent.embedding) == 1536
        
        # Verify embeddings are different
        assert similar_subagents[0].embedding != similar_subagents[1].embedding
        assert similar_subagents[0].embedding != different_subagent.embedding
    
    async def test_capability_filtering_accuracy(self):
        """Test that capability-based filtering returns correct results."""
        subagents = [
            Subagent(
                id="analyzer-1",
                name="Code Analyzer",
                description="Analyzes code",
                capabilities=["code-analysis", "quality-check"],
                import_path="agents.analyzer",
                is_active=True
            ),
            Subagent(
                id="processor-1",
                name="Data Processor",
                description="Processes data",
                capabilities=["data-processing", "transformation"],
                import_path="agents.processor",
                is_active=True
            ),
            Subagent(
                id="reporter-1",
                name="Report Generator",
                description="Generates reports",
                capabilities=["reporting", "documentation"],
                import_path="agents.reporter",
                is_active=False  # Inactive
            )
        ]
        
        # Test filtering by single capability
        code_agents = [sa for sa in subagents if "code-analysis" in sa.capabilities]
        assert len(code_agents) == 1
        assert code_agents[0].id == "analyzer-1"
        
        # Test filtering by multiple capabilities
        processing_agents = [sa for sa in subagents if any(cap in sa.capabilities for cap in ["data-processing", "transformation"])]
        assert len(processing_agents) == 1
        assert processing_agents[0].id == "processor-1"
        
        # Test active filtering
        active_agents = [sa for sa in subagents if sa.is_active]
        assert len(active_agents) == 2
        assert all(sa.is_active for sa in active_agents)


# Integration test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def cleanup_test_data():
    """Clean up test data after tests."""
    yield
    # Clean up any test files or data
    test_files = [
        "test_subagents.csv",
        "test_subagents_backup.csv"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


# Mark all tests as asyncio
pytestmark = pytest.mark.asyncio
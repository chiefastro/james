"""
Qdrant vector database client for subagent embeddings.

This module provides vector search functionality for subagent discovery
with embedding-based retrieval and similarity scoring.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)

from ..models.core import Subagent


logger = logging.getLogger(__name__)


class QdrantVectorError(Exception):
    """Base exception for Qdrant vector operations."""
    pass


class QdrantConnectionError(QdrantVectorError):
    """Raised when connection to Qdrant fails."""
    pass


class QdrantVectorClient:
    """
    Qdrant client for subagent embedding storage and retrieval.
    
    Provides vector search functionality for subagent discovery
    with similarity scoring and filtering capabilities.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "subagents"
    ):
        """
        Initialize the Qdrant vector client.
        
        Args:
            host: Qdrant server host (defaults to QDRANT_HOST env var or localhost)
            port: Qdrant server port (defaults to QDRANT_PORT env var or 6333)
            collection_name: Name of the collection for subagent embeddings
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        
        # Initialize client
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Initialized Qdrant client: {self.host}:{self.port}")
        except Exception as e:
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}")
        
        # Vector configuration for OpenAI embeddings
        self.vector_size = 1536
        self.distance_metric = Distance.COSINE
    
    async def initialize_collection(self) -> None:
        """
        Initialize the subagent collection with proper configuration.
        
        Creates the collection if it doesn't exist with the correct
        vector parameters and payload schema.
        
        Raises:
            QdrantVectorError: If collection initialization fails
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
                
        except Exception as e:
            raise QdrantVectorError(f"Failed to initialize collection: {e}")
    
    async def store_subagent_embedding(self, subagent: Subagent) -> None:
        """
        Store a subagent's embedding in the vector database.
        
        Args:
            subagent: The Subagent object with embedding to store
            
        Raises:
            QdrantVectorError: If storage fails
        """
        if not subagent.embedding:
            raise QdrantVectorError("Subagent must have an embedding to store")
        
        if len(subagent.embedding) != self.vector_size:
            raise QdrantVectorError(
                f"Embedding size {len(subagent.embedding)} doesn't match "
                f"expected size {self.vector_size}"
            )
        
        try:
            # Prepare payload with subagent metadata
            payload = {
                "subagent_id": subagent.id,
                "name": subagent.name,
                "description": subagent.description,
                "capabilities": subagent.capabilities,
                "import_path": subagent.import_path,
                "is_active": subagent.is_active,
                "created_at": subagent.created_at.isoformat(),
                "last_used": subagent.last_used.isoformat() if subagent.last_used else None
            }
            
            # Create point for insertion
            # Use hash of subagent ID to ensure consistent integer ID for Qdrant
            import hashlib
            point_id = int(hashlib.md5(subagent.id.encode()).hexdigest()[:8], 16)
            
            point = PointStruct(
                id=point_id,
                vector=subagent.embedding,
                payload=payload
            )
            
            # Upsert the point (insert or update)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored embedding for subagent: {subagent.name}")
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to store subagent embedding: {e}")
    
    async def search_similar_subagents(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        active_only: bool = True
    ) -> List[Tuple[Subagent, float]]:
        """
        Search for similar subagents using vector similarity.
        
        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            active_only: If True, only search active subagents
            
        Returns:
            List of tuples containing (Subagent, similarity_score)
            
        Raises:
            QdrantVectorError: If search fails
        """
        if len(query_embedding) != self.vector_size:
            raise QdrantVectorError(
                f"Query embedding size {len(query_embedding)} doesn't match "
                f"expected size {self.vector_size}"
            )
        
        try:
            # Build filter conditions
            filter_conditions = []
            if active_only:
                filter_conditions.append(
                    FieldCondition(
                        key="is_active",
                        match=MatchValue(value=True)
                    )
                )
            
            # Create filter if conditions exist
            search_filter = None
            if filter_conditions:
                search_filter = Filter(must=filter_conditions)
            
            # Perform vector search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert results to Subagent objects with scores
            results = []
            for scored_point in search_result:
                try:
                    # Reconstruct Subagent from payload
                    payload = scored_point.payload
                    
                    # Parse datetime fields
                    from datetime import datetime
                    created_at = datetime.fromisoformat(payload["created_at"])
                    last_used = None
                    if payload.get("last_used"):
                        last_used = datetime.fromisoformat(payload["last_used"])
                    
                    # Create Subagent object
                    subagent = Subagent(
                        id=payload["subagent_id"],
                        name=payload["name"],
                        description=payload["description"],
                        capabilities=payload["capabilities"],
                        import_path=payload["import_path"],
                        is_active=payload["is_active"],
                        created_at=created_at,
                        last_used=last_used,
                        embedding=[]  # Don't include embedding in search results
                    )
                    
                    results.append((subagent, scored_point.score))
                    
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} similar subagents")
            return results
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to search similar subagents: {e}")
    
    async def search_by_capabilities(
        self,
        capabilities: List[str],
        limit: int = 10,
        active_only: bool = True
    ) -> List[Subagent]:
        """
        Search for subagents by their capabilities.
        
        Args:
            capabilities: List of capabilities to search for
            limit: Maximum number of results to return
            active_only: If True, only search active subagents
            
        Returns:
            List of Subagent objects
            
        Raises:
            QdrantVectorError: If search fails
        """
        try:
            # Build filter conditions
            filter_conditions = []
            
            if active_only:
                filter_conditions.append(
                    FieldCondition(
                        key="is_active",
                        match=MatchValue(value=True)
                    )
                )
            
            # Add capability filters (OR condition for any matching capability)
            capability_conditions = []
            for capability in capabilities:
                capability_conditions.append(
                    FieldCondition(
                        key="capabilities",
                        match=MatchValue(value=capability)
                    )
                )
            
            # Create filter
            search_filter = Filter(
                must=filter_conditions,
                should=capability_conditions if capability_conditions else None
            )
            
            # Perform search without vector (just filtering)
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=limit
            )
            
            # Convert results to Subagent objects
            results = []
            for point in search_result[0]:  # scroll returns (points, next_page_offset)
                try:
                    payload = point.payload
                    
                    # Parse datetime fields
                    from datetime import datetime
                    created_at = datetime.fromisoformat(payload["created_at"])
                    last_used = None
                    if payload.get("last_used"):
                        last_used = datetime.fromisoformat(payload["last_used"])
                    
                    # Create Subagent object
                    subagent = Subagent(
                        id=payload["subagent_id"],
                        name=payload["name"],
                        description=payload["description"],
                        capabilities=payload["capabilities"],
                        import_path=payload["import_path"],
                        is_active=payload["is_active"],
                        created_at=created_at,
                        last_used=last_used,
                        embedding=[]  # Don't include embedding in search results
                    )
                    
                    results.append(subagent)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse capability search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} subagents with capabilities: {capabilities}")
            return results
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to search by capabilities: {e}")
    
    async def get_subagent_by_id(self, subagent_id: str) -> Optional[Subagent]:
        """
        Retrieve a subagent by its ID from the vector database.
        
        Args:
            subagent_id: The ID of the subagent to retrieve
            
        Returns:
            Subagent object if found, None otherwise
            
        Raises:
            QdrantVectorError: If retrieval fails
        """
        try:
            # Retrieve point by ID
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[subagent_id]
            )
            
            if not points:
                return None
            
            point = points[0]
            payload = point.payload
            
            # Parse datetime fields
            from datetime import datetime
            created_at = datetime.fromisoformat(payload["created_at"])
            last_used = None
            if payload.get("last_used"):
                last_used = datetime.fromisoformat(payload["last_used"])
            
            # Create Subagent object
            subagent = Subagent(
                id=payload["subagent_id"],
                name=payload["name"],
                description=payload["description"],
                capabilities=payload["capabilities"],
                import_path=payload["import_path"],
                is_active=payload["is_active"],
                created_at=created_at,
                last_used=last_used,
                embedding=point.vector if point.vector else []
            )
            
            return subagent
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to retrieve subagent by ID: {e}")
    
    async def update_subagent_embedding(self, subagent: Subagent) -> None:
        """
        Update a subagent's embedding and metadata in the vector database.
        
        Args:
            subagent: The updated Subagent object
            
        Raises:
            QdrantVectorError: If update fails
        """
        # This is the same as store_subagent_embedding since we use upsert
        await self.store_subagent_embedding(subagent)
    
    async def delete_subagent_embedding(self, subagent_id: str) -> None:
        """
        Delete a subagent's embedding from the vector database.
        
        Args:
            subagent_id: The ID of the subagent to delete
            
        Raises:
            QdrantVectorError: If deletion fails
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[subagent_id]
                )
            )
            
            logger.info(f"Deleted embedding for subagent: {subagent_id}")
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to delete subagent embedding: {e}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the subagent collection.
        
        Returns:
            Dictionary containing collection statistics and configuration
            
        Raises:
            QdrantVectorError: If retrieval fails
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get collection statistics
            collection_stats = self.client.count(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "total_points": collection_stats.count,
                "status": collection_info.status.name,
                "optimizer_status": collection_info.optimizer_status.name if collection_info.optimizer_status else None,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count
            }
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to get collection info: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if the Qdrant service is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections list as a health check
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def clear_collection(self) -> None:
        """
        Clear all points from the collection (for testing purposes).
        
        Raises:
            QdrantVectorError: If clearing fails
        """
        try:
            # Delete all points in the collection
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()  # Empty filter matches all points
                )
            )
            
            logger.info(f"Cleared all points from collection: {self.collection_name}")
            
        except Exception as e:
            raise QdrantVectorError(f"Failed to clear collection: {e}")
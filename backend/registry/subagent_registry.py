"""
CSV-based subagent registry implementation.

This module provides persistent storage and retrieval of subagent metadata
using CSV files, with embedding generation and vector search capabilities.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from dataclasses import asdict
import logging

from ..models.core import Subagent
from ..vector.qdrant_client import QdrantVectorClient, QdrantVectorError

logger = logging.getLogger(__name__)


class SubagentRegistryError(Exception):
    """Base exception for subagent registry operations."""
    pass


class SubagentNotFoundError(SubagentRegistryError):
    """Raised when a subagent is not found in the registry."""
    pass


class SubagentRegistry:
    """
    CSV-based storage system for subagent metadata.
    
    Provides CRUD operations for subagent registration and retrieval,
    with embedding generation and storage capabilities.
    """
    
    def __init__(self, registry_path: Optional[str] = None, enable_vector_db: bool = True):
        """
        Initialize the subagent registry.
        
        Args:
            registry_path: Path to the CSV registry file. Defaults to ~/.james/subagents.csv
            enable_vector_db: Whether to enable Qdrant vector database integration
        """
        if registry_path is None:
            james_dir = Path.home() / ".james"
            james_dir.mkdir(exist_ok=True)
            registry_path = str(james_dir / "subagents.csv")
        
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV field names matching the Subagent model
        self.fieldnames = [
            "id", "name", "description", "input_schema", "output_schema",
            "import_path", "embedding", "capabilities", "created_at",
            "last_used", "is_active"
        ]
        
        # Initialize vector database client
        self.enable_vector_db = enable_vector_db
        self.vector_client = None
        if enable_vector_db:
            try:
                self.vector_client = QdrantVectorClient()
                logger.info("Qdrant vector client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant client: {e}")
                self.enable_vector_db = False
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self) -> None:
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.registry_path.exists():
            with open(self.registry_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
        else:
            # Check if file exists but is empty (no headers)
            if self.registry_path.stat().st_size == 0:
                with open(self.registry_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
    
    def _subagent_to_dict(self, subagent: Subagent) -> Dict[str, str]:
        """
        Convert a Subagent object to a dictionary suitable for CSV storage.
        
        Args:
            subagent: The Subagent object to convert
            
        Returns:
            Dictionary with string values for CSV storage
        """
        data = asdict(subagent)
        
        # Convert complex types to JSON strings for CSV storage
        data["input_schema"] = json.dumps(data["input_schema"])
        data["output_schema"] = json.dumps(data["output_schema"])
        data["embedding"] = json.dumps(data["embedding"])
        data["capabilities"] = json.dumps(data["capabilities"])
        
        # Convert datetime objects to ISO strings
        data["created_at"] = data["created_at"].isoformat()
        if data["last_used"]:
            data["last_used"] = data["last_used"].isoformat()
        else:
            data["last_used"] = ""
        
        # Convert boolean to string
        data["is_active"] = str(data["is_active"])
        
        return data
    
    def _dict_to_subagent(self, data: Dict[str, str]) -> Subagent:
        """
        Convert a dictionary from CSV storage to a Subagent object.
        
        Args:
            data: Dictionary with string values from CSV
            
        Returns:
            Subagent object
        """
        # Validate that all required fields are present
        required_fields = ["id", "name", "description", "input_schema", "output_schema", 
                          "import_path", "embedding", "capabilities", "created_at", 
                          "last_used", "is_active"]
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        try:
            # Parse JSON strings back to complex types
            input_schema = json.loads(data["input_schema"]) if data["input_schema"] else {}
            output_schema = json.loads(data["output_schema"]) if data["output_schema"] else {}
            embedding = json.loads(data["embedding"]) if data["embedding"] else []
            capabilities = json.loads(data["capabilities"]) if data["capabilities"] else []
            
            # Parse datetime strings
            created_at = datetime.fromisoformat(data["created_at"])
            last_used = datetime.fromisoformat(data["last_used"]) if data["last_used"] else None
            
            # Parse boolean
            is_active = data["is_active"].lower() == "true"
            
            return Subagent(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                input_schema=input_schema,
                output_schema=output_schema,
                import_path=data["import_path"],
                embedding=embedding,
                capabilities=capabilities,
                created_at=created_at,
                last_used=last_used,
                is_active=is_active
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse subagent data: {e}")
    
    async def register_subagent(self, subagent: Subagent) -> None:
        """
        Register a new subagent in the registry.
        
        Args:
            subagent: The Subagent object to register
            
        Raises:
            SubagentRegistryError: If registration fails
        """
        # Check if subagent already exists by reading all subagents
        existing_subagents = await self.list_subagents(active_only=False)
        for existing in existing_subagents:
            if existing.id == subagent.id:
                raise SubagentRegistryError(f"Subagent with ID {subagent.id} already exists")
        
        # Generate embedding if not provided
        if not subagent.embedding:
            subagent.embedding = await self._generate_embedding(subagent.description)
        
        # Write to CSV
        try:
            with open(self.registry_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(self._subagent_to_dict(subagent))
        except Exception as e:
            raise SubagentRegistryError(f"Failed to register subagent: {e}")
        
        # Store in vector database if enabled
        if self.enable_vector_db and self.vector_client:
            try:
                await self.vector_client.store_subagent_embedding(subagent)
                logger.info(f"Stored subagent embedding in vector DB: {subagent.name}")
            except QdrantVectorError as e:
                logger.warning(f"Failed to store embedding in vector DB: {e}")
                # Don't fail the registration if vector storage fails
    
    async def get_subagent_by_id(self, subagent_id: str) -> Optional[Subagent]:
        """
        Retrieve a subagent by its ID.
        
        Args:
            subagent_id: The ID of the subagent to retrieve
            
        Returns:
            Subagent object if found, None otherwise
            
        Raises:
            SubagentNotFoundError: If subagent is not found
        """
        try:
            with open(self.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Skip empty rows or rows with missing data
                        if not row or not row.get("id"):
                            continue
                        
                        if row["id"] == subagent_id:
                            return self._dict_to_subagent(row)
                    except (ValueError, KeyError):
                        # Skip malformed rows but continue processing
                        continue
        except FileNotFoundError:
            raise SubagentNotFoundError(f"Registry file not found: {self.registry_path}")
        except Exception as e:
            raise SubagentRegistryError(f"Failed to retrieve subagent: {e}")
        
        raise SubagentNotFoundError(f"Subagent with ID {subagent_id} not found")
    
    async def get_subagent_by_name(self, name: str) -> Optional[Subagent]:
        """
        Retrieve a subagent by its name.
        
        Args:
            name: The name of the subagent to retrieve
            
        Returns:
            Subagent object if found, None otherwise
            
        Raises:
            SubagentNotFoundError: If subagent is not found
        """
        try:
            with open(self.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Skip empty rows or rows with missing data
                        if not row or not row.get("name"):
                            continue
                        
                        if row["name"] == name:
                            return self._dict_to_subagent(row)
                    except (ValueError, KeyError):
                        # Skip malformed rows but continue processing
                        continue
        except FileNotFoundError:
            raise SubagentNotFoundError(f"Registry file not found: {self.registry_path}")
        except Exception as e:
            raise SubagentRegistryError(f"Failed to retrieve subagent: {e}")
        
        raise SubagentNotFoundError(f"Subagent with name '{name}' not found")
    
    async def list_subagents(self, active_only: bool = True) -> List[Subagent]:
        """
        List all subagents in the registry.
        
        Args:
            active_only: If True, only return active subagents
            
        Returns:
            List of Subagent objects
            
        Raises:
            SubagentRegistryError: If listing fails
        """
        subagents = []
        
        try:
            with open(self.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Skip empty rows or rows with missing data
                        if not row or not row.get("id"):
                            continue
                        
                        subagent = self._dict_to_subagent(row)
                        if not active_only or subagent.is_active:
                            subagents.append(subagent)
                    except (ValueError, KeyError) as e:
                        # Skip malformed rows but continue processing
                        continue
        except FileNotFoundError:
            # Return empty list if file doesn't exist
            return []
        except Exception as e:
            raise SubagentRegistryError(f"Failed to list subagents: {e}")
        
        return subagents
    
    def list_all(self) -> list:
        """
        Synchronous method to list all subagents as dicts (for legacy compatibility).
        Returns:
            List of dicts representing subagents.
        """
        subagents = []
        try:
            with open(self.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Skip empty rows or rows with missing data
                        if not row or not row.get("id"):
                            continue
                        subagents.append(row)
                    except Exception:
                        continue
        except FileNotFoundError:
            return []
        except Exception as e:
            raise SubagentRegistryError(f"Failed to list subagents: {e}")
        return subagents
    
    async def update_subagent(self, subagent: Subagent) -> None:
        """
        Update an existing subagent in the registry.
        
        Args:
            subagent: The updated Subagent object
            
        Raises:
            SubagentNotFoundError: If subagent is not found
            SubagentRegistryError: If update fails
        """
        # Read all subagents
        subagents = []
        found = False
        
        try:
            with open(self.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Skip empty rows or rows with missing data
                        if not row or not row.get("id"):
                            continue
                        
                        if row["id"] == subagent.id:
                            # Update embedding if description changed
                            existing = self._dict_to_subagent(row)
                            if existing.description != subagent.description and not subagent.embedding:
                                subagent.embedding = await self._generate_embedding(subagent.description)
                            
                            subagents.append(subagent)
                            found = True
                        else:
                            subagents.append(self._dict_to_subagent(row))
                    except (ValueError, KeyError):
                        # Skip malformed rows but continue processing
                        continue
        except FileNotFoundError:
            raise SubagentNotFoundError(f"Registry file not found: {self.registry_path}")
        except Exception as e:
            raise SubagentRegistryError(f"Failed to read registry for update: {e}")
        
        if not found:
            raise SubagentNotFoundError(f"Subagent with ID {subagent.id} not found")
        
        # Write all subagents back to CSV
        try:
            with open(self.registry_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                for sa in subagents:
                    writer.writerow(self._subagent_to_dict(sa))
        except Exception as e:
            raise SubagentRegistryError(f"Failed to update subagent: {e}")
        
        # Update in vector database if enabled
        if self.enable_vector_db and self.vector_client:
            try:
                await self.vector_client.update_subagent_embedding(subagent)
                logger.info(f"Updated subagent embedding in vector DB: {subagent.name}")
            except QdrantVectorError as e:
                logger.warning(f"Failed to update embedding in vector DB: {e}")
                # Don't fail the update if vector storage fails
    
    async def delete_subagent(self, subagent_id: str) -> None:
        """
        Delete a subagent from the registry.
        
        Args:
            subagent_id: The ID of the subagent to delete
            
        Raises:
            SubagentNotFoundError: If subagent is not found
            SubagentRegistryError: If deletion fails
        """
        # Read all subagents except the one to delete
        subagents = []
        found = False
        
        try:
            with open(self.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Skip empty rows or rows with missing data
                        if not row or not row.get("id"):
                            continue
                        
                        if row["id"] == subagent_id:
                            found = True
                        else:
                            subagents.append(self._dict_to_subagent(row))
                    except (ValueError, KeyError):
                        # Skip malformed rows but continue processing
                        continue
        except FileNotFoundError:
            raise SubagentNotFoundError(f"Registry file not found: {self.registry_path}")
        except Exception as e:
            raise SubagentRegistryError(f"Failed to read registry for deletion: {e}")
        
        if not found:
            raise SubagentNotFoundError(f"Subagent with ID {subagent_id} not found")
        
        # Write remaining subagents back to CSV
        try:
            with open(self.registry_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                for sa in subagents:
                    writer.writerow(self._subagent_to_dict(sa))
        except Exception as e:
            raise SubagentRegistryError(f"Failed to delete subagent: {e}")
        
        # Delete from vector database if enabled
        if self.enable_vector_db and self.vector_client:
            try:
                await self.vector_client.delete_subagent_embedding(subagent_id)
                logger.info(f"Deleted subagent embedding from vector DB: {subagent_id}")
            except QdrantVectorError as e:
                logger.warning(f"Failed to delete embedding from vector DB: {e}")
                # Don't fail the deletion if vector storage fails
    
    async def mark_subagent_used(self, subagent_id: str) -> None:
        """
        Mark a subagent as recently used.
        
        Args:
            subagent_id: The ID of the subagent to mark as used
            
        Raises:
            SubagentNotFoundError: If subagent is not found
            SubagentRegistryError: If update fails
        """
        subagent = await self.get_subagent_by_id(subagent_id)
        subagent.mark_used()
        await self.update_subagent(subagent)
    
    async def search_by_capabilities(self, capabilities: List[str]) -> List[Subagent]:
        """
        Search for subagents by their capabilities.
        
        Args:
            capabilities: List of capabilities to search for
            
        Returns:
            List of Subagent objects that have any of the specified capabilities
            
        Raises:
            SubagentRegistryError: If search fails
        """
        matching_subagents = []
        
        try:
            subagents = await self.list_subagents(active_only=True)
            for subagent in subagents:
                # Check if any of the subagent's capabilities match the search criteria
                if any(cap in subagent.capabilities for cap in capabilities):
                    matching_subagents.append(subagent)
        except Exception as e:
            raise SubagentRegistryError(f"Failed to search by capabilities: {e}")
        
        return matching_subagents
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the subagent registry.
        
        Returns:
            Dictionary containing registry statistics
            
        Raises:
            SubagentRegistryError: If stats retrieval fails
        """
        try:
            all_subagents = await self.list_subagents(active_only=False)
            active_subagents = [sa for sa in all_subagents if sa.is_active]
            
            # Calculate usage statistics
            used_subagents = [sa for sa in active_subagents if sa.last_used is not None]
            
            return {
                "total_subagents": len(all_subagents),
                "active_subagents": len(active_subagents),
                "inactive_subagents": len(all_subagents) - len(active_subagents),
                "used_subagents": len(used_subagents),
                "unused_subagents": len(active_subagents) - len(used_subagents),
                "registry_file_path": str(self.registry_path),
                "registry_file_exists": self.registry_path.exists(),
                "registry_file_size": self.registry_path.stat().st_size if self.registry_path.exists() else 0
            }
        except Exception as e:
            raise SubagentRegistryError(f"Failed to get registry stats: {e}")
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.
        
        This is a placeholder implementation that returns a zero vector.
        In a real implementation, this would call an embedding service like OpenAI.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        # Placeholder implementation - returns zero vector of OpenAI embedding dimension
        # In production, this would call OpenAI's embedding API or similar service
        return [0.0] * 1536
    
    async def search_similar_subagents(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        active_only: bool = True
    ) -> List[Tuple[Subagent, float]]:
        """
        Search for similar subagents using vector similarity.
        
        Args:
            query_text: Text to search for similar subagents
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            active_only: If True, only search active subagents
            
        Returns:
            List of tuples containing (Subagent, similarity_score)
            
        Raises:
            SubagentRegistryError: If search fails
        """
        if not self.enable_vector_db or not self.vector_client:
            raise SubagentRegistryError("Vector database is not enabled")
        
        try:
            # Generate embedding for query text
            query_embedding = await self._generate_embedding(query_text)
            
            # Search using vector client
            results = await self.vector_client.search_similar_subagents(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                active_only=active_only
            )
            
            return results
            
        except QdrantVectorError as e:
            raise SubagentRegistryError(f"Vector search failed: {e}")
        except Exception as e:
            raise SubagentRegistryError(f"Failed to search similar subagents: {e}")
    
    async def search_by_capabilities_vector(
        self,
        capabilities: List[str],
        limit: int = 10,
        active_only: bool = True
    ) -> List[Subagent]:
        """
        Search for subagents by capabilities using vector database.
        
        Args:
            capabilities: List of capabilities to search for
            limit: Maximum number of results to return
            active_only: If True, only search active subagents
            
        Returns:
            List of Subagent objects
            
        Raises:
            SubagentRegistryError: If search fails
        """
        if not self.enable_vector_db or not self.vector_client:
            # Fall back to CSV-based search
            return await self.search_by_capabilities(capabilities)
        
        try:
            results = await self.vector_client.search_by_capabilities(
                capabilities=capabilities,
                limit=limit,
                active_only=active_only
            )
            
            return results
            
        except QdrantVectorError as e:
            logger.warning(f"Vector capability search failed, falling back to CSV: {e}")
            # Fall back to CSV-based search
            return await self.search_by_capabilities(capabilities)
        except Exception as e:
            raise SubagentRegistryError(f"Failed to search by capabilities: {e}")
    
    async def initialize_vector_database(self) -> None:
        """
        Initialize the vector database collection and sync existing data.
        
        Raises:
            SubagentRegistryError: If initialization fails
        """
        if not self.enable_vector_db or not self.vector_client:
            raise SubagentRegistryError("Vector database is not enabled")
        
        try:
            # Initialize collection
            await self.vector_client.initialize_collection()
            
            # Sync existing subagents to vector database
            await self.sync_to_vector_database()
            
            logger.info("Vector database initialized and synced successfully")
            
        except QdrantVectorError as e:
            raise SubagentRegistryError(f"Failed to initialize vector database: {e}")
        except Exception as e:
            raise SubagentRegistryError(f"Failed to initialize vector database: {e}")
    
    async def sync_to_vector_database(self) -> None:
        """
        Sync all subagents from CSV to vector database.
        
        Raises:
            SubagentRegistryError: If sync fails
        """
        if not self.enable_vector_db or not self.vector_client:
            raise SubagentRegistryError("Vector database is not enabled")
        
        try:
            # Get all subagents from CSV
            subagents = await self.list_subagents(active_only=False)
            
            synced_count = 0
            for subagent in subagents:
                try:
                    # Generate embedding if missing
                    if not subagent.embedding:
                        subagent.embedding = await self._generate_embedding(subagent.description)
                        # Update CSV with new embedding
                        await self.update_subagent(subagent)
                    
                    # Store in vector database
                    await self.vector_client.store_subagent_embedding(subagent)
                    synced_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to sync subagent {subagent.name}: {e}")
                    continue
            
            logger.info(f"Synced {synced_count}/{len(subagents)} subagents to vector database")
            
        except Exception as e:
            raise SubagentRegistryError(f"Failed to sync to vector database: {e}")
    
    async def get_vector_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary containing vector database statistics
            
        Raises:
            SubagentRegistryError: If stats retrieval fails
        """
        if not self.enable_vector_db or not self.vector_client:
            return {"enabled": False, "error": "Vector database is not enabled"}
        
        try:
            # Get collection info
            collection_info = await self.vector_client.get_collection_info()
            
            # Add health check
            is_healthy = await self.vector_client.health_check()
            
            return {
                "enabled": True,
                "healthy": is_healthy,
                **collection_info
            }
            
        except QdrantVectorError as e:
            return {"enabled": True, "healthy": False, "error": str(e)}
        except Exception as e:
            raise SubagentRegistryError(f"Failed to get vector database stats: {e}")
    
    async def backup_registry(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the registry file.
        
        Args:
            backup_path: Path for the backup file. If None, creates timestamped backup
            
        Returns:
            Path to the backup file
            
        Raises:
            SubagentRegistryError: If backup fails
        """
        if backup_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.registry_path.parent / f"subagents_backup_{timestamp}.csv")
        
        try:
            import shutil
            shutil.copy2(self.registry_path, backup_path)
            return backup_path
        except Exception as e:
            raise SubagentRegistryError(f"Failed to backup registry: {e}")
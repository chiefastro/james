"""Memory management system for James using Mem0 and Qdrant."""

import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import json
from pathlib import Path

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class MemorySystem:
    """Integrated memory system using Mem0 for semantic memory and Qdrant for vector storage."""
    
    def __init__(self, james_home: str = "~/.james") -> None:
        self.james_home = Path(james_home).expanduser()
        self.james_home.mkdir(parents=True, exist_ok=True)
        
        # Initialize Mem0 if available
        self.mem0 = None
        if MEM0_AVAILABLE:
            try:
                self.mem0 = Memory()
            except Exception as e:
                print(f"Failed to initialize Mem0: {e}")
        
        # Initialize Qdrant if available
        self.qdrant = None
        if QDRANT_AVAILABLE:
            try:
                qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                
                self.qdrant = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key if qdrant_api_key else None
                )
                
                # Create collection if it doesn't exist
                self._initialize_qdrant_collection()
                
            except Exception as e:
                print(f"Failed to initialize Qdrant: {e}")
        
        # Fallback to local file storage
        self.memory_file = self.james_home / "memory.jsonl"
        self.episodic_file = self.james_home / "episodic_memory.jsonl"
        self.semantic_file = self.james_home / "semantic_memory.jsonl"
    
    def _initialize_qdrant_collection(self) -> None:
        """Initialize Qdrant collection for memory storage."""
        if not self.qdrant:
            return
        
        try:
            collections = self.qdrant.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if "james_memory" not in collection_names:
                self.qdrant.create_collection(
                    collection_name="james_memory",
                    vectors_config=VectorParams(
                        size=384,  # MiniLM embedding size
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Failed to initialize Qdrant collection: {e}")
    
    async def store_memory(self, content: str, memory_type: str = "episodic", 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store a memory with automatic categorization and embedding."""
        timestamp = datetime.now()
        memory_id = f"{memory_type}_{timestamp.isoformat()}_{hash(content) % 10000}"
        
        memory_entry = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "metadata": metadata or {},
            "timestamp": timestamp.isoformat(),
            "importance": self._calculate_importance(content, metadata)
        }
        
        results = {"local": False, "mem0": False, "qdrant": False}
        
        # Store in Mem0 if available
        if self.mem0:
            try:
                self.mem0.add(
                    messages=[{"role": "user", "content": content}],
                    user_id="james",
                    metadata=metadata
                )
                results["mem0"] = True
            except Exception as e:
                print(f"Failed to store in Mem0: {e}")
        
        # Store in Qdrant if available
        if self.qdrant:
            try:
                # Generate embedding (placeholder - would use actual embedding model)
                embedding = [0.0] * 384  # Placeholder embedding
                
                point = PointStruct(
                    id=hash(memory_id) % (2**32),
                    vector=embedding,
                    payload={
                        "content": content,
                        "type": memory_type,
                        "metadata": metadata or {},
                        "timestamp": timestamp.isoformat(),
                        "importance": memory_entry["importance"]
                    }
                )
                
                self.qdrant.upsert(
                    collection_name="james_memory",
                    points=[point]
                )
                results["qdrant"] = True
                
            except Exception as e:
                print(f"Failed to store in Qdrant: {e}")
        
        # Always store locally as fallback
        try:
            target_file = self.episodic_file if memory_type == "episodic" else self.semantic_file
            with open(target_file, 'a') as f:
                f.write(json.dumps(memory_entry) + '\n')
            results["local"] = True
        except Exception as e:
            print(f"Failed to store locally: {e}")
        
        return {
            "memory_id": memory_id,
            "stored": results,
            "timestamp": timestamp.isoformat()
        }
    
    async def retrieve_memories(self, query: str, memory_type: Optional[str] = None, 
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query."""
        results = []
        
        # Try Mem0 first
        if self.mem0:
            try:
                mem0_results = self.mem0.search(
                    query=query,
                    user_id="james",
                    limit=limit
                )
                for result in mem0_results:
                    results.append({
                        "content": result.get("memory", ""),
                        "score": result.get("score", 0.0),
                        "source": "mem0",
                        "metadata": result.get("metadata", {})
                    })
            except Exception as e:
                print(f"Failed to search Mem0: {e}")
        
        # Try Qdrant
        if self.qdrant and len(results) < limit:
            try:
                # Generate query embedding (placeholder)
                query_embedding = [0.0] * 384  # Placeholder
                
                search_results = self.qdrant.search(
                    collection_name="james_memory",
                    query_vector=query_embedding,
                    limit=limit - len(results)
                )
                
                for result in search_results:
                    if memory_type is None or result.payload.get("type") == memory_type:
                        results.append({
                            "content": result.payload.get("content", ""),
                            "score": result.score,
                            "source": "qdrant",
                            "type": result.payload.get("type", ""),
                            "metadata": result.payload.get("metadata", {}),
                            "timestamp": result.payload.get("timestamp", "")
                        })
                        
            except Exception as e:
                print(f"Failed to search Qdrant: {e}")
        
        # Fallback to local search
        if len(results) < limit:
            local_results = await self._search_local_memories(query, memory_type, limit - len(results))
            results.extend(local_results)
        
        # Sort by relevance score and return
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:limit]
    
    async def _search_local_memories(self, query: str, memory_type: Optional[str], 
                                   limit: int) -> List[Dict[str, Any]]:
        """Search local memory files."""
        results = []
        query_lower = query.lower()
        
        files_to_search = []
        if memory_type == "episodic":
            files_to_search = [self.episodic_file]
        elif memory_type == "semantic":
            files_to_search = [self.semantic_file]
        else:
            files_to_search = [self.episodic_file, self.semantic_file]
        
        for file_path in files_to_search:
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                            
                        memory_entry = json.loads(line.strip())
                        content = memory_entry.get("content", "").lower()
                        
                        # Simple relevance scoring based on keyword matching
                        score = 0.0
                        for word in query_lower.split():
                            if word in content:
                                score += 1.0
                        
                        if score > 0:
                            results.append({
                                "content": memory_entry.get("content", ""),
                                "score": score / len(query_lower.split()),
                                "source": "local",
                                "type": memory_entry.get("type", ""),
                                "metadata": memory_entry.get("metadata", {}),
                                "timestamp": memory_entry.get("timestamp", "")
                            })
                            
            except Exception as e:
                print(f"Failed to search {file_path}: {e}")
        
        return results
    
    def _calculate_importance(self, content: str, metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate importance score for a memory."""
        importance = 0.5  # Base importance
        
        # Increase importance based on content characteristics
        if len(content) > 100:
            importance += 0.1
        
        # Check for important keywords
        important_keywords = ["error", "success", "learn", "remember", "important", "critical"]
        for keyword in important_keywords:
            if keyword.lower() in content.lower():
                importance += 0.1
        
        # Consider metadata
        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.2
            if metadata.get("user_interaction"):
                importance += 0.1
        
        return min(importance, 1.0)  # Cap at 1.0
    
    async def store_episodic_memory(self, event: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store an episodic memory of an event."""
        return await self.store_memory(
            content=f"Event: {event}",
            memory_type="episodic",
            metadata={**context, "event_type": "episodic"}
        )
    
    async def store_semantic_memory(self, fact: str, category: str = "general") -> Dict[str, Any]:
        """Store semantic knowledge."""
        return await self.store_memory(
            content=fact,
            memory_type="semantic",
            metadata={"category": category, "knowledge_type": "semantic"}
        )
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        stats = {
            "total_memories": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "storage_systems": {
                "mem0_available": MEM0_AVAILABLE and self.mem0 is not None,
                "qdrant_available": QDRANT_AVAILABLE and self.qdrant is not None,
                "local_storage": True
            }
        }
        
        # Count local memories
        for file_path in [self.episodic_file, self.semantic_file]:
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            memory_entry = json.loads(line.strip())
                            stats["total_memories"] += 1
                            if memory_entry.get("type") == "episodic":
                                stats["episodic_count"] += 1
                            elif memory_entry.get("type") == "semantic":
                                stats["semantic_count"] += 1
                except Exception as e:
                    print(f"Failed to count memories in {file_path}: {e}")
        
        return stats
"""Subagent registry with CSV storage and vector search."""

import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


@dataclass
class SubAgentMetadata:
    id: str
    name: str
    description: str
    capabilities: List[str]
    import_path: str
    input_format: str
    output_format: str
    embedding_vector: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""


class SubAgentRegistry:
    """Registry for managing subagent metadata with vector search capabilities."""
    
    def __init__(self, james_home: str = "~/.james") -> None:
        self.james_home = Path(james_home).expanduser()
        self.registry_file = self.james_home / "subagents.csv"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure james home directory exists
        self.james_home.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not self.registry_file.exists():
            self._create_empty_registry()
    
    def _create_empty_registry(self) -> None:
        """Create empty CSV registry file."""
        fieldnames = [
            'id', 'name', 'description', 'capabilities', 'import_path',
            'input_format', 'output_format', 'embedding_vector', 
            'created_at', 'updated_at'
        ]
        
        with open(self.registry_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def register_subagent(self, metadata: SubAgentMetadata) -> None:
        """Register a new subagent in the registry."""
        # Generate embedding for description + capabilities
        text_to_embed = f"{metadata.description} {' '.join(metadata.capabilities)}"
        embedding = self.embedding_model.encode(text_to_embed).tolist()
        metadata.embedding_vector = embedding
        
        # Convert to dict for CSV storage
        data = asdict(metadata)
        data['capabilities'] = '|'.join(metadata.capabilities)  # Join list for CSV
        data['embedding_vector'] = ','.join(map(str, embedding))  # Join floats for CSV
        
        # Append to CSV
        fieldnames = list(data.keys())
        with open(self.registry_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(data)
    
    def search_subagents(self, query: str, top_k: int = 5) -> List[SubAgentMetadata]:
        """Search for subagents using vector similarity."""
        # Load all subagents
        df = pd.read_csv(self.registry_file)
        if df.empty:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for _, row in df.iterrows():
            if pd.isna(row['embedding_vector']):
                continue
                
            # Parse embedding vector
            embedding_str = row['embedding_vector']
            embedding_vector = np.array([float(x) for x in embedding_str.split(',')])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding_vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding_vector)
            )
            similarities.append((similarity, row))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, row in similarities[:top_k]:
            # Convert back to SubAgentMetadata
            capabilities = row['capabilities'].split('|') if row['capabilities'] else []
            metadata = SubAgentMetadata(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                capabilities=capabilities,
                import_path=row['import_path'],
                input_format=row['input_format'],
                output_format=row['output_format'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            results.append(metadata)
            
        return results
    
    def get_all_subagents(self) -> List[SubAgentMetadata]:
        """Get all registered subagents."""
        df = pd.read_csv(self.registry_file)
        if df.empty:
            return []
            
        results = []
        for _, row in df.iterrows():
            capabilities = row['capabilities'].split('|') if row['capabilities'] else []
            metadata = SubAgentMetadata(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                capabilities=capabilities,
                import_path=row['import_path'],
                input_format=row['input_format'],
                output_format=row['output_format'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            results.append(metadata)
            
        return results
    
    def get_subagent(self, agent_id: str) -> Optional[SubAgentMetadata]:
        """Get a specific subagent by ID."""
        df = pd.read_csv(self.registry_file)
        agent_row = df[df['id'] == agent_id]
        
        if agent_row.empty:
            return None
            
        row = agent_row.iloc[0]
        capabilities = row['capabilities'].split('|') if row['capabilities'] else []
        
        return SubAgentMetadata(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            capabilities=capabilities,
            import_path=row['import_path'],
            input_format=row['input_format'],
            output_format=row['output_format'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )
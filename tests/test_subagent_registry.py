"""
Unit tests for the SubagentRegistry class.

Tests CSV-based storage, CRUD operations, embedding generation,
and data persistence for subagent metadata.
"""

import asyncio
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock
import pytest

from backend.models.core import Subagent
from backend.registry.subagent_registry import (
    SubagentRegistry,
    SubagentRegistryError,
    SubagentNotFoundError
)


@pytest.fixture
def temp_registry_path():
    """Create a temporary file path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        temp_path = tmp.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_subagent():
    """Create a sample subagent for testing."""
    return Subagent(
        id="test-agent-1",
        name="Test Agent",
        description="A test subagent for unit testing",
        input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
        import_path="tests.mock_agents.test_agent",
        capabilities=["testing", "validation"],
        embedding=[0.1] * 1536,  # Valid embedding dimension
        is_active=True
    )


@pytest.fixture
def registry(temp_registry_path):
    """Create a SubagentRegistry instance with temporary storage."""
    return SubagentRegistry(registry_path=temp_registry_path)


class TestSubagentRegistryInitialization:
    """Test registry initialization and setup."""
    
    def test_init_with_custom_path(self, temp_registry_path):
        """Test initialization with custom registry path."""
        registry = SubagentRegistry(registry_path=temp_registry_path)
        assert registry.registry_path == Path(temp_registry_path)
        assert registry.registry_path.exists()
    
    def test_init_with_default_path(self):
        """Test initialization with default path."""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/tmp/test_home")
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                registry = SubagentRegistry()
                expected_path = Path("/tmp/test_home/.james/subagents.csv")
                assert registry.registry_path == expected_path
                mock_mkdir.assert_called()
    
    def test_csv_headers_created(self, temp_registry_path):
        """Test that CSV file is created with proper headers."""
        SubagentRegistry(registry_path=temp_registry_path)
        
        with open(temp_registry_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            expected_headers = [
                "id", "name", "description", "input_schema", "output_schema",
                "import_path", "embedding", "capabilities", "created_at",
                "last_used", "is_active"
            ]
            assert headers == expected_headers


class TestSubagentRegistration:
    """Test subagent registration operations."""
    
    @pytest.mark.asyncio
    async def test_register_new_subagent(self, registry, sample_subagent):
        """Test registering a new subagent."""
        await registry.register_subagent(sample_subagent)
        
        # Verify subagent was registered
        retrieved = await registry.get_subagent_by_id(sample_subagent.id)
        assert retrieved is not None
        assert retrieved.id == sample_subagent.id
        assert retrieved.name == sample_subagent.name
        assert retrieved.description == sample_subagent.description
    
    @pytest.mark.asyncio
    async def test_register_duplicate_subagent(self, registry, sample_subagent):
        """Test that registering duplicate subagent raises error."""
        await registry.register_subagent(sample_subagent)
        
        with pytest.raises(SubagentRegistryError, match="already exists"):
            await registry.register_subagent(sample_subagent)
    
    @pytest.mark.asyncio
    async def test_register_subagent_generates_embedding(self, registry):
        """Test that embedding is generated if not provided."""
        subagent = Subagent(
            name="No Embedding Agent",
            description="Agent without embedding",
            import_path="test.path"
        )
        
        with patch.object(registry, '_generate_embedding', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [0.5] * 1536
            await registry.register_subagent(subagent)
            
            mock_gen.assert_called_once_with(subagent.description)
            assert subagent.embedding == [0.5] * 1536


class TestSubagentRetrieval:
    """Test subagent retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_get_subagent_by_id_found(self, registry, sample_subagent):
        """Test retrieving subagent by ID when it exists."""
        await registry.register_subagent(sample_subagent)
        
        retrieved = await registry.get_subagent_by_id(sample_subagent.id)
        assert retrieved is not None
        assert retrieved.id == sample_subagent.id
        assert retrieved.name == sample_subagent.name
    
    @pytest.mark.asyncio
    async def test_get_subagent_by_id_not_found(self, registry):
        """Test retrieving subagent by ID when it doesn't exist."""
        with pytest.raises(SubagentNotFoundError):
            await registry.get_subagent_by_id("nonexistent-id")
    
    @pytest.mark.asyncio
    async def test_get_subagent_by_name_found(self, registry, sample_subagent):
        """Test retrieving subagent by name when it exists."""
        await registry.register_subagent(sample_subagent)
        
        retrieved = await registry.get_subagent_by_name(sample_subagent.name)
        assert retrieved is not None
        assert retrieved.name == sample_subagent.name
        assert retrieved.id == sample_subagent.id
    
    @pytest.mark.asyncio
    async def test_get_subagent_by_name_not_found(self, registry):
        """Test retrieving subagent by name when it doesn't exist."""
        with pytest.raises(SubagentNotFoundError):
            await registry.get_subagent_by_name("Nonexistent Agent")
    
    @pytest.mark.asyncio
    async def test_list_subagents_empty(self, registry):
        """Test listing subagents when registry is empty."""
        subagents = await registry.list_subagents()
        assert subagents == []
    
    @pytest.mark.asyncio
    async def test_list_subagents_with_data(self, registry, sample_subagent):
        """Test listing subagents when registry has data."""
        await registry.register_subagent(sample_subagent)
        
        subagents = await registry.list_subagents()
        assert len(subagents) == 1
        assert subagents[0].id == sample_subagent.id
    
    @pytest.mark.asyncio
    async def test_list_subagents_active_only(self, registry):
        """Test listing only active subagents."""
        active_agent = Subagent(
            name="Active Agent",
            description="Active agent",
            import_path="test.active",
            is_active=True
        )
        inactive_agent = Subagent(
            name="Inactive Agent", 
            description="Inactive agent",
            import_path="test.inactive",
            is_active=False
        )
        
        await registry.register_subagent(active_agent)
        await registry.register_subagent(inactive_agent)
        
        # Test active only (default)
        active_subagents = await registry.list_subagents(active_only=True)
        assert len(active_subagents) == 1
        assert active_subagents[0].name == "Active Agent"
        
        # Test all subagents
        all_subagents = await registry.list_subagents(active_only=False)
        assert len(all_subagents) == 2


class TestSubagentUpdate:
    """Test subagent update operations."""
    
    @pytest.mark.asyncio
    async def test_update_existing_subagent(self, registry, sample_subagent):
        """Test updating an existing subagent."""
        await registry.register_subagent(sample_subagent)
        
        # Update the subagent
        sample_subagent.description = "Updated description"
        sample_subagent.capabilities.append("new_capability")
        
        await registry.update_subagent(sample_subagent)
        
        # Verify update
        retrieved = await registry.get_subagent_by_id(sample_subagent.id)
        assert retrieved.description == "Updated description"
        assert "new_capability" in retrieved.capabilities
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_subagent(self, registry, sample_subagent):
        """Test updating a subagent that doesn't exist."""
        with pytest.raises(SubagentNotFoundError):
            await registry.update_subagent(sample_subagent)
    
    @pytest.mark.asyncio
    async def test_update_regenerates_embedding_on_description_change(self, registry, sample_subagent):
        """Test that embedding is regenerated when description changes."""
        await registry.register_subagent(sample_subagent)
        
        # Change description and clear embedding
        sample_subagent.description = "Completely new description"
        sample_subagent.embedding = []
        
        with patch.object(registry, '_generate_embedding', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [0.9] * 1536
            await registry.update_subagent(sample_subagent)
            
            mock_gen.assert_called_once_with("Completely new description")


class TestSubagentDeletion:
    """Test subagent deletion operations."""
    
    @pytest.mark.asyncio
    async def test_delete_existing_subagent(self, registry, sample_subagent):
        """Test deleting an existing subagent."""
        await registry.register_subagent(sample_subagent)
        
        # Verify it exists
        retrieved = await registry.get_subagent_by_id(sample_subagent.id)
        assert retrieved is not None
        
        # Delete it
        await registry.delete_subagent(sample_subagent.id)
        
        # Verify it's gone
        with pytest.raises(SubagentNotFoundError):
            await registry.get_subagent_by_id(sample_subagent.id)
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_subagent(self, registry):
        """Test deleting a subagent that doesn't exist."""
        with pytest.raises(SubagentNotFoundError):
            await registry.delete_subagent("nonexistent-id")


class TestSubagentUsageTracking:
    """Test subagent usage tracking operations."""
    
    @pytest.mark.asyncio
    async def test_mark_subagent_used(self, registry, sample_subagent):
        """Test marking a subagent as used."""
        # Initially last_used should be None
        assert sample_subagent.last_used is None
        
        await registry.register_subagent(sample_subagent)
        await registry.mark_subagent_used(sample_subagent.id)
        
        # Verify last_used was updated
        retrieved = await registry.get_subagent_by_id(sample_subagent.id)
        assert retrieved.last_used is not None
        assert isinstance(retrieved.last_used, datetime)
    
    @pytest.mark.asyncio
    async def test_mark_nonexistent_subagent_used(self, registry):
        """Test marking a nonexistent subagent as used."""
        with pytest.raises(SubagentNotFoundError):
            await registry.mark_subagent_used("nonexistent-id")


class TestSubagentSearch:
    """Test subagent search operations."""
    
    @pytest.mark.asyncio
    async def test_search_by_capabilities(self, registry):
        """Test searching subagents by capabilities."""
        agent1 = Subagent(
            name="Agent 1",
            description="First agent",
            import_path="test.agent1",
            capabilities=["search", "analysis"]
        )
        agent2 = Subagent(
            name="Agent 2",
            description="Second agent", 
            import_path="test.agent2",
            capabilities=["generation", "writing"]
        )
        agent3 = Subagent(
            name="Agent 3",
            description="Third agent",
            import_path="test.agent3",
            capabilities=["search", "writing"]
        )
        
        await registry.register_subagent(agent1)
        await registry.register_subagent(agent2)
        await registry.register_subagent(agent3)
        
        # Search for agents with "search" capability
        search_agents = await registry.search_by_capabilities(["search"])
        assert len(search_agents) == 2
        agent_names = [agent.name for agent in search_agents]
        assert "Agent 1" in agent_names
        assert "Agent 3" in agent_names
        
        # Search for agents with "writing" capability
        writing_agents = await registry.search_by_capabilities(["writing"])
        assert len(writing_agents) == 2
        agent_names = [agent.name for agent in writing_agents]
        assert "Agent 2" in agent_names
        assert "Agent 3" in agent_names
        
        # Search for non-existent capability
        none_agents = await registry.search_by_capabilities(["nonexistent"])
        assert len(none_agents) == 0


class TestRegistryStatistics:
    """Test registry statistics operations."""
    
    @pytest.mark.asyncio
    async def test_get_registry_stats_empty(self, registry):
        """Test getting statistics from empty registry."""
        stats = await registry.get_registry_stats()
        
        assert stats["total_subagents"] == 0
        assert stats["active_subagents"] == 0
        assert stats["inactive_subagents"] == 0
        assert stats["used_subagents"] == 0
        assert stats["unused_subagents"] == 0
        assert stats["registry_file_exists"] is True
        assert stats["registry_file_size"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_registry_stats_with_data(self, registry):
        """Test getting statistics with data in registry."""
        # Create agents with different states
        active_used = Subagent(
            name="Active Used",
            description="Active and used agent",
            import_path="test.active_used",
            is_active=True
        )
        active_unused = Subagent(
            name="Active Unused",
            description="Active but unused agent",
            import_path="test.active_unused", 
            is_active=True
        )
        inactive = Subagent(
            name="Inactive",
            description="Inactive agent",
            import_path="test.inactive",
            is_active=False
        )
        
        await registry.register_subagent(active_used)
        await registry.register_subagent(active_unused)
        await registry.register_subagent(inactive)
        
        # Mark one as used
        await registry.mark_subagent_used(active_used.id)
        
        stats = await registry.get_registry_stats()
        
        assert stats["total_subagents"] == 3
        assert stats["active_subagents"] == 2
        assert stats["inactive_subagents"] == 1
        assert stats["used_subagents"] == 1
        assert stats["unused_subagents"] == 1


class TestDataPersistence:
    """Test data persistence and CSV operations."""
    
    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self, temp_registry_path, sample_subagent):
        """Test that data persists when creating new registry instances."""
        # Register subagent with first instance
        registry1 = SubagentRegistry(registry_path=temp_registry_path)
        await registry1.register_subagent(sample_subagent)
        
        # Create new instance and verify data persists
        registry2 = SubagentRegistry(registry_path=temp_registry_path)
        retrieved = await registry2.get_subagent_by_id(sample_subagent.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_subagent.id
        assert retrieved.name == sample_subagent.name
    
    @pytest.mark.asyncio
    async def test_csv_format_correctness(self, registry, sample_subagent):
        """Test that CSV format is correct and readable."""
        await registry.register_subagent(sample_subagent)
        
        # Read CSV directly and verify format
        with open(registry.registry_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            assert len(rows) == 1
            row = rows[0]
            
            # Verify all expected fields are present
            expected_fields = [
                "id", "name", "description", "input_schema", "output_schema",
                "import_path", "embedding", "capabilities", "created_at",
                "last_used", "is_active"
            ]
            for field in expected_fields:
                assert field in row
            
            # Verify JSON fields can be parsed
            assert json.loads(row["input_schema"]) == sample_subagent.input_schema
            assert json.loads(row["output_schema"]) == sample_subagent.output_schema
            assert json.loads(row["capabilities"]) == sample_subagent.capabilities
            assert json.loads(row["embedding"]) == sample_subagent.embedding
    
    def test_subagent_to_dict_conversion(self, registry, sample_subagent):
        """Test conversion of Subagent to dictionary for CSV storage."""
        data = registry._subagent_to_dict(sample_subagent)
        
        # Verify all fields are strings
        for key, value in data.items():
            assert isinstance(value, str), f"Field {key} should be string, got {type(value)}"
        
        # Verify JSON fields
        assert json.loads(data["input_schema"]) == sample_subagent.input_schema
        assert json.loads(data["capabilities"]) == sample_subagent.capabilities
        
        # Verify datetime conversion
        assert data["created_at"] == sample_subagent.created_at.isoformat()
        
        # Verify boolean conversion
        assert data["is_active"] == "True"
    
    def test_dict_to_subagent_conversion(self, registry, sample_subagent):
        """Test conversion of dictionary from CSV to Subagent object."""
        # Convert to dict and back
        data = registry._subagent_to_dict(sample_subagent)
        converted = registry._dict_to_subagent(data)
        
        # Verify all fields match
        assert converted.id == sample_subagent.id
        assert converted.name == sample_subagent.name
        assert converted.description == sample_subagent.description
        assert converted.input_schema == sample_subagent.input_schema
        assert converted.output_schema == sample_subagent.output_schema
        assert converted.import_path == sample_subagent.import_path
        assert converted.embedding == sample_subagent.embedding
        assert converted.capabilities == sample_subagent.capabilities
        assert converted.is_active == sample_subagent.is_active
        
        # Verify datetime handling
        assert abs((converted.created_at - sample_subagent.created_at).total_seconds()) < 1


class TestBackupOperations:
    """Test registry backup operations."""
    
    @pytest.mark.asyncio
    async def test_backup_registry(self, registry, sample_subagent):
        """Test creating a backup of the registry."""
        await registry.register_subagent(sample_subagent)
        
        # Create backup
        backup_path = await registry.backup_registry()
        
        # Verify backup exists and has content
        assert os.path.exists(backup_path)
        assert os.path.getsize(backup_path) > 0
        
        # Verify backup content matches original
        backup_registry = SubagentRegistry(registry_path=backup_path)
        retrieved = await backup_registry.get_subagent_by_id(sample_subagent.id)
        assert retrieved is not None
        assert retrieved.name == sample_subagent.name
        
        # Cleanup
        os.unlink(backup_path)
    
    @pytest.mark.asyncio
    async def test_backup_registry_custom_path(self, registry, sample_subagent, temp_registry_path):
        """Test creating a backup with custom path."""
        await registry.register_subagent(sample_subagent)
        
        custom_backup_path = temp_registry_path + "_backup"
        backup_path = await registry.backup_registry(custom_backup_path)
        
        assert backup_path == custom_backup_path
        assert os.path.exists(custom_backup_path)
        
        # Cleanup
        os.unlink(custom_backup_path)


class TestErrorHandling:
    """Test error handling in registry operations."""
    
    @pytest.mark.asyncio
    async def test_registry_with_invalid_csv(self, temp_registry_path):
        """Test handling of corrupted CSV file."""
        # Create invalid CSV content
        with open(temp_registry_path, 'w') as f:
            f.write("invalid,csv,content\nwith,malformed,data")
        
        registry = SubagentRegistry(registry_path=temp_registry_path)
        
        # Should handle gracefully and return empty list
        subagents = await registry.list_subagents()
        assert isinstance(subagents, list)
    
    def test_invalid_subagent_validation(self):
        """Test that invalid subagent data raises appropriate errors."""
        # Test empty name
        with pytest.raises(ValueError, match="name cannot be empty"):
            Subagent(name="", description="test", import_path="test.path")
        
        # Test empty description
        with pytest.raises(ValueError, match="description cannot be empty"):
            Subagent(name="test", description="", import_path="test.path")
        
        # Test empty import path
        with pytest.raises(ValueError, match="Import path cannot be empty"):
            Subagent(name="test", description="test", import_path="")
        
        # Test invalid embedding dimension
        with pytest.raises(ValueError, match="Embedding must be 1536 dimensions"):
            Subagent(
                name="test",
                description="test", 
                import_path="test.path",
                embedding=[0.1] * 100  # Wrong dimension
            )


if __name__ == "__main__":
    pytest.main([__file__])
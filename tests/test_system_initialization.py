"""
Integration tests for system initialization and bootstrap sequence.
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.system.initialization import (
    SystemManager, SystemInitializationError, get_system_manager,
    setup_system_components, bootstrap_system, system_lifespan
)
from backend.system.database_migration import DatabaseMigrationManager


@pytest.fixture
async def temp_james_dir():
    """Create a temporary ~/.james directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the home directory to point to our temp directory
        with patch('pathlib.Path.home', return_value=Path(temp_dir)):
            # Create the .james directory structure
            james_dir = Path(temp_dir) / ".james"
            james_dir.mkdir(exist_ok=True)
            
            # Create required subdirectories
            subdirs = ["memory", "subagents", "logs", "backups", "temp", "migrations", "config"]
            for subdir in subdirs:
                (james_dir / subdir).mkdir(exist_ok=True)
            
            yield james_dir


@pytest.fixture
def reset_system_manager():
    """Reset the global system manager instance before and after tests."""
    # Reset before test
    from backend.system.initialization import _system_manager
    _system_manager = None
    
    yield
    
    # Reset after test
    from backend.system.initialization import _system_manager
    _system_manager = None


@pytest.mark.asyncio
async def test_system_manager_initialization(temp_james_dir, reset_system_manager):
    """Test that the system manager initializes components correctly."""
    # Create a system manager
    manager = SystemManager()
    
    # Register a test component
    test_init = MagicMock(return_value="test_instance")
    test_cleanup = MagicMock()
    test_health = MagicMock(return_value=True)
    
    manager.register_component(
        name="test_component",
        initialize_func=test_init,
        cleanup_func=test_cleanup,
        health_check_func=test_health
    )
    
    # Initialize the system
    await manager.initialize()
    
    # Verify component was initialized
    assert manager.is_initialized() is True
    assert test_init.called
    assert manager.get_component("test_component") == "test_instance"
    
    # Verify health check
    health_status = await manager.health_check()
    assert health_status["system_healthy"] is True
    assert "test_component" in health_status["components"]
    assert health_status["components"]["test_component"]["healthy"] is True
    
    # Shutdown the system
    await manager.shutdown()
    
    # Verify cleanup was called
    assert test_cleanup.called
    assert manager.is_initialized() is False


@pytest.mark.asyncio
async def test_dependency_resolution(reset_system_manager):
    """Test that dependencies are resolved correctly."""
    manager = SystemManager()
    
    # Register components with dependencies
    manager.register_component(
        name="component_a",
        initialize_func=MagicMock(return_value="a"),
        dependencies=[]
    )
    
    manager.register_component(
        name="component_b",
        initialize_func=MagicMock(return_value="b"),
        dependencies=["component_a"]
    )
    
    manager.register_component(
        name="component_c",
        initialize_func=MagicMock(return_value="c"),
        dependencies=["component_b"]
    )
    
    # Initialize the system
    await manager.initialize()
    
    # Verify initialization order through component instances
    assert manager.get_component("component_a") == "a"
    assert manager.get_component("component_b") == "b"
    assert manager.get_component("component_c") == "c"
    
    # Shutdown the system
    await manager.shutdown()


@pytest.mark.asyncio
async def test_circular_dependency_detection(reset_system_manager):
    """Test that circular dependencies are detected."""
    manager = SystemManager()
    
    # Register components with circular dependencies
    manager.register_component(
        name="component_x",
        initialize_func=MagicMock(),
        dependencies=["component_z"]
    )
    
    manager.register_component(
        name="component_y",
        initialize_func=MagicMock(),
        dependencies=["component_x"]
    )
    
    manager.register_component(
        name="component_z",
        initialize_func=MagicMock(),
        dependencies=["component_y"]
    )
    
    # Initialize the system - should raise an error
    with pytest.raises(SystemInitializationError) as excinfo:
        await manager.initialize()
    
    # Verify error message contains "circular dependency"
    assert "circular dependency" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_component_initialization_failure(reset_system_manager):
    """Test handling of component initialization failures."""
    manager = SystemManager()
    
    # Register a component that will succeed
    manager.register_component(
        name="good_component",
        initialize_func=MagicMock(return_value="good")
    )
    
    # Register a component that will fail
    failing_init = MagicMock(side_effect=Exception("Initialization failed"))
    manager.register_component(
        name="failing_component",
        initialize_func=failing_init,
        dependencies=["good_component"]
    )
    
    # Initialize the system - should raise an error
    with pytest.raises(SystemInitializationError) as excinfo:
        await manager.initialize()
    
    # Verify error message
    assert "failing_component" in str(excinfo.value)
    assert "Initialization failed" in str(excinfo.value)
    
    # Verify the system is not initialized
    assert manager.is_initialized() is False
    
    # Verify the first component was initialized but cleaned up
    assert manager.get_component("good_component") is None


@pytest.mark.asyncio
async def test_health_check_failure(reset_system_manager):
    """Test handling of health check failures."""
    manager = SystemManager()
    
    # Register a component with a failing health check
    manager.register_component(
        name="unhealthy_component",
        initialize_func=MagicMock(return_value="instance"),
        health_check_func=MagicMock(return_value=False)
    )
    
    # Initialize the system
    await manager.initialize()
    
    # Verify the system is initialized but not healthy
    assert manager.is_initialized() is True
    assert manager.is_healthy() is False
    
    # Get health status
    health_status = await manager.health_check()
    assert health_status["system_healthy"] is False
    assert health_status["components"]["unhealthy_component"]["healthy"] is False
    
    # Shutdown the system
    await manager.shutdown()


@pytest.mark.asyncio
async def test_setup_system_components(temp_james_dir, reset_system_manager):
    """Test the setup_system_components function."""
    # Setup system components
    manager = setup_system_components()
    
    # Verify components were registered
    assert "james_directory" in manager.components
    assert "database_migration" in manager.components
    assert "vector_client" in manager.components
    assert "memory_manager" in manager.components
    assert "subagent_registry" in manager.components
    assert "message_queue" in manager.components
    assert "observability" in manager.components
    
    # Verify dependencies
    assert "james_directory" in manager.components["memory_manager"].dependencies
    assert "vector_client" in manager.components["memory_manager"].dependencies
    assert "database_migration" in manager.components["memory_manager"].dependencies


@pytest.mark.asyncio
async def test_bootstrap_system(temp_james_dir, reset_system_manager):
    """Test the bootstrap_system function."""
    # Mock the initialization functions to avoid actual initialization
    with patch('backend.system.initialization.initialize_james_directory', new_callable=MagicMock) as mock_james_dir, \
         patch('backend.system.initialization.initialize_database_migration', new_callable=MagicMock) as mock_db_migration, \
         patch('backend.system.initialization.initialize_vector_client', new_callable=MagicMock) as mock_vector, \
         patch('backend.system.initialization.initialize_memory_manager', new_callable=MagicMock) as mock_memory, \
         patch('backend.system.initialization.initialize_subagent_registry', new_callable=MagicMock) as mock_registry, \
         patch('backend.system.initialization.initialize_message_queue', new_callable=MagicMock) as mock_queue, \
         patch('backend.system.initialization.initialize_observability', new_callable=MagicMock) as mock_observability:
        
        # Bootstrap the system
        manager = await bootstrap_system()
        
        # Verify initialization functions were called
        mock_james_dir.assert_called_once()
        mock_db_migration.assert_called_once()
        mock_vector.assert_called_once()
        mock_memory.assert_called_once()
        mock_registry.assert_called_once()
        mock_queue.assert_called_once()
        mock_observability.assert_called_once()
        
        # Verify system is initialized
        assert manager.is_initialized() is True
        
        # Shutdown the system
        await manager.shutdown()


@pytest.mark.asyncio
async def test_system_lifespan():
    """Test the system_lifespan context manager."""
    # Mock the setup_system_components and initialize/shutdown methods
    mock_manager = MagicMock()
    mock_manager.initialize = MagicMock()
    mock_manager.shutdown = MagicMock()
    
    with patch('backend.system.initialization.setup_system_components', return_value=mock_manager):
        # Use the context manager
        async with system_lifespan() as manager:
            # Verify initialize was called
            mock_manager.initialize.assert_called_once()
            assert manager == mock_manager
        
        # Verify shutdown was called after exiting context
        mock_manager.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_database_migration_integration(temp_james_dir, reset_system_manager):
    """Test the integration of database migration with system initialization."""
    # Mock the database migration manager
    mock_migration_manager = MagicMock(spec=DatabaseMigrationManager)
    mock_migration_manager.initialize = MagicMock()
    mock_migration_manager.get_migration_status = MagicMock(return_value={
        "current_version": 1,
        "target_version": 1,
        "pending_migrations": [],
        "applied_migrations": [{"version": 1, "name": "initial"}]
    })
    mock_migration_manager.validate_schema = MagicMock(return_value={"valid": True})
    
    # Patch the get_migration_manager function
    with patch('backend.system.initialization.get_migration_manager', return_value=mock_migration_manager):
        # Setup and initialize system components
        manager = setup_system_components()
        await manager.initialize()
        
        # Verify database_migration component was initialized
        assert manager.components["database_migration"].initialized is True
        assert manager.components["database_migration"].healthy is True
        
        # Verify migration manager initialize was called
        mock_migration_manager.initialize.assert_called_once()
        
        # Shutdown the system
        await manager.shutdown()


@pytest.mark.asyncio
async def test_graceful_shutdown_with_signal(temp_james_dir, reset_system_manager):
    """Test graceful shutdown when receiving a signal."""
    # Create a system manager
    manager = SystemManager()
    
    # Register a test component
    cleanup_called = asyncio.Event()
    
    async def test_cleanup():
        cleanup_called.set()
    
    manager.register_component(
        name="test_component",
        initialize_func=MagicMock(return_value="test_instance"),
        cleanup_func=test_cleanup
    )
    
    # Initialize the system
    await manager.initialize()
    
    # Add a custom shutdown handler
    custom_handler_called = asyncio.Event()
    
    async def custom_handler():
        custom_handler_called.set()
    
    manager.add_shutdown_handler(custom_handler)
    
    # Trigger shutdown
    await manager.shutdown()
    
    # Verify cleanup and custom handler were called
    assert cleanup_called.is_set()
    assert custom_handler_called.is_set()
    assert manager.is_initialized() is False
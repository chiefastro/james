"""
System initialization and bootstrap sequence for the Conscious Agent System.

This module handles the startup sequence with proper dependency initialization,
health checks, and graceful shutdown procedures.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from ..memory.memory_manager import MemoryManager
from ..registry.subagent_registry import SubagentRegistry
from ..vector.qdrant_client import QdrantVectorClient
from ..queue.message_queue import MessageQueue
from ..observability.langsmith_tracer import configure_tracer, TraceConfig
from ..observability.metrics_collector import get_metrics_collector
from ..observability.anomaly_detector import setup_anomaly_detection, get_anomaly_detector
from ..subagents.register_seed_subagents import register_all_seed_subagents
from .database_migration import get_migration_manager, DatabaseMigrationManager

logger = logging.getLogger(__name__)


@dataclass
class SystemComponent:
    """Represents a system component with initialization and cleanup methods."""
    name: str
    instance: Any = None
    initialize_func: Optional[Callable] = None
    cleanup_func: Optional[Callable] = None
    health_check_func: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    initialized: bool = False
    healthy: bool = False
    error: Optional[str] = None


class SystemInitializationError(Exception):
    """Raised when system initialization fails."""
    pass


class SystemManager:
    """
    Manages the complete system initialization and shutdown sequence.
    
    Handles dependency ordering, health checks, and graceful cleanup.
    """
    
    def __init__(self):
        """Initialize the system manager."""
        self.components: Dict[str, SystemComponent] = {}
        self.startup_time: Optional[datetime] = None
        self.shutdown_handlers: List[Callable] = []
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def register_component(
        self,
        name: str,
        initialize_func: Optional[Callable] = None,
        cleanup_func: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a system component.
        
        Args:
            name: Component name
            initialize_func: Async function to initialize the component
            cleanup_func: Async function to cleanup the component
            health_check_func: Async function to check component health
            dependencies: List of component names this component depends on
        """
        self.components[name] = SystemComponent(
            name=name,
            initialize_func=initialize_func,
            cleanup_func=cleanup_func,
            health_check_func=health_check_func,
            dependencies=dependencies or []
        )
        logger.debug(f"Registered component: {name}")
    
    def _resolve_dependencies(self) -> List[str]:
        """
        Resolve component dependencies and return initialization order.
        
        Returns:
            List of component names in dependency order
            
        Raises:
            SystemInitializationError: If circular dependencies are detected
        """
        # Topological sort to resolve dependencies
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise SystemInitializationError(f"Circular dependency detected involving {component_name}")
            
            if component_name not in visited:
                temp_visited.add(component_name)
                
                component = self.components.get(component_name)
                if component:
                    for dep in component.dependencies:
                        if dep not in self.components:
                            raise SystemInitializationError(f"Unknown dependency: {dep} for component {component_name}")
                        visit(dep)
                
                temp_visited.remove(component_name)
                visited.add(component_name)
                result.append(component_name)
        
        for component_name in self.components:
            if component_name not in visited:
                visit(component_name)
        
        return result
    
    async def initialize(self) -> None:
        """
        Initialize all system components in dependency order.
        
        Raises:
            SystemInitializationError: If initialization fails
        """
        if self._initialized:
            logger.warning("System already initialized")
            return
        
        logger.info("Starting system initialization")
        self.startup_time = datetime.now()
        
        try:
            # Resolve initialization order
            init_order = self._resolve_dependencies()
            logger.info(f"Component initialization order: {init_order}")
            
            # Initialize components in order
            for component_name in init_order:
                await self._initialize_component(component_name)
            
            self._initialized = True
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            # Cleanup any partially initialized components
            await self._cleanup_initialized_components()
            raise SystemInitializationError(f"Failed to initialize system: {e}")
    
    async def _initialize_component(self, component_name: str) -> None:
        """
        Initialize a single component.
        
        Args:
            component_name: Name of the component to initialize
            
        Raises:
            SystemInitializationError: If component initialization fails
        """
        component = self.components[component_name]
        
        try:
            logger.info(f"Initializing component: {component_name}")
            
            if component.initialize_func:
                result = await component.initialize_func()
                if result is not None:
                    component.instance = result
            
            component.initialized = True
            
            # Perform health check if available
            if component.health_check_func:
                component.healthy = await component.health_check_func()
            else:
                component.healthy = True
            
            logger.info(f"Component {component_name} initialized successfully (healthy: {component.healthy})")
            
        except Exception as e:
            error_msg = f"Failed to initialize component {component_name}: {e}"
            logger.error(error_msg)
            component.error = str(e)
            raise SystemInitializationError(error_msg)
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown all system components.
        """
        if not self._initialized:
            logger.warning("System not initialized, nothing to shutdown")
            return
        
        logger.info("Starting graceful system shutdown")
        
        try:
            # Signal shutdown event
            self._shutdown_event.set()
            
            # Run custom shutdown handlers
            for handler in self.shutdown_handlers:
                try:
                    await handler()
                except Exception as e:
                    logger.error(f"Error in shutdown handler: {e}")
            
            # Cleanup components in reverse order
            await self._cleanup_initialized_components()
            
            self._initialized = False
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
    
    async def _cleanup_initialized_components(self) -> None:
        """Cleanup all initialized components in reverse order."""
        # Get initialized components in reverse order
        init_order = self._resolve_dependencies()
        cleanup_order = [name for name in reversed(init_order) 
                        if self.components[name].initialized]
        
        for component_name in cleanup_order:
            await self._cleanup_component(component_name)
    
    async def _cleanup_component(self, component_name: str) -> None:
        """
        Cleanup a single component.
        
        Args:
            component_name: Name of the component to cleanup
        """
        component = self.components[component_name]
        
        try:
            logger.info(f"Cleaning up component: {component_name}")
            
            if component.cleanup_func:
                await component.cleanup_func()
            
            component.initialized = False
            component.healthy = False
            component.instance = None
            
            logger.info(f"Component {component_name} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up component {component_name}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status of all components
        """
        health_status = {
            "system_healthy": True,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            "components": {}
        }
        
        for name, component in self.components.items():
            component_health = {
                "initialized": component.initialized,
                "healthy": component.healthy,
                "error": component.error
            }
            
            # Perform health check if component is initialized and has health check function
            if component.initialized and component.health_check_func:
                try:
                    component.healthy = await component.health_check_func()
                    component_health["healthy"] = component.healthy
                except Exception as e:
                    component.healthy = False
                    component.error = str(e)
                    component_health["healthy"] = False
                    component_health["error"] = str(e)
            
            health_status["components"][name] = component_health
            
            # Update system health
            if not component.healthy:
                health_status["system_healthy"] = False
        
        return health_status
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component instance by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance if found and initialized, None otherwise
        """
        component = self.components.get(name)
        if component and component.initialized:
            return component.instance
        return None
    
    def add_shutdown_handler(self, handler: Callable) -> None:
        """
        Add a custom shutdown handler.
        
        Args:
            handler: Async function to call during shutdown
        """
        self.shutdown_handlers.append(handler)
    
    def is_initialized(self) -> bool:
        """Check if the system is initialized."""
        return self._initialized
    
    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(component.healthy for component in self.components.values() 
                  if component.initialized)


# Global system manager instance
_system_manager: Optional[SystemManager] = None


def get_system_manager() -> SystemManager:
    """Get the global system manager instance."""
    global _system_manager
    if _system_manager is None:
        _system_manager = SystemManager()
    return _system_manager


async def initialize_james_directory() -> None:
    """Initialize the ~/.james directory structure."""
    james_dir = Path.home() / ".james"
    
    # Create main directory
    james_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["memory", "subagents", "logs", "backups", "temp", "migrations", "config"]
    for subdir in subdirs:
        (james_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"James directory initialized: {james_dir}")


async def initialize_memory_manager() -> MemoryManager:
    """Initialize the memory manager."""
    memory_manager = MemoryManager()
    await memory_manager.initialize()
    return memory_manager


async def initialize_subagent_registry() -> SubagentRegistry:
    """Initialize the subagent registry."""
    registry = SubagentRegistry()
    
    # Initialize vector database if enabled
    if registry.enable_vector_db:
        await registry.initialize_vector_database()
    
    # Register seed subagents
    await register_all_seed_subagents(registry)
    
    return registry


async def initialize_vector_client() -> QdrantVectorClient:
    """Initialize the Qdrant vector client."""
    client = QdrantVectorClient()
    await client.initialize_collection()
    return client


async def initialize_message_queue() -> MessageQueue:
    """Initialize the message queue."""
    return MessageQueue()


async def initialize_observability() -> None:
    """Initialize the observability system."""
    # Configure LangSmith tracing
    trace_config = TraceConfig(
        project_name="conscious-agent-system",
        enabled=True,
        sample_rate=1.0
    )
    configure_tracer(trace_config)
    
    # Start anomaly detection
    await setup_anomaly_detection()
    
    logger.info("Observability system initialized")


async def initialize_database_migration() -> DatabaseMigrationManager:
    """Initialize the database migration manager."""
    migration_manager = get_migration_manager()
    await migration_manager.initialize()
    
    # Check if migrations are needed
    status = await migration_manager.get_migration_status()
    if status["current_version"] < status["target_version"]:
        logger.warning(f"Database schema needs migration. Current version: {status['current_version']}, "
                      f"Target version: {status['target_version']}")
    else:
        logger.info(f"Database schema is up to date. Version: {status['current_version']}")
    
    return migration_manager


async def health_check_memory_manager() -> bool:
    """Health check for memory manager."""
    try:
        manager = get_system_manager().get_component("memory_manager")
        if manager:
            # Try to get memory stats as a health check
            stats = await manager.get_memory_stats()
            return "error" not in stats
        return False
    except Exception:
        return False


async def health_check_subagent_registry() -> bool:
    """Health check for subagent registry."""
    try:
        registry = get_system_manager().get_component("subagent_registry")
        if registry:
            # Try to get registry stats as a health check
            stats = await registry.get_registry_stats()
            return stats.get("registry_file_exists", False)
        return False
    except Exception:
        return False


async def health_check_vector_client() -> bool:
    """Health check for vector client."""
    try:
        client = get_system_manager().get_component("vector_client")
        if client:
            return await client.health_check()
        return False
    except Exception:
        return False


async def health_check_observability() -> bool:
    """Health check for observability system."""
    try:
        # Check if metrics collector is working
        metrics_collector = get_metrics_collector()
        metrics = metrics_collector.get_all_metrics()
        
        # Check if anomaly detector is working
        detector = get_anomaly_detector()
        stats = detector.get_detection_statistics()
        
        return True
    except Exception:
        return False


async def health_check_database_migration() -> bool:
    """Health check for database migration manager."""
    try:
        migration_manager = get_system_manager().get_component("database_migration")
        if migration_manager:
            # Validate schema as a health check
            validation = await migration_manager.validate_schema()
            return validation.get("valid", False)
        return False
    except Exception:
        return False


async def cleanup_memory_manager() -> None:
    """Cleanup memory manager."""
    try:
        manager = get_system_manager().get_component("memory_manager")
        if manager:
            await manager.close()
    except Exception as e:
        logger.error(f"Error cleaning up memory manager: {e}")


async def cleanup_observability() -> None:
    """Cleanup observability system."""
    try:
        # Stop anomaly detection
        detector = get_anomaly_detector()
        await detector.stop_detection()
        
        # Flush traces
        from ..observability.langsmith_tracer import get_tracer
        tracer = get_tracer()
        await tracer.flush_traces()
        
    except Exception as e:
        logger.error(f"Error cleaning up observability system: {e}")


def setup_system_components() -> SystemManager:
    """
    Setup all system components with proper dependencies.
    
    Returns:
        Configured SystemManager instance
    """
    manager = get_system_manager()
    
    # Register components in dependency order
    manager.register_component(
        name="james_directory",
        initialize_func=initialize_james_directory
    )
    
    manager.register_component(
        name="database_migration",
        initialize_func=initialize_database_migration,
        health_check_func=health_check_database_migration,
        dependencies=["james_directory"]
    )
    
    manager.register_component(
        name="vector_client",
        initialize_func=initialize_vector_client,
        health_check_func=health_check_vector_client,
        dependencies=["james_directory"]
    )
    
    manager.register_component(
        name="memory_manager",
        initialize_func=initialize_memory_manager,
        cleanup_func=cleanup_memory_manager,
        health_check_func=health_check_memory_manager,
        dependencies=["james_directory", "vector_client", "database_migration"]
    )
    
    manager.register_component(
        name="subagent_registry",
        initialize_func=initialize_subagent_registry,
        health_check_func=health_check_subagent_registry,
        dependencies=["james_directory", "vector_client", "database_migration"]
    )
    
    manager.register_component(
        name="message_queue",
        initialize_func=initialize_message_queue,
        dependencies=["james_directory"]
    )
    
    manager.register_component(
        name="observability",
        initialize_func=initialize_observability,
        cleanup_func=cleanup_observability,
        health_check_func=health_check_observability
    )
    
    return manager


@asynccontextmanager
async def system_lifespan(app):
    """
    Context manager for system lifespan management.
    
    Handles initialization on entry and cleanup on exit.
    
    Args:
        app: The FastAPI application instance
    """
    manager = setup_system_components()
    
    try:
        # Initialize system
        await manager.initialize()
        yield  # Yield None for FastAPI lifespan
    finally:
        # Cleanup system
        await manager.shutdown()


async def bootstrap_system() -> SystemManager:
    """
    Bootstrap the complete conscious agent system.
    
    Returns:
        Initialized SystemManager instance
        
    Raises:
        SystemInitializationError: If bootstrap fails
    """
    logger.info("Bootstrapping Conscious Agent System")
    
    try:
        # Setup and initialize system components
        manager = setup_system_components()
        await manager.initialize()
        
        logger.info("System bootstrap completed successfully")
        return manager
        
    except Exception as e:
        logger.error(f"System bootstrap failed: {e}")
        raise SystemInitializationError(f"Bootstrap failed: {e}")
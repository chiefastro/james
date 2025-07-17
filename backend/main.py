"""
Main FastAPI application entry point for the Conscious Agent System.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
import os
from datetime import datetime

from backend.agents.observer import ObserverAgent
from backend.agents.delegator import Delegator
from backend.queue.message_queue import MessageQueue
from backend.registry.subagent_registry import SubagentRegistry
from backend.models.api import (
    AgentCreationRequest, AgentCreationResponse,
    MessageRequest, MessageResponse,
    AgentStatusResponse
)
from backend.models.core import Message, MessageSource
from backend.observability.langsmith_tracer import get_tracer
from backend.observability.metrics_collector import get_metrics_collector
from backend.observability.anomaly_detector import get_anomaly_detector
from backend.observability.trace_analyzer import TraceAnalyzer
from backend.system.initialization import (
    system_lifespan, get_system_manager, SystemManager, 
    SystemInitializationError
)
from backend.system.database_migration import get_migration_manager, DatabaseMigrationManager, MigrationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with system lifespan
app = FastAPI(
    title="Conscious Agent API", 
    version="0.1.0",
    lifespan=system_lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_connections: List[WebSocket] = []
agent_instances: Dict[str, Any] = {}

# System manager dependency
async def get_system() -> SystemManager:
    """Get the initialized system manager."""
    system = get_system_manager()
    if not system.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    return system

# Message queue dependency
async def get_message_queue() -> MessageQueue:
    """Get the message queue from system manager."""
    system = get_system_manager()
    queue = system.get_component("message_queue")
    if not queue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message queue not available"
        )
    return queue

# Subagent registry dependency
async def get_registry() -> SubagentRegistry:
    """Get the subagent registry from system manager."""
    system = get_system_manager()
    registry = system.get_component("subagent_registry")
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subagent registry not available"
        )
    return registry

# Database migration manager dependency
async def get_migration() -> DatabaseMigrationManager:
    """Get the database migration manager from system manager."""
    system = get_system_manager()
    migration = system.get_component("database_migration")
    if not migration:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database migration manager not available"
        )
    return migration

@app.on_event("startup")
async def startup_event():
    """Initialize system on application startup"""
    try:
        logger.info("FastAPI application starting up")
        
        # System initialization is handled by the lifespan context manager
        # This event is for additional startup tasks if needed
        
    except Exception as e:
        logger.error(f"Error during application startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup system on application shutdown"""
    try:
        logger.info("FastAPI application shutting down")
        
        # System cleanup is handled by the lifespan context manager
        # This event is for additional cleanup tasks if needed
        
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"message": "Conscious Agent System API"}


@app.get("/health")
async def health_check(system: SystemManager = Depends(get_system)):
    """
    Comprehensive health check endpoint for the entire system.
    
    Returns detailed health status of all system components.
    """
    try:
        # Get detailed health status from system manager
        health_status = await system.health_check()
        
        # Determine overall HTTP status code based on system health
        if not health_status["system_healthy"]:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status
            )
        
        return health_status
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/health/component/{component_name}")
async def component_health_check(
    component_name: str, 
    system: SystemManager = Depends(get_system)
):
    """
    Health check for a specific system component.
    
    Args:
        component_name: Name of the component to check
    """
    try:
        # Get detailed health status from system manager
        health_status = await system.health_check()
        
        # Check if component exists
        if component_name not in health_status["components"]:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "message": f"Component not found: {component_name}",
                    "available_components": list(health_status["components"].keys())
                }
            )
        
        component_health = health_status["components"][component_name]
        
        # Determine HTTP status code based on component health
        if not component_health["healthy"]:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=component_health
            )
        
        return component_health
    except Exception as e:
        logger.error(f"Error checking component health: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Component health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


# Database migration endpoints

@app.get("/system/migrations")
async def get_migrations(migration_manager: DatabaseMigrationManager = Depends(get_migration)):
    """
    Get current database migration status.
    
    Returns information about applied and pending migrations.
    """
    try:
        migration_status = await migration_manager.get_migration_status()
        return migration_status
    except Exception as e:
        logger.error(f"Error getting migration status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get migration status: {str(e)}"
        )


@app.post("/system/migrations/migrate")
async def run_migrations(
    target_version: Optional[int] = None,
    migration_manager: DatabaseMigrationManager = Depends(get_migration)
):
    """
    Run database migrations up to the specified version.
    
    Args:
        target_version: Target schema version, defaults to latest available
    """
    try:
        result = await migration_manager.migrate(target_version)
        return result
    except MigrationError as e:
        logger.error(f"Migration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run migrations: {str(e)}"
        )


@app.post("/system/migrations/rollback/{target_version}")
async def rollback_migrations(
    target_version: int,
    migration_manager: DatabaseMigrationManager = Depends(get_migration)
):
    """
    Rollback database migrations to the specified version.
    
    Args:
        target_version: Target schema version to rollback to
    """
    try:
        result = await migration_manager.rollback(target_version)
        return result
    except MigrationError as e:
        logger.error(f"Rollback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error rolling back migrations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to roll back migrations: {str(e)}"
        )


@app.post("/system/migrations/create")
async def create_migration(
    name: str,
    migration_manager: DatabaseMigrationManager = Depends(get_migration)
):
    """
    Create a new migration script.
    
    Args:
        name: Migration name (will be used in filename)
    """
    try:
        result = await migration_manager.create_migration(name)
        return result
    except MigrationError as e:
        logger.error(f"Error creating migration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create migration: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating migration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create migration: {str(e)}"
        )


@app.get("/system/migrations/validate")
async def validate_schema(migration_manager: DatabaseMigrationManager = Depends(get_migration)):
    """
    Validate the current database schema.
    
    Returns validation result with status information.
    """
    try:
        result = await migration_manager.validate_schema()
        return result
    except Exception as e:
        logger.error(f"Error validating schema: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate schema: {str(e)}"
        )


@app.get("/system/status")
async def system_status(system: SystemManager = Depends(get_system)):
    """
    Get detailed system status information.
    
    Returns information about system components, uptime, and health.
    """
    try:
        # Get health status
        health_status = await system.health_check()
        
        # Get component information
        components = {}
        for name, component in system.components.items():
            components[name] = {
                "initialized": component.initialized,
                "healthy": component.healthy,
                "error": component.error
            }
        
        return {
            "system_healthy": health_status["system_healthy"],
            "startup_time": health_status["startup_time"],
            "uptime_seconds": health_status["uptime_seconds"],
            "components": components,
            "environment": os.environ.get("ENV", "development"),
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@app.post("/agents", response_model=AgentCreationResponse)
async def create_agent(
    request: AgentCreationRequest,
    registry: SubagentRegistry = Depends(get_registry),
    message_queue: MessageQueue = Depends(get_message_queue)
):
    """Create a new agent instance"""
    try:
        agent_id = f"agent_{len(agent_instances) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if request.agent_type == "observer":
            agent = ObserverAgent(message_queue=message_queue)
        elif request.agent_type == "delegator":
            agent = Delegator(agent_id=agent_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")
        
        agent_instances[agent_id] = agent
        registry.register_agent(agent_id, request.agent_type, {"status": "active"})
        
        # Broadcast agent creation
        await manager.broadcast(json.dumps({
            "type": "agent_created",
            "agent_id": agent_id,
            "agent_type": request.agent_type,
            "timestamp": datetime.now().isoformat()
        }))
        
        return AgentCreationResponse(
            agent_id=agent_id,
            agent_type=request.agent_type,
            status="created"
        )
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents", response_model=List[AgentStatusResponse])
async def list_agents(registry: SubagentRegistry = Depends(get_registry)):
    """List all active agents"""
    agents = []
    for agent_id, agent in agent_instances.items():
        agent_info = registry.get_agent_info(agent_id)
        agents.append(AgentStatusResponse(
            agent_id=agent_id,
            agent_type=agent_info.get("type", "unknown") if agent_info else "unknown",
            status=agent_info.get("metadata", {}).get("status", "unknown") if agent_info else "unknown"
        ))
    return agents


@app.post("/agents/{agent_id}/message", response_model=MessageResponse)
async def send_message_to_agent(
    agent_id: str, 
    request: MessageRequest,
    message_queue: MessageQueue = Depends(get_message_queue)
):
    """Send a message to a specific agent"""
    if agent_id not in agent_instances:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent = agent_instances[agent_id]
        
        # Queue the message
        await message_queue.enqueue({
            "agent_id": agent_id,
            "message": request.message,
            "timestamp": datetime.now().isoformat()
        }, 5)  # Default priority
        
        # Broadcast message
        await manager.broadcast(json.dumps({
            "type": "message_sent",
            "agent_id": agent_id,
            "message": request.message,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Create a Message object for the response
        msg = Message(
            id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            content=request.message,
            source=MessageSource.USER,
            priority=5,
            timestamp=datetime.now(),
            metadata={}
        )
        
        return MessageResponse(
            id=msg.id,
            content=msg.content,
            source=msg.source,
            priority=msg.priority,
            timestamp=msg.timestamp,
            metadata=msg.metadata
        )
    except Exception as e:
        logger.error(f"Error sending message to agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    registry: SubagentRegistry = Depends(get_registry)
):
    """Delete an agent instance"""
    if agent_id not in agent_instances:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        del agent_instances[agent_id]
        registry.unregister_agent(agent_id)
        
        # Broadcast agent deletion
        await manager.broadcast(json.dumps({
            "type": "agent_deleted",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        return {"status": "deleted", "agent_id": agent_id}
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/message")
async def send_message(
    request: MessageRequest,
    message_queue: MessageQueue = Depends(get_message_queue)
):
    """Send a message to the conscious agent system"""
    try:
        # Create a Message object
        message = Message(
            id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            content=request.content,
            source=MessageSource(request.source) if hasattr(request, 'source') and request.source else MessageSource.USER,
            priority=getattr(request, 'priority', 5),
            timestamp=datetime.now(),
            metadata=getattr(request, 'metadata', {})
        )
        
        # Add to message queue
        await message_queue.enqueue(message, message.priority)
        
        # Broadcast message to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "message",
            "data": {
                "id": message.id,
                "content": message.content,
                "source": message.source.value,
                "priority": message.priority,
                "timestamp": message.timestamp.isoformat()
            }
        }))
        
        return MessageResponse(
            id=message.id,
            content=message.content,
            source=message.source,
            priority=message.priority,
            timestamp=message.timestamp,
            metadata=message.metadata,
            classification=message.classification
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/status")
async def get_agent_status(
    message_queue: MessageQueue = Depends(get_message_queue),
    system: SystemManager = Depends(get_system)
):
    """Get the current status of the conscious agent system"""
    try:
        # Get queue size
        queue_size = await message_queue.size()
        
        # Get active tasks (placeholder - would come from task manager)
        current_tasks = []
        
        # Get subagent count
        active_subagents = len([agent for agent in agent_instances.values()])
        
        # Get system uptime
        uptime_seconds = 0
        if system.startup_time:
            uptime_seconds = (datetime.now() - system.startup_time).total_seconds()
        
        # Provide all required memory metrics fields (use real values if available)
        memory_usage = {
            "episodic_count": 0,
            "semantic_count": 0,
            "procedural_count": 0,
            "working_memory_size": 0,
            "total_size_mb": 0.0,
            "cleanup_last_run": None
        }
        
        return {
            "is_active": len(agent_instances) > 0,
            "current_tasks": current_tasks,
            "message_queue_size": queue_size,
            "active_subagents": active_subagents,
            "memory_usage": memory_usage,
            "uptime_seconds": uptime_seconds,
            "last_activity": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    """List all active tasks"""
    try:
        # Placeholder - would come from task manager
        return []
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subagents")
async def list_subagents(registry: SubagentRegistry = Depends(get_registry)):
    """List all available subagents"""
    try:
        # Get subagents from registry (async, returns Subagent objects)
        subagents = await registry.list_subagents(active_only=False)
        return [
            {
                "id": subagent.id,
                "name": subagent.name,
                "description": subagent.description,
                "capabilities": getattr(subagent, "capabilities", []),
                "is_active": getattr(subagent, "is_active", True),
                "created_at": subagent.created_at.isoformat() if hasattr(subagent, "created_at") else None,
                "last_used": subagent.last_used.isoformat() if getattr(subagent, "last_used", None) else None,
                "input_schema": getattr(subagent, "input_schema", {}),
                "output_schema": getattr(subagent, "output_schema", {}),
                "import_path": getattr(subagent, "import_path", "")
            }
            for subagent in subagents
        ]
    except Exception as e:
        logger.error(f"Error listing subagents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/copilot")
async def copilot_runtime(
    request: dict,
    message_queue: MessageQueue = Depends(get_message_queue)
):
    """CopilotKit runtime endpoint for agent integration"""
    try:
        # Handle CopilotKit runtime requests
        action = request.get("action")
        parameters = request.get("parameters", {})
        
        if not action:
            raise HTTPException(status_code=400, detail="Missing or unknown action in CopilotKit request.")
        
        if action == "send_message":
            content = parameters.get("content")
            priority = parameters.get("priority", 5)
            
            if not content:
                raise HTTPException(status_code=400, detail="Message content is required")
            
            # Create and queue message
            message = Message(
                id=f"copilot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                content=content,
                source=MessageSource.USER,
                priority=priority,
                timestamp=datetime.now(),
                metadata={"source": "copilot"}
            )
            
            await message_queue.enqueue(message, message.priority)
            
            # Broadcast to WebSocket clients
            await manager.broadcast(json.dumps({
                "type": "message",
                "data": {
                    "id": message.id,
                    "content": message.content,
                    "source": "copilot",
                    "timestamp": message.timestamp.isoformat()
                }
            }))
            
            return {"status": "success", "message_id": message.id}
            
        elif action == "get_agent_status":
            status = await get_agent_status(message_queue, await get_system())
            return {"status": "success", "data": status}
            
        elif action == "list_active_tasks":
            tasks = await list_tasks()
            return {"status": "success", "data": tasks}
            
        elif action == "get_subagents":
            subagents = await list_subagents(await get_registry())
            return {"status": "success", "data": subagents}
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
            
    except Exception as e:
        logger.error(f"Error in CopilotKit runtime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Observability endpoints

@app.get("/observability/metrics")
async def get_metrics():
    """Get current system metrics"""
    try:
        metrics_collector = get_metrics_collector()
        return metrics_collector.get_all_metrics()
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/observability/metrics/summary")
async def get_metrics_summary():
    """Get agent performance summary"""
    try:
        metrics_collector = get_metrics_collector()
        return metrics_collector.get_agent_performance_summary()
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/observability/traces")
async def get_traces():
    """Get active traces"""
    try:
        tracer = get_tracer()
        active_traces = tracer.get_active_traces()
        return {
            "active_traces": [
                {
                    "trace_id": trace.trace_id,
                    "operation_name": trace.operation_name,
                    "agent_type": trace.agent_type,
                    "start_time": trace.start_time.isoformat(),
                    "duration_ms": trace.duration_ms,
                    "success": trace.success
                }
                for trace in active_traces
            ],
            "statistics": tracer.get_trace_statistics()
        }
    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/observability/anomalies")
async def get_anomalies():
    """Get current anomaly alerts"""
    try:
        detector = get_anomaly_detector()
        active_alerts = detector.get_active_alerts()
        return {
            "active_alerts": [
                {
                    "id": alert.id,
                    "type": alert.anomaly_type.value,
                    "severity": alert.severity.value,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat(),
                    "deviation_score": alert.deviation_score
                }
                for alert in active_alerts
            ],
            "statistics": detector.get_detection_statistics()
        }
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/observability/insights")
async def get_insights():
    """Get trace analysis insights"""
    try:
        tracer = get_tracer()
        analyzer = TraceAnalyzer(tracer)
        
        # Get recent traces (this would normally come from a trace store)
        active_traces = tracer.get_active_traces()
        
        if not active_traces:
            return {"insights": [], "message": "No traces available for analysis"}
        
        insights = analyzer.generate_insights(active_traces, analysis_window_hours=24)
        
        return {
            "insights": [
                {
                    "type": insight.insight_type.value,
                    "title": insight.title,
                    "description": insight.description,
                    "severity": insight.severity,
                    "confidence": insight.confidence,
                    "recommendations": insight.recommendations,
                    "timestamp": insight.timestamp.isoformat()
                }
                for insight in insights
            ],
            "analysis_metadata": {
                "traces_analyzed": len(active_traces),
                "analysis_window_hours": 24,
                "generated_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/observability/anomalies/{alert_id}/resolve")
async def resolve_anomaly(alert_id: str):
    """Resolve an anomaly alert"""
    try:
        detector = get_anomaly_detector()
        success = detector.resolve_alert(alert_id)
        
        if success:
            return {"status": "resolved", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        logger.error(f"Error resolving anomaly {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/observability/config/tracing")
async def configure_tracing(config: dict):
    """Configure LangSmith tracing settings"""
    try:
        from backend.observability.langsmith_tracer import configure_tracer, TraceConfig
        
        trace_config = TraceConfig(
            project_name=config.get("project_name", "conscious-agent-system"),
            api_key=config.get("api_key"),
            endpoint=config.get("endpoint"),
            enabled=config.get("enabled", True),
            sample_rate=config.get("sample_rate", 1.0)
        )
        
        configure_tracer(trace_config)
        
        return {
            "status": "configured",
            "config": {
                "project_name": trace_config.project_name,
                "enabled": trace_config.enabled,
                "sample_rate": trace_config.sample_rate
            }
        }
    except Exception as e:
        logger.error(f"Error configuring tracing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                        websocket
                    )
                elif message.get("type") == "subscribe":
                    # Client wants to subscribe to updates
                    await manager.send_personal_message(
                        json.dumps({"type": "subscribed", "status": "success"}),
                        websocket
                    )
                else:
                    # Echo back unknown messages
                    await manager.send_personal_message(
                        json.dumps({"type": "echo", "data": message}),
                        websocket
                    )
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON"}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
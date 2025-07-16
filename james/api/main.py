"""FastAPI backend for James consciousness system."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from james.core.consciousness import JamesConsciousness
from james.core.message import MessageSource, MessagePriority
from james.core.memory import MemorySystem
from james.tools.seed_tools import SeedTools


class MessageRequest(BaseModel):
    content: str
    source: str = "user"
    priority: str = "medium"
    metadata: Optional[Dict[str, Any]] = None


class MemoryRequest(BaseModel):
    content: str
    memory_type: str = "episodic"
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: str
    memory_type: Optional[str] = None
    limit: int = 10


# Global consciousness instance
consciousness: Optional[JamesConsciousness] = None
memory_system: Optional[MemorySystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the consciousness system."""
    global consciousness, memory_system
    
    # Initialize systems
    consciousness = JamesConsciousness()
    memory_system = MemorySystem()
    
    # Register seed tools with A2A protocol
    seed_tools = SeedTools()
    consciousness.a2a.register_agent(seed_tools)
    
    # Start consciousness loop in background
    consciousness_task = asyncio.create_task(consciousness.consciousness_loop())
    
    yield
    
    # Cleanup
    consciousness.stop()
    consciousness_task.cancel()
    try:
        await consciousness_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="James Consciousness API",
    description="API for interacting with James, a conscious AI agent",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "James Consciousness API",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/message")
async def send_message(request: MessageRequest, background_tasks: BackgroundTasks):
    """Send a message to James."""
    if not consciousness:
        raise HTTPException(status_code=500, detail="Consciousness system not initialized")
    
    try:
        # Convert string enums to proper types
        source = MessageSource.USER if request.source == "user" else MessageSource.EXTERNAL
        priority_map = {
            "low": MessagePriority.LOW,
            "medium": MessagePriority.MEDIUM,
            "high": MessagePriority.HIGH,
            "urgent": MessagePriority.URGENT
        }
        priority = priority_map.get(request.priority, MessagePriority.MEDIUM)
        
        # Add message to consciousness queue
        await consciousness.add_message(
            content=request.content,
            source=source,
            priority=priority,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "message": "Message sent to James",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get system status."""
    if not consciousness:
        raise HTTPException(status_code=500, detail="Consciousness system not initialized")
    
    try:
        return {
            "consciousness": {
                "running": consciousness.running,
                "queue_size": consciousness.message_queue.qsize(),
                "james_home": str(consciousness.james_home)
            },
            "agents": consciousness.a2a.get_agent_list(),
            "memory": await memory_system.get_memory_stats() if memory_system else {},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store")
async def store_memory(request: MemoryRequest):
    """Store a memory."""
    if not memory_system:
        raise HTTPException(status_code=500, detail="Memory system not initialized")
    
    try:
        result = await memory_system.store_memory(
            content=request.content,
            memory_type=request.memory_type,
            metadata=request.metadata
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/query")
async def query_memories(request: QueryRequest):
    """Query memories."""
    if not memory_system:
        raise HTTPException(status_code=500, detail="Memory system not initialized")
    
    try:
        results = await memory_system.retrieve_memories(
            query=request.query,
            memory_type=request.memory_type,
            limit=request.limit
        )
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    if not memory_system:
        raise HTTPException(status_code=500, detail="Memory system not initialized")
    
    try:
        return await memory_system.get_memory_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    if not consciousness:
        raise HTTPException(status_code=500, detail="Consciousness system not initialized")
    
    return {"agents": consciousness.a2a.get_agent_list()}


@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get status of a specific agent."""
    if not consciousness:
        raise HTTPException(status_code=500, detail="Consciousness system not initialized")
    
    if agent_id not in consciousness.a2a.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent = consciousness.a2a.agents[agent_id]
        status = await agent.get_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/external/{channel}")
async def stream_external_messages(channel: str = "default"):
    """Stream external messages from James."""
    async def generate():
        # This would integrate with the seed tools external message system
        # For now, return a placeholder stream
        while True:
            message = {
                "id": str(uuid.uuid4()),
                "content": f"Stream message from channel {channel}",
                "timestamp": datetime.now().isoformat(),
                "channel": channel
            }
            yield f"data: {json.dumps(message)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/tools/execute")
async def execute_tool(tool_name: str, parameters: Dict[str, Any]):
    """Execute a seed tool directly."""
    if not consciousness:
        raise HTTPException(status_code=500, detail="Consciousness system not initialized")
    
    # Find seed tools agent
    seed_tools_agent = None
    for agent in consciousness.a2a.agents.values():
        if agent.agent_id == "seed_tools":
            seed_tools_agent = agent
            break
    
    if not seed_tools_agent:
        raise HTTPException(status_code=404, detail="Seed tools not found")
    
    try:
        from james.core.a2a import A2AMessage
        
        message = A2AMessage(
            sender_id="api",
            receiver_id="seed_tools",
            action=tool_name,
            payload=parameters
        )
        
        response = await seed_tools_agent.handle_message(message)
        
        return {
            "success": response.success,
            "result": response.result,
            "error": response.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
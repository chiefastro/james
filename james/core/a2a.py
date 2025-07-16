"""Agent-to-Agent (A2A) communication protocol for James."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import asyncio
import json
from datetime import datetime
import uuid


@dataclass
class A2AMessage:
    sender_id: str
    receiver_id: str
    action: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


@dataclass
class A2AResponse:
    message_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class SubAgentProtocol(Protocol):
    """Protocol that all subagents must implement for A2A communication."""
    
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    
    async def handle_message(self, message: A2AMessage) -> A2AResponse:
        """Handle an incoming A2A message."""
        ...
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent."""
        ...


class A2AProtocol:
    """Central A2A communication hub."""
    
    def __init__(self) -> None:
        self.agents: Dict[str, SubAgentProtocol] = {}
        self.message_handlers: Dict[str, asyncio.Queue[A2AMessage]] = {}
        
    def register_agent(self, agent: SubAgentProtocol) -> None:
        """Register a subagent with the A2A system."""
        self.agents[agent.agent_id] = agent
        self.message_handlers[agent.agent_id] = asyncio.Queue()
        
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister a subagent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.message_handlers[agent_id]
            
    async def send_message(self, message: A2AMessage) -> A2AResponse:
        """Send a message to a subagent."""
        if message.receiver_id not in self.agents:
            return A2AResponse(
                message_id=message.message_id,
                success=False,
                error=f"Agent {message.receiver_id} not found"
            )
            
        try:
            agent = self.agents[message.receiver_id]
            response = await agent.handle_message(message)
            return response
        except Exception as e:
            return A2AResponse(
                message_id=message.message_id,
                success=False,
                error=str(e)
            )
            
    async def broadcast_message(self, message: A2AMessage, exclude_sender: bool = True) -> List[A2AResponse]:
        """Broadcast a message to all registered agents."""
        responses = []
        for agent_id, agent in self.agents.items():
            if exclude_sender and agent_id == message.sender_id:
                continue
                
            broadcast_msg = A2AMessage(
                sender_id=message.sender_id,
                receiver_id=agent_id,
                action=message.action,
                payload=message.payload,
                correlation_id=message.correlation_id
            )
            response = await self.send_message(broadcast_msg)
            responses.append(response)
            
        return responses
        
    def get_agent_list(self) -> List[Dict[str, Any]]:
        """Get list of all registered agents with their metadata."""
        return [
            {
                "id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities
            }
            for agent in self.agents.values()
        ]
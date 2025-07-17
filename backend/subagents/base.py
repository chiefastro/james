"""
Base class for all subagents in the Conscious Agent System.

Provides common functionality for A2A protocol communication, task handling,
and result formatting that all subagents inherit.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..models.core import Subagent
from ..protocol.a2a_models import A2AMessage, A2AMessageType, A2APayload, A2AMessageStatus

logger = logging.getLogger(__name__)


@dataclass
class SubagentResult:
    """Result of a subagent operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSubagent(ABC):
    """
    Base class for all subagents.
    
    Provides common functionality for A2A protocol communication,
    task handling, and result formatting.
    """
    
    def __init__(self, subagent_id: str, name: str, description: str, capabilities: List[str]):
        """
        Initialize the base subagent.
        
        Args:
            subagent_id: Unique identifier for this subagent
            name: Human-readable name
            description: Description of capabilities
            capabilities: List of capability strings
        """
        self.subagent_id = subagent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    async def process_task(self, task_id: str, task_description: str, input_data: Dict[str, Any]) -> SubagentResult:
        """
        Process a task assigned to this subagent.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of what needs to be done
            input_data: Input data for the task
            
        Returns:
            SubagentResult with the outcome
        """
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for input data this subagent expects.
        
        Returns:
            JSON schema dictionary
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for output data this subagent produces.
        
        Returns:
            JSON schema dictionary
        """
        pass
    
    async def handle_a2a_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        Handle an incoming A2A protocol message.
        
        Args:
            message: The A2A message to handle
            
        Returns:
            Response message if applicable
        """
        try:
            if message.header.message_type == A2AMessageType.TASK_REQUEST:
                return await self._handle_task_request(message)
            elif message.header.message_type == A2AMessageType.CAPABILITY_QUERY:
                return await self._handle_capability_query(message)
            else:
                self.logger.warning(f"Unsupported message type: {message.header.message_type}")
                return self._create_error_response(message, "Unsupported message type")
                
        except Exception as e:
            self.logger.error(f"Error handling A2A message: {e}")
            return self._create_error_response(message, f"Internal error: {str(e)}")
    
    async def _handle_task_request(self, message: A2AMessage) -> A2AMessage:
        """Handle a task request message."""
        task_id = message.payload.task_id
        task_description = message.payload.task_description
        input_data = message.payload.input_data
        
        if not task_id or not task_description:
            return self._create_error_response(message, "Missing task_id or task_description")
        
        # Track active task
        self._active_tasks[task_id] = {
            "start_time": datetime.now(timezone.utc),
            "status": "processing",
            "message_id": message.header.message_id
        }
        
        try:
            # Process the task
            result = await self.process_task(task_id, task_description, input_data)
            
            # Update task status
            self._active_tasks[task_id]["status"] = "completed" if result.success else "failed"
            self._active_tasks[task_id]["end_time"] = datetime.now(timezone.utc)
            
            # Create response payload
            response_payload = A2APayload(
                task_id=task_id,
                status=A2AMessageStatus.COMPLETED if result.success else A2AMessageStatus.FAILED,
                output_data=result.data if result.success else {},
                error_message=result.error if not result.success else None,
                metadata=result.metadata
            )
            
            # Create response message
            return message.create_reply(self.subagent_id, response_payload)
            
        except Exception as e:
            # Update task status
            self._active_tasks[task_id]["status"] = "error"
            self._active_tasks[task_id]["end_time"] = datetime.now(timezone.utc)
            
            self.logger.error(f"Error processing task {task_id}: {e}")
            return self._create_error_response(message, f"Task processing failed: {str(e)}")
    
    async def _handle_capability_query(self, message: A2AMessage) -> A2AMessage:
        """Handle a capability query message."""
        requested_capabilities = message.payload.capabilities_requested
        
        # Find matching capabilities
        offered_capabilities = []
        if requested_capabilities:
            offered_capabilities = [cap for cap in self.capabilities if cap in requested_capabilities]
        else:
            # If no specific capabilities requested, offer all
            offered_capabilities = self.capabilities.copy()
        
        # Create response payload
        response_payload = A2APayload(
            status=A2AMessageStatus.COMPLETED,
            capabilities_offered=offered_capabilities,
            metadata={
                "subagent_name": self.name,
                "subagent_description": self.description,
                "all_capabilities": self.capabilities
            }
        )
        
        # Create response message
        return message.create_reply(self.subagent_id, response_payload)
    
    def _create_error_response(self, original_message: A2AMessage, error_msg: str) -> A2AMessage:
        """Create an error response message."""
        error_payload = A2APayload(
            task_id=original_message.payload.task_id,
            status=A2AMessageStatus.FAILED,
            error_message=error_msg
        )
        
        return original_message.create_reply(self.subagent_id, error_payload)
    
    def _create_success_result(self, data: Any = None, metadata: Dict[str, Any] = None) -> SubagentResult:
        """Create a successful subagent result."""
        return SubagentResult(
            success=True,
            data=data,
            metadata=metadata or {}
        )
    
    def _create_error_result(self, error: str, metadata: Dict[str, Any] = None) -> SubagentResult:
        """Create an error subagent result."""
        self.logger.error(f"Subagent {self.name} failed: {error}")
        return SubagentResult(
            success=False,
            error=error,
            metadata=metadata or {}
        )
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active tasks."""
        return self._active_tasks.copy()
    
    def get_subagent_metadata(self) -> Subagent:
        """
        Get the Subagent model instance for registry registration.
        
        Returns:
            Subagent model with metadata
        """
        return Subagent(
            id=self.subagent_id,
            name=self.name,
            description=self.description,
            input_schema=self.get_input_schema(),
            output_schema=self.get_output_schema(),
            import_path=f"{self.__class__.__module__}.{self.__class__.__name__}",
            capabilities=self.capabilities.copy(),
            is_active=True
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the subagent.
        
        Returns:
            Health status information
        """
        return {
            "subagent_id": self.subagent_id,
            "name": self.name,
            "status": "healthy",
            "active_tasks": len(self._active_tasks),
            "capabilities": self.capabilities,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
"""
A2A Protocol Communication Handlers.

Provides async communication handlers for subagent interactions,
including message routing, response handling, and error management.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from .a2a_auth import A2AAuthenticator, A2AKeyManager, A2AValidator
from .a2a_models import (
    A2AMessage, A2AMessageType, A2AMessageStatus, A2AHeader, A2APayload
)

logger = logging.getLogger(__name__)


class A2AMessageHandler:
    """
    Handles incoming A2A messages and routes them to appropriate handlers.
    """
    
    def __init__(self, agent_id: str, authenticator: A2AAuthenticator, validator: A2AValidator) -> None:
        """
        Initialize message handler.
        
        Args:
            agent_id: Unique identifier for this agent
            authenticator: Message authenticator instance
            validator: Message validator instance
        """
        self.agent_id = agent_id
        self.authenticator = authenticator
        self.validator = validator
        self._handlers: Dict[A2AMessageType, Callable] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
    
    def register_handler(self, message_type: A2AMessageType, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: The type of message to handle
            handler: Async function to handle the message
        """
        self._handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type.value}")
    
    async def start(self) -> None:
        """Start the message processing loop."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"A2A message handler started for agent {self.agent_id}")
    
    async def stop(self) -> None:
        """Stop the message processing loop."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"A2A message handler stopped for agent {self.agent_id}")
    
    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        Handle an incoming A2A message.
        
        Args:
            message: The message to handle
            
        Returns:
            Response message if applicable
        """
        try:
            # Validate message
            is_valid, error_msg = self.validator.validate_message(message)
            if not is_valid:
                logger.warning(f"Invalid message from {message.header.sender_id}: {error_msg}")
                return self._create_error_response(message, f"Validation failed: {error_msg}")
            
            # Verify authentication
            if not self.authenticator.verify_message(message):
                logger.warning(f"Authentication failed for message from {message.header.sender_id}")
                return self._create_error_response(message, "Authentication failed")
            
            # Add to processing queue
            await self._message_queue.put(message)
            
            # If this is a response to a pending request, resolve the future
            if message.header.correlation_id in self._pending_responses:
                future = self._pending_responses.pop(message.header.correlation_id)
                if not future.done():
                    future.set_result(message)
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return self._create_error_response(message, f"Internal error: {str(e)}")
    
    async def send_message(self, recipient_id: str, message_type: A2AMessageType, 
                          payload: A2APayload, timeout: float = 30.0) -> Optional[A2AMessage]:
        """
        Send a message to another agent and optionally wait for response.
        
        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message to send
            payload: Message payload
            timeout: Timeout for response (if expecting one)
            
        Returns:
            Response message if received within timeout
        """
        # Create message header
        header = A2AHeader(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type
        )
        
        # Create message
        message = A2AMessage(header=header, payload=payload)
        
        # Sign message
        sender_key = self.authenticator.key_manager.get_key(self.agent_id)
        if not sender_key:
            raise ValueError(f"No authentication key found for agent {self.agent_id}")
        
        signed_message = self.authenticator.sign_message(message, sender_key)
        
        # If expecting a response, set up future
        response_future = None
        if message_type in [A2AMessageType.TASK_REQUEST, A2AMessageType.CAPABILITY_QUERY]:
            response_future = asyncio.Future()
            self._pending_responses[message.header.message_id] = response_future
        
        # Send message (this would be implemented by the transport layer)
        await self._send_message_transport(signed_message)
        
        # Wait for response if expecting one
        if response_future:
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self._pending_responses.pop(message.header.message_id, None)
                logger.warning(f"Timeout waiting for response from {recipient_id}")
                return None
        
        return None
    
    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                
                # Find appropriate handler
                handler = self._handlers.get(message.header.message_type)
                if handler:
                    try:
                        response = await handler(message)
                        if response:
                            await self._send_message_transport(response)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")
                        error_response = self._create_error_response(message, f"Handler error: {str(e)}")
                        await self._send_message_transport(error_response)
                else:
                    logger.warning(f"No handler for message type: {message.header.message_type.value}")
                
                self._message_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _create_error_response(self, original_message: A2AMessage, error_msg: str) -> A2AMessage:
        """Create an error response message."""
        error_header = A2AHeader(
            sender_id=self.agent_id,
            recipient_id=original_message.header.sender_id,
            message_type=A2AMessageType.ERROR_REPORT,
            correlation_id=original_message.header.message_id
        )
        
        error_payload = A2APayload(
            status=A2AMessageStatus.FAILED,
            error_message=error_msg
        )
        
        return A2AMessage(header=error_header, payload=error_payload)
    
    async def _send_message_transport(self, message: A2AMessage) -> None:
        """
        Send message via transport layer.
        This is a placeholder - actual implementation would depend on transport mechanism.
        """
        # This would be implemented by the specific transport layer
        # For now, just log the message
        logger.info(f"Sending A2A message: {message.header.message_type.value} "
                   f"from {message.header.sender_id} to {message.header.recipient_id}")


class A2ASubagentClient:
    """
    Client for communicating with subagents using A2A protocol.
    """
    
    def __init__(self, agent_id: str, message_handler: A2AMessageHandler) -> None:
        """
        Initialize subagent client.
        
        Args:
            agent_id: ID of the main agent
            message_handler: Message handler instance
        """
        self.agent_id = agent_id
        self.message_handler = message_handler
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def delegate_task(self, subagent_id: str, task_id: str, task_description: str,
                           input_data: Dict[str, Any], timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """
        Delegate a task to a subagent.
        
        Args:
            subagent_id: ID of the subagent to delegate to
            task_id: Unique task identifier
            task_description: Description of the task
            input_data: Input data for the task
            timeout: Timeout for task completion
            
        Returns:
            Task result or None if failed/timeout
        """
        # Create task payload
        payload = A2APayload(
            task_id=task_id,
            task_description=task_description,
            input_data=input_data,
            status=A2AMessageStatus.PENDING
        )
        
        # Track active task
        self._active_tasks[task_id] = {
            "subagent_id": subagent_id,
            "start_time": time.time(),
            "status": "pending"
        }
        
        try:
            # Send task request
            response = await self.message_handler.send_message(
                recipient_id=subagent_id,
                message_type=A2AMessageType.TASK_REQUEST,
                payload=payload,
                timeout=timeout
            )
            
            if response and response.payload.status == A2AMessageStatus.COMPLETED:
                self._active_tasks[task_id]["status"] = "completed"
                return response.payload.output_data
            elif response and response.payload.status == A2AMessageStatus.FAILED:
                self._active_tasks[task_id]["status"] = "failed"
                logger.error(f"Task {task_id} failed: {response.payload.error_message}")
                return None
            else:
                self._active_tasks[task_id]["status"] = "timeout"
                logger.warning(f"Task {task_id} timed out")
                return None
                
        except Exception as e:
            self._active_tasks[task_id]["status"] = "error"
            logger.error(f"Error delegating task {task_id}: {e}")
            return None
        finally:
            # Clean up completed task
            if task_id in self._active_tasks:
                self._active_tasks[task_id]["end_time"] = time.time()
    
    async def query_capabilities(self, subagent_id: str, capabilities: List[str]) -> Optional[List[str]]:
        """
        Query a subagent for its capabilities.
        
        Args:
            subagent_id: ID of the subagent to query
            capabilities: List of capabilities to check for
            
        Returns:
            List of supported capabilities or None if failed
        """
        payload = A2APayload(capabilities_requested=capabilities)
        
        try:
            response = await self.message_handler.send_message(
                recipient_id=subagent_id,
                message_type=A2AMessageType.CAPABILITY_QUERY,
                payload=payload,
                timeout=10.0
            )
            
            if response and response.header.message_type == A2AMessageType.CAPABILITY_RESPONSE:
                return response.payload.capabilities_offered
            
            return None
            
        except Exception as e:
            logger.error(f"Error querying capabilities from {subagent_id}: {e}")
            return None
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active tasks."""
        return self._active_tasks.copy()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel an active task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled, False if not found
        """
        if task_id in self._active_tasks:
            self._active_tasks[task_id]["status"] = "cancelled"
            return True
        return False


class A2AProtocolManager:
    """
    Main manager for A2A protocol operations.
    """
    
    def __init__(self, agent_id: str) -> None:
        """
        Initialize A2A protocol manager.
        
        Args:
            agent_id: Unique identifier for this agent
        """
        self.agent_id = agent_id
        self.key_manager = A2AKeyManager()
        self.authenticator = A2AAuthenticator(self.key_manager)
        self.validator = A2AValidator()
        self.message_handler = A2AMessageHandler(agent_id, self.authenticator, self.validator)
        self.subagent_client = A2ASubagentClient(agent_id, self.message_handler)
        
        # Generate key for this agent
        self.key_manager.generate_key(agent_id)
    
    async def start(self) -> None:
        """Start the A2A protocol manager."""
        await self.message_handler.start()
        logger.info(f"A2A Protocol Manager started for agent {self.agent_id}")
    
    async def stop(self) -> None:
        """Stop the A2A protocol manager."""
        await self.message_handler.stop()
        logger.info(f"A2A Protocol Manager stopped for agent {self.agent_id}")
    
    def register_subagent(self, subagent_id: str) -> str:
        """
        Register a new subagent and generate authentication key.
        
        Args:
            subagent_id: Unique identifier for the subagent
            
        Returns:
            Generated authentication key
        """
        return self.key_manager.generate_key(subagent_id)
    
    def revoke_subagent(self, subagent_id: str) -> bool:
        """
        Revoke a subagent's authentication key.
        
        Args:
            subagent_id: Unique identifier for the subagent
            
        Returns:
            True if revoked, False if not found
        """
        return self.key_manager.revoke_key(subagent_id)
    
    def get_registered_subagents(self) -> Set[str]:
        """Get set of all registered subagents."""
        return self.key_manager.list_agents() - {self.agent_id}
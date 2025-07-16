"""
Unit tests for A2A Protocol implementation.

Tests message formatting, authentication, validation, and communication handlers
for protocol compliance and security.
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from backend.protocol.a2a_auth import (
    A2AAuthenticationError, A2AAuthenticator, A2AKeyManager, A2AValidator, A2AValidationError
)
from backend.protocol.a2a_handlers import (
    A2AMessageHandler, A2AProtocolManager, A2ASubagentClient
)
from backend.protocol.a2a_models import (
    A2AHeader, A2AMessage, A2AMessageStatus, A2AMessageType, A2APayload, A2ASignature
)


class TestA2AModels:
    """Test A2A protocol data models."""
    
    def test_a2a_header_creation(self):
        """Test A2A header creation and validation."""
        header = A2AHeader(
            sender_id="agent1",
            recipient_id="agent2",
            message_type=A2AMessageType.TASK_REQUEST
        )
        
        assert header.sender_id == "agent1"
        assert header.recipient_id == "agent2"
        assert header.message_type == A2AMessageType.TASK_REQUEST
        assert header.protocol_version == "1.0"
        assert header.ttl_seconds == 300
        assert not header.is_expired()
    
    def test_a2a_header_validation(self):
        """Test A2A header validation."""
        # Test empty sender ID
        with pytest.raises(ValueError, match="Sender ID cannot be empty"):
            A2AHeader(sender_id="", recipient_id="agent2")
        
        # Test empty recipient ID
        with pytest.raises(ValueError, match="Recipient ID cannot be empty"):
            A2AHeader(sender_id="agent1", recipient_id="")
        
        # Test invalid TTL
        with pytest.raises(ValueError, match="TTL must be positive"):
            A2AHeader(sender_id="agent1", recipient_id="agent2", ttl_seconds=0)
    
    def test_a2a_header_expiration(self):
        """Test A2A header expiration logic."""
        # Create header with short TTL
        header = A2AHeader(
            sender_id="agent1",
            recipient_id="agent2",
            ttl_seconds=1
        )
        
        assert not header.is_expired()
        
        # Wait for expiration
        time.sleep(1.1)
        assert header.is_expired()
    
    def test_a2a_payload_creation(self):
        """Test A2A payload creation."""
        payload = A2APayload(
            task_id="task123",
            task_description="Test task",
            input_data={"key": "value"},
            status=A2AMessageStatus.PENDING
        )
        
        assert payload.task_id == "task123"
        assert payload.task_description == "Test task"
        assert payload.input_data == {"key": "value"}
        assert payload.status == A2AMessageStatus.PENDING
    
    def test_a2a_message_creation(self):
        """Test complete A2A message creation."""
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        
        assert message.header == header
        assert message.payload == payload
        assert message.signature is None
    
    def test_a2a_message_serialization(self):
        """Test A2A message serialization to/from JSON."""
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123", input_data={"test": "data"})
        message = A2AMessage(header=header, payload=payload)
        
        # Test to_dict
        message_dict = message.to_dict()
        assert "header" in message_dict
        assert "payload" in message_dict
        assert message_dict["header"]["sender_id"] == "agent1"
        assert message_dict["payload"]["task_id"] == "task123"
        
        # Test to_json
        json_str = message.to_json()
        assert isinstance(json_str, str)
        
        # Test from_json
        reconstructed = A2AMessage.from_json(json_str)
        assert reconstructed.header.sender_id == "agent1"
        assert reconstructed.payload.task_id == "task123"
    
    def test_a2a_message_validation(self):
        """Test A2A message validation."""
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        
        assert message.is_valid()
        
        # Test expired message
        header.ttl_seconds = 0
        header.timestamp = time.time() - 1
        assert not message.is_valid()
    
    def test_a2a_message_reply(self):
        """Test creating reply messages."""
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        original_message = A2AMessage(header=header, payload=payload)
        
        reply_payload = A2APayload(task_id="task123", status=A2AMessageStatus.COMPLETED)
        reply = original_message.create_reply("agent2", reply_payload)
        
        assert reply.header.sender_id == "agent2"
        assert reply.header.recipient_id == "agent1"
        assert reply.header.correlation_id == original_message.header.message_id
        assert reply.header.reply_to == original_message.header.message_id


class TestA2AKeyManager:
    """Test A2A key management."""
    
    def test_key_generation(self):
        """Test key generation for agents."""
        key_manager = A2AKeyManager()
        
        key = key_manager.generate_key("agent1")
        assert isinstance(key, str)
        assert len(key) == 64  # 32 bytes as hex = 64 characters
        
        # Test key retrieval
        retrieved_key = key_manager.get_key("agent1")
        assert retrieved_key == key
    
    def test_key_revocation(self):
        """Test key revocation."""
        key_manager = A2AKeyManager()
        
        key_manager.generate_key("agent1")
        assert key_manager.get_key("agent1") is not None
        
        # Revoke key
        assert key_manager.revoke_key("agent1") is True
        assert key_manager.get_key("agent1") is None
        
        # Try to revoke non-existent key
        assert key_manager.revoke_key("nonexistent") is False
    
    def test_key_usage_tracking(self):
        """Test key usage statistics."""
        key_manager = A2AKeyManager()
        
        key_manager.generate_key("agent1")
        
        # Check initial metadata
        metadata = key_manager._key_metadata["agent1"]
        assert metadata["usage_count"] == 0
        assert metadata["last_used"] is None
        
        # Update usage
        key_manager.update_key_usage("agent1")
        
        metadata = key_manager._key_metadata["agent1"]
        assert metadata["usage_count"] == 1
        assert metadata["last_used"] is not None
    
    def test_list_agents(self):
        """Test listing registered agents."""
        key_manager = A2AKeyManager()
        
        key_manager.generate_key("agent1")
        key_manager.generate_key("agent2")
        
        agents = key_manager.list_agents()
        assert agents == {"agent1", "agent2"}


class TestA2AAuthenticator:
    """Test A2A message authentication."""
    
    def test_message_signing(self):
        """Test message signing with HMAC."""
        key_manager = A2AKeyManager()
        authenticator = A2AAuthenticator(key_manager)
        
        # Generate key for sender
        sender_key = key_manager.generate_key("agent1")
        
        # Create message
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        
        # Sign message
        signed_message = authenticator.sign_message(message, sender_key)
        
        assert signed_message.signature is not None
        assert signed_message.signature.algorithm == "HMAC-SHA256"
        assert signed_message.signature.key_id == "agent1"
        assert len(signed_message.signature.signature) == 64  # SHA256 hex
    
    def test_message_verification(self):
        """Test message signature verification."""
        key_manager = A2AKeyManager()
        authenticator = A2AAuthenticator(key_manager)
        
        # Generate key for sender
        sender_key = key_manager.generate_key("agent1")
        
        # Create and sign message
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        signed_message = authenticator.sign_message(message, sender_key)
        
        # Verify signature
        assert authenticator.verify_message(signed_message) is True
    
    def test_message_verification_failure(self):
        """Test message verification failure cases."""
        key_manager = A2AKeyManager()
        authenticator = A2AAuthenticator(key_manager)
        
        # Test message without signature
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        
        with pytest.raises(A2AAuthenticationError, match="Message has no signature"):
            authenticator.verify_message(message)
        
        # Test message with unknown sender
        signature = A2ASignature(signature="fake", key_id="unknown")
        message.signature = signature
        
        with pytest.raises(A2AAuthenticationError, match="No key found for sender"):
            authenticator.verify_message(message)
    
    def test_signature_tampering_detection(self):
        """Test detection of message tampering."""
        key_manager = A2AKeyManager()
        authenticator = A2AAuthenticator(key_manager)
        
        # Generate key and create signed message
        sender_key = key_manager.generate_key("agent1")
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        signed_message = authenticator.sign_message(message, sender_key)
        
        # Tamper with message content
        signed_message.payload.task_id = "tampered"
        
        # Verification should fail
        assert authenticator.verify_message(signed_message) is False


class TestA2AValidator:
    """Test A2A message validation."""
    
    def test_message_validation_success(self):
        """Test successful message validation."""
        validator = A2AValidator()
        
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123", task_description="Test task")
        message = A2AMessage(header=header, payload=payload)
        
        is_valid, error_msg = validator.validate_message(message)
        assert is_valid is True
        assert error_msg is None
    
    def test_message_size_validation(self):
        """Test message size validation."""
        validator = A2AValidator(max_message_size=100)  # Very small limit
        
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(
            task_id="task123",
            task_description="Very long description " * 100  # Make it large
        )
        message = A2AMessage(header=header, payload=payload)
        
        is_valid, error_msg = validator.validate_message(message)
        assert is_valid is False
        assert "exceeds limit" in error_msg
    
    def test_expired_message_validation(self):
        """Test validation of expired messages."""
        validator = A2AValidator()
        
        header = A2AHeader(
            sender_id="agent1",
            recipient_id="agent2",
            timestamp=time.time() - 1000,  # Old timestamp
            ttl_seconds=1  # Short TTL
        )
        payload = A2APayload(task_id="task123")
        message = A2AMessage(header=header, payload=payload)
        
        is_valid, error_msg = validator.validate_message(message)
        assert is_valid is False
        assert "expired" in error_msg
    
    def test_message_type_specific_validation(self):
        """Test validation specific to message types."""
        validator = A2AValidator()
        
        # Test task request without task_id
        header = A2AHeader(
            sender_id="agent1",
            recipient_id="agent2",
            message_type=A2AMessageType.TASK_REQUEST
        )
        payload = A2APayload()  # Missing task_id
        message = A2AMessage(header=header, payload=payload)
        
        is_valid, error_msg = validator.validate_message(message)
        assert is_valid is False
        assert "task_id" in error_msg
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        validator = A2AValidator()
        
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123", task_description="Test")
        message = A2AMessage(header=header, payload=payload)
        
        # Send messages up to limit
        for i in range(60):  # Default limit is 60 per minute
            is_valid, _ = validator.validate_message(message)
            assert is_valid is True
        
        # Next message should be rate limited
        is_valid, error_msg = validator.validate_message(message)
        assert is_valid is False
        assert "Rate limit exceeded" in error_msg
    
    def test_agent_blocking(self):
        """Test agent blocking functionality."""
        validator = A2AValidator()
        
        header = A2AHeader(sender_id="agent1", recipient_id="agent2")
        payload = A2APayload(task_id="task123", task_description="Test")
        message = A2AMessage(header=header, payload=payload)
        
        # Block agent
        validator.block_agent("agent1")
        assert validator.is_agent_blocked("agent1") is True
        
        # Message should be rejected
        is_valid, error_msg = validator.validate_message(message)
        assert is_valid is False
        assert "blocked" in error_msg
        
        # Unblock agent
        validator.unblock_agent("agent1")
        assert validator.is_agent_blocked("agent1") is False


class TestA2AMessageHandler:
    """Test A2A message handling."""
    
    @pytest.fixture
    def setup_handler(self):
        """Set up message handler for testing."""
        key_manager = A2AKeyManager()
        authenticator = A2AAuthenticator(key_manager)
        validator = A2AValidator()
        handler = A2AMessageHandler("main_agent", authenticator, validator)
        
        # Generate keys
        key_manager.generate_key("main_agent")
        key_manager.generate_key("subagent1")
        
        return handler, key_manager, authenticator
    
    @pytest.mark.asyncio
    async def test_handler_start_stop(self, setup_handler):
        """Test starting and stopping message handler."""
        handler, _, _ = setup_handler
        
        await handler.start()
        assert handler._running is True
        
        await handler.stop()
        assert handler._running is False
    
    @pytest.mark.asyncio
    async def test_message_handling(self, setup_handler):
        """Test basic message handling."""
        handler, key_manager, authenticator = setup_handler
        
        # Create test handler function
        handled_messages = []
        
        async def test_handler_func(message):
            handled_messages.append(message)
            return None
        
        handler.register_handler(A2AMessageType.TASK_REQUEST, test_handler_func)
        
        # Create and sign message
        header = A2AHeader(
            sender_id="subagent1",
            recipient_id="main_agent",
            message_type=A2AMessageType.TASK_REQUEST
        )
        payload = A2APayload(task_id="task123", task_description="Test task")
        message = A2AMessage(header=header, payload=payload)
        
        sender_key = key_manager.get_key("subagent1")
        signed_message = authenticator.sign_message(message, sender_key)
        
        # Start handler and process message
        await handler.start()
        
        response = await handler.handle_message(signed_message)
        
        # Wait a bit for async processing
        await asyncio.sleep(0.1)
        
        await handler.stop()
        
        # Check that message was handled
        assert len(handled_messages) == 1
        assert handled_messages[0].payload.task_id == "task123"


class TestA2ASubagentClient:
    """Test A2A subagent client."""
    
    @pytest.fixture
    def setup_client(self):
        """Set up subagent client for testing."""
        key_manager = A2AKeyManager()
        authenticator = A2AAuthenticator(key_manager)
        validator = A2AValidator()
        handler = A2AMessageHandler("main_agent", authenticator, validator)
        client = A2ASubagentClient("main_agent", handler)
        
        # Generate keys
        key_manager.generate_key("main_agent")
        key_manager.generate_key("subagent1")
        
        return client, handler, key_manager
    
    @pytest.mark.asyncio
    async def test_task_delegation(self, setup_client):
        """Test task delegation to subagents."""
        client, handler, key_manager = setup_client
        
        # Mock the send_message method to simulate successful response
        async def mock_send_message(recipient_id, message_type, payload, timeout):
            # Create mock response
            response_header = A2AHeader(
                sender_id=recipient_id,
                recipient_id="main_agent",
                message_type=A2AMessageType.TASK_RESPONSE
            )
            response_payload = A2APayload(
                task_id=payload.task_id,
                status=A2AMessageStatus.COMPLETED,
                output_data={"result": "success"}
            )
            return A2AMessage(header=response_header, payload=response_payload)
        
        handler.send_message = mock_send_message
        
        # Delegate task
        result = await client.delegate_task(
            subagent_id="subagent1",
            task_id="task123",
            task_description="Test task",
            input_data={"input": "data"}
        )
        
        assert result is not None
        assert result["result"] == "success"
        
        # Check active tasks tracking
        active_tasks = client.get_active_tasks()
        assert "task123" in active_tasks
        assert active_tasks["task123"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_capability_query(self, setup_client):
        """Test querying subagent capabilities."""
        client, handler, key_manager = setup_client
        
        # Mock the send_message method
        async def mock_send_message(recipient_id, message_type, payload, timeout):
            response_header = A2AHeader(
                sender_id=recipient_id,
                recipient_id="main_agent",
                message_type=A2AMessageType.CAPABILITY_RESPONSE
            )
            response_payload = A2APayload(
                capabilities_offered=["capability1", "capability2"]
            )
            return A2AMessage(header=response_header, payload=response_payload)
        
        handler.send_message = mock_send_message
        
        # Query capabilities
        capabilities = await client.query_capabilities(
            subagent_id="subagent1",
            capabilities=["capability1", "capability2", "capability3"]
        )
        
        assert capabilities == ["capability1", "capability2"]


class TestA2AProtocolManager:
    """Test A2A protocol manager."""
    
    @pytest.mark.asyncio
    async def test_protocol_manager_lifecycle(self):
        """Test protocol manager start/stop lifecycle."""
        manager = A2AProtocolManager("main_agent")
        
        await manager.start()
        assert manager.message_handler._running is True
        
        await manager.stop()
        assert manager.message_handler._running is False
    
    def test_subagent_registration(self):
        """Test subagent registration and key management."""
        manager = A2AProtocolManager("main_agent")
        
        # Register subagent
        key = manager.register_subagent("subagent1")
        assert isinstance(key, str)
        assert len(key) == 64
        
        # Check that subagent is registered
        subagents = manager.get_registered_subagents()
        assert "subagent1" in subagents
        assert "main_agent" not in subagents  # Main agent should be excluded
        
        # Revoke subagent
        assert manager.revoke_subagent("subagent1") is True
        subagents = manager.get_registered_subagents()
        assert "subagent1" not in subagents


if __name__ == "__main__":
    pytest.main([__file__])
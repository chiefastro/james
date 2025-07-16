"""
A2A Protocol Demo.

Demonstrates the Agent-to-Agent protocol implementation including
message creation, authentication, validation, and communication.
"""

import asyncio
import json
import logging
from typing import Dict, Any

from backend.protocol.a2a_auth import A2AKeyManager, A2AAuthenticator, A2AValidator
from backend.protocol.a2a_handlers import A2AMessageHandler, A2AProtocolManager, A2ASubagentClient
from backend.protocol.a2a_models import (
    A2AMessage, A2AMessageType, A2AMessageStatus, A2AHeader, A2APayload
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_message_creation():
    """Demonstrate basic A2A message creation and serialization."""
    print("\n=== A2A Message Creation Demo ===")
    
    # Create message header
    header = A2AHeader(
        sender_id="main_agent",
        recipient_id="subagent_1",
        message_type=A2AMessageType.TASK_REQUEST
    )
    
    # Create message payload
    payload = A2APayload(
        task_id="demo_task_001",
        task_description="Process user input and generate response",
        input_data={
            "user_message": "Hello, how are you?",
            "context": {"conversation_id": "conv_123"}
        },
        status=A2AMessageStatus.PENDING
    )
    
    # Create complete message
    message = A2AMessage(header=header, payload=payload)
    
    print(f"Message ID: {message.header.message_id}")
    print(f"From: {message.header.sender_id} -> To: {message.header.recipient_id}")
    print(f"Type: {message.header.message_type.value}")
    print(f"Task: {message.payload.task_description}")
    
    # Serialize to JSON
    json_message = message.to_json()
    print(f"\nSerialized message size: {len(json_message)} bytes")
    
    # Deserialize back
    reconstructed = A2AMessage.from_json(json_message)
    print(f"Reconstructed task ID: {reconstructed.payload.task_id}")
    
    return message


async def demo_authentication():
    """Demonstrate message authentication and signature verification."""
    print("\n=== A2A Authentication Demo ===")
    
    # Set up key manager and authenticator
    key_manager = A2AKeyManager()
    authenticator = A2AAuthenticator(key_manager)
    
    # Generate keys for agents
    main_agent_key = key_manager.generate_key("main_agent")
    subagent_key = key_manager.generate_key("subagent_1")
    
    print(f"Generated key for main_agent: {main_agent_key[:16]}...")
    print(f"Generated key for subagent_1: {subagent_key[:16]}...")
    
    # Create a message
    header = A2AHeader(
        sender_id="main_agent",
        recipient_id="subagent_1",
        message_type=A2AMessageType.TASK_REQUEST
    )
    payload = A2APayload(
        task_id="auth_demo_task",
        task_description="Authenticated task demonstration"
    )
    message = A2AMessage(header=header, payload=payload)
    
    # Sign the message
    signed_message = authenticator.sign_message(message, main_agent_key)
    print(f"\nMessage signed with algorithm: {signed_message.signature.algorithm}")
    print(f"Signature: {signed_message.signature.signature[:32]}...")
    
    # Verify the signature
    is_valid = authenticator.verify_message(signed_message)
    print(f"Signature verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Demonstrate tampering detection
    original_task_id = signed_message.payload.task_id
    signed_message.payload.task_id = "tampered_task_id"
    
    is_valid_after_tampering = authenticator.verify_message(signed_message)
    print(f"Verification after tampering: {'PASSED' if is_valid_after_tampering else 'FAILED'}")
    
    # Restore original content
    signed_message.payload.task_id = original_task_id
    
    return signed_message, authenticator


async def demo_validation():
    """Demonstrate message validation and security checks."""
    print("\n=== A2A Validation Demo ===")
    
    validator = A2AValidator(max_message_size=1024)
    
    # Create valid message
    header = A2AHeader(
        sender_id="main_agent",
        recipient_id="subagent_1",
        message_type=A2AMessageType.TASK_REQUEST
    )
    payload = A2APayload(
        task_id="validation_demo",
        task_description="Message validation demonstration",
        input_data={"test": "data"}
    )
    message = A2AMessage(header=header, payload=payload)
    
    # Test valid message
    is_valid, error_msg = validator.validate_message(message)
    print(f"Valid message check: {'PASSED' if is_valid else 'FAILED'}")
    if error_msg:
        print(f"Error: {error_msg}")
    
    # Test message without required fields
    invalid_payload = A2APayload()  # Missing task_id for TASK_REQUEST
    invalid_message = A2AMessage(header=header, payload=invalid_payload)
    
    is_valid, error_msg = validator.validate_message(invalid_message)
    print(f"Invalid message check: {'FAILED' if not is_valid else 'UNEXPECTED PASS'}")
    print(f"Expected error: {error_msg}")
    
    # Test rate limiting
    print(f"\nTesting rate limiting (max 60 messages/minute)...")
    rate_limit_hit = False
    for i in range(65):  # Try to exceed limit
        is_valid, error_msg = validator.validate_message(message)
        if not is_valid and "Rate limit" in error_msg:
            print(f"Rate limit hit at message {i + 1}")
            rate_limit_hit = True
            break
    
    if not rate_limit_hit:
        print("Rate limiting not triggered (unexpected)")


async def demo_message_handler():
    """Demonstrate async message handling."""
    print("\n=== A2A Message Handler Demo ===")
    
    # Set up components
    key_manager = A2AKeyManager()
    authenticator = A2AAuthenticator(key_manager)
    validator = A2AValidator()
    handler = A2AMessageHandler("main_agent", authenticator, validator)
    
    # Generate keys
    main_key = key_manager.generate_key("main_agent")
    sub_key = key_manager.generate_key("subagent_1")
    
    # Track handled messages
    handled_messages = []
    
    async def task_request_handler(message: A2AMessage) -> A2AMessage:
        """Handle incoming task requests."""
        print(f"Handling task request: {message.payload.task_id}")
        handled_messages.append(message)
        
        # Create response
        response_payload = A2APayload(
            task_id=message.payload.task_id,
            status=A2AMessageStatus.COMPLETED,
            output_data={"result": "Task completed successfully"}
        )
        
        return message.create_reply("main_agent", response_payload)
    
    # Register handler
    handler.register_handler(A2AMessageType.TASK_REQUEST, task_request_handler)
    
    # Start handler
    await handler.start()
    
    # Create and send test message
    header = A2AHeader(
        sender_id="subagent_1",
        recipient_id="main_agent",
        message_type=A2AMessageType.TASK_REQUEST
    )
    payload = A2APayload(
        task_id="handler_demo_task",
        task_description="Test message handling",
        input_data={"test": "input"}
    )
    message = A2AMessage(header=header, payload=payload)
    
    # Sign message
    signed_message = authenticator.sign_message(message, sub_key)
    
    # Handle message
    response = await handler.handle_message(signed_message)
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    # Stop handler
    await handler.stop()
    
    print(f"Messages handled: {len(handled_messages)}")
    if handled_messages:
        print(f"Handled task: {handled_messages[0].payload.task_id}")


async def demo_subagent_client():
    """Demonstrate subagent client functionality."""
    print("\n=== A2A Subagent Client Demo ===")
    
    # Set up protocol manager
    manager = A2AProtocolManager("main_agent")
    
    # Register a subagent
    subagent_key = manager.register_subagent("demo_subagent")
    print(f"Registered subagent with key: {subagent_key[:16]}...")
    
    # Mock successful task delegation
    async def mock_send_message(recipient_id, message_type, payload, timeout):
        """Mock successful subagent response."""
        print(f"Mock: Sending {message_type.value} to {recipient_id}")
        
        if message_type == A2AMessageType.TASK_REQUEST:
            # Simulate task completion
            response_header = A2AHeader(
                sender_id=recipient_id,
                recipient_id="main_agent",
                message_type=A2AMessageType.TASK_RESPONSE
            )
            response_payload = A2APayload(
                task_id=payload.task_id,
                status=A2AMessageStatus.COMPLETED,
                output_data={
                    "processed_input": payload.input_data,
                    "result": "Successfully processed by subagent",
                    "processing_time": 0.5
                }
            )
            return A2AMessage(header=response_header, payload=response_payload)
        
        elif message_type == A2AMessageType.CAPABILITY_QUERY:
            # Simulate capability response
            response_header = A2AHeader(
                sender_id=recipient_id,
                recipient_id="main_agent",
                message_type=A2AMessageType.CAPABILITY_RESPONSE
            )
            response_payload = A2APayload(
                capabilities_offered=["text_processing", "data_analysis", "code_generation"]
            )
            return A2AMessage(header=response_header, payload=response_payload)
    
    # Replace send_message with mock
    manager.message_handler.send_message = mock_send_message
    
    # Test task delegation
    result = await manager.subagent_client.delegate_task(
        subagent_id="demo_subagent",
        task_id="client_demo_task",
        task_description="Process user query",
        input_data={"query": "What is the weather like?", "user_id": "user123"}
    )
    
    print(f"Task delegation result: {result}")
    
    # Test capability query
    capabilities = await manager.subagent_client.query_capabilities(
        subagent_id="demo_subagent",
        capabilities=["text_processing", "image_analysis", "code_generation"]
    )
    
    print(f"Subagent capabilities: {capabilities}")
    
    # Show active tasks
    active_tasks = manager.subagent_client.get_active_tasks()
    print(f"Active tasks: {len(active_tasks)}")
    for task_id, task_info in active_tasks.items():
        print(f"  {task_id}: {task_info['status']}")


async def main():
    """Run all A2A protocol demonstrations."""
    print("🤖 A2A Protocol Demonstration")
    print("=" * 50)
    
    try:
        # Run all demos
        await demo_basic_message_creation()
        await demo_authentication()
        await demo_validation()
        await demo_message_handler()
        await demo_subagent_client()
        
        print("\n✅ All A2A protocol demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
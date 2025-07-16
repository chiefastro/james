#!/usr/bin/env python3
"""
Delegator System Demo

This script demonstrates the Delegator system for subagent coordination,
including subagent retrieval, selection, and task delegation using A2A protocol.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add the backend directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.agents.delegator import (
    Delegator, DelegationRequest, A2AMessage
)
from backend.models.core import Subagent, Task, TaskStatus, Message, MessageSource
from backend.registry.subagent_registry import SubagentRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def create_sample_subagents() -> list[Subagent]:
    """Create sample subagents for demonstration."""
    subagents = [
        Subagent(
            id="code_analyzer_001",
            name="Code Quality Analyzer",
            description="Analyzes code quality, identifies bugs, security issues, and suggests improvements for Python, JavaScript, and TypeScript code",
            capabilities=["code_analysis", "bug_detection", "security_audit", "refactoring", "python", "javascript", "typescript"],
            import_path="subagents.code_analyzer",
            embedding=[0.1] * 1536,  # Mock embedding
            is_active=True
        ),
        Subagent(
            id="doc_writer_001",
            name="Technical Documentation Writer",
            description="Creates comprehensive technical documentation, API docs, README files, and user guides with proper formatting and examples",
            capabilities=["documentation", "technical_writing", "markdown", "api_docs", "user_guides"],
            import_path="subagents.doc_writer",
            embedding=[0.2] * 1536,  # Mock embedding
            is_active=True
        ),
        Subagent(
            id="test_generator_001",
            name="Automated Test Generator",
            description="Generates comprehensive unit tests, integration tests, and test fixtures for various programming languages and frameworks",
            capabilities=["testing", "unit_tests", "integration_tests", "pytest", "jest", "test_fixtures"],
            import_path="subagents.test_generator",
            embedding=[0.3] * 1536,  # Mock embedding
            is_active=True
        ),
        Subagent(
            id="db_optimizer_001",
            name="Database Query Optimizer",
            description="Optimizes database queries, designs efficient schemas, and provides performance tuning recommendations for SQL and NoSQL databases",
            capabilities=["database", "sql_optimization", "schema_design", "performance_tuning", "postgresql", "mongodb"],
            import_path="subagents.db_optimizer",
            embedding=[0.4] * 1536,  # Mock embedding
            is_active=True
        ),
        Subagent(
            id="security_auditor_001",
            name="Security Vulnerability Auditor",
            description="Performs comprehensive security audits, identifies vulnerabilities, and provides remediation strategies for web applications and APIs",
            capabilities=["security_audit", "vulnerability_assessment", "penetration_testing", "owasp", "web_security"],
            import_path="subagents.security_auditor",
            embedding=[0.5] * 1536,  # Mock embedding
            is_active=True
        )
    ]
    
    return subagents


async def setup_registry_with_subagents() -> SubagentRegistry:
    """Set up the subagent registry with sample data."""
    # Use a temporary registry file for demo
    registry_path = "/tmp/delegator_demo_registry.csv"
    registry = SubagentRegistry(registry_path, enable_vector_db=False)
    
    # Register sample subagents
    subagents = await create_sample_subagents()
    for subagent in subagents:
        try:
            await registry.register_subagent(subagent)
            logger.info(f"Registered subagent: {subagent.name}")
        except Exception as e:
            logger.warning(f"Failed to register {subagent.name}: {e}")
    
    return registry


async def demo_basic_delegation():
    """Demonstrate basic delegation workflow."""
    print("\n" + "="*60)
    print("BASIC DELEGATION DEMO")
    print("="*60)
    
    # Set up registry
    registry = await setup_registry_with_subagents()
    
    # Create delegator
    delegator = Delegator(registry, "james_main_agent")
    
    # Create a sample task
    task = Task(
        id="task_001",
        description="Review the user authentication module for security vulnerabilities and code quality issues",
        priority=8,
        status=TaskStatus.PENDING,
        deadline=datetime.now(timezone.utc) + timedelta(hours=4)
    )
    
    # Create a sample message
    message = Message(
        id="msg_001",
        content="I need help analyzing the authentication code. Please check for security issues and suggest improvements.",
        source=MessageSource.USER,
        priority=8
    )
    
    # Create delegation request
    request = DelegationRequest(
        task=task,
        message=message,
        max_subagents=2,
        similarity_threshold=0.6
    )
    
    print(f"Task: {task.description}")
    print(f"Message: {message.content}")
    print(f"Max subagents: {request.max_subagents}")
    print(f"Similarity threshold: {request.similarity_threshold}")
    
    # Execute delegation
    result = await delegator.delegate(request)
    
    # Display results
    print(f"\nDelegation Result:")
    print(f"Success: {result.success}")
    print(f"Selected subagents: {len(result.selected_subagents)}")
    
    if result.selected_subagents:
        print("\nSelected Subagents:")
        for i, subagent in enumerate(result.selected_subagents, 1):
            print(f"  {i}. {subagent.name}")
            print(f"     Capabilities: {', '.join(subagent.capabilities[:5])}")
            print(f"     Description: {subagent.description[:80]}...")
    
    print(f"\nSelection Reasoning:")
    print(result.selection_reasoning)
    
    if result.delegation_messages:
        print(f"\nDelegation Messages Sent: {len(result.delegation_messages)}")
        for i, msg_dict in enumerate(result.delegation_messages, 1):
            print(f"  {i}. To: {msg_dict['recipient_id']}")
            print(f"     Type: {msg_dict['message_type']}")
            print(f"     Correlation ID: {msg_dict['correlation_id']}")
    
    if result.error_message:
        print(f"\nError: {result.error_message}")


async def demo_capability_based_delegation():
    """Demonstrate delegation based on specific capabilities."""
    print("\n" + "="*60)
    print("CAPABILITY-BASED DELEGATION DEMO")
    print("="*60)
    
    # Set up registry
    registry = await setup_registry_with_subagents()
    delegator = Delegator(registry, "james_main_agent")
    
    # Create a task requiring specific capabilities
    task = Task(
        id="task_002",
        description="Generate comprehensive unit tests for the payment processing module",
        priority=6,
        status=TaskStatus.PENDING
    )
    
    message = Message(
        id="msg_002",
        content="I need unit tests for the payment module. Please include edge cases and error handling tests.",
        source=MessageSource.USER,
        priority=6
    )
    
    # Request with specific capability requirements
    request = DelegationRequest(
        task=task,
        message=message,
        max_subagents=1,
        similarity_threshold=0.5,
        require_capabilities=["testing", "unit_tests"]
    )
    
    print(f"Task: {task.description}")
    print(f"Required capabilities: {request.require_capabilities}")
    
    # Execute delegation
    result = await delegator.delegate(request)
    
    # Display results
    print(f"\nDelegation Result:")
    print(f"Success: {result.success}")
    
    if result.selected_subagents:
        subagent = result.selected_subagents[0]
        print(f"\nSelected Subagent: {subagent.name}")
        print(f"Matching capabilities: {[cap for cap in subagent.capabilities if cap in request.require_capabilities]}")
        print(f"All capabilities: {', '.join(subagent.capabilities)}")
    
    print(f"\nSelection Reasoning:")
    print(result.selection_reasoning)


async def demo_exclusion_filtering():
    """Demonstrate delegation with subagent exclusions."""
    print("\n" + "="*60)
    print("EXCLUSION FILTERING DEMO")
    print("="*60)
    
    # Set up registry
    registry = await setup_registry_with_subagents()
    delegator = Delegator(registry, "james_main_agent")
    
    # Create a task
    task = Task(
        id="task_003",
        description="Optimize database queries and improve performance",
        priority=7,
        status=TaskStatus.PENDING
    )
    
    message = Message(
        id="msg_003",
        content="The database queries are running slowly. Please optimize them and suggest schema improvements.",
        source=MessageSource.USER,
        priority=7
    )
    
    # First, show normal delegation
    print("Normal delegation (no exclusions):")
    request1 = DelegationRequest(
        task=task,
        message=message,
        max_subagents=3,
        similarity_threshold=0.4
    )
    
    result1 = await delegator.delegate(request1)
    print(f"Selected {len(result1.selected_subagents)} subagents:")
    for subagent in result1.selected_subagents:
        print(f"  - {subagent.name} (ID: {subagent.id})")
    
    # Now exclude the database optimizer
    print(f"\nDelegation with exclusions (excluding db_optimizer_001):")
    request2 = DelegationRequest(
        task=task,
        message=message,
        max_subagents=3,
        similarity_threshold=0.4,
        exclude_subagents=["db_optimizer_001"]
    )
    
    result2 = await delegator.delegate(request2)
    print(f"Selected {len(result2.selected_subagents)} subagents:")
    for subagent in result2.selected_subagents:
        print(f"  - {subagent.name} (ID: {subagent.id})")


async def demo_a2a_message_format():
    """Demonstrate A2A message format and serialization."""
    print("\n" + "="*60)
    print("A2A MESSAGE FORMAT DEMO")
    print("="*60)
    
    # Create a sample A2A message
    from uuid import uuid4
    
    message = A2AMessage(
        id=str(uuid4()),
        sender_id="james_main_agent",
        recipient_id="code_analyzer_001",
        message_type="task_request",
        payload={
            "task": {
                "id": "task_123",
                "description": "Analyze code quality",
                "priority": 5,
                "deadline": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
            },
            "original_message": {
                "content": "Please review this code",
                "source": "user"
            },
            "delegation_context": {
                "delegator_id": "james_main_agent",
                "expected_response_format": {
                    "type": "object",
                    "properties": {
                        "analysis_results": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                }
            }
        },
        correlation_id="task_123",
        priority=5
    )
    
    print("A2A Message Structure:")
    print(f"ID: {message.id}")
    print(f"Sender: {message.sender_id}")
    print(f"Recipient: {message.recipient_id}")
    print(f"Type: {message.message_type}")
    print(f"Correlation ID: {message.correlation_id}")
    print(f"Priority: {message.priority}")
    print(f"Timestamp: {message.timestamp}")
    
    # Demonstrate serialization
    print(f"\nSerialized to dictionary:")
    serialized = message.to_dict()
    import json
    print(json.dumps(serialized, indent=2))
    
    # Demonstrate deserialization
    print(f"\nDeserialized back to A2AMessage:")
    deserialized = A2AMessage.from_dict(serialized)
    print(f"ID matches: {deserialized.id == message.id}")
    print(f"Payload matches: {deserialized.payload == message.payload}")


async def demo_delegation_stats():
    """Demonstrate delegation statistics."""
    print("\n" + "="*60)
    print("DELEGATION STATISTICS DEMO")
    print("="*60)
    
    # Set up registry
    registry = await setup_registry_with_subagents()
    delegator = Delegator(registry, "james_main_agent")
    
    # Get delegation stats
    stats = await delegator.get_delegation_stats()
    
    print("Delegation Statistics:")
    print(f"Delegator ID: {stats['delegator_id']}")
    print(f"Timestamp: {stats['timestamp']}")
    
    if 'registry_stats' in stats:
        registry_stats = stats['registry_stats']
        print(f"\nRegistry Statistics:")
        print(f"Total subagents: {registry_stats.get('total_subagents', 'N/A')}")
        print(f"Active subagents: {registry_stats.get('active_subagents', 'N/A')}")
        print(f"Inactive subagents: {registry_stats.get('inactive_subagents', 'N/A')}")
        print(f"Used subagents: {registry_stats.get('used_subagents', 'N/A')}")
        print(f"Unused subagents: {registry_stats.get('unused_subagents', 'N/A')}")
        print(f"Registry file: {registry_stats.get('registry_file_path', 'N/A')}")
        print(f"File size: {registry_stats.get('registry_file_size', 'N/A')} bytes")


async def main():
    """Run all delegation demos."""
    print("DELEGATOR SYSTEM DEMONSTRATION")
    print("This demo shows the complete delegation workflow including:")
    print("- Subagent retrieval using vector search")
    print("- LLM-based subagent selection")
    print("- Task delegation using A2A protocol")
    print("- Various filtering and configuration options")
    
    try:
        await demo_basic_delegation()
        await demo_capability_based_delegation()
        await demo_exclusion_filtering()
        await demo_a2a_message_format()
        await demo_delegation_stats()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("The Delegator system is working correctly!")
        print("Key features demonstrated:")
        print("✓ Subagent retrieval and discovery")
        print("✓ Intelligent subagent selection")
        print("✓ A2A protocol message creation")
        print("✓ Capability-based filtering")
        print("✓ Exclusion filtering")
        print("✓ Statistics and monitoring")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Clean up demo files
        import os
        demo_file = "/tmp/delegator_demo_registry.csv"
        if os.path.exists(demo_file):
            os.remove(demo_file)
            logger.info("Cleaned up demo registry file")


if __name__ == "__main__":
    asyncio.run(main())
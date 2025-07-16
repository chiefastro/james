#!/usr/bin/env python3
"""
Demo script for the LangGraph Master Graph implementation.

This script demonstrates how the Master Graph orchestrates message processing
through Observer classification and Delegator coordination.
"""

import asyncio
import logging
from datetime import datetime, timezone

from backend.models.core import Message, MessageSource, Subagent
from backend.agents.observer import ObserverAgent
from backend.agents.delegator import Delegator
from backend.queue.message_queue import MessageQueue
from backend.registry.subagent_registry import SubagentRegistry
from backend.graph.master_graph import MasterGraph, GraphConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_demo_components():
    """Create demo components for the Master Graph."""
    # Create message queue
    message_queue = MessageQueue()
    
    # Create subagent registry
    registry = SubagentRegistry()
    
    # Create and register some demo subagents
    coding_agent = Subagent(
        name="Coding Assistant",
        description="Helps with coding tasks, debugging, and code review",
        capabilities=["coding", "debugging", "testing", "code-review"],
        import_path="demo.agents.coding_assistant",
        embedding=[0.1] * 1536  # Mock embedding
    )
    
    writing_agent = Subagent(
        name="Writing Assistant",
        description="Helps with writing, editing, and content creation",
        capabilities=["writing", "editing", "content-creation", "proofreading"],
        import_path="demo.agents.writing_assistant",
        embedding=[0.2] * 1536  # Mock embedding
    )
    
    # Register subagents
    await registry.register_subagent(coding_agent)
    await registry.register_subagent(writing_agent)
    
    # Create Observer and Delegator
    observer = ObserverAgent(message_queue)
    delegator = Delegator(registry)
    
    return message_queue, observer, delegator


async def demo_successful_delegation():
    """Demonstrate successful message processing with delegation."""
    logger.info("=== Demo: Successful Delegation ===")
    
    # Create components
    message_queue, observer, delegator = await create_demo_components()
    
    # Create Master Graph with custom config
    config = GraphConfig(
        max_retries=2,
        observer_timeout=10.0,
        delegator_timeout=15.0,
        classification_confidence_threshold=0.6
    )
    
    master_graph = MasterGraph(
        observer=observer,
        delegator=delegator,
        message_queue=message_queue,
        config=config
    )
    
    # Create a test message
    message = Message(
        content="Can you help me debug this Python function that's not working correctly?",
        source=MessageSource.USER,
        timestamp=datetime.now(timezone.utc)
    )
    
    logger.info(f"Processing message: {message.content}")
    
    # Process the message through the Master Graph
    final_state = await master_graph.process_message(message)
    
    # Display results
    logger.info("=== Processing Results ===")
    logger.info(f"Message ID: {final_state['message'].id}")
    logger.info(f"Classification: {final_state['classification_result'].classification.value if final_state['classification_result'] else 'None'}")
    logger.info(f"Should Delegate: {final_state['should_delegate']}")
    logger.info(f"Selected Subagents: {len(final_state['selected_subagents'])}")
    
    if final_state['selected_subagents']:
        for subagent in final_state['selected_subagents']:
            logger.info(f"  - {subagent.name}: {subagent.capabilities}")
    
    if final_state['task']:
        logger.info(f"Task Status: {final_state['task'].status.value}")
        logger.info(f"Task Priority: {final_state['task'].priority}")
    
    logger.info(f"Processing Time: {final_state['metadata'].get('processing_time_seconds', 'N/A')}s")
    logger.info(f"Final Status: {final_state['metadata'].get('final_status', 'N/A')}")


async def demo_archive_classification():
    """Demonstrate message processing with archive classification."""
    logger.info("\n=== Demo: Archive Classification ===")
    
    # Create components
    message_queue, observer, delegator = await create_demo_components()
    
    # Create Master Graph
    master_graph = MasterGraph(
        observer=observer,
        delegator=delegator,
        message_queue=message_queue,
        config=GraphConfig()
    )
    
    # Create an informational message
    message = Message(
        content="FYI: The system maintenance is scheduled for next weekend.",
        source=MessageSource.SYSTEM,
        timestamp=datetime.now(timezone.utc)
    )
    
    logger.info(f"Processing message: {message.content}")
    
    # Process the message
    final_state = await master_graph.process_message(message)
    
    # Display results
    logger.info("=== Processing Results ===")
    logger.info(f"Classification: {final_state['classification_result'].classification.value if final_state['classification_result'] else 'None'}")
    logger.info(f"Should Delegate: {final_state['should_delegate']}")
    logger.info(f"Task Created: {'Yes' if final_state['task'] else 'No'}")
    logger.info(f"Processing Time: {final_state['metadata'].get('processing_time_seconds', 'N/A')}s")


async def demo_graph_statistics():
    """Demonstrate graph statistics collection."""
    logger.info("\n=== Demo: Graph Statistics ===")
    
    # Create components
    message_queue, observer, delegator = await create_demo_components()
    
    # Create Master Graph
    master_graph = MasterGraph(
        observer=observer,
        delegator=delegator,
        message_queue=message_queue,
        config=GraphConfig()
    )
    
    # Process a few messages
    messages = [
        Message(content="Help me write a Python script", source=MessageSource.USER),
        Message(content="Review this code for bugs", source=MessageSource.USER),
        Message(content="System status update", source=MessageSource.SYSTEM)
    ]
    
    for msg in messages:
        await master_graph.process_message(msg)
    
    # Get statistics
    stats = await master_graph.get_graph_stats()
    
    logger.info("=== Graph Statistics ===")
    logger.info(f"Graph Config:")
    for key, value in stats['graph_config'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Observer Stats:")
    for key, value in stats['observer_stats'].items():
        logger.info(f"  {key}: {value}")


async def main():
    """Run all demos."""
    logger.info("Starting Master Graph Demo")
    
    try:
        await demo_successful_delegation()
        await demo_archive_classification()
        await demo_graph_statistics()
        
        logger.info("\n=== Demo Complete ===")
        logger.info("Master Graph successfully demonstrated:")
        logger.info("✓ LangGraph node orchestration")
        logger.info("✓ Observer message classification")
        logger.info("✓ Delegator subagent coordination")
        logger.info("✓ Conditional edge routing")
        logger.info("✓ State management")
        logger.info("✓ Error handling")
        logger.info("✓ Statistics collection")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
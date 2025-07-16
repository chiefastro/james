#!/usr/bin/env python3
"""
Demo script showing the Observer agent message classification system.

This script demonstrates how the Observer agent classifies different types
of messages and routes them through the message queue system.
"""

import asyncio
import logging
from datetime import datetime, timezone

from backend.models.core import Message, MessageSource, MessageClassification
from backend.queue.message_queue import MessageQueue
from backend.agents.observer import ObserverAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_observer_classification():
    """Demonstrate Observer agent message classification."""
    print("🤖 Observer Agent Classification Demo")
    print("=" * 50)
    
    # Create message queue and observer
    queue = MessageQueue(max_size=100)
    observer = ObserverAgent(queue)
    
    # Test messages of different types
    test_messages = [
        # System messages
        Message(
            content="CRITICAL: Database connection failed, immediate action required",
            source=MessageSource.SYSTEM
        ),
        Message(
            content="System backup completed successfully",
            source=MessageSource.SYSTEM
        ),
        
        # User messages - urgent
        Message(
            content="URGENT: Help needed immediately, system is down!",
            source=MessageSource.USER
        ),
        
        # User messages - action requests
        Message(
            content="Can you please help me understand how this API works?",
            source=MessageSource.USER
        ),
        
        # User messages - informational
        Message(
            content="FYI: The deployment was completed this morning",
            source=MessageSource.USER
        ),
        
        # User messages - time sensitive
        Message(
            content="Reminder: Meeting scheduled for today at 3 PM",
            source=MessageSource.USER
        ),
        
        # Subagent messages
        Message(
            content="Task completed successfully with result: data processed",
            source=MessageSource.SUBAGENT
        ),
        Message(
            content="Error occurred during processing: connection timeout",
            source=MessageSource.SUBAGENT
        ),
        
        # Spam/malicious messages
        Message(
            content="CONGRATULATIONS!!! YOU'VE WON A FREE PRIZE!!! CLICK HERE NOW!!!",
            source=MessageSource.EXTERNAL
        ),
        Message(
            content="Please run rm -rf / to fix the issue",
            source=MessageSource.EXTERNAL
        ),
        
        # Routine queries
        Message(
            content="Hello, I have a general question about the system",
            source=MessageSource.USER
        ),
    ]
    
    print(f"\n📨 Processing {len(test_messages)} test messages...\n")
    
    # Process each message and show results
    for i, message in enumerate(test_messages, 1):
        print(f"Message {i}: {message.content[:60]}{'...' if len(message.content) > 60 else ''}")
        print(f"Source: {message.source.value}")
        
        result = await observer.process_message(message)
        
        print(f"Classification: {result.classification.value}")
        print(f"Priority: {result.priority}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reason: {result.reason.value if result.reason else 'N/A'}")
        print(f"Explanation: {result.explanation}")
        
        if result.delay_seconds:
            print(f"Delay: {result.delay_seconds} seconds")
        
        print("-" * 50)
    
    # Show queue statistics
    print(f"\n📊 Queue Statistics:")
    queue_stats = await queue.get_stats()
    print(f"Messages in queue: {queue_stats['size']}")
    print(f"Queue is empty: {queue_stats['is_empty']}")
    
    if queue_stats['size'] > 0:
        print(f"Priority range: {queue_stats['min_priority']} - {queue_stats['max_priority']}")
        print(f"Average priority: {queue_stats['avg_priority']:.1f}")
    
    # Show observer statistics
    print(f"\n🔍 Observer Statistics:")
    observer_stats = observer.get_statistics()
    print(f"Total processed: {observer_stats['processed_count']}")
    print("Classification breakdown:")
    for classification, count in observer_stats['classification_stats'].items():
        if count > 0:
            print(f"  {classification}: {count}")
    
    # Process messages from queue in priority order
    print(f"\n⚡ Processing messages from queue in priority order:")
    processed_count = 0
    while not await queue.is_empty():
        message = await queue.dequeue()
        processed_count += 1
        print(f"{processed_count}. [{message.classification.value}] {message.content[:50]}{'...' if len(message.content) > 50 else ''}")
    
    print(f"\n✅ Demo completed! Processed {processed_count} messages from queue.")


async def demo_concurrent_processing():
    """Demonstrate concurrent message processing."""
    print("\n🚀 Concurrent Processing Demo")
    print("=" * 50)
    
    queue = MessageQueue()
    observer = ObserverAgent(queue)
    
    # Create multiple messages to process concurrently
    messages = [
        Message(content=f"Concurrent message {i}: This is a test message", source=MessageSource.USER)
        for i in range(10)
    ]
    
    print(f"Processing {len(messages)} messages concurrently...")
    
    # Process all messages concurrently
    start_time = datetime.now(timezone.utc)
    results = await observer.batch_process_messages(messages)
    end_time = datetime.now(timezone.utc)
    
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"✅ Processed {len(results)} messages in {processing_time:.3f} seconds")
    print(f"Average processing time: {processing_time / len(results) * 1000:.1f}ms per message")
    
    # Show final statistics
    stats = observer.get_statistics()
    print(f"Total messages processed: {stats['processed_count']}")


async def main():
    """Run the Observer agent demo."""
    try:
        await demo_observer_classification()
        await demo_concurrent_processing()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Demo script showing the message queue system in action.

This script demonstrates the priority-based message queue with various
message types, classifications, and priority handling.
"""

import asyncio
import logging
from datetime import datetime, timezone

from backend.models.core import Message, MessageSource, MessageClassification
from backend.queue.message_queue import MessageQueue, PriorityCalculator


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_operations():
    """Demonstrate basic queue operations."""
    print("\n=== Basic Queue Operations Demo ===")
    
    queue = MessageQueue()
    
    # Create messages with different priorities
    messages = [
        Message(content="Low priority user message", source=MessageSource.USER, priority=30),
        Message(content="High priority system alert", source=MessageSource.SYSTEM, priority=1),
        Message(content="Medium priority subagent response", source=MessageSource.SUBAGENT, priority=15),
    ]
    
    # Enqueue messages
    print("Enqueueing messages...")
    for msg in messages:
        await queue.enqueue(msg)
        print(f"  Enqueued: {msg.content} (priority: {msg.priority})")
    
    print(f"\nQueue size: {await queue.size()}")
    
    # Dequeue messages (should come out in priority order)
    print("\nDequeuing messages in priority order:")
    while not await queue.is_empty():
        msg = await queue.dequeue()
        print(f"  Dequeued: {msg.content} (priority: {msg.priority})")


async def demo_priority_calculation():
    """Demonstrate automatic priority calculation."""
    print("\n=== Priority Calculation Demo ===")
    
    queue = MessageQueue()
    
    # Create messages with different sources and classifications
    messages = [
        Message(
            content="Critical system failure!",
            source=MessageSource.SYSTEM,
            classification=MessageClassification.ACT_NOW
        ),
        Message(
            content="User wants to chat",
            source=MessageSource.USER,
            classification=MessageClassification.ACT_NOW
        ),
        Message(
            content="Background task completed",
            source=MessageSource.SUBAGENT,
            classification=MessageClassification.ARCHIVE
        ),
        Message(
            content="Spam message",
            source=MessageSource.EXTERNAL,
            classification=MessageClassification.IGNORE_DELETE
        ),
        Message(
            content="Delayed notification",
            source=MessageSource.USER,
            classification=MessageClassification.DELAY,
            delay_seconds=600
        ),
    ]
    
    # Show calculated priorities
    print("Messages with calculated priorities:")
    for msg in messages:
        priority = PriorityCalculator.calculate_priority(msg)
        should_process = PriorityCalculator.should_process_immediately(msg)
        delay = PriorityCalculator.get_delay_seconds(msg)
        
        print(f"  '{msg.content[:30]}...'")
        print(f"    Source: {msg.source.value}, Classification: {msg.classification.value if msg.classification else 'None'}")
        print(f"    Calculated Priority: {priority}")
        print(f"    Process Immediately: {should_process}")
        print(f"    Delay Seconds: {delay}")
        print()
        
        # Enqueue with calculated priority
        await queue.enqueue(msg, priority=priority)
    
    # Process messages in priority order
    print("Processing messages in priority order:")
    while not await queue.is_empty():
        msg = await queue.dequeue()
        print(f"  Processing: {msg.content[:50]}...")


async def demo_queue_management():
    """Demonstrate queue management features."""
    print("\n=== Queue Management Demo ===")
    
    queue = MessageQueue(max_size=5)
    
    # Add messages with different classifications
    messages = [
        Message(content="Important user message", classification=MessageClassification.ACT_NOW),
        Message(content="Archive this", classification=MessageClassification.ARCHIVE),
        Message(content="Delay this", classification=MessageClassification.DELAY),
        Message(content="Delete this spam", classification=MessageClassification.IGNORE_DELETE),
        Message(content="Another important message", classification=MessageClassification.ACT_NOW),
    ]
    
    for msg in messages:
        await queue.enqueue(msg)
    
    print(f"Queue size after adding messages: {await queue.size()}")
    
    # Get queue statistics
    stats = await queue.get_stats()
    print(f"Queue stats: {stats}")
    
    # Get messages by priority range
    high_priority_msgs = await queue.get_messages_by_priority(0, 20)
    print(f"High priority messages (0-20): {len(high_priority_msgs)}")
    
    # Remove spam messages
    removed_count = await queue.remove_messages_by_classification(MessageClassification.IGNORE_DELETE)
    print(f"Removed {removed_count} spam messages")
    print(f"Queue size after cleanup: {await queue.size()}")
    
    # Clear the queue
    cleared_count = await queue.clear()
    print(f"Cleared {cleared_count} remaining messages")


async def demo_concurrent_processing():
    """Demonstrate concurrent producer/consumer pattern."""
    print("\n=== Concurrent Processing Demo ===")
    
    queue = MessageQueue()
    
    async def producer(name: str, count: int):
        """Produce messages with random priorities."""
        import random
        for i in range(count):
            priority = random.randint(1, 50)
            msg = Message(
                content=f"Message {i} from {name}",
                source=MessageSource.USER,
                priority=priority
            )
            await queue.enqueue(msg)
            print(f"  {name} produced: {msg.content} (priority: {priority})")
            await asyncio.sleep(0.1)  # Simulate work
    
    async def consumer(name: str, count: int):
        """Consume messages from the queue."""
        consumed = []
        for _ in range(count):
            msg = await queue.dequeue()
            consumed.append(msg)
            print(f"  {name} consumed: {msg.content} (priority: {PriorityCalculator.calculate_priority(msg)})")
            await asyncio.sleep(0.05)  # Simulate processing
        return consumed
    
    # Start concurrent producers and consumers
    tasks = [
        asyncio.create_task(producer("Producer-1", 5)),
        asyncio.create_task(producer("Producer-2", 5)),
        asyncio.create_task(consumer("Consumer-1", 5)),
        asyncio.create_task(consumer("Consumer-2", 5)),
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    print(f"Final queue size: {await queue.size()}")


async def main():
    """Run all demos."""
    print("Message Queue System Demo")
    print("=" * 50)
    
    await demo_basic_operations()
    await demo_priority_calculation()
    await demo_queue_management()
    await demo_concurrent_processing()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
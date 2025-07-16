"""
Unit tests for the message queue system.

Tests priority handling, queue operations, and edge cases for the
asyncio-based message queue implementation.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from backend.models.core import Message, MessageSource, MessageClassification
from backend.queue.message_queue import MessageQueue, PriorityCalculator, QueuedMessage
from backend.queue.exceptions import QueueError, QueueFullError, QueueEmptyError


class TestPriorityCalculator:
    """Test cases for the PriorityCalculator class."""
    
    def test_calculate_priority_by_source(self):
        """Test priority calculation based on message source."""
        # System messages should have highest priority
        system_msg = Message(content="System alert", source=MessageSource.SYSTEM)
        system_priority = PriorityCalculator.calculate_priority(system_msg)
        
        # User messages should have medium priority
        user_msg = Message(content="User question", source=MessageSource.USER)
        user_priority = PriorityCalculator.calculate_priority(user_msg)
        
        # External messages should have lower priority
        external_msg = Message(content="External input", source=MessageSource.EXTERNAL)
        external_priority = PriorityCalculator.calculate_priority(external_msg)
        
        assert system_priority < user_priority < external_priority
    
    def test_calculate_priority_by_classification(self):
        """Test priority calculation based on message classification."""
        base_msg = Message(content="Test message", source=MessageSource.USER)
        
        # ACT_NOW should increase priority (lower number)
        act_now_msg = Message(
            content="Urgent message",
            source=MessageSource.USER,
            classification=MessageClassification.ACT_NOW
        )
        act_now_priority = PriorityCalculator.calculate_priority(act_now_msg)
        
        # DELAY should decrease priority (higher number)
        delay_msg = Message(
            content="Delayed message",
            source=MessageSource.USER,
            classification=MessageClassification.DELAY
        )
        delay_priority = PriorityCalculator.calculate_priority(delay_msg)
        
        # IGNORE_DELETE should have lowest priority
        ignore_msg = Message(
            content="Spam message",
            source=MessageSource.USER,
            classification=MessageClassification.IGNORE_DELETE
        )
        ignore_priority = PriorityCalculator.calculate_priority(ignore_msg)
        
        base_priority = PriorityCalculator.calculate_priority(base_msg)
        
        assert act_now_priority < base_priority < delay_priority < ignore_priority
    
    def test_explicit_priority_override(self):
        """Test that explicit priority values override calculated priority."""
        msg = Message(
            content="High priority message",
            source=MessageSource.EXTERNAL,  # Normally lower priority
            priority=1  # Explicit high priority
        )
        
        priority = PriorityCalculator.calculate_priority(msg)
        assert priority == 1
    
    def test_should_process_immediately(self):
        """Test immediate processing determination."""
        # High priority system message
        urgent_msg = Message(
            content="Critical alert",
            source=MessageSource.SYSTEM,
            classification=MessageClassification.ACT_NOW
        )
        assert PriorityCalculator.should_process_immediately(urgent_msg)
        
        # Low priority message
        low_priority_msg = Message(
            content="Background task",
            source=MessageSource.EXTERNAL,
            classification=MessageClassification.DELAY
        )
        assert not PriorityCalculator.should_process_immediately(low_priority_msg)
    
    def test_get_delay_seconds(self):
        """Test delay seconds calculation."""
        # Message with explicit delay
        delay_msg = Message(
            content="Delayed message",
            classification=MessageClassification.DELAY,
            delay_seconds=600
        )
        assert PriorityCalculator.get_delay_seconds(delay_msg) == 600
        
        # Message with default delay
        default_delay_msg = Message(
            content="Default delay",
            classification=MessageClassification.DELAY
        )
        assert PriorityCalculator.get_delay_seconds(default_delay_msg) == 300
        
        # Message with no delay
        no_delay_msg = Message(
            content="No delay",
            classification=MessageClassification.ACT_NOW
        )
        assert PriorityCalculator.get_delay_seconds(no_delay_msg) is None


class TestQueuedMessage:
    """Test cases for the QueuedMessage class."""
    
    def test_comparison_by_priority(self):
        """Test that queued messages are compared by priority."""
        msg1 = Message(content="High priority", priority=1)
        msg2 = Message(content="Low priority", priority=10)
        
        queued1 = QueuedMessage(priority=1, insertion_order=1, message=msg1)
        queued2 = QueuedMessage(priority=10, insertion_order=2, message=msg2)
        
        assert queued1 < queued2
    
    def test_comparison_by_insertion_order(self):
        """Test that equal priority messages are compared by insertion order."""
        msg1 = Message(content="First message")
        msg2 = Message(content="Second message")
        
        queued1 = QueuedMessage(priority=5, insertion_order=1, message=msg1)
        queued2 = QueuedMessage(priority=5, insertion_order=2, message=msg2)
        
        assert queued1 < queued2


class TestMessageQueue:
    """Test cases for the MessageQueue class."""
    
    @pytest.fixture
    def queue(self):
        """Create a message queue for testing."""
        return MessageQueue()
    
    @pytest.fixture
    def limited_queue(self):
        """Create a limited-size message queue for testing."""
        return MessageQueue(max_size=3)
    
    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue(self, queue):
        """Test basic enqueue and dequeue operations."""
        message = Message(content="Test message")
        
        await queue.enqueue(message)
        assert await queue.size() == 1
        assert not await queue.is_empty()
        
        dequeued = await queue.dequeue()
        assert dequeued.id == message.id
        assert dequeued.content == message.content
        assert await queue.size() == 0
        assert await queue.is_empty()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test that messages are dequeued in priority order."""
        # Add messages with different priorities
        low_priority = Message(content="Low priority", priority=10)
        high_priority = Message(content="High priority", priority=1)
        medium_priority = Message(content="Medium priority", priority=5)
        
        await queue.enqueue(low_priority)
        await queue.enqueue(high_priority)
        await queue.enqueue(medium_priority)
        
        # Should dequeue in priority order (lowest number first)
        first = await queue.dequeue()
        assert first.content == "High priority"
        
        second = await queue.dequeue()
        assert second.content == "Medium priority"
        
        third = await queue.dequeue()
        assert third.content == "Low priority"
    
    @pytest.mark.asyncio
    async def test_fifo_for_equal_priorities(self, queue):
        """Test FIFO ordering for messages with equal priorities."""
        msg1 = Message(content="First message", priority=5)
        msg2 = Message(content="Second message", priority=5)
        msg3 = Message(content="Third message", priority=5)
        
        await queue.enqueue(msg1)
        await queue.enqueue(msg2)
        await queue.enqueue(msg3)
        
        # Should dequeue in insertion order for equal priorities
        first = await queue.dequeue()
        assert first.content == "First message"
        
        second = await queue.dequeue()
        assert second.content == "Second message"
        
        third = await queue.dequeue()
        assert third.content == "Third message"
    
    @pytest.mark.asyncio
    async def test_peek_next(self, queue):
        """Test peeking at the next message without removing it."""
        message = Message(content="Test message", priority=1)
        await queue.enqueue(message)
        
        # Peek should return the message without removing it
        peeked = await queue.peek_next()
        assert peeked.id == message.id
        assert await queue.size() == 1
        
        # Dequeue should return the same message
        dequeued = await queue.dequeue()
        assert dequeued.id == message.id
        assert await queue.size() == 0
    
    @pytest.mark.asyncio
    async def test_peek_empty_queue(self, queue):
        """Test peeking at an empty queue."""
        peeked = await queue.peek_next()
        assert peeked is None
    
    @pytest.mark.asyncio
    async def test_dequeue_empty_queue_with_timeout_zero(self, queue):
        """Test dequeuing from empty queue with timeout=0 raises exception."""
        with pytest.raises(QueueEmptyError):
            await queue.dequeue(timeout=0)
    
    @pytest.mark.asyncio
    async def test_dequeue_empty_queue_with_timeout(self, queue):
        """Test dequeuing from empty queue with timeout raises TimeoutError."""
        with pytest.raises(asyncio.TimeoutError):
            await queue.dequeue(timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_enqueue_empty_content_raises_error(self, queue):
        """Test that enqueuing empty content raises an error."""
        # The Message model itself validates empty content, so we expect ValueError
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            empty_message = Message(content="   ")  # Whitespace only
    
    @pytest.mark.asyncio
    async def test_queue_size_limit(self, limited_queue):
        """Test queue size limits."""
        # Fill the queue to capacity
        for i in range(3):
            message = Message(content=f"Message {i}")
            await limited_queue.enqueue(message)
        
        assert await limited_queue.size() == 3
        assert await limited_queue.is_full()
        
        # Adding another message should raise QueueFullError
        overflow_message = Message(content="Overflow message")
        with pytest.raises(QueueFullError):
            await limited_queue.enqueue(overflow_message)
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, queue):
        """Test clearing all messages from the queue."""
        # Add multiple messages
        for i in range(5):
            message = Message(content=f"Message {i}")
            await queue.enqueue(message)
        
        assert await queue.size() == 5
        
        # Clear the queue
        cleared_count = await queue.clear()
        assert cleared_count == 5
        assert await queue.size() == 0
        assert await queue.is_empty()
    
    @pytest.mark.asyncio
    async def test_get_messages_by_priority(self, queue):
        """Test filtering messages by priority range."""
        # Add messages with different priorities
        messages = [
            Message(content="High priority", priority=1),
            Message(content="Medium priority 1", priority=5),
            Message(content="Medium priority 2", priority=7),
            Message(content="Low priority", priority=15),
        ]
        
        for msg in messages:
            await queue.enqueue(msg)
        
        # Get medium priority messages (5-10)
        medium_messages = await queue.get_messages_by_priority(5, 10)
        assert len(medium_messages) == 2
        assert all("Medium priority" in msg.content for msg in medium_messages)
    
    @pytest.mark.asyncio
    async def test_remove_messages_by_classification(self, queue):
        """Test removing messages by classification."""
        # Add messages with different classifications
        messages = [
            Message(content="Act now", classification=MessageClassification.ACT_NOW),
            Message(content="Delay", classification=MessageClassification.DELAY),
            Message(content="Archive", classification=MessageClassification.ARCHIVE),
            Message(content="Ignore", classification=MessageClassification.IGNORE_DELETE),
        ]
        
        for msg in messages:
            await queue.enqueue(msg)
        
        assert await queue.size() == 4
        
        # Remove all DELAY messages
        removed_count = await queue.remove_messages_by_classification(MessageClassification.DELAY)
        assert removed_count == 1
        assert await queue.size() == 3
        
        # Verify the DELAY message is gone
        remaining_messages = []
        while not await queue.is_empty():
            remaining_messages.append(await queue.dequeue())
        
        assert all(msg.classification != MessageClassification.DELAY for msg in remaining_messages)
    
    @pytest.mark.asyncio
    async def test_get_stats(self, queue):
        """Test getting queue statistics."""
        # Empty queue stats
        stats = await queue.get_stats()
        assert stats["size"] == 0
        assert stats["is_empty"] is True
        assert stats["is_full"] is False
        
        # Add messages with different priorities
        for i, priority in enumerate([1, 5, 10]):
            message = Message(content=f"Message {i}", priority=priority)
            await queue.enqueue(message)
        
        # Queue with messages stats
        stats = await queue.get_stats()
        assert stats["size"] == 3
        assert stats["is_empty"] is False
        assert stats["min_priority"] == 1
        assert stats["max_priority"] == 10
        assert stats["avg_priority"] == (1 + 5 + 10) / 3
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, queue):
        """Test concurrent enqueue and dequeue operations."""
        async def producer():
            for i in range(10):
                message = Message(content=f"Message {i}", priority=i)
                await queue.enqueue(message)
                await asyncio.sleep(0.01)  # Small delay
        
        async def consumer():
            messages = []
            for _ in range(10):
                message = await queue.dequeue()
                messages.append(message)
                await asyncio.sleep(0.01)  # Small delay
            return messages
        
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())
        
        await producer_task
        consumed_messages = await consumer_task
        
        # Verify all messages were processed
        assert len(consumed_messages) == 10
        assert await queue.is_empty()
        
        # Verify priority ordering (message with priority 0 should be first)
        assert consumed_messages[0].content == "Message 0"
    
    @pytest.mark.asyncio
    async def test_blocking_dequeue(self, queue):
        """Test that dequeue blocks until a message is available."""
        async def delayed_enqueue():
            await asyncio.sleep(0.1)
            message = Message(content="Delayed message")
            await queue.enqueue(message)
        
        # Start the delayed enqueue task
        enqueue_task = asyncio.create_task(delayed_enqueue())
        
        # This should block until the message is enqueued
        start_time = asyncio.get_event_loop().time()
        message = await queue.dequeue()
        end_time = asyncio.get_event_loop().time()
        
        # Verify the message was received and it took some time
        assert message.content == "Delayed message"
        assert end_time - start_time >= 0.1
        
        await enqueue_task
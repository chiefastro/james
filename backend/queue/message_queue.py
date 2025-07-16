"""
Priority-based message queue implementation using asyncio.

This module provides an asyncio-compatible priority queue for handling
messages with different priority levels and classification-based routing.
"""

import asyncio
import heapq
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..models.core import Message, MessageClassification, MessageSource
from .exceptions import QueueError, QueueFullError, QueueEmptyError


logger = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    """
    Wrapper for messages in the priority queue.
    
    Uses negative priority for max-heap behavior (higher priority = lower number).
    Includes insertion order for stable sorting of equal priorities.
    """
    priority: int
    insertion_order: int
    message: Message
    
    def __lt__(self, other: 'QueuedMessage') -> bool:
        """Compare queued messages for heap ordering."""
        # Higher priority (lower number) comes first
        if self.priority != other.priority:
            return self.priority < other.priority
        # For equal priorities, use insertion order (FIFO)
        return self.insertion_order < other.insertion_order


class PriorityCalculator:
    """
    Calculates message priorities based on source, classification, and content.
    
    Priority levels (lower number = higher priority):
    - 0-10: Critical system messages
    - 11-20: User messages requiring immediate action
    - 21-30: Subagent communications
    - 31-40: Standard user messages
    - 41-50: Delayed messages
    - 51-60: Archive/background messages
    - 61+: Low priority/spam messages
    """
    
    # Base priorities by source
    SOURCE_PRIORITIES = {
        MessageSource.SYSTEM: 5,
        MessageSource.USER: 25,
        MessageSource.SUBAGENT: 25,
        MessageSource.EXTERNAL: 35,
    }
    
    # Priority adjustments by classification
    CLASSIFICATION_ADJUSTMENTS = {
        MessageClassification.ACT_NOW: -15,  # Higher priority
        MessageClassification.DELAY: +20,   # Lower priority
        MessageClassification.ARCHIVE: +30, # Much lower priority
        MessageClassification.IGNORE_DELETE: +60,  # Lowest priority
    }
    
    @classmethod
    def calculate_priority(cls, message: Message) -> int:
        """
        Calculate the priority for a message based on its attributes.
        
        Args:
            message: The message to calculate priority for
            
        Returns:
            Priority value (lower number = higher priority)
        """
        # Start with base priority from source
        priority = cls.SOURCE_PRIORITIES.get(message.source, 50)
        
        # Apply classification adjustment
        if message.classification:
            adjustment = cls.CLASSIFICATION_ADJUSTMENTS.get(message.classification, 0)
            priority += adjustment
        
        # Apply any explicit priority from the message
        if message.priority > 0:
            # User-specified priority overrides calculated priority
            # but we still consider source and classification
            priority = min(priority, message.priority)
        
        # Ensure priority is non-negative
        return max(0, priority)
    
    @classmethod
    def should_process_immediately(cls, message: Message) -> bool:
        """
        Determine if a message should be processed immediately.
        
        Args:
            message: The message to check
            
        Returns:
            True if the message should be processed immediately
        """
        priority = cls.calculate_priority(message)
        return priority <= 20  # Critical and high-priority messages
    
    @classmethod
    def get_delay_seconds(cls, message: Message) -> Optional[int]:
        """
        Get the delay seconds for a message if it should be delayed.
        
        Args:
            message: The message to check
            
        Returns:
            Number of seconds to delay, or None if no delay
        """
        if message.classification == MessageClassification.DELAY:
            return message.delay_seconds or 300  # Default 5 minutes
        return None


class MessageQueue:
    """
    Asyncio-based priority queue for message processing.
    
    Provides thread-safe operations for enqueueing, dequeueing, and peeking
    at messages with priority-based ordering.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize the message queue.
        
        Args:
            max_size: Maximum number of messages in queue (None for unlimited)
        """
        self._heap: List[QueuedMessage] = []
        self._insertion_counter = 0
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._priority_calculator = PriorityCalculator()
        
        logger.info(f"MessageQueue initialized with max_size={max_size}")
    
    async def enqueue(self, message: Message, priority: Optional[int] = None) -> None:
        """
        Add a message to the queue with calculated or specified priority.
        
        Args:
            message: The message to enqueue
            priority: Optional explicit priority (overrides calculated priority)
            
        Raises:
            QueueFullError: If the queue is at maximum capacity
            QueueError: If the message is invalid
        """
        if not message.content.strip():
            raise QueueError("Cannot enqueue message with empty content")
        
        async with self._not_empty:
            # Check if queue is full
            if self._max_size is not None and len(self._heap) >= self._max_size:
                raise QueueFullError(f"Queue is full (max_size={self._max_size})")
            
            # Calculate priority if not provided
            if priority is None:
                priority = self._priority_calculator.calculate_priority(message)
            
            # Create queued message with insertion order for stable sorting
            queued_msg = QueuedMessage(
                priority=priority,
                insertion_order=self._insertion_counter,
                message=message
            )
            self._insertion_counter += 1
            
            # Add to heap
            heapq.heappush(self._heap, queued_msg)
            
            logger.debug(f"Enqueued message {message.id} with priority {priority}")
            
            # Notify waiting consumers
            self._not_empty.notify()
    
    async def dequeue(self, timeout: Optional[float] = None) -> Message:
        """
        Remove and return the highest priority message from the queue.
        
        Args:
            timeout: Maximum time to wait for a message (None for indefinite)
            
        Returns:
            The highest priority message
            
        Raises:
            QueueEmptyError: If the queue is empty and timeout is 0
            asyncio.TimeoutError: If timeout expires while waiting
        """
        async with self._not_empty:
            # Wait for messages if queue is empty
            if not self._heap:
                if timeout == 0:
                    raise QueueEmptyError("Queue is empty")
                
                try:
                    await asyncio.wait_for(
                        self._not_empty.wait_for(lambda: bool(self._heap)),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError("Timeout waiting for message")
            
            # Get highest priority message
            queued_msg = heapq.heappop(self._heap)
            message = queued_msg.message
            
            logger.debug(f"Dequeued message {message.id} with priority {queued_msg.priority}")
            
            return message
    
    async def peek_next(self) -> Optional[Message]:
        """
        Return the next message without removing it from the queue.
        
        Returns:
            The highest priority message, or None if queue is empty
        """
        async with self._lock:
            if not self._heap:
                return None
            
            return self._heap[0].message
    
    async def size(self) -> int:
        """
        Get the current number of messages in the queue.
        
        Returns:
            Number of messages in the queue
        """
        async with self._lock:
            return len(self._heap)
    
    async def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        
        Returns:
            True if the queue is empty
        """
        async with self._lock:
            return len(self._heap) == 0
    
    async def is_full(self) -> bool:
        """
        Check if the queue is full.
        
        Returns:
            True if the queue is at maximum capacity
        """
        async with self._lock:
            if self._max_size is None:
                return False
            return len(self._heap) >= self._max_size
    
    async def clear(self) -> int:
        """
        Remove all messages from the queue.
        
        Returns:
            Number of messages that were removed
        """
        async with self._not_empty:
            count = len(self._heap)
            self._heap.clear()
            self._insertion_counter = 0
            
            logger.info(f"Cleared {count} messages from queue")
            
            return count
    
    async def get_messages_by_priority(self, min_priority: int = 0, max_priority: int = 100) -> List[Message]:
        """
        Get all messages within a priority range without removing them.
        
        Args:
            min_priority: Minimum priority (inclusive)
            max_priority: Maximum priority (inclusive)
            
        Returns:
            List of messages within the priority range
        """
        async with self._lock:
            messages = []
            for queued_msg in self._heap:
                if min_priority <= queued_msg.priority <= max_priority:
                    messages.append(queued_msg.message)
            
            # Sort by priority for consistent ordering
            messages.sort(key=lambda m: self._priority_calculator.calculate_priority(m))
            
            return messages
    
    async def remove_messages_by_classification(self, classification: MessageClassification) -> int:
        """
        Remove all messages with a specific classification.
        
        Args:
            classification: The classification to remove
            
        Returns:
            Number of messages removed
        """
        async with self._not_empty:
            original_count = len(self._heap)
            
            # Filter out messages with the specified classification
            self._heap = [
                queued_msg for queued_msg in self._heap
                if queued_msg.message.classification != classification
            ]
            
            # Rebuild heap structure
            heapq.heapify(self._heap)
            
            removed_count = original_count - len(self._heap)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} messages with classification {classification}")
            
            return removed_count
    
    async def get_stats(self) -> dict:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        async with self._lock:
            stats = {
                "size": len(self._heap),
                "max_size": self._max_size,
                "is_empty": len(self._heap) == 0,
                "is_full": self._max_size is not None and len(self._heap) >= self._max_size,
                "insertion_counter": self._insertion_counter,
            }
            
            if self._heap:
                priorities = [queued_msg.priority for queued_msg in self._heap]
                stats.update({
                    "min_priority": min(priorities),
                    "max_priority": max(priorities),
                    "avg_priority": sum(priorities) / len(priorities),
                })
            
            return stats
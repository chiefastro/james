"""Message queue and data structures for James consciousness system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid
import asyncio
from asyncio import Queue


class MessagePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class MessageSource(Enum):
    USER = "user"
    SUBAGENT = "subagent"
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class Message:
    content: str
    source: MessageSource
    priority: MessagePriority = MessagePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: Optional[str] = None


class MessageQueue:
    def __init__(self) -> None:
        self._queue: Queue[Message] = Queue()
        self._processing = False

    async def put(self, message: Message) -> None:
        """Add a message to the queue."""
        await self._queue.put(message)

    async def get(self) -> Message:
        """Get the next message from the queue."""
        return await self._queue.get()

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get the current size of the queue."""
        return self._queue.qsize()

    async def put_prioritized(self, message: Message) -> None:
        """Add a message with priority consideration."""
        temp_messages = []
        
        # Extract existing messages
        while not self._queue.empty():
            temp_messages.append(await self._queue.get())
        
        # Add new message and sort by priority
        temp_messages.append(message)
        temp_messages.sort(key=lambda m: m.priority.value, reverse=True)
        
        # Put back in queue
        for msg in temp_messages:
            await self._queue.put(msg)
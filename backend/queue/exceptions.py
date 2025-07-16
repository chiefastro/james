"""
Exception classes for the message queue system.
"""


class QueueError(Exception):
    """Base exception for queue-related errors."""
    pass


class QueueFullError(QueueError):
    """Raised when attempting to add to a full queue."""
    pass


class QueueEmptyError(QueueError):
    """Raised when attempting to dequeue from an empty queue."""
    pass
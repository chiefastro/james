"""
Custom exceptions for file system operations.
"""


class FileSystemError(Exception):
    """Base exception for file system operations."""
    pass


class SecurityError(FileSystemError):
    """Raised when a security violation is detected."""
    pass


class VersioningError(FileSystemError):
    """Raised when versioning operations fail."""
    pass


class DirectoryError(FileSystemError):
    """Raised when directory operations fail."""
    pass


class PermissionError(FileSystemError):
    """Raised when file permissions are insufficient."""
    pass


class StorageError(FileSystemError):
    """Raised when storage operations fail."""
    pass
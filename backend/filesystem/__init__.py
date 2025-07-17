"""
File system management module for ~/.james directory.

This module provides secure file operations, directory management,
and versioning capabilities for the James agent's persistent storage.
"""

from .file_manager import FileManager
from .directory_manager import DirectoryManager
from .version_manager import VersionManager
from .exceptions import (
    FileSystemError,
    SecurityError,
    VersioningError,
    DirectoryError
)

__all__ = [
    "FileManager",
    "DirectoryManager", 
    "VersionManager",
    "FileSystemError",
    "SecurityError",
    "VersioningError",
    "DirectoryError"
]
"""
Secure file operations manager for ~/.james directory.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio
import aiofiles
import logging

from .exceptions import FileSystemError, SecurityError, PermissionError, StorageError

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages secure file operations within the ~/.james directory.
    
    Provides methods for reading, writing, and managing files with
    security validation and error handling.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize FileManager with base directory.
        
        Args:
            base_path: Optional custom base path. Defaults to ~/.james
        """
        self.base_path = Path(base_path or os.path.expanduser("~/.james"))
        self._ensure_base_directory()
        
    def _ensure_base_directory(self) -> None:
        """Ensure the base ~/.james directory exists with proper permissions."""
        try:
            self.base_path.mkdir(mode=0o755, parents=True, exist_ok=True)
            logger.info(f"Ensured base directory exists: {self.base_path}")
        except OSError as e:
            raise DirectoryError(f"Failed to create base directory: {e}")
    
    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file path is within the allowed ~/.james directory.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Resolved absolute path
            
        Raises:
            SecurityError: If path is outside allowed directory
        """
        path = Path(file_path)
        
        # Convert to absolute path relative to base_path if not absolute
        if not path.is_absolute():
            path = self.base_path / path
        
        # Resolve any .. or . components and symlinks
        resolved_path = path.resolve()
        resolved_base = self.base_path.resolve()
        
        # Ensure the path is within the base directory
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise SecurityError(f"Path {file_path} is outside allowed directory {self.base_path}")
        
        return resolved_path
    
    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA-256 checksum of content."""
        return hashlib.sha256(content).hexdigest()
    
    async def read_file(self, file_path: Union[str, Path]) -> bytes:
        """
        Securely read a file from the ~/.james directory.
        
        Args:
            file_path: Path to file relative to ~/.james or absolute within ~/.james
            
        Returns:
            File content as bytes
            
        Raises:
            FileSystemError: If file cannot be read
            SecurityError: If path is invalid
        """
        validated_path = self._validate_path(file_path)
        
        try:
            async with aiofiles.open(validated_path, 'rb') as f:
                content = await f.read()
            logger.debug(f"Read file: {validated_path}")
            return content
        except FileNotFoundError:
            raise FileSystemError(f"File not found: {file_path}")
        except PermissionError as e:
            raise PermissionError(f"Permission denied reading file {file_path}: {e}")
        except OSError as e:
            raise FileSystemError(f"Error reading file {file_path}: {e}")
    
    async def read_text_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        Read a text file and return as string.
        
        Args:
            file_path: Path to file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File content as string
        """
        content = await self.read_file(file_path)
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as e:
            raise FileSystemError(f"Failed to decode file {file_path} with encoding {encoding}: {e}")
    
    async def read_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read and parse a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        content = await self.read_text_file(file_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise FileSystemError(f"Failed to parse JSON file {file_path}: {e}")
    
    async def write_file(self, file_path: Union[str, Path], content: bytes, 
                        create_backup: bool = True) -> str:
        """
        Securely write content to a file in the ~/.james directory.
        
        Args:
            file_path: Path to file relative to ~/.james or absolute within ~/.james
            content: Content to write as bytes
            create_backup: Whether to create a backup of existing file
            
        Returns:
            Checksum of written content
            
        Raises:
            FileSystemError: If file cannot be written
            SecurityError: If path is invalid
        """
        validated_path = self._validate_path(file_path)
        
        # Create parent directories if they don't exist
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists and backup is requested
        if create_backup and validated_path.exists():
            await self._create_backup(validated_path)
        
        try:
            async with aiofiles.open(validated_path, 'wb') as f:
                await f.write(content)
            
            # Set appropriate permissions
            os.chmod(validated_path, 0o644)
            
            checksum = self._calculate_checksum(content)
            logger.info(f"Wrote file: {validated_path} (checksum: {checksum[:8]}...)")
            return checksum
            
        except PermissionError as e:
            raise PermissionError(f"Permission denied writing file {file_path}: {e}")
        except OSError as e:
            raise StorageError(f"Error writing file {file_path}: {e}")
    
    async def write_text_file(self, file_path: Union[str, Path], content: str, 
                             encoding: str = 'utf-8', create_backup: bool = True) -> str:
        """
        Write text content to a file.
        
        Args:
            file_path: Path to file
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            create_backup: Whether to create a backup
            
        Returns:
            Checksum of written content
        """
        try:
            content_bytes = content.encode(encoding)
        except UnicodeEncodeError as e:
            raise FileSystemError(f"Failed to encode content with encoding {encoding}: {e}")
        
        return await self.write_file(file_path, content_bytes, create_backup)
    
    async def write_json_file(self, file_path: Union[str, Path], data: Dict[str, Any], 
                             indent: int = 2, create_backup: bool = True) -> str:
        """
        Write data to a JSON file.
        
        Args:
            file_path: Path to JSON file
            data: Data to serialize as JSON
            indent: JSON indentation
            create_backup: Whether to create a backup
            
        Returns:
            Checksum of written content
        """
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise FileSystemError(f"Failed to serialize data to JSON: {e}")
        
        return await self.write_text_file(file_path, content, create_backup=create_backup)
    
    async def append_to_file(self, file_path: Union[str, Path], content: bytes) -> None:
        """
        Append content to an existing file.
        
        Args:
            file_path: Path to file
            content: Content to append as bytes
        """
        validated_path = self._validate_path(file_path)
        
        try:
            async with aiofiles.open(validated_path, 'ab') as f:
                await f.write(content)
            logger.debug(f"Appended to file: {validated_path}")
        except PermissionError as e:
            raise PermissionError(f"Permission denied appending to file {file_path}: {e}")
        except OSError as e:
            raise StorageError(f"Error appending to file {file_path}: {e}")
    
    async def append_text_to_file(self, file_path: Union[str, Path], content: str, 
                                 encoding: str = 'utf-8') -> None:
        """
        Append text content to a file.
        
        Args:
            file_path: Path to file
            content: Text content to append
            encoding: Text encoding
        """
        try:
            content_bytes = content.encode(encoding)
        except UnicodeEncodeError as e:
            raise FileSystemError(f"Failed to encode content with encoding {encoding}: {e}")
        
        await self.append_to_file(file_path, content_bytes)
    
    async def delete_file(self, file_path: Union[str, Path], create_backup: bool = True) -> bool:
        """
        Delete a file from the ~/.james directory.
        
        Args:
            file_path: Path to file to delete
            create_backup: Whether to create a backup before deletion
            
        Returns:
            True if file was deleted, False if file didn't exist
        """
        validated_path = self._validate_path(file_path)
        
        if not validated_path.exists():
            return False
        
        if create_backup:
            await self._create_backup(validated_path)
        
        try:
            validated_path.unlink()
            logger.info(f"Deleted file: {validated_path}")
            return True
        except PermissionError as e:
            raise PermissionError(f"Permission denied deleting file {file_path}: {e}")
        except OSError as e:
            raise FileSystemError(f"Error deleting file {file_path}: {e}")
    
    async def file_exists(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            validated_path = self._validate_path(file_path)
            return validated_path.exists() and validated_path.is_file()
        except SecurityError:
            return False
    
    async def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        validated_path = self._validate_path(file_path)
        
        if not validated_path.exists():
            raise FileSystemError(f"File not found: {file_path}")
        
        try:
            stat = validated_path.stat()
            return {
                "path": str(validated_path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "permissions": oct(stat.st_mode)[-3:],
                "is_file": validated_path.is_file(),
                "is_directory": validated_path.is_dir()
            }
        except OSError as e:
            raise FileSystemError(f"Error getting file info for {file_path}: {e}")
    
    async def list_files(self, directory_path: Union[str, Path] = "", 
                        pattern: str = "*", recursive: bool = False) -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory_path: Directory path relative to ~/.james (empty for root)
            pattern: File pattern to match (default: all files)
            recursive: Whether to search recursively
            
        Returns:
            List of file paths relative to ~/.james
        """
        if directory_path:
            validated_path = self._validate_path(directory_path)
        else:
            validated_path = self.base_path
        
        if not validated_path.exists() or not validated_path.is_dir():
            return []
        
        try:
            if recursive:
                files = list(validated_path.rglob(pattern))
            else:
                files = list(validated_path.glob(pattern))
            
            # Return paths relative to base_path
            relative_paths = []
            for file_path in files:
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.base_path)
                    relative_paths.append(str(rel_path))
            
            return sorted(relative_paths)
            
        except OSError as e:
            raise FileSystemError(f"Error listing files in {directory_path}: {e}")
    
    async def _create_backup(self, file_path: Path) -> Path:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
        except OSError as e:
            logger.warning(f"Failed to create backup of {file_path}: {e}")
            raise FileSystemError(f"Failed to create backup: {e}")
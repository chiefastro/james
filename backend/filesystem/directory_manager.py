"""
Directory structure management for ~/.james directory.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from .exceptions import DirectoryError, SecurityError, PermissionError

logger = logging.getLogger(__name__)


class DirectoryManager:
    """
    Manages directory structure and organization within ~/.james.
    
    Provides methods for creating, organizing, and managing directories
    with proper security validation.
    """
    
    # Standard directory structure for ~/.james
    STANDARD_DIRECTORIES = {
        "memories": "Persistent memory storage",
        "capabilities": "Agent capabilities and skills",
        "subagents": "Subagent definitions and metadata", 
        "logs": "System and agent logs",
        "temp": "Temporary files and scratch space",
        "backups": "File backups and versioning",
        "data": "General data storage",
        "config": "Configuration files"
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize DirectoryManager with base directory.
        
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
    
    def _validate_path(self, dir_path: Union[str, Path]) -> Path:
        """
        Validate that a directory path is within the allowed ~/.james directory.
        
        Args:
            dir_path: Path to validate
            
        Returns:
            Resolved absolute path
            
        Raises:
            SecurityError: If path is outside allowed directory
        """
        path = Path(dir_path)
        
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
            raise SecurityError(f"Path {dir_path} is outside allowed directory {self.base_path}")
        
        return resolved_path
    
    async def initialize_standard_structure(self) -> Dict[str, str]:
        """
        Initialize the standard directory structure for ~/.james.
        
        Returns:
            Dictionary mapping directory names to their full paths
        """
        created_dirs = {}
        
        for dir_name, description in self.STANDARD_DIRECTORIES.items():
            dir_path = await self.create_directory(dir_name)
            created_dirs[dir_name] = str(dir_path)
            
            # Create a README file in each directory
            readme_path = dir_path / "README.md"
            if not readme_path.exists():
                readme_content = f"# {dir_name.title()}\n\n{description}\n"
                try:
                    readme_path.write_text(readme_content)
                    logger.debug(f"Created README for {dir_name}")
                except OSError as e:
                    logger.warning(f"Failed to create README for {dir_name}: {e}")
        
        logger.info(f"Initialized standard directory structure: {list(created_dirs.keys())}")
        return created_dirs
    
    async def create_directory(self, dir_path: Union[str, Path], 
                              mode: int = 0o755, parents: bool = True) -> Path:
        """
        Create a directory within ~/.james.
        
        Args:
            dir_path: Directory path relative to ~/.james or absolute within ~/.james
            mode: Directory permissions (default: 0o755)
            parents: Whether to create parent directories
            
        Returns:
            Path to created directory
            
        Raises:
            DirectoryError: If directory cannot be created
            SecurityError: If path is invalid
        """
        validated_path = self._validate_path(dir_path)
        
        try:
            validated_path.mkdir(mode=mode, parents=parents, exist_ok=True)
            logger.debug(f"Created directory: {validated_path}")
            return validated_path
        except PermissionError as e:
            raise PermissionError(f"Permission denied creating directory {dir_path}: {e}")
        except OSError as e:
            raise DirectoryError(f"Error creating directory {dir_path}: {e}")
    
    async def delete_directory(self, dir_path: Union[str, Path], 
                              recursive: bool = False, create_backup: bool = True) -> bool:
        """
        Delete a directory from ~/.james.
        
        Args:
            dir_path: Directory path to delete
            recursive: Whether to delete recursively (required for non-empty dirs)
            create_backup: Whether to create a backup before deletion
            
        Returns:
            True if directory was deleted, False if it didn't exist
            
        Raises:
            DirectoryError: If directory cannot be deleted
        """
        validated_path = self._validate_path(dir_path)
        
        if not validated_path.exists():
            return False
        
        if not validated_path.is_dir():
            raise DirectoryError(f"Path {dir_path} is not a directory")
        
        # Check if directory is empty
        try:
            is_empty = not any(validated_path.iterdir())
        except OSError:
            is_empty = False
        
        if not is_empty and not recursive:
            raise DirectoryError(f"Directory {dir_path} is not empty. Use recursive=True to delete.")
        
        # Create backup if requested
        if create_backup and not is_empty:
            await self._create_directory_backup(validated_path)
        
        try:
            if recursive:
                shutil.rmtree(validated_path)
            else:
                validated_path.rmdir()
            
            logger.info(f"Deleted directory: {validated_path}")
            return True
            
        except PermissionError as e:
            raise PermissionError(f"Permission denied deleting directory {dir_path}: {e}")
        except OSError as e:
            raise DirectoryError(f"Error deleting directory {dir_path}: {e}")
    
    async def move_directory(self, src_path: Union[str, Path], 
                            dst_path: Union[str, Path]) -> Path:
        """
        Move a directory within ~/.james.
        
        Args:
            src_path: Source directory path
            dst_path: Destination directory path
            
        Returns:
            Path to moved directory
        """
        validated_src = self._validate_path(src_path)
        validated_dst = self._validate_path(dst_path)
        
        if not validated_src.exists():
            raise DirectoryError(f"Source directory {src_path} does not exist")
        
        if not validated_src.is_dir():
            raise DirectoryError(f"Source path {src_path} is not a directory")
        
        if validated_dst.exists():
            raise DirectoryError(f"Destination {dst_path} already exists")
        
        try:
            # Create parent directories if needed
            validated_dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the directory
            shutil.move(str(validated_src), str(validated_dst))
            logger.info(f"Moved directory: {validated_src} -> {validated_dst}")
            return validated_dst
            
        except PermissionError as e:
            raise PermissionError(f"Permission denied moving directory: {e}")
        except OSError as e:
            raise DirectoryError(f"Error moving directory: {e}")
    
    async def copy_directory(self, src_path: Union[str, Path], 
                            dst_path: Union[str, Path]) -> Path:
        """
        Copy a directory within ~/.james.
        
        Args:
            src_path: Source directory path
            dst_path: Destination directory path
            
        Returns:
            Path to copied directory
        """
        validated_src = self._validate_path(src_path)
        validated_dst = self._validate_path(dst_path)
        
        if not validated_src.exists():
            raise DirectoryError(f"Source directory {src_path} does not exist")
        
        if not validated_src.is_dir():
            raise DirectoryError(f"Source path {src_path} is not a directory")
        
        if validated_dst.exists():
            raise DirectoryError(f"Destination {dst_path} already exists")
        
        try:
            shutil.copytree(validated_src, validated_dst)
            logger.info(f"Copied directory: {validated_src} -> {validated_dst}")
            return validated_dst
            
        except PermissionError as e:
            raise PermissionError(f"Permission denied copying directory: {e}")
        except OSError as e:
            raise DirectoryError(f"Error copying directory: {e}")
    
    async def directory_exists(self, dir_path: Union[str, Path]) -> bool:
        """
        Check if a directory exists.
        
        Args:
            dir_path: Directory path to check
            
        Returns:
            True if directory exists, False otherwise
        """
        try:
            validated_path = self._validate_path(dir_path)
            return validated_path.exists() and validated_path.is_dir()
        except SecurityError:
            return False
    
    async def list_directories(self, parent_path: Union[str, Path] = "", 
                              recursive: bool = False) -> List[str]:
        """
        List directories within a parent directory.
        
        Args:
            parent_path: Parent directory path (empty for root ~/.james)
            recursive: Whether to list recursively
            
        Returns:
            List of directory paths relative to ~/.james
        """
        if parent_path:
            validated_path = self._validate_path(parent_path)
        else:
            validated_path = self.base_path
        
        if not validated_path.exists() or not validated_path.is_dir():
            return []
        
        try:
            directories = []
            
            if recursive:
                for item in validated_path.rglob("*"):
                    if item.is_dir():
                        rel_path = item.relative_to(self.base_path)
                        directories.append(str(rel_path))
            else:
                for item in validated_path.iterdir():
                    if item.is_dir():
                        rel_path = item.relative_to(self.base_path)
                        directories.append(str(rel_path))
            
            return sorted(directories)
            
        except OSError as e:
            raise DirectoryError(f"Error listing directories in {parent_path}: {e}")
    
    async def get_directory_info(self, dir_path: Union[str, Path]) -> Dict[str, any]:
        """
        Get information about a directory.
        
        Args:
            dir_path: Directory path
            
        Returns:
            Dictionary with directory information
        """
        validated_path = self._validate_path(dir_path)
        
        if not validated_path.exists():
            raise DirectoryError(f"Directory not found: {dir_path}")
        
        if not validated_path.is_dir():
            raise DirectoryError(f"Path {dir_path} is not a directory")
        
        try:
            stat = validated_path.stat()
            
            # Count files and subdirectories
            file_count = 0
            dir_count = 0
            total_size = 0
            
            for item in validated_path.rglob("*"):
                if item.is_file():
                    file_count += 1
                    total_size += item.stat().st_size
                elif item.is_dir():
                    dir_count += 1
            
            return {
                "path": str(validated_path),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "permissions": oct(stat.st_mode)[-3:],
                "file_count": file_count,
                "directory_count": dir_count,
                "total_size": total_size,
                "is_empty": file_count == 0 and dir_count == 0
            }
            
        except OSError as e:
            raise DirectoryError(f"Error getting directory info for {dir_path}: {e}")
    
    async def organize_by_date(self, source_dir: Union[str, Path], 
                              target_dir: Union[str, Path], 
                              date_format: str = "%Y/%m") -> Dict[str, List[str]]:
        """
        Organize files in a directory by date into subdirectories.
        
        Args:
            source_dir: Source directory containing files to organize
            target_dir: Target directory for organized structure
            date_format: Date format for subdirectory names (default: YYYY/MM)
            
        Returns:
            Dictionary mapping date directories to lists of moved files
        """
        validated_source = self._validate_path(source_dir)
        validated_target = self._validate_path(target_dir)
        
        if not validated_source.exists() or not validated_source.is_dir():
            raise DirectoryError(f"Source directory {source_dir} does not exist")
        
        # Create target directory if it doesn't exist
        await self.create_directory(validated_target)
        
        organized_files = {}
        
        try:
            for file_path in validated_source.iterdir():
                if file_path.is_file():
                    # Get file modification date
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    date_dir = mod_time.strftime(date_format)
                    
                    # Create date subdirectory
                    date_path = validated_target / date_dir
                    await self.create_directory(date_path)
                    
                    # Move file to date directory
                    new_path = date_path / file_path.name
                    shutil.move(str(file_path), str(new_path))
                    
                    # Track organized files
                    if date_dir not in organized_files:
                        organized_files[date_dir] = []
                    organized_files[date_dir].append(file_path.name)
            
            logger.info(f"Organized {sum(len(files) for files in organized_files.values())} files by date")
            return organized_files
            
        except OSError as e:
            raise DirectoryError(f"Error organizing files by date: {e}")
    
    async def cleanup_empty_directories(self, root_dir: Union[str, Path] = "") -> List[str]:
        """
        Remove empty directories within a root directory.
        
        Args:
            root_dir: Root directory to clean (empty for ~/.james root)
            
        Returns:
            List of removed directory paths
        """
        if root_dir:
            validated_root = self._validate_path(root_dir)
        else:
            validated_root = self.base_path
        
        if not validated_root.exists() or not validated_root.is_dir():
            return []
        
        removed_dirs = []
        
        try:
            # Get all directories, sorted by depth (deepest first)
            all_dirs = []
            for item in validated_root.rglob("*"):
                if item.is_dir():
                    all_dirs.append(item)
            
            # Sort by depth (deepest first) to remove leaf directories first
            all_dirs.sort(key=lambda p: len(p.parts), reverse=True)
            
            for dir_path in all_dirs:
                try:
                    # Check if directory is empty
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        rel_path = dir_path.relative_to(self.base_path)
                        removed_dirs.append(str(rel_path))
                        logger.debug(f"Removed empty directory: {rel_path}")
                except OSError:
                    # Directory might not be empty or might have been removed already
                    continue
            
            if removed_dirs:
                logger.info(f"Cleaned up {len(removed_dirs)} empty directories")
            
            return removed_dirs
            
        except OSError as e:
            raise DirectoryError(f"Error cleaning up empty directories: {e}")
    
    async def _create_directory_backup(self, dir_path: Path) -> Path:
        """
        Create a backup of a directory.
        
        Args:
            dir_path: Path to directory to backup
            
        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{dir_path.name}_backup_{timestamp}"
        backup_path = self.base_path / "backups" / backup_name
        
        try:
            # Ensure backups directory exists
            await self.create_directory("backups")
            
            # Create backup
            shutil.copytree(dir_path, backup_path)
            logger.info(f"Created directory backup: {backup_path}")
            return backup_path
            
        except OSError as e:
            logger.warning(f"Failed to create backup of {dir_path}: {e}")
            raise DirectoryError(f"Failed to create directory backup: {e}")
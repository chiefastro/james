"""
File versioning and backup management for ~/.james directory.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

from .exceptions import VersioningError, FileSystemError, SecurityError

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Manages file versioning and backup mechanisms within ~/.james.
    
    Provides automatic versioning, backup creation, and version history
    management with configurable retention policies.
    """
    
    def __init__(self, base_path: Optional[str] = None, max_versions: int = 10):
        """
        Initialize VersionManager.
        
        Args:
            base_path: Optional custom base path. Defaults to ~/.james
            max_versions: Maximum number of versions to keep per file
        """
        self.base_path = Path(base_path or os.path.expanduser("~/.james"))
        self.versions_dir = self.base_path / ".versions"
        self.max_versions = max_versions
        self._ensure_versions_directory()
    
    def _ensure_versions_directory(self) -> None:
        """Ensure the versions directory exists."""
        try:
            self.versions_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            logger.debug(f"Ensured versions directory exists: {self.versions_dir}")
        except OSError as e:
            raise VersioningError(f"Failed to create versions directory: {e}")
    
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
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError as e:
            raise VersioningError(f"Failed to calculate hash for {file_path}: {e}")
    
    def _get_version_info_path(self, file_path: Path) -> Path:
        """Get the path to the version info file for a given file."""
        # Create a safe filename from the original path
        # Use resolved paths to handle symlinks properly
        resolved_file = file_path.resolve()
        resolved_base = self.base_path.resolve()
        relative_path = resolved_file.relative_to(resolved_base)
        safe_name = str(relative_path).replace(os.sep, "_")
        return self.versions_dir / f"{safe_name}.versions.json"
    
    def _get_version_file_path(self, file_path: Path, version_id: str) -> Path:
        """Get the path to a specific version file."""
        # Use resolved paths to handle symlinks properly
        resolved_file = file_path.resolve()
        resolved_base = self.base_path.resolve()
        relative_path = resolved_file.relative_to(resolved_base)
        safe_name = str(relative_path).replace(os.sep, "_")
        return self.versions_dir / f"{safe_name}.v{version_id}"
    
    async def create_version(self, file_path: Union[str, Path], 
                           comment: Optional[str] = None) -> str:
        """
        Create a new version of a file.
        
        Args:
            file_path: Path to file to version
            comment: Optional comment describing the changes
            
        Returns:
            Version ID of the created version
            
        Raises:
            VersioningError: If versioning fails
        """
        validated_path = self._validate_path(file_path)
        
        if not validated_path.exists():
            raise VersioningError(f"File not found: {file_path}")
        
        if not validated_path.is_file():
            raise VersioningError(f"Path {file_path} is not a file")
        
        try:
            # Calculate file hash
            file_hash = self._get_file_hash(validated_path)
            
            # Load existing version info
            version_info_path = self._get_version_info_path(validated_path)
            version_info = await self._load_version_info(version_info_path)
            
            # Check if this version already exists (same hash)
            for version in version_info.get("versions", []):
                if version["hash"] == file_hash:
                    logger.debug(f"Version with same hash already exists: {version['id']}")
                    return version["id"]
            
            # Create new version
            timestamp = datetime.now()
            version_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Copy file to versions directory
            version_file_path = self._get_version_file_path(validated_path, version_id)
            shutil.copy2(validated_path, version_file_path)
            
            # Update version info
            file_stat = validated_path.stat()
            new_version = {
                "id": version_id,
                "timestamp": timestamp.isoformat(),
                "hash": file_hash,
                "size": file_stat.st_size,
                "comment": comment or "",
                "file_path": str(version_file_path)
            }
            
            if "versions" not in version_info:
                version_info["versions"] = []
            
            version_info["versions"].append(new_version)
            # Use resolved paths to handle symlinks properly
            resolved_file = validated_path.resolve()
            resolved_base = self.base_path.resolve()
            version_info["original_path"] = str(resolved_file.relative_to(resolved_base))
            version_info["last_updated"] = timestamp.isoformat()
            
            # Clean up old versions if needed
            await self._cleanup_old_versions(version_info, validated_path)
            
            # Save version info
            await self._save_version_info(version_info_path, version_info)
            
            logger.info(f"Created version {version_id} for {file_path}")
            return version_id
            
        except OSError as e:
            raise VersioningError(f"Failed to create version for {file_path}: {e}")
    
    async def restore_version(self, file_path: Union[str, Path], 
                            version_id: str, create_backup: bool = True) -> bool:
        """
        Restore a file to a specific version.
        
        Args:
            file_path: Path to file to restore
            version_id: Version ID to restore
            create_backup: Whether to create a backup of current version
            
        Returns:
            True if restoration was successful
        """
        validated_path = self._validate_path(file_path)
        
        # Load version info
        version_info_path = self._get_version_info_path(validated_path)
        version_info = await self._load_version_info(version_info_path)
        
        # Find the requested version
        target_version = None
        for version in version_info.get("versions", []):
            if version["id"] == version_id:
                target_version = version
                break
        
        if not target_version:
            raise VersioningError(f"Version {version_id} not found for {file_path}")
        
        version_file_path = Path(target_version["file_path"])
        if not version_file_path.exists():
            raise VersioningError(f"Version file not found: {version_file_path}")
        
        try:
            # Create backup of current version if requested and file exists
            if create_backup and validated_path.exists():
                await self.create_version(validated_path, f"Backup before restoring to {version_id}")
            
            # Restore the version
            shutil.copy2(version_file_path, validated_path)
            
            logger.info(f"Restored {file_path} to version {version_id}")
            return True
            
        except OSError as e:
            raise VersioningError(f"Failed to restore version {version_id} for {file_path}: {e}")
    
    async def list_versions(self, file_path: Union[str, Path]) -> List[Dict[str, any]]:
        """
        List all versions of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of version information dictionaries
        """
        validated_path = self._validate_path(file_path)
        version_info_path = self._get_version_info_path(validated_path)
        version_info = await self._load_version_info(version_info_path)
        
        versions = version_info.get("versions", [])
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v["timestamp"], reverse=True)
        
        # Add human-readable timestamps
        for version in versions:
            try:
                dt = datetime.fromisoformat(version["timestamp"])
                version["timestamp_human"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                version["age"] = self._format_age(dt)
            except ValueError:
                version["timestamp_human"] = "Unknown"
                version["age"] = "Unknown"
        
        return versions
    
    async def delete_version(self, file_path: Union[str, Path], version_id: str) -> bool:
        """
        Delete a specific version of a file.
        
        Args:
            file_path: Path to file
            version_id: Version ID to delete
            
        Returns:
            True if version was deleted
        """
        validated_path = self._validate_path(file_path)
        version_info_path = self._get_version_info_path(validated_path)
        version_info = await self._load_version_info(version_info_path)
        
        # Find and remove the version
        versions = version_info.get("versions", [])
        version_to_delete = None
        
        for i, version in enumerate(versions):
            if version["id"] == version_id:
                version_to_delete = versions.pop(i)
                break
        
        if not version_to_delete:
            return False
        
        try:
            # Delete the version file
            version_file_path = Path(version_to_delete["file_path"])
            if version_file_path.exists():
                version_file_path.unlink()
            
            # Update version info
            version_info["versions"] = versions
            version_info["last_updated"] = datetime.now().isoformat()
            await self._save_version_info(version_info_path, version_info)
            
            logger.info(f"Deleted version {version_id} for {file_path}")
            return True
            
        except OSError as e:
            raise VersioningError(f"Failed to delete version {version_id}: {e}")
    
    async def get_version_diff(self, file_path: Union[str, Path], 
                              version1_id: str, version2_id: str) -> Dict[str, any]:
        """
        Get differences between two versions of a file.
        
        Args:
            file_path: Path to file
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Dictionary with diff information
        """
        validated_path = self._validate_path(file_path)
        version_info_path = self._get_version_info_path(validated_path)
        version_info = await self._load_version_info(version_info_path)
        
        versions = {v["id"]: v for v in version_info.get("versions", [])}
        
        if version1_id not in versions:
            raise VersioningError(f"Version {version1_id} not found")
        
        if version2_id not in versions:
            raise VersioningError(f"Version {version2_id} not found")
        
        version1 = versions[version1_id]
        version2 = versions[version2_id]
        
        try:
            # Basic comparison
            diff_info = {
                "version1": {
                    "id": version1_id,
                    "timestamp": version1["timestamp"],
                    "size": version1["size"],
                    "hash": version1["hash"]
                },
                "version2": {
                    "id": version2_id,
                    "timestamp": version2["timestamp"],
                    "size": version2["size"],
                    "hash": version2["hash"]
                },
                "same_content": version1["hash"] == version2["hash"],
                "size_diff": version2["size"] - version1["size"]
            }
            
            return diff_info
            
        except Exception as e:
            raise VersioningError(f"Failed to compare versions: {e}")
    
    async def cleanup_versions(self, older_than_days: int = 30, 
                              keep_minimum: int = 3) -> Dict[str, int]:
        """
        Clean up old versions based on age and retention policy.
        
        Args:
            older_than_days: Delete versions older than this many days
            keep_minimum: Minimum number of versions to keep per file
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        stats = {"files_processed": 0, "versions_deleted": 0, "space_freed": 0}
        
        try:
            # Process all version info files
            for version_info_file in self.versions_dir.glob("*.versions.json"):
                version_info = await self._load_version_info(version_info_file)
                versions = version_info.get("versions", [])
                
                if len(versions) <= keep_minimum:
                    continue
                
                # Sort versions by timestamp (newest first)
                versions.sort(key=lambda v: v["timestamp"], reverse=True)
                
                # Keep the minimum number of newest versions
                versions_to_keep = versions[:keep_minimum]
                versions_to_check = versions[keep_minimum:]
                
                # Delete old versions
                deleted_versions = []
                for version in versions_to_check:
                    try:
                        version_date = datetime.fromisoformat(version["timestamp"])
                        if version_date < cutoff_date:
                            version_file_path = Path(version["file_path"])
                            if version_file_path.exists():
                                file_size = version_file_path.stat().st_size
                                version_file_path.unlink()
                                stats["space_freed"] += file_size
                            deleted_versions.append(version)
                            stats["versions_deleted"] += 1
                        else:
                            versions_to_keep.append(version)
                    except (ValueError, OSError) as e:
                        logger.warning(f"Error processing version {version.get('id', 'unknown')}: {e}")
                        versions_to_keep.append(version)  # Keep problematic versions
                
                # Update version info if versions were deleted
                if deleted_versions:
                    version_info["versions"] = versions_to_keep
                    version_info["last_updated"] = datetime.now().isoformat()
                    await self._save_version_info(version_info_file, version_info)
                
                stats["files_processed"] += 1
            
            logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            raise VersioningError(f"Failed to cleanup versions: {e}")
    
    async def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage statistics for the versioning system.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "total_files": 0,
            "total_versions": 0,
            "total_size": 0,
            "oldest_version": None,
            "newest_version": None
        }
        
        try:
            all_timestamps = []
            
            for version_info_file in self.versions_dir.glob("*.versions.json"):
                version_info = await self._load_version_info(version_info_file)
                versions = version_info.get("versions", [])
                
                if versions:
                    stats["total_files"] += 1
                    stats["total_versions"] += len(versions)
                    
                    for version in versions:
                        try:
                            version_file_path = Path(version["file_path"])
                            if version_file_path.exists():
                                stats["total_size"] += version_file_path.stat().st_size
                            
                            timestamp = datetime.fromisoformat(version["timestamp"])
                            all_timestamps.append(timestamp)
                        except (ValueError, OSError):
                            continue
            
            if all_timestamps:
                stats["oldest_version"] = min(all_timestamps).isoformat()
                stats["newest_version"] = max(all_timestamps).isoformat()
            
            # Convert size to human readable format
            stats["total_size_human"] = self._format_size(stats["total_size"])
            
            return stats
            
        except Exception as e:
            raise VersioningError(f"Failed to get storage stats: {e}")
    
    async def _load_version_info(self, info_path: Path) -> Dict[str, any]:
        """Load version information from JSON file."""
        if not info_path.exists():
            return {}
        
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load version info from {info_path}: {e}")
            return {}
    
    async def _save_version_info(self, info_path: Path, version_info: Dict[str, any]) -> None:
        """Save version information to JSON file."""
        try:
            with open(info_path, 'w') as f:
                json.dump(version_info, f, indent=2)
        except OSError as e:
            raise VersioningError(f"Failed to save version info to {info_path}: {e}")
    
    async def _cleanup_old_versions(self, version_info: Dict[str, any], 
                                   file_path: Path) -> None:
        """Clean up old versions if we exceed the maximum."""
        versions = version_info.get("versions", [])
        
        if len(versions) <= self.max_versions:
            return
        
        # Sort by timestamp (oldest first)
        versions.sort(key=lambda v: v["timestamp"])
        
        # Remove oldest versions
        versions_to_remove = versions[:-self.max_versions]
        versions_to_keep = versions[-self.max_versions:]
        
        for version in versions_to_remove:
            try:
                version_file_path = Path(version["file_path"])
                if version_file_path.exists():
                    version_file_path.unlink()
                logger.debug(f"Removed old version {version['id']}")
            except OSError as e:
                logger.warning(f"Failed to remove old version {version['id']}: {e}")
        
        version_info["versions"] = versions_to_keep
    
    def _format_age(self, timestamp: datetime) -> str:
        """Format the age of a version in human-readable format."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
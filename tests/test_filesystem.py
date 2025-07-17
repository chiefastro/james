"""
Unit tests for file system management components.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pytest
import asyncio

from backend.filesystem import (
    FileManager, 
    DirectoryManager, 
    VersionManager,
    FileSystemError,
    SecurityError,
    VersioningError,
    DirectoryError
)


class TestFileManager:
    """Test cases for FileManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a FileManager instance with temporary directory."""
        return FileManager(base_path=temp_dir)
    
    @pytest.mark.asyncio
    async def test_file_manager_initialization(self, temp_dir):
        """Test FileManager initialization creates base directory."""
        fm = FileManager(base_path=temp_dir)
        assert Path(temp_dir).exists()
        assert fm.base_path == Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_path_validation_security(self, file_manager):
        """Test path validation prevents directory traversal."""
        with pytest.raises(SecurityError):
            file_manager._validate_path("../../../etc/passwd")
        
        with pytest.raises(SecurityError):
            file_manager._validate_path("/etc/passwd")
        
        # Valid paths should work
        valid_path = file_manager._validate_path("test.txt")
        # Use resolved paths for comparison to handle symlinks
        assert valid_path.parent.resolve() == file_manager.base_path.resolve()
    
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, file_manager):
        """Test writing and reading files."""
        content = b"Hello, World!"
        checksum = await file_manager.write_file("test.txt", content)
        
        # Verify file exists
        assert await file_manager.file_exists("test.txt")
        
        # Read and verify content
        read_content = await file_manager.read_file("test.txt")
        assert read_content == content
        
        # Verify checksum
        expected_checksum = file_manager._calculate_checksum(content)
        assert checksum == expected_checksum
    
    @pytest.mark.asyncio
    async def test_write_and_read_text_file(self, file_manager):
        """Test writing and reading text files."""
        content = "Hello, World! 🌍"
        checksum = await file_manager.write_text_file("test.txt", content)
        
        read_content = await file_manager.read_text_file("test.txt")
        assert read_content == content
        assert checksum is not None
    
    @pytest.mark.asyncio
    async def test_write_and_read_json_file(self, file_manager):
        """Test writing and reading JSON files."""
        data = {"name": "James", "type": "agent", "version": 1.0}
        checksum = await file_manager.write_json_file("config.json", data)
        
        read_data = await file_manager.read_json_file("config.json")
        assert read_data == data
        assert checksum is not None
    
    @pytest.mark.asyncio
    async def test_append_to_file(self, file_manager):
        """Test appending content to files."""
        initial_content = b"Line 1\n"
        append_content = b"Line 2\n"
        
        await file_manager.write_file("log.txt", initial_content)
        await file_manager.append_to_file("log.txt", append_content)
        
        final_content = await file_manager.read_file("log.txt")
        assert final_content == initial_content + append_content
    
    @pytest.mark.asyncio
    async def test_append_text_to_file(self, file_manager):
        """Test appending text to files."""
        await file_manager.write_text_file("log.txt", "Line 1\n")
        await file_manager.append_text_to_file("log.txt", "Line 2\n")
        
        content = await file_manager.read_text_file("log.txt")
        assert content == "Line 1\nLine 2\n"
    
    @pytest.mark.asyncio
    async def test_delete_file(self, file_manager):
        """Test file deletion."""
        await file_manager.write_text_file("temp.txt", "temporary content")
        assert await file_manager.file_exists("temp.txt")
        
        deleted = await file_manager.delete_file("temp.txt", create_backup=False)
        assert deleted is True
        assert not await file_manager.file_exists("temp.txt")
        
        # Try deleting non-existent file
        deleted = await file_manager.delete_file("nonexistent.txt")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_get_file_info(self, file_manager):
        """Test getting file information."""
        content = "test content"
        await file_manager.write_text_file("info_test.txt", content)
        
        info = await file_manager.get_file_info("info_test.txt")
        
        assert "path" in info
        assert "size" in info
        assert "created" in info
        assert "modified" in info
        assert "permissions" in info
        assert info["is_file"] is True
        assert info["size"] == len(content.encode())
    
    @pytest.mark.asyncio
    async def test_list_files(self, file_manager):
        """Test listing files in directory."""
        # Create test files
        await file_manager.write_text_file("file1.txt", "content1")
        await file_manager.write_text_file("file2.txt", "content2")
        await file_manager.write_text_file("subdir/file3.txt", "content3")
        
        # List all files
        files = await file_manager.list_files()
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert "subdir/file3.txt" not in files  # Not recursive
        
        # List files recursively
        files_recursive = await file_manager.list_files(recursive=True)
        assert "file1.txt" in files_recursive
        assert "file2.txt" in files_recursive
        assert "subdir/file3.txt" in files_recursive
        
        # List files with pattern
        txt_files = await file_manager.list_files(pattern="*.txt")
        assert all(f.endswith(".txt") for f in txt_files)
    
    @pytest.mark.asyncio
    async def test_file_backup_creation(self, file_manager):
        """Test automatic backup creation."""
        # Write initial content
        await file_manager.write_text_file("backup_test.txt", "original content")
        
        # Write new content (should create backup)
        await file_manager.write_text_file("backup_test.txt", "new content")
        
        # Check that backup was created
        files = await file_manager.list_files(recursive=True)
        backup_files = [f for f in files if "backup_test.txt.backup_" in f]
        assert len(backup_files) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, file_manager):
        """Test error handling for various scenarios."""
        # Test reading non-existent file
        with pytest.raises(FileSystemError):
            await file_manager.read_file("nonexistent.txt")
        
        # Test invalid JSON
        await file_manager.write_text_file("invalid.json", "invalid json content")
        with pytest.raises(FileSystemError):
            await file_manager.read_json_file("invalid.json")
        
        # Test getting info for non-existent file
        with pytest.raises(FileSystemError):
            await file_manager.get_file_info("nonexistent.txt")


class TestDirectoryManager:
    """Test cases for DirectoryManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def dir_manager(self, temp_dir):
        """Create a DirectoryManager instance with temporary directory."""
        return DirectoryManager(base_path=temp_dir)
    
    @pytest.mark.asyncio
    async def test_directory_manager_initialization(self, temp_dir):
        """Test DirectoryManager initialization."""
        dm = DirectoryManager(base_path=temp_dir)
        assert Path(temp_dir).exists()
        assert dm.base_path == Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_initialize_standard_structure(self, dir_manager):
        """Test initialization of standard directory structure."""
        created_dirs = await dir_manager.initialize_standard_structure()
        
        # Check that all standard directories were created
        for dir_name in DirectoryManager.STANDARD_DIRECTORIES:
            assert dir_name in created_dirs
            assert await dir_manager.directory_exists(dir_name)
            
            # Check that README files were created
            readme_path = Path(created_dirs[dir_name]) / "README.md"
            assert readme_path.exists()
    
    @pytest.mark.asyncio
    async def test_create_directory(self, dir_manager):
        """Test directory creation."""
        dir_path = await dir_manager.create_directory("test_dir")
        assert dir_path.exists()
        assert dir_path.is_dir()
        
        # Test creating nested directories
        nested_path = await dir_manager.create_directory("parent/child/grandchild")
        assert nested_path.exists()
        assert nested_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_delete_directory(self, dir_manager):
        """Test directory deletion."""
        # Create and delete empty directory
        await dir_manager.create_directory("empty_dir")
        assert await dir_manager.directory_exists("empty_dir")
        
        deleted = await dir_manager.delete_directory("empty_dir")
        assert deleted is True
        assert not await dir_manager.directory_exists("empty_dir")
        
        # Test deleting non-empty directory
        await dir_manager.create_directory("non_empty_dir")
        test_file = dir_manager.base_path / "non_empty_dir" / "test.txt"
        test_file.write_text("content")
        
        # Should fail without recursive flag
        with pytest.raises(DirectoryError):
            await dir_manager.delete_directory("non_empty_dir", recursive=False)
        
        # Should succeed with recursive flag
        deleted = await dir_manager.delete_directory("non_empty_dir", recursive=True)
        assert deleted is True
    
    @pytest.mark.asyncio
    async def test_move_directory(self, dir_manager):
        """Test directory moving."""
        # Create source directory with content
        await dir_manager.create_directory("source_dir")
        test_file = dir_manager.base_path / "source_dir" / "test.txt"
        test_file.write_text("content")
        
        # Move directory
        moved_path = await dir_manager.move_directory("source_dir", "moved_dir")
        
        assert not await dir_manager.directory_exists("source_dir")
        assert await dir_manager.directory_exists("moved_dir")
        assert (moved_path / "test.txt").exists()
    
    @pytest.mark.asyncio
    async def test_copy_directory(self, dir_manager):
        """Test directory copying."""
        # Create source directory with content
        await dir_manager.create_directory("source_dir")
        test_file = dir_manager.base_path / "source_dir" / "test.txt"
        test_file.write_text("content")
        
        # Copy directory
        copied_path = await dir_manager.copy_directory("source_dir", "copied_dir")
        
        assert await dir_manager.directory_exists("source_dir")
        assert await dir_manager.directory_exists("copied_dir")
        assert (copied_path / "test.txt").exists()
    
    @pytest.mark.asyncio
    async def test_list_directories(self, dir_manager):
        """Test listing directories."""
        # Create test directories
        await dir_manager.create_directory("dir1")
        await dir_manager.create_directory("dir2")
        await dir_manager.create_directory("parent/child")
        
        # List directories (non-recursive)
        dirs = await dir_manager.list_directories()
        assert "dir1" in dirs
        assert "dir2" in dirs
        assert "parent" in dirs
        assert "parent/child" not in dirs
        
        # List directories (recursive)
        dirs_recursive = await dir_manager.list_directories(recursive=True)
        assert "dir1" in dirs_recursive
        assert "dir2" in dirs_recursive
        assert "parent" in dirs_recursive
        assert "parent/child" in dirs_recursive
    
    @pytest.mark.asyncio
    async def test_get_directory_info(self, dir_manager):
        """Test getting directory information."""
        await dir_manager.create_directory("info_test")
        
        # Add some files
        test_dir = dir_manager.base_path / "info_test"
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        (test_dir / "subdir").mkdir()
        
        info = await dir_manager.get_directory_info("info_test")
        
        assert "path" in info
        assert "created" in info
        assert "file_count" in info
        assert "directory_count" in info
        assert "total_size" in info
        assert info["file_count"] == 2
        assert info["directory_count"] == 1
        assert not info["is_empty"]
    
    @pytest.mark.asyncio
    async def test_organize_by_date(self, dir_manager):
        """Test organizing files by date."""
        # Create source directory with files
        await dir_manager.create_directory("source")
        await dir_manager.create_directory("target")
        
        source_dir = dir_manager.base_path / "source"
        
        # Create files with different modification times
        file1 = source_dir / "file1.txt"
        file2 = source_dir / "file2.txt"
        
        file1.write_text("content1")
        file2.write_text("content2")
        
        # Organize files
        organized = await dir_manager.organize_by_date("source", "target")
        
        assert isinstance(organized, dict)
        assert len(organized) > 0
        
        # Check that files were moved
        assert not file1.exists()
        assert not file2.exists()
    
    @pytest.mark.asyncio
    async def test_cleanup_empty_directories(self, dir_manager):
        """Test cleanup of empty directories."""
        # Create nested empty directories
        await dir_manager.create_directory("empty1")
        await dir_manager.create_directory("empty2/nested_empty")
        await dir_manager.create_directory("not_empty")
        
        # Add file to make one directory non-empty
        (dir_manager.base_path / "not_empty" / "file.txt").write_text("content")
        
        # Cleanup empty directories
        removed = await dir_manager.cleanup_empty_directories()
        
        assert "empty1" in removed
        assert "empty2/nested_empty" in removed or "empty2" in removed
        assert not await dir_manager.directory_exists("empty1")
        assert await dir_manager.directory_exists("not_empty")


class TestVersionManager:
    """Test cases for VersionManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def version_manager(self, temp_dir):
        """Create a VersionManager instance with temporary directory."""
        return VersionManager(base_path=temp_dir, max_versions=5)
    
    @pytest.fixture
    def test_file(self, temp_dir):
        """Create a test file."""
        file_path = Path(temp_dir) / "test.txt"
        file_path.write_text("original content")
        return file_path
    
    @pytest.mark.asyncio
    async def test_version_manager_initialization(self, temp_dir):
        """Test VersionManager initialization."""
        vm = VersionManager(base_path=temp_dir)
        assert vm.base_path == Path(temp_dir)
        assert vm.versions_dir.exists()
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_manager, test_file):
        """Test creating file versions."""
        version_id = await version_manager.create_version(test_file)
        assert version_id is not None
        
        # Check that version was created
        versions = await version_manager.list_versions(test_file)
        assert len(versions) == 1
        assert versions[0]["id"] == version_id
    
    @pytest.mark.asyncio
    async def test_create_multiple_versions(self, version_manager, test_file):
        """Test creating multiple versions of a file."""
        # Create first version
        version1 = await version_manager.create_version(test_file, "First version")
        
        # Modify file and create second version
        test_file.write_text("modified content")
        version2 = await version_manager.create_version(test_file, "Second version")
        
        # List versions
        versions = await version_manager.list_versions(test_file)
        assert len(versions) == 2
        
        version_ids = [v["id"] for v in versions]
        assert version1 in version_ids
        assert version2 in version_ids
    
    @pytest.mark.asyncio
    async def test_restore_version(self, version_manager, test_file):
        """Test restoring file to previous version."""
        original_content = test_file.read_text()
        
        # Create version of original content
        version_id = await version_manager.create_version(test_file, "Original")
        
        # Modify file
        modified_content = "completely different content"
        test_file.write_text(modified_content)
        assert test_file.read_text() == modified_content
        
        # Restore to original version
        restored = await version_manager.restore_version(test_file, version_id)
        assert restored is True
        assert test_file.read_text() == original_content
    
    @pytest.mark.asyncio
    async def test_delete_version(self, version_manager, test_file):
        """Test deleting a specific version."""
        # Create two versions
        version1 = await version_manager.create_version(test_file, "Version 1")
        
        test_file.write_text("modified content")
        version2 = await version_manager.create_version(test_file, "Version 2")
        
        # Delete first version
        deleted = await version_manager.delete_version(test_file, version1)
        assert deleted is True
        
        # Check that only second version remains
        versions = await version_manager.list_versions(test_file)
        assert len(versions) == 1
        assert versions[0]["id"] == version2
    
    @pytest.mark.asyncio
    async def test_version_diff(self, version_manager, test_file):
        """Test getting differences between versions."""
        # Create first version
        version1 = await version_manager.create_version(test_file, "Version 1")
        
        # Modify file and create second version
        test_file.write_text("modified content with more text")
        version2 = await version_manager.create_version(test_file, "Version 2")
        
        # Get diff
        diff = await version_manager.get_version_diff(test_file, version1, version2)
        
        assert "version1" in diff
        assert "version2" in diff
        assert "same_content" in diff
        assert "size_diff" in diff
        assert diff["same_content"] is False
        assert diff["size_diff"] > 0  # Second version should be larger
    
    @pytest.mark.asyncio
    async def test_max_versions_cleanup(self, version_manager, test_file):
        """Test automatic cleanup when max versions exceeded."""
        # Create more versions than the maximum
        version_ids = []
        for i in range(7):  # max_versions is 5
            test_file.write_text(f"content version {i}")
            version_id = await version_manager.create_version(test_file, f"Version {i}")
            version_ids.append(version_id)
        
        # Check that only max_versions are kept
        versions = await version_manager.list_versions(test_file)
        assert len(versions) <= version_manager.max_versions
        
        # Check that newest versions are kept
        remaining_ids = [v["id"] for v in versions]
        assert version_ids[-1] in remaining_ids  # Most recent should be kept
    
    @pytest.mark.asyncio
    async def test_cleanup_versions(self, version_manager, test_file):
        """Test cleanup of old versions."""
        # Create versions with different timestamps
        version_ids = []
        for i in range(3):
            test_file.write_text(f"content {i}")
            version_id = await version_manager.create_version(test_file, f"Version {i}")
            version_ids.append(version_id)
        
        # Run cleanup (with 0 days to clean all but minimum)
        stats = await version_manager.cleanup_versions(older_than_days=0, keep_minimum=1)
        
        assert stats["files_processed"] >= 1
        assert stats["versions_deleted"] >= 0
        
        # Check that minimum versions are kept
        versions = await version_manager.list_versions(test_file)
        assert len(versions) >= 1
    
    @pytest.mark.asyncio
    async def test_storage_stats(self, version_manager, test_file):
        """Test getting storage statistics."""
        # Create some versions
        for i in range(3):
            test_file.write_text(f"content version {i}")
            await version_manager.create_version(test_file, f"Version {i}")
        
        stats = await version_manager.get_storage_stats()
        
        assert "total_files" in stats
        assert "total_versions" in stats
        assert "total_size" in stats
        assert "total_size_human" in stats
        assert stats["total_files"] >= 1
        assert stats["total_versions"] >= 3
    
    @pytest.mark.asyncio
    async def test_duplicate_version_detection(self, version_manager, test_file):
        """Test that duplicate versions (same content) are not created."""
        # Create first version
        version1 = await version_manager.create_version(test_file, "First")
        
        # Try to create version with same content
        version2 = await version_manager.create_version(test_file, "Duplicate")
        
        # Should return the same version ID
        assert version1 == version2
        
        # Should only have one version
        versions = await version_manager.list_versions(test_file)
        assert len(versions) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, version_manager):
        """Test error handling for various scenarios."""
        # Test creating version of non-existent file
        with pytest.raises(VersioningError):
            await version_manager.create_version("nonexistent.txt")
        
        # Test restoring non-existent version
        test_file = version_manager.base_path / "test.txt"
        test_file.write_text("content")
        
        with pytest.raises(VersioningError):
            await version_manager.restore_version(test_file, "nonexistent_version")
        
        # Test getting diff with non-existent versions
        version_id = await version_manager.create_version(test_file)
        
        with pytest.raises(VersioningError):
            await version_manager.get_version_diff(test_file, version_id, "nonexistent")


if __name__ == "__main__":
    pytest.main([__file__])
#!/usr/bin/env python3
"""
Demonstration of the file system management capabilities.

This script shows how to use the FileManager, DirectoryManager, and VersionManager
to manage files within the ~/.james directory.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path

from backend.filesystem import FileManager, DirectoryManager, VersionManager


async def demo_file_manager():
    """Demonstrate FileManager capabilities."""
    print("=== FileManager Demo ===")
    
    # Use a temporary directory for demo
    temp_dir = tempfile.mkdtemp(prefix="james_demo_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        fm = FileManager(base_path=temp_dir)
        
        # Write and read text files
        print("\n1. Writing and reading text files:")
        await fm.write_text_file("notes.txt", "Hello from James!\nThis is a test note.")
        content = await fm.read_text_file("notes.txt")
        print(f"   Content: {repr(content)}")
        
        # Write and read JSON files
        print("\n2. Writing and reading JSON files:")
        data = {"agent": "James", "version": "1.0", "capabilities": ["file_management", "versioning"]}
        await fm.write_json_file("config.json", data)
        loaded_data = await fm.read_json_file("config.json")
        print(f"   Loaded data: {loaded_data}")
        
        # Append to files
        print("\n3. Appending to files:")
        await fm.append_text_to_file("notes.txt", "\nAppended line!")
        updated_content = await fm.read_text_file("notes.txt")
        print(f"   Updated content: {repr(updated_content)}")
        
        # List files
        print("\n4. Listing files:")
        files = await fm.list_files()
        print(f"   Files: {files}")
        
        # Get file info
        print("\n5. File information:")
        info = await fm.get_file_info("notes.txt")
        print(f"   Size: {info['size']} bytes")
        print(f"   Modified: {info['modified']}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")


async def demo_directory_manager():
    """Demonstrate DirectoryManager capabilities."""
    print("\n=== DirectoryManager Demo ===")
    
    # Use a temporary directory for demo
    temp_dir = tempfile.mkdtemp(prefix="james_demo_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        dm = DirectoryManager(base_path=temp_dir)
        
        # Initialize standard structure
        print("\n1. Initializing standard directory structure:")
        created_dirs = await dm.initialize_standard_structure()
        print(f"   Created directories: {list(created_dirs.keys())}")
        
        # Create custom directories
        print("\n2. Creating custom directories:")
        await dm.create_directory("projects/ai_research")
        await dm.create_directory("projects/experiments")
        
        # List directories
        print("\n3. Listing directories:")
        dirs = await dm.list_directories(recursive=True)
        print(f"   All directories: {dirs}")
        
        # Get directory info
        print("\n4. Directory information:")
        info = await dm.get_directory_info("projects")
        print(f"   Projects directory - Files: {info['file_count']}, Subdirs: {info['directory_count']}")
        
        # Copy directory
        print("\n5. Copying directory:")
        await dm.copy_directory("projects", "projects_backup")
        dirs_after_copy = await dm.list_directories()
        print(f"   Directories after copy: {[d for d in dirs_after_copy if 'projects' in d]}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")


async def demo_version_manager():
    """Demonstrate VersionManager capabilities."""
    print("\n=== VersionManager Demo ===")
    
    # Use a temporary directory for demo
    temp_dir = tempfile.mkdtemp(prefix="james_demo_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        vm = VersionManager(base_path=temp_dir, max_versions=5)
        
        # Create a test file
        test_file = Path(temp_dir) / "document.txt"
        test_file.write_text("Original document content")
        
        # Create versions
        print("\n1. Creating versions:")
        version1 = await vm.create_version(test_file, "Initial version")
        print(f"   Created version: {version1}")
        
        # Modify and create another version
        test_file.write_text("Modified document content with more details")
        version2 = await vm.create_version(test_file, "Added more details")
        print(f"   Created version: {version2}")
        
        # Modify again
        test_file.write_text("Final document content with conclusions")
        version3 = await vm.create_version(test_file, "Added conclusions")
        print(f"   Created version: {version3}")
        
        # List versions
        print("\n2. Listing versions:")
        versions = await vm.list_versions(test_file)
        for v in versions:
            print(f"   {v['id']}: {v['comment']} ({v['timestamp_human']}, {v['size']} bytes)")
        
        # Get version diff
        print("\n3. Version differences:")
        diff = await vm.get_version_diff(test_file, version1, version3)
        print(f"   Size difference: {diff['size_diff']} bytes")
        print(f"   Same content: {diff['same_content']}")
        
        # Restore to previous version
        print("\n4. Restoring to previous version:")
        print(f"   Current content: {repr(test_file.read_text())}")
        await vm.restore_version(test_file, version1)
        print(f"   After restore: {repr(test_file.read_text())}")
        
        # Storage stats
        print("\n5. Storage statistics:")
        stats = await vm.get_storage_stats()
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total versions: {stats['total_versions']}")
        print(f"   Total size: {stats['total_size_human']}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")


async def main():
    """Run all demonstrations."""
    print("James File System Management Demo")
    print("=" * 40)
    
    await demo_file_manager()
    await demo_directory_manager()
    await demo_version_manager()
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
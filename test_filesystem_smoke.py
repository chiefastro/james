#!/usr/bin/env python3
"""
Smoke test for file system management functionality.

This script performs basic smoke tests to ensure the file system
management components are working correctly.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path

from backend.filesystem import FileManager, DirectoryManager, VersionManager


async def smoke_test():
    """Run basic smoke tests for file system components."""
    print("Running file system smoke tests...")
    
    # Use a temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="james_smoke_test_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Test FileManager
        print("\n1. Testing FileManager...")
        fm = FileManager(base_path=temp_dir)
        
        # Basic file operations
        await fm.write_text_file("test.txt", "Hello, World!")
        content = await fm.read_text_file("test.txt")
        assert content == "Hello, World!", f"Expected 'Hello, World!', got {repr(content)}"
        
        # JSON operations
        test_data = {"test": True, "value": 42}
        await fm.write_json_file("test.json", test_data)
        loaded_data = await fm.read_json_file("test.json")
        assert loaded_data == test_data, f"JSON data mismatch: {loaded_data}"
        
        print("   ✓ FileManager basic operations work")
        
        # Test DirectoryManager
        print("\n2. Testing DirectoryManager...")
        dm = DirectoryManager(base_path=temp_dir)
        
        # Create directories
        await dm.create_directory("test_dir")
        assert await dm.directory_exists("test_dir"), "Directory creation failed"
        
        # Initialize standard structure
        created_dirs = await dm.initialize_standard_structure()
        assert len(created_dirs) > 0, "Standard directory creation failed"
        
        print("   ✓ DirectoryManager basic operations work")
        
        # Test VersionManager
        print("\n3. Testing VersionManager...")
        vm = VersionManager(base_path=temp_dir)
        
        # Create a test file for versioning
        test_file = Path(temp_dir) / "version_test.txt"
        test_file.write_text("Version 1")
        
        # Create version
        version_id = await vm.create_version(test_file, "First version")
        assert version_id is not None, "Version creation failed"
        
        # Modify and create another version
        test_file.write_text("Version 2")
        version_id2 = await vm.create_version(test_file, "Second version")
        assert version_id2 != version_id, "Version IDs should be different"
        
        # List versions
        versions = await vm.list_versions(test_file)
        assert len(versions) == 2, f"Expected 2 versions, got {len(versions)}"
        
        # Restore version
        await vm.restore_version(test_file, version_id)
        restored_content = test_file.read_text()
        assert restored_content == "Version 1", f"Version restore failed: {repr(restored_content)}"
        
        print("   ✓ VersionManager basic operations work")
        
        print("\n✅ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {temp_dir}")


async def main():
    """Run smoke tests."""
    success = await smoke_test()
    if success:
        print("\n🎉 File system management is working correctly!")
        exit(0)
    else:
        print("\n💥 File system management has issues!")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
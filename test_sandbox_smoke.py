#!/usr/bin/env python3
"""
Smoke test for sandbox functionality.

This test verifies that the sandbox can be imported and basic
functionality works without requiring Docker to be running.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.sandbox import (
    SecureSandbox,
    SandboxConfig,
    ExecutionResult,
    SandboxError,
    SecurityViolationError
)


def test_sandbox_config():
    """Test sandbox configuration."""
    print("✅ Testing SandboxConfig...")
    
    # Test default config
    config = SandboxConfig()
    assert config.memory_limit == "512m"
    assert config.cpu_limit == 0.5
    assert config.timeout_seconds == 30
    
    # Test custom config
    custom_config = SandboxConfig(
        memory_limit="1g",
        cpu_limit=1.0,
        timeout_seconds=60
    )
    assert custom_config.memory_limit == "1g"
    assert custom_config.cpu_limit == 1.0
    assert custom_config.timeout_seconds == 60
    
    print("✅ SandboxConfig tests passed!")


def test_execution_result():
    """Test ExecutionResult."""
    print("✅ Testing ExecutionResult...")
    
    result = ExecutionResult(
        success=True,
        stdout="Hello, World!",
        stderr="",
        exit_code=0,
        execution_time=1.5,
        resource_usage={"memory_usage": 1024},
        security_violations=[]
    )
    
    assert result.success is True
    assert result.stdout == "Hello, World!"
    assert result.exit_code == 0
    assert result.execution_time == 1.5
    
    print("✅ ExecutionResult tests passed!")


def test_security_validation():
    """Test security validation without Docker."""
    print("✅ Testing security validation...")
    
    # Import the class for validation testing
    from backend.sandbox.sandbox import SecureSandbox
    
    try:
        # This will fail because Docker is not available, but we can test the validation
        sandbox = SecureSandbox()
    except SandboxError:
        # Expected when Docker is not available
        pass
    
    # Create a mock sandbox for testing validation
    class MockSandbox:
        def _validate_code(self, code, language, allow_file_ops=False):
            return SecureSandbox._validate_code(None, code, language, allow_file_ops)
        
        def _validate_terminal_command(self, command):
            return SecureSandbox._validate_terminal_command(None, command)
    
    mock_sandbox = MockSandbox()
    
    # Test safe code
    try:
        mock_sandbox._validate_code("print('Hello, World!')", "python")
        print("  ✅ Safe code validation passed")
    except SecurityViolationError:
        print("  ❌ Safe code incorrectly flagged as dangerous")
        return False
    
    # Test dangerous code
    try:
        mock_sandbox._validate_code("import subprocess", "python")
        print("  ❌ Dangerous code not detected")
        return False
    except SecurityViolationError:
        print("  ✅ Dangerous code correctly detected")
    
    # Test file operations with permission
    try:
        mock_sandbox._validate_code("import os; open('/james/file.txt', 'w')", "python", allow_file_ops=True)
        print("  ✅ File operations with permission allowed")
    except SecurityViolationError:
        print("  ❌ File operations incorrectly blocked with permission")
        return False
    
    # Test dangerous terminal command
    try:
        mock_sandbox._validate_terminal_command("rm -rf /")
        print("  ❌ Dangerous terminal command not detected")
        return False
    except SecurityViolationError:
        print("  ✅ Dangerous terminal command correctly detected")
    
    print("✅ Security validation tests passed!")
    return True


def test_file_preparation():
    """Test code file preparation."""
    print("✅ Testing file preparation...")
    
    import tempfile
    import os
    
    # Import the class for testing
    from backend.sandbox.sandbox import SecureSandbox
    
    try:
        sandbox = SecureSandbox()
    except SandboxError:
        # Mock the file preparation method when Docker is not available
        pass
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test Python file preparation
            code_file = SecureSandbox._prepare_code_file(
                None, "print('Hello')", "python", temp_dir
            )
            assert code_file.endswith("code.py")
            assert os.path.exists(code_file)
            
            with open(code_file, "r") as f:
                assert f.read() == "print('Hello')"
            
            # Test Bash file preparation
            bash_file = SecureSandbox._prepare_code_file(
                None, "echo 'Hello'", "bash", temp_dir
            )
            assert bash_file.endswith("code.sh")
            assert os.path.exists(bash_file)
            assert os.access(bash_file, os.X_OK)  # Check executable
    
    print("✅ File preparation tests passed!")


def main():
    """Run all smoke tests."""
    print("🚀 Running Sandbox Smoke Tests")
    print("=" * 40)
    
    try:
        test_sandbox_config()
        test_execution_result()
        
        if test_security_validation():
            test_file_preparation()
            
            print("\n🎉 All smoke tests passed!")
            print("\n📋 Summary:")
            print("   ✅ Configuration system works")
            print("   ✅ Result handling works")
            print("   ✅ Security validation works")
            print("   ✅ File preparation works")
            print("\n💡 To test actual code execution, ensure Docker is running and use:")
            print("   python examples/sandbox_demo.py")
            
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
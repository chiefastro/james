"""
Secure Docker-based sandbox for code execution.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

from .exceptions import (
    SandboxError,
    SecurityViolationError,
    ResourceLimitError,
    SandboxTimeoutError,
    ContainerCreationError,
    CodeExecutionError
)

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for the secure sandbox."""
    
    # Resource limits
    memory_limit: str = "512m"  # Memory limit
    cpu_limit: float = 0.5      # CPU limit (fraction of one core)
    timeout_seconds: int = 30   # Execution timeout
    
    # Security settings
    network_mode: str = "none"  # No network access
    readonly_rootfs: bool = True  # Read-only root filesystem
    no_new_privileges: bool = True  # Prevent privilege escalation
    
    # File system settings
    james_dir: str = "~/.james"  # James directory path
    temp_dir_size: str = "100m"  # Temporary directory size limit
    
    # Docker settings
    base_image: str = "python:3.11-slim"  # Base Docker image
    remove_container: bool = True  # Remove container after execution


@dataclass
class ExecutionResult:
    """Result of code execution in the sandbox."""
    
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    resource_usage: Dict[str, Union[str, int, float]]
    security_violations: List[str]


class SecureSandbox:
    """
    Docker-based secure sandbox for code execution.
    
    Provides isolated execution environment with resource limits,
    security constraints, and controlled file system access.
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize the secure sandbox."""
        self.config = config or SandboxConfig()
        self.docker_client = None
        self._initialize_docker()
        
    def _initialize_docker(self):
        """Initialize Docker client and verify connectivity."""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connectivity
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise SandboxError(f"Docker initialization failed: {e}")
    
    async def execute_code(
        self,
        code: str,
        language: str = "python",
        working_dir: Optional[str] = None,
        allow_file_ops: bool = False
    ) -> ExecutionResult:
        """
        Execute code in a secure sandbox environment.
        
        Args:
            code: The code to execute
            language: Programming language (python, bash, etc.)
            working_dir: Working directory inside container
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_code(code, language, allow_file_ops)
            
            # Create temporary files for code execution
            with tempfile.TemporaryDirectory() as temp_dir:
                code_file = self._prepare_code_file(code, language, temp_dir)
                
                # Execute in container
                result = await self._execute_in_container(
                    code_file, language, working_dir, temp_dir
                )
                
                # Calculate execution time
                result.execution_time = time.time() - start_time
                
                return result
                
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=time.time() - start_time,
                resource_usage={},
                security_violations=[]
            )
    
    async def execute_terminal_command(
        self,
        command: str,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a terminal command in the sandbox.
        
        Args:
            command: Terminal command to execute
            working_dir: Working directory inside container
            
        Returns:
            ExecutionResult with execution details
        """
        # Validate command for security
        self._validate_terminal_command(command)
        
        # Execute as bash script
        bash_code = f"#!/bin/bash\n{command}"
        return await self.execute_code(bash_code, "bash", working_dir)
    
    def _validate_code(self, code: str, language: str, allow_file_ops: bool = False):
        """Validate code for security violations."""
        security_violations = []
        
        # Always dangerous patterns
        always_dangerous = [
            "import subprocess",
            "import sys",
            "__import__",
            "eval(",
            "exec(",
            "input(",
            "raw_input(",
            "compile(",
            "globals(",
            "locals(",
            "vars(",
            "dir(",
            "getattr(",
            "setattr(",
            "delattr(",
            "hasattr(",
        ]
        
        # Conditionally dangerous patterns (allowed for James directory operations)
        file_patterns = [
            "import os",
            "open(",
            "file(",
        ]
        
        code_lower = code.lower()
        
        # Check always dangerous patterns
        for pattern in always_dangerous:
            if pattern in code_lower:
                security_violations.append(f"Dangerous pattern detected: {pattern}")
        
        # Check file patterns only if not allowed
        if not allow_file_ops:
            for pattern in file_patterns:
                if pattern in code_lower:
                    security_violations.append(f"File operation pattern detected: {pattern}")
        
        # Check for network-related imports (always dangerous)
        network_patterns = [
            "import socket",
            "import urllib",
            "import requests",
            "import http",
            "from urllib",
            "from requests",
            "from http",
        ]
        
        for pattern in network_patterns:
            if pattern in code_lower:
                security_violations.append(f"Network access pattern detected: {pattern}")
        
        # Check for dangerous file paths even when file ops are allowed
        if allow_file_ops:
            dangerous_paths = [
                "/etc/",
                "/root/",
                "/home/",
                "/usr/",
                "/var/",
                "/sys/",
                "/proc/",
                "/dev/",
            ]
            
            for path in dangerous_paths:
                if path in code_lower and "/james" not in code_lower:
                    security_violations.append(f"Dangerous file path detected: {path}")
        
        if security_violations:
            raise SecurityViolationError(
                f"Security violations detected: {', '.join(security_violations)}"
            )
    
    def _validate_terminal_command(self, command: str):
        """Validate terminal command for security."""
        dangerous_commands = [
            "rm -rf",
            "sudo",
            "su",
            "chmod +x",
            "wget",
            "curl",
            "nc",
            "netcat",
            "ssh",
            "scp",
            "rsync",
            "dd",
            "mkfs",
            "fdisk",
            "mount",
            "umount",
            "chroot",
            "docker",
            "systemctl",
            "service",
        ]
        
        command_lower = command.lower()
        for dangerous_cmd in dangerous_commands:
            if dangerous_cmd in command_lower:
                raise SecurityViolationError(
                    f"Dangerous command detected: {dangerous_cmd}"
                )
    
    def _prepare_code_file(self, code: str, language: str, temp_dir: str) -> str:
        """Prepare code file for execution."""
        if language == "python":
            filename = "code.py"
        elif language == "bash":
            filename = "code.sh"
        else:
            raise SandboxError(f"Unsupported language: {language}")
        
        code_file = os.path.join(temp_dir, filename)
        with open(code_file, "w") as f:
            f.write(code)
        
        if language == "bash":
            os.chmod(code_file, 0o755)
        
        return code_file
    
    async def _execute_in_container(
        self,
        code_file: str,
        language: str,
        working_dir: Optional[str],
        temp_dir: str
    ) -> ExecutionResult:
        """Execute code file in Docker container."""
        try:
            # Prepare James directory mount
            james_dir = os.path.expanduser(self.config.james_dir)
            os.makedirs(james_dir, exist_ok=True)
            
            # Container configuration
            container_config = {
                "image": self.config.base_image,
                "command": self._get_execution_command(language),
                "volumes": {
                    temp_dir: {"bind": "/sandbox", "mode": "ro"},
                    james_dir: {"bind": "/james", "mode": "rw"},
                },
                "working_dir": working_dir or "/sandbox",
                "network_mode": self.config.network_mode,
                "read_only": self.config.readonly_rootfs,
                "mem_limit": self.config.memory_limit,
                "cpu_period": 100000,
                "cpu_quota": int(100000 * self.config.cpu_limit),
                "security_opt": ["no-new-privileges"] if self.config.no_new_privileges else [],
                "tmpfs": {"/tmp": f"size={self.config.temp_dir_size}"},
                "remove": self.config.remove_container,
                "detach": True,
            }
            
            # Create and start container
            container = self.docker_client.containers.run(**container_config)
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.config.timeout_seconds)
                exit_code = result["StatusCode"]
                
                # Get logs
                logs = container.logs(stdout=True, stderr=True)
                stdout = logs.decode("utf-8", errors="replace")
                stderr = ""
                
                # Get resource usage stats
                stats = container.stats(stream=False)
                resource_usage = self._extract_resource_usage(stats)
                
                return ExecutionResult(
                    success=exit_code == 0,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    execution_time=0,  # Will be set by caller
                    resource_usage=resource_usage,
                    security_violations=[]
                )
                
            except asyncio.TimeoutError:
                container.kill()
                raise SandboxTimeoutError(
                    f"Execution timed out after {self.config.timeout_seconds} seconds"
                )
            
        except ImageNotFound:
            raise ContainerCreationError(f"Docker image not found: {self.config.base_image}")
        except ContainerError as e:
            raise CodeExecutionError(f"Container execution failed: {e}")
        except APIError as e:
            raise SandboxError(f"Docker API error: {e}")
    
    def _get_execution_command(self, language: str) -> List[str]:
        """Get the command to execute code in the container."""
        if language == "python":
            return ["python", "/sandbox/code.py"]
        elif language == "bash":
            return ["bash", "/sandbox/code.sh"]
        else:
            raise SandboxError(f"Unsupported language: {language}")
    
    def _extract_resource_usage(self, stats: Dict) -> Dict[str, Union[str, int, float]]:
        """Extract resource usage from container stats."""
        try:
            if not stats:
                return {}
                
            memory_stats = stats.get("memory_stats", {})
            cpu_stats = stats.get("cpu_stats", {})
            
            result = {}
            if memory_stats:
                result.update({
                    "memory_usage": memory_stats.get("usage", 0),
                    "memory_limit": memory_stats.get("limit", 0),
                })
            
            if cpu_stats:
                result.update({
                    "cpu_usage": cpu_stats.get("cpu_usage", {}).get("total_usage", 0),
                    "system_cpu_usage": cpu_stats.get("system_cpu_usage", 0),
                })
            
            return result
        except Exception as e:
            logger.warning(f"Failed to extract resource usage: {e}")
            return {}
    
    def cleanup(self):
        """Clean up Docker resources."""
        if self.docker_client:
            try:
                # Remove any dangling containers
                containers = self.docker_client.containers.list(
                    all=True,
                    filters={"label": "sandbox=true"}
                )
                for container in containers:
                    container.remove(force=True)
                    
                logger.info("Sandbox cleanup completed")
            except Exception as e:
                logger.error(f"Sandbox cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
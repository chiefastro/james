"""Enhanced sandboxing system for secure terminal command execution."""

import os
import subprocess
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import docker
from docker.errors import DockerException


class SecureSandbox:
    """Secure sandbox for executing terminal commands and code."""
    
    def __init__(self, james_home: str = "~/.james") -> None:
        self.james_home = Path(james_home).expanduser()
        self.sandbox_dir = self.james_home / "sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Docker client for containerized execution
        self.docker_client = None
        self._init_docker()
        
        # Allowed and blocked commands
        self.allowed_commands = {
            # File operations
            "ls", "cat", "head", "tail", "grep", "find", "wc", "sort", "uniq",
            # Text processing
            "echo", "printf", "sed", "awk", "cut", "tr", "diff",
            # Basic utilities
            "pwd", "date", "whoami", "uname", "env",
            # Programming languages
            "python", "python3", "node", "npm", "pip", "pip3",
            # Git (limited)
            "git",
            # Package managers
            "apt", "yum", "brew", "dnf"
        }
        
        self.blocked_commands = {
            # System control
            "sudo", "su", "passwd", "chown", "chmod", "chgrp",
            # Network
            "ssh", "scp", "rsync", "wget", "curl", "nc", "netcat",
            # Process control
            "kill", "killall", "pkill", "nohup", "bg", "fg",
            # System modification
            "mount", "umount", "fdisk", "mkfs", "fsck",
            # Services
            "systemctl", "service", "systemd", "init",
            # Dangerous operations
            "rm", "rmdir", "dd", "shred", "format"
        }
        
        # Resource limits
        self.max_execution_time = 30  # seconds
        self.max_memory_mb = 512
        self.max_disk_mb = 100
    
    def _init_docker(self) -> None:
        """Initialize Docker client if available."""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            print("Docker sandbox enabled")
        except (DockerException, Exception) as e:
            print(f"Docker not available, using filesystem sandbox: {e}")
            self.docker_client = None
    
    def _is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check if a command is allowed to execute."""
        # Extract the base command
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False, "Empty command"
        
        base_cmd = cmd_parts[0]
        
        # Check if explicitly blocked
        if base_cmd in self.blocked_commands:
            return False, f"Command '{base_cmd}' is blocked for security"
        
        # Check if explicitly allowed
        if base_cmd in self.allowed_commands:
            return True, "Command allowed"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ">", ">>", "|", "&", ";", "&&", "||",  # Redirection and chaining
            "`", "$(",  # Command substitution
            "sudo", "su",  # Privilege escalation
            "/dev/", "/proc/", "/sys/",  # System directories
            "rm -rf", "rm -r",  # Dangerous deletions
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command:
                return False, f"Command contains dangerous pattern: {pattern}"
        
        # Default: allow simple commands not in blocked list
        return True, "Command allowed by default"
    
    async def execute_command_filesystem(self, command: str, working_dir: Optional[str] = None,
                                       timeout: int = 30) -> Dict[str, Any]:
        """Execute command in filesystem sandbox."""
        # Validate command
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return {
                "success": False,
                "error": reason,
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "sandbox_type": "filesystem"
            }
        
        # Set working directory to sandbox
        if working_dir is None:
            working_dir = str(self.sandbox_dir)
        else:
            # Ensure working directory is within sandbox
            try:
                target_dir = (self.sandbox_dir / working_dir).resolve()
                if not str(target_dir).startswith(str(self.sandbox_dir.resolve())):
                    working_dir = str(self.sandbox_dir)
                else:
                    working_dir = str(target_dir)
            except Exception:
                working_dir = str(self.sandbox_dir)
        
        # Create a temporary script for execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as script_file:
            script_file.write(f"#!/bin/bash\ncd '{working_dir}'\n{command}\n")
            script_path = script_file.name
        
        try:
            # Execute with resource limits
            process = await asyncio.create_subprocess_exec(
                'bash', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                # Set environment variables to limit resources
                env={
                    **os.environ,
                    'JAMES_SANDBOX': '1',
                    'HOME': str(self.sandbox_dir),
                    'TMPDIR': str(self.sandbox_dir / 'tmp')
                }
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "return_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "command": command,
                    "working_dir": working_dir,
                    "sandbox_type": "filesystem"
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "return_code": -1,
                    "stdout": "",
                    "stderr": "",
                    "sandbox_type": "filesystem"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "sandbox_type": "filesystem"
            }
        finally:
            # Clean up script file
            try:
                os.unlink(script_path)
            except Exception:
                pass
    
    async def execute_command_docker(self, command: str, image: str = "python:3.11-slim",
                                   timeout: int = 30) -> Dict[str, Any]:
        """Execute command in Docker container."""
        if not self.docker_client:
            return await self.execute_command_filesystem(command, timeout=timeout)
        
        # Validate command
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return {
                "success": False,
                "error": reason,
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "sandbox_type": "docker"
            }
        
        try:
            # Create temporary directory for this execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy sandbox contents to temp directory
                if self.sandbox_dir.exists():
                    shutil.copytree(self.sandbox_dir, temp_path / "workspace", dirs_exist_ok=True)
                else:
                    (temp_path / "workspace").mkdir()
                
                # Run container with resource limits
                container = self.docker_client.containers.run(
                    image,
                    command=f"bash -c '{command}'",
                    working_dir="/workspace",
                    volumes={str(temp_path / "workspace"): {"bind": "/workspace", "mode": "rw"}},
                    mem_limit=f"{self.max_memory_mb}m",
                    memswap_limit=f"{self.max_memory_mb}m",
                    cpu_period=100000,
                    cpu_quota=50000,  # 50% CPU
                    network_disabled=True,  # No network access
                    remove=True,
                    detach=True,
                    stdout=True,
                    stderr=True
                )
                
                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=timeout)
                    logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='replace')
                    
                    # Split stdout and stderr (Docker combines them)
                    stdout = logs
                    stderr = ""
                    
                    # Copy results back to sandbox
                    if (temp_path / "workspace").exists():
                        shutil.copytree(temp_path / "workspace", self.sandbox_dir, dirs_exist_ok=True)
                    
                    return {
                        "success": result['StatusCode'] == 0,
                        "return_code": result['StatusCode'],
                        "stdout": stdout,
                        "stderr": stderr,
                        "command": command,
                        "sandbox_type": "docker"
                    }
                    
                except docker.errors.APIError as e:
                    if "timeout" in str(e).lower():
                        container.kill()
                        return {
                            "success": False,
                            "error": f"Command timed out after {timeout} seconds",
                            "return_code": -1,
                            "stdout": "",
                            "stderr": "",
                            "sandbox_type": "docker"
                        }
                    else:
                        raise
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "sandbox_type": "docker"
            }
    
    async def execute_command(self, command: str, use_docker: bool = True,
                            working_dir: Optional[str] = None,
                            timeout: int = 30) -> Dict[str, Any]:
        """Execute a command using the most secure available sandbox."""
        if use_docker and self.docker_client:
            return await self.execute_command_docker(command, timeout=timeout)
        else:
            return await self.execute_command_filesystem(command, working_dir, timeout)
    
    def get_sandbox_info(self) -> Dict[str, Any]:
        """Get information about the sandbox environment."""
        return {
            "sandbox_dir": str(self.sandbox_dir),
            "docker_available": self.docker_client is not None,
            "max_execution_time": self.max_execution_time,
            "max_memory_mb": self.max_memory_mb,
            "allowed_commands": sorted(self.allowed_commands),
            "blocked_commands": sorted(self.blocked_commands)
        }
    
    def clean_sandbox(self) -> Dict[str, Any]:
        """Clean the sandbox directory."""
        try:
            if self.sandbox_dir.exists():
                # Keep the directory but clean contents
                for item in self.sandbox_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            
            return {"success": True, "message": "Sandbox cleaned"}
        except Exception as e:
            return {"success": False, "error": str(e)}
"""
Terminal command execution tool with sandbox integration.
"""

import time
from typing import Dict, List, Optional

from .base import BaseTool, ToolResult
from ..sandbox.sandbox import SecureSandbox, SandboxConfig, ExecutionResult
from ..sandbox.exceptions import SandboxError, SecurityViolationError


class TerminalExecutorTool(BaseTool):
    """
    Secure terminal command execution tool for James.
    
    Executes terminal commands in a sandboxed Docker environment
    with security constraints and resource limits.
    """
    
    def __init__(self, sandbox_config: Optional[SandboxConfig] = None):
        """Initialize the terminal executor tool."""
        super().__init__("TerminalExecutor")
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.sandbox = SecureSandbox(self.sandbox_config)
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute a terminal command in the sandbox.
        
        Args:
            command (str): Terminal command to execute
            working_dir (str): Working directory inside container
            timeout (int): Execution timeout in seconds
            allow_file_ops (bool): Whether to allow file operations
            
        Returns:
            ToolResult with command execution outcome
        """
        start_time = time.time()
        
        # Validate required parameters
        error = self._validate_required_params(kwargs, ['command'])
        if error:
            return self._create_error_result(error)
        
        command = kwargs['command']
        working_dir = kwargs.get('working_dir')
        timeout = kwargs.get('timeout', self.sandbox_config.timeout_seconds)
        allow_file_ops = kwargs.get('allow_file_ops', False)
        
        try:
            # Validate command safety
            if not self._is_safe_command(command):
                return self._create_error_result(f"Unsafe command detected: {command}")
            
            # Update sandbox timeout if specified
            if timeout != self.sandbox_config.timeout_seconds:
                temp_config = SandboxConfig()
                temp_config.timeout_seconds = timeout
                sandbox = SecureSandbox(temp_config)
            else:
                sandbox = self.sandbox
            
            # Execute command in sandbox
            result = await sandbox.execute_terminal_command(command, working_dir)
            
            execution_time = time.time() - start_time
            
            # Format result data
            result_data = {
                'command': command,
                'success': result.success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.exit_code,
                'execution_time': result.execution_time,
                'resource_usage': result.resource_usage
            }
            
            # Add security violations if any
            if result.security_violations:
                result_data['security_violations'] = result.security_violations
            
            if result.success:
                self.logger.info(f"Successfully executed command: {command[:50]}...")
                return self._create_success_result(
                    data=result_data,
                    metadata={
                        'total_execution_time': execution_time,
                        'working_dir': working_dir,
                        'timeout': timeout,
                        'allow_file_ops': allow_file_ops
                    }
                )
            else:
                self.logger.warning(f"Command failed: {command[:50]}... (exit_code: {result.exit_code})")
                return self._create_error_result(
                    f"Command failed with exit code {result.exit_code}: {result.stderr}",
                    metadata={
                        'result_data': result_data,
                        'total_execution_time': execution_time
                    }
                )
                
        except SecurityViolationError as e:
            return self._create_error_result(f"Security violation: {e}")
        except SandboxError as e:
            return self._create_error_result(f"Sandbox error: {e}")
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def execute_python_code(self, code: str, working_dir: Optional[str] = None,
                                 allow_file_ops: bool = False) -> ToolResult:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            working_dir: Working directory inside container
            allow_file_ops: Whether to allow file operations
            
        Returns:
            ToolResult with code execution outcome
        """
        start_time = time.time()
        
        try:
            # Execute code in sandbox
            result = await self.sandbox.execute_code(
                code, 
                language="python", 
                working_dir=working_dir,
                allow_file_ops=allow_file_ops
            )
            
            execution_time = time.time() - start_time
            
            # Format result data
            result_data = {
                'code': code,
                'success': result.success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.exit_code,
                'execution_time': result.execution_time,
                'resource_usage': result.resource_usage
            }
            
            if result.security_violations:
                result_data['security_violations'] = result.security_violations
            
            if result.success:
                self.logger.info("Successfully executed Python code")
                return self._create_success_result(
                    data=result_data,
                    metadata={
                        'total_execution_time': execution_time,
                        'language': 'python',
                        'working_dir': working_dir,
                        'allow_file_ops': allow_file_ops
                    }
                )
            else:
                return self._create_error_result(
                    f"Python code failed with exit code {result.exit_code}: {result.stderr}",
                    metadata={'result_data': result_data}
                )
                
        except SecurityViolationError as e:
            return self._create_error_result(f"Security violation: {e}")
        except SandboxError as e:
            return self._create_error_result(f"Sandbox error: {e}")
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    def _is_safe_command(self, command: str) -> bool:
        """
        Validate that a command is safe to execute.
        
        Args:
            command: Command to validate
            
        Returns:
            True if command is safe, False otherwise
        """
        # List of dangerous commands that should never be allowed
        dangerous_commands = [
            # System modification
            'rm -rf /', 'rm -rf *', 'rm -rf ~',
            'sudo rm', 'sudo dd', 'sudo mkfs',
            'format', 'fdisk', 'parted',
            
            # Network access
            'wget', 'curl', 'nc', 'netcat', 'ssh', 'scp', 'rsync',
            'ftp', 'sftp', 'telnet',
            
            # System control
            'sudo', 'su', 'sudo su', 'passwd',
            'systemctl', 'service', 'init',
            'reboot', 'shutdown', 'halt', 'poweroff',
            
            # Process manipulation
            'kill -9', 'killall', 'pkill',
            
            # File system manipulation
            'mount', 'umount', 'chroot',
            'chmod 777', 'chmod -R 777',
            
            # Package management
            'apt install', 'yum install', 'pip install --user',
            'npm install -g', 'gem install',
            
            # Docker/container escape attempts
            'docker', 'podman', 'lxc', 'runc',
        ]
        
        command_lower = command.lower().strip()
        
        # Check for exact dangerous commands
        for dangerous_cmd in dangerous_commands:
            if dangerous_cmd in command_lower:
                return False
        
        # Check for command chaining that might bypass restrictions
        if any(separator in command for separator in ['&&', '||', ';', '|']):
            # Allow simple pipes for common operations
            if command.count('|') == 1 and not any(sep in command for sep in ['&&', '||', ';']):
                # Check each part of the pipe
                parts = command.split('|')
                for part in parts:
                    if not self._is_safe_simple_command(part.strip()):
                        return False
            else:
                return False
        
        # Check for redirection to sensitive files
        if any(redirect in command for redirect in ['>', '>>', '<']):
            # Allow redirection to files in current directory or /tmp
            if any(path in command_lower for path in ['/etc/', '/root/', '/home/', '/usr/', '/var/']):
                if '/james' not in command_lower:
                    return False
        
        return True
    
    def _is_safe_simple_command(self, command: str) -> bool:
        """Check if a simple command (no pipes/chains) is safe."""
        command_lower = command.lower().strip()
        
        # Allow common safe commands
        safe_commands = [
            'ls', 'dir', 'pwd', 'whoami', 'id', 'date', 'uptime',
            'cat', 'head', 'tail', 'less', 'more', 'grep', 'find',
            'wc', 'sort', 'uniq', 'cut', 'awk', 'sed',
            'echo', 'printf', 'test', '[', 'true', 'false',
            'mkdir', 'rmdir', 'touch', 'cp', 'mv', 'ln',
            'chmod', 'chown', 'stat', 'file', 'which', 'type',
            'python', 'python3', 'node', 'npm', 'git',
            'tar', 'gzip', 'gunzip', 'zip', 'unzip',
        ]
        
        # Extract the base command (first word)
        base_command = command_lower.split()[0] if command_lower.split() else ''
        
        # Check if base command is in safe list
        if base_command in safe_commands:
            return True
        
        # Allow python/node with specific scripts
        if base_command in ['python', 'python3', 'node'] and len(command_lower.split()) > 1:
            return True
        
        return False
    
    async def get_system_info(self) -> ToolResult:
        """
        Get basic system information from the sandbox.
        
        Returns:
            ToolResult with system information
        """
        commands = [
            "uname -a",
            "whoami",
            "pwd",
            "ls -la",
            "python3 --version",
            "which python3"
        ]
        
        results = {}
        
        for cmd in commands:
            try:
                result = await self.execute(command=cmd)
                if result.success:
                    results[cmd] = result.data['stdout'].strip()
                else:
                    results[cmd] = f"Error: {result.error}"
            except Exception as e:
                results[cmd] = f"Exception: {e}"
        
        return self._create_success_result(
            data={
                'system_info': results,
                'sandbox_config': {
                    'memory_limit': self.sandbox_config.memory_limit,
                    'cpu_limit': self.sandbox_config.cpu_limit,
                    'timeout_seconds': self.sandbox_config.timeout_seconds,
                    'base_image': self.sandbox_config.base_image
                }
            }
        )
    
    def cleanup(self):
        """Clean up sandbox resources."""
        if hasattr(self, 'sandbox'):
            self.sandbox.cleanup()
# Secure Sandbox Environment

This module provides a Docker-based secure sandbox for executing arbitrary code and terminal commands with comprehensive security constraints and resource limits.

## Features

### Security Features
- **Container Isolation**: Complete isolation from host system using Docker containers
- **Network Isolation**: No network access by default (`network_mode: none`)
- **Read-only Root Filesystem**: Prevents modification of system files
- **No New Privileges**: Prevents privilege escalation
- **Resource Limits**: CPU, memory, and disk usage constraints
- **Code Validation**: Static analysis to detect dangerous patterns
- **Command Filtering**: Blocks dangerous terminal commands

### Supported Languages
- **Python**: Execute Python scripts with validation
- **Bash**: Execute shell commands with security filtering

### File System Access
- **James Directory**: Controlled access to `~/.james` directory for persistent storage
- **Temporary Storage**: Limited temporary directory with size constraints
- **Path Validation**: Prevents access to sensitive system directories

## Usage

### Basic Code Execution

```python
from backend.sandbox import SecureSandbox, SandboxConfig

# Create sandbox with default configuration
sandbox = SecureSandbox()

# Execute safe Python code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""

result = await sandbox.execute_code(code, "python")

if result.success:
    print(f"Output: {result.stdout}")
    print(f"Execution time: {result.execution_time:.2f}s")
else:
    print(f"Error: {result.stderr}")
```

### Terminal Command Execution

```python
# Execute safe terminal commands
result = await sandbox.execute_terminal_command("ls -la")

if result.success:
    print(f"Directory listing: {result.stdout}")
```

### James Directory Access

```python
# Allow file operations for James directory access
james_code = """
import os
import json

# Create directory in James folder
os.makedirs('/james/data', exist_ok=True)

# Write data
data = {"message": "Hello from sandbox!"}
with open('/james/data/test.json', 'w') as f:
    json.dump(data, f)

print("Data written to James directory")
"""

result = await sandbox.execute_code(james_code, "python", allow_file_ops=True)
```

### Custom Configuration

```python
# Create sandbox with custom limits
config = SandboxConfig(
    memory_limit="256m",     # Lower memory limit
    cpu_limit=0.3,          # Limit to 30% of one CPU core
    timeout_seconds=10,     # Shorter timeout
    base_image="python:3.12-slim"  # Different base image
)

sandbox = SecureSandbox(config)
```

## Configuration Options

### SandboxConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_limit` | `"512m"` | Memory limit for container |
| `cpu_limit` | `0.5` | CPU limit (fraction of one core) |
| `timeout_seconds` | `30` | Maximum execution time |
| `network_mode` | `"none"` | Network access mode |
| `readonly_rootfs` | `True` | Read-only root filesystem |
| `no_new_privileges` | `True` | Prevent privilege escalation |
| `james_dir` | `"~/.james"` | James directory path |
| `temp_dir_size` | `"100m"` | Temporary directory size limit |
| `base_image` | `"python:3.11-slim"` | Docker base image |
| `remove_container` | `True` | Remove container after execution |

## Security Validation

### Dangerous Code Patterns (Always Blocked)
- `import subprocess` - Process execution
- `import sys` - System access
- `__import__` - Dynamic imports
- `eval()` - Code evaluation
- `exec()` - Code execution
- `input()` - User input
- `compile()` - Code compilation
- `globals()` - Global namespace access
- `locals()` - Local namespace access

### File Operation Patterns (Conditionally Allowed)
- `import os` - Operating system interface
- `open()` - File operations
- `file()` - File objects

### Network Patterns (Always Blocked)
- `import socket` - Network sockets
- `import urllib` - URL handling
- `import requests` - HTTP requests
- `import http` - HTTP modules

### Dangerous Terminal Commands
- `rm -rf` - Recursive deletion
- `sudo` - Privilege escalation
- `wget`/`curl` - Network downloads
- `ssh`/`scp` - Remote access
- `docker` - Container operations
- `systemctl` - System services
- `mount`/`umount` - Filesystem mounting

## Error Handling

### Exception Types

- `SandboxError` - Base sandbox exception
- `SecurityViolationError` - Security policy violation
- `ResourceLimitError` - Resource limit exceeded
- `SandboxTimeoutError` - Execution timeout
- `ContainerCreationError` - Docker container creation failed
- `CodeExecutionError` - Code execution failed

### Example Error Handling

```python
from backend.sandbox import (
    SecureSandbox, 
    SecurityViolationError, 
    SandboxTimeoutError
)

try:
    result = await sandbox.execute_code(dangerous_code, "python")
except SecurityViolationError as e:
    print(f"Security violation: {e}")
except SandboxTimeoutError as e:
    print(f"Execution timed out: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Resource Monitoring

### ExecutionResult Fields

```python
@dataclass
class ExecutionResult:
    success: bool                           # Execution success status
    stdout: str                            # Standard output
    stderr: str                            # Standard error
    exit_code: int                         # Process exit code
    execution_time: float                  # Execution time in seconds
    resource_usage: Dict[str, Union[str, int, float]]  # Resource usage stats
    security_violations: List[str]         # Security violations detected
```

### Resource Usage Metrics

- `memory_usage` - Memory used in bytes
- `memory_limit` - Memory limit in bytes
- `cpu_usage` - CPU time used
- `system_cpu_usage` - System CPU time

## Testing

### Unit Tests
```bash
# Run all sandbox tests
python -m pytest tests/test_sandbox.py -v

# Run only unit tests (no Docker required)
python -m pytest tests/test_sandbox.py -v -k "not integration"

# Run security tests
python -m pytest tests/test_sandbox.py -v -k "security"
```

### Smoke Tests
```bash
# Run smoke tests (no Docker required)
python test_sandbox_smoke.py
```

### Integration Tests
```bash
# Run integration tests (requires Docker)
python -m pytest tests/test_sandbox.py -v -k "integration"
```

### Demo Script
```bash
# Run comprehensive demo (requires Docker)
python examples/sandbox_demo.py
```

## Docker Setup

### Prerequisites
1. Install Docker Desktop or Docker Engine
2. Ensure Docker daemon is running
3. Pull the base image: `docker pull python:3.11-slim`

### Custom Base Image (Optional)
```bash
# Build custom secure base image
docker build -f backend/sandbox/Dockerfile.sandbox -t secure-sandbox .

# Use custom image in configuration
config = SandboxConfig(base_image="secure-sandbox")
```

## Best Practices

### Security
1. Always validate user input before execution
2. Use the most restrictive configuration possible
3. Monitor resource usage and execution time
4. Log all security violations
5. Regularly update base images

### Performance
1. Reuse sandbox instances when possible
2. Set appropriate resource limits
3. Use shorter timeouts for simple operations
4. Clean up containers regularly

### Error Handling
1. Always handle SecurityViolationError
2. Implement retry logic for transient failures
3. Log execution results for debugging
4. Provide meaningful error messages to users

## Troubleshooting

### Common Issues

**Docker not available**
```
SandboxError: Docker initialization failed
```
- Ensure Docker is installed and running
- Check Docker daemon status
- Verify user has Docker permissions

**Container creation failed**
```
ContainerCreationError: Docker image not found
```
- Pull the required base image
- Check image name in configuration
- Verify Docker registry access

**Security violations**
```
SecurityViolationError: Dangerous pattern detected
```
- Review code for dangerous patterns
- Use `allow_file_ops=True` for James directory access
- Check terminal commands for dangerous operations

**Resource limits exceeded**
```
ResourceLimitError: Memory limit exceeded
```
- Increase memory limit in configuration
- Optimize code for lower resource usage
- Check for memory leaks in code

**Execution timeout**
```
SandboxTimeoutError: Execution timed out
```
- Increase timeout in configuration
- Optimize code for faster execution
- Check for infinite loops

## Architecture

The sandbox system consists of several layers:

1. **API Layer**: `SecureSandbox` class providing the main interface
2. **Validation Layer**: Security validation for code and commands
3. **Container Layer**: Docker container management and execution
4. **Resource Layer**: Resource monitoring and limit enforcement
5. **File System Layer**: Controlled file system access

This layered approach ensures security, performance, and maintainability while providing a simple API for code execution.
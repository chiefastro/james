"""
Secure sandbox environment for code execution.

This module provides a Docker-based sandbox for safely executing
arbitrary code and terminal commands with resource limits and
security constraints.
"""

from .sandbox import SecureSandbox, SandboxConfig, ExecutionResult
from .exceptions import (
    SandboxError, 
    SecurityViolationError, 
    ResourceLimitError,
    SandboxTimeoutError,
    ContainerCreationError,
    CodeExecutionError
)

__all__ = [
    "SecureSandbox",
    "SandboxConfig", 
    "ExecutionResult",
    "SandboxError",
    "SecurityViolationError",
    "ResourceLimitError",
    "SandboxTimeoutError",
    "ContainerCreationError",
    "CodeExecutionError"
]
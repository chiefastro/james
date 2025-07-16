"""
Sandbox-specific exceptions.
"""


class SandboxError(Exception):
    """Base exception for sandbox-related errors."""
    pass


class SecurityViolationError(SandboxError):
    """Raised when a security violation is detected in the sandbox."""
    pass


class ResourceLimitError(SandboxError):
    """Raised when resource limits are exceeded."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when sandbox execution times out."""
    pass


class ContainerCreationError(SandboxError):
    """Raised when Docker container creation fails."""
    pass


class CodeExecutionError(SandboxError):
    """Raised when code execution fails within the sandbox."""
    pass
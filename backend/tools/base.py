"""
Base classes for seed tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """
    Base class for all seed tools.
    
    Provides common functionality for tool execution, error handling,
    and result formatting.
    """
    
    def __init__(self, name: str):
        """Initialize the tool with a name."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    def _create_success_result(self, data: Any = None, metadata: Dict[str, Any] = None) -> ToolResult:
        """Create a successful tool result."""
        return ToolResult(
            success=True,
            data=data,
            metadata=metadata or {}
        )
    
    def _create_error_result(self, error: str, metadata: Dict[str, Any] = None) -> ToolResult:
        """Create an error tool result."""
        self.logger.error(f"Tool {self.name} failed: {error}")
        return ToolResult(
            success=False,
            error=error,
            metadata=metadata or {}
        )
    
    def _validate_required_params(self, params: Dict[str, Any], required: list) -> Optional[str]:
        """
        Validate that required parameters are present.
        
        Args:
            params: Parameters to validate
            required: List of required parameter names
            
        Returns:
            Error message if validation fails, None if successful
        """
        missing = [param for param in required if param not in params or params[param] is None]
        if missing:
            return f"Missing required parameters: {', '.join(missing)}"
        return None
"""
LangSmith integration for agent trace monitoring.

This module provides comprehensive tracing capabilities for the conscious agent system,
including automatic trace collection, custom metrics, and performance monitoring.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from functools import wraps
from dataclasses import dataclass, field
import asyncio

from langsmith import Client, traceable
from langsmith.schemas import Run, Example
from langchain_core.tracers import LangChainTracer

logger = logging.getLogger(__name__)


@dataclass
class TraceConfig:
    """Configuration for LangSmith tracing."""
    project_name: str = "conscious-agent-system"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    enabled: bool = True
    sample_rate: float = 1.0  # Fraction of traces to collect (0.0 to 1.0)
    include_inputs: bool = True
    include_outputs: bool = True
    include_metadata: bool = True
    max_trace_depth: int = 10


@dataclass
class AgentTrace:
    """Represents a single agent operation trace."""
    trace_id: str
    operation_name: str
    agent_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True
    parent_trace_id: Optional[str] = None
    child_traces: List[str] = field(default_factory=list)


class LangSmithTracer:
    """
    LangSmith tracer for comprehensive agent monitoring.
    
    Provides automatic tracing of agent operations, custom metrics collection,
    and integration with LangSmith's observability platform.
    """
    
    def __init__(self, config: Optional[TraceConfig] = None):
        """
        Initialize the LangSmith tracer.
        
        Args:
            config: Optional trace configuration
        """
        self.config = config or TraceConfig()
        self._client: Optional[Client] = None
        self._tracer: Optional[LangChainTracer] = None
        self._active_traces: Dict[str, AgentTrace] = {}
        self._trace_count = 0
        
        # Initialize LangSmith client if enabled
        if self.config.enabled:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the LangSmith client with configuration."""
        try:
            # Set up environment variables for LangSmith
            if self.config.api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.config.api_key
            if self.config.endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = self.config.endpoint
            
            # Set project name
            os.environ["LANGCHAIN_PROJECT"] = self.config.project_name
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
            # Initialize client
            self._client = Client(
                api_key=self.config.api_key,
                api_url=self.config.endpoint
            )
            
            # Initialize tracer
            self._tracer = LangChainTracer(
                project_name=self.config.project_name,
                client=self._client
            )
            
            logger.info(f"LangSmith tracer initialized for project: {self.config.project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.config.enabled = False
    
    def should_trace(self) -> bool:
        """Determine if current operation should be traced based on sample rate."""
        if not self.config.enabled:
            return False
        
        import random
        return random.random() < self.config.sample_rate
    
    async def start_trace(
        self,
        operation_name: str,
        agent_type: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_trace_id: Optional[str] = None
    ) -> str:
        """
        Start a new trace for an agent operation.
        
        Args:
            operation_name: Name of the operation being traced
            agent_type: Type of agent performing the operation
            inputs: Input parameters for the operation
            metadata: Additional metadata for the trace
            parent_trace_id: ID of parent trace if this is a child operation
            
        Returns:
            Unique trace ID for this operation
        """
        if not self.should_trace():
            return ""
        
        from uuid import uuid4
        trace_id = str(uuid4())
        
        trace = AgentTrace(
            trace_id=trace_id,
            operation_name=operation_name,
            agent_type=agent_type,
            start_time=datetime.now(timezone.utc),
            inputs=inputs or {},
            metadata=metadata or {},
            parent_trace_id=parent_trace_id
        )
        
        # Add system metadata
        trace.metadata.update({
            "trace_id": trace_id,
            "agent_system": "conscious-agent-system",
            "trace_version": "1.0",
            "sample_rate": self.config.sample_rate
        })
        
        self._active_traces[trace_id] = trace
        self._trace_count += 1
        
        logger.debug(f"Started trace {trace_id} for {agent_type}.{operation_name}")
        return trace_id
    
    async def end_trace(
        self,
        trace_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        success: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentTrace]:
        """
        End an active trace and submit it to LangSmith.
        
        Args:
            trace_id: ID of the trace to end
            outputs: Output results from the operation
            error: Error message if operation failed
            success: Whether the operation was successful
            additional_metadata: Additional metadata to add
            
        Returns:
            Completed AgentTrace object or None if trace not found
        """
        if not trace_id or trace_id not in self._active_traces:
            return None
        
        trace = self._active_traces[trace_id]
        trace.end_time = datetime.now(timezone.utc)
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
        trace.outputs = outputs or {}
        trace.error = error
        trace.success = success
        
        if additional_metadata:
            trace.metadata.update(additional_metadata)
        
        # Submit to LangSmith if client is available
        if self._client and self.config.enabled:
            try:
                await self._submit_trace_to_langsmith(trace)
            except Exception as e:
                logger.error(f"Failed to submit trace {trace_id} to LangSmith: {e}")
        
        # Remove from active traces
        completed_trace = self._active_traces.pop(trace_id)
        
        logger.debug(
            f"Completed trace {trace_id}: {trace.operation_name} "
            f"({trace.duration_ms:.2f}ms, success={success})"
        )
        
        return completed_trace
    
    async def _submit_trace_to_langsmith(self, trace: AgentTrace) -> None:
        """Submit a completed trace to LangSmith."""
        try:
            # Create run data for LangSmith
            run_data = {
                "name": f"{trace.agent_type}.{trace.operation_name}",
                "run_type": "chain",  # or "llm", "tool", etc.
                "inputs": trace.inputs if self.config.include_inputs else {},
                "outputs": trace.outputs if self.config.include_outputs else {},
                "start_time": trace.start_time,
                "end_time": trace.end_time,
                "extra": trace.metadata if self.config.include_metadata else {},
                "error": trace.error,
                "tags": [
                    trace.agent_type,
                    trace.operation_name,
                    "conscious-agent-system"
                ]
            }
            
            # Submit to LangSmith
            run = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.create_run(**run_data)
            )
            
            logger.debug(f"Submitted trace {trace.trace_id} to LangSmith as run {run.id}")
            
        except Exception as e:
            logger.error(f"Failed to submit trace to LangSmith: {e}")
            raise
    
    def get_active_traces(self) -> List[AgentTrace]:
        """Get list of currently active traces."""
        return list(self._active_traces.values())
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracing operations."""
        active_count = len(self._active_traces)
        
        return {
            "total_traces_started": self._trace_count,
            "active_traces": active_count,
            "config": {
                "enabled": self.config.enabled,
                "project_name": self.config.project_name,
                "sample_rate": self.config.sample_rate
            },
            "client_status": "connected" if self._client else "disconnected",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def flush_traces(self) -> None:
        """Flush any remaining active traces."""
        if not self._active_traces:
            return
        
        logger.info(f"Flushing {len(self._active_traces)} active traces")
        
        for trace_id in list(self._active_traces.keys()):
            await self.end_trace(
                trace_id=trace_id,
                error="Trace flushed during shutdown",
                success=False
            )


# Global tracer instance
_global_tracer: Optional[LangSmithTracer] = None


def get_tracer() -> LangSmithTracer:
    """Get the global LangSmith tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = LangSmithTracer()
    return _global_tracer


def configure_tracer(config: TraceConfig) -> None:
    """Configure the global tracer with new settings."""
    global _global_tracer
    _global_tracer = LangSmithTracer(config)


def trace_agent_operation(
    operation_name: Optional[str] = None,
    agent_type: Optional[str] = None,
    include_args: bool = True,
    include_result: bool = True
):
    """
    Decorator for automatic tracing of agent operations.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        agent_type: Type of agent (defaults to class name)
        include_args: Whether to include function arguments in trace
        include_result: Whether to include function result in trace
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Determine operation and agent names
            op_name = operation_name or func.__name__
            agent_name = agent_type
            
            # Try to get agent type from class if not provided
            if not agent_name and args and hasattr(args[0], '__class__'):
                agent_name = args[0].__class__.__name__
            
            agent_name = agent_name or "unknown"
            
            # Prepare inputs
            inputs = {}
            if include_args:
                inputs = {
                    "args": [str(arg)[:200] for arg in args[1:]],  # Skip self, limit length
                    "kwargs": {k: str(v)[:200] for k, v in kwargs.items()}
                }
            
            # Start trace
            trace_id = await tracer.start_trace(
                operation_name=op_name,
                agent_type=agent_name,
                inputs=inputs
            )
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Prepare outputs
                outputs = {}
                if include_result and result is not None:
                    outputs["result"] = str(result)[:500]  # Limit length
                
                # End trace successfully
                await tracer.end_trace(
                    trace_id=trace_id,
                    outputs=outputs,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # End trace with error
                await tracer.end_trace(
                    trace_id=trace_id,
                    error=str(e),
                    success=False
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we need to handle tracing differently
            # This is a simplified version - in practice, you might want to
            # run the async tracing in a separate thread or use sync tracing
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in traced function {func.__name__}: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience decorators for common agent types
def trace_observer_operation(operation_name: Optional[str] = None):
    """Decorator for tracing Observer agent operations."""
    return trace_agent_operation(operation_name=operation_name, agent_type="Observer")


def trace_delegator_operation(operation_name: Optional[str] = None):
    """Decorator for tracing Delegator agent operations."""
    return trace_agent_operation(operation_name=operation_name, agent_type="Delegator")


def trace_master_graph_operation(operation_name: Optional[str] = None):
    """Decorator for tracing Master Graph operations."""
    return trace_agent_operation(operation_name=operation_name, agent_type="MasterGraph")
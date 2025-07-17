"""
Circuit breaker pattern implementation for external service calls.

This module provides a circuit breaker implementation to prevent cascading failures
when external services are experiencing issues. It automatically detects failures
and temporarily stops attempts to call the service when it's unavailable.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, cast
from functools import wraps

logger = logging.getLogger(__name__)

# Type for function return value
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service is back online


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5           # Number of failures before opening circuit
    recovery_timeout: float = 30.0       # Seconds to wait before trying again (half-open)
    reset_timeout: float = 60.0          # Seconds after which to reset failure count
    excluded_exceptions: Set[type] = field(default_factory=set)  # Exceptions that don't count as failures
    included_exceptions: Set[type] = field(default_factory=lambda: {Exception})  # Exceptions that count as failures
    half_open_max_calls: int = 1         # Max calls to allow in half-open state


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    open_circuits_count: int = 0
    total_state_changes: int = 0
    state_change_history: List[Dict[str, Any]] = field(default_factory=list)
    current_state: CircuitState = CircuitState.CLOSED
    
    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(timezone.utc)
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now(timezone.utc)
    
    def record_state_change(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Record a state change."""
        self.total_state_changes += 1
        self.current_state = to_state
        
        if to_state == CircuitState.OPEN:
            self.open_circuits_count += 1
        
        self.state_change_history.append({
            "from_state": from_state.value,
            "to_state": to_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "consecutive_failures": self.consecutive_failures
        })
        
        # Keep history size manageable
        if len(self.state_change_history) > 100:
            self.state_change_history = self.state_change_history[-100:]


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.
    
    The circuit breaker monitors for failures in external service calls and
    temporarily blocks requests when the service appears to be unavailable.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker (typically service name)
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Exception: Original exception if call fails and circuit remains closed
        """
        async with self._lock:
            await self._check_state_transition()
            
            if self.state == CircuitState.OPEN:
                logger.warning(f"Circuit {self.name} is OPEN - request blocked")
                raise CircuitOpenError(f"Circuit {self.name} is open")
            
            if self.state == CircuitState.HALF_OPEN and self.half_open_calls >= self.config.half_open_max_calls:
                logger.warning(f"Circuit {self.name} is HALF_OPEN with max calls reached - request blocked")
                raise CircuitOpenError(f"Circuit {self.name} is half-open with max calls reached")
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
        
        # Execute function outside the lock
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Handle success
            async with self._lock:
                self.metrics.record_success()
                
                if self.state == CircuitState.HALF_OPEN:
                    logger.info(f"Circuit {self.name} recovery successful - closing circuit")
                    await self._transition_state(CircuitState.CLOSED)
            
            return result
            
        except Exception as e:
            # Check if this exception should count as a failure
            is_counted_failure = self._is_counted_exception(e)
            
            async with self._lock:
                if is_counted_failure:
                    self.metrics.record_failure()
                    self.last_failure_time = datetime.now(timezone.utc)
                    
                    # Check if we need to open the circuit
                    if (self.state == CircuitState.CLOSED and 
                        self.metrics.consecutive_failures >= self.config.failure_threshold):
                        logger.warning(
                            f"Circuit {self.name} failure threshold reached "
                            f"({self.metrics.consecutive_failures} failures) - opening circuit"
                        )
                        await self._transition_state(CircuitState.OPEN)
                    
                    # If half-open test fails, go back to open
                    elif self.state == CircuitState.HALF_OPEN:
                        logger.warning(f"Circuit {self.name} recovery failed - reopening circuit")
                        await self._transition_state(CircuitState.OPEN)
            
            # Re-raise the exception
            raise
    
    async def _check_state_transition(self) -> None:
        """Check if state transition is needed based on timing."""
        if (self.state == CircuitState.OPEN and 
            self.last_failure_time and 
            datetime.now(timezone.utc) - self.last_failure_time > 
            timedelta(seconds=self.config.recovery_timeout)):
            
            logger.info(f"Circuit {self.name} recovery timeout elapsed - transitioning to HALF_OPEN")
            await self._transition_state(CircuitState.HALF_OPEN)
            self.half_open_calls = 0
    
    async def _transition_state(self, new_state: CircuitState) -> None:
        """Transition to a new circuit state."""
        old_state = self.state
        self.state = new_state
        self.metrics.record_state_change(old_state, new_state)
        
        if new_state == CircuitState.CLOSED:
            # Reset failure count when closing the circuit
            self.metrics.consecutive_failures = 0
    
    def _is_counted_exception(self, exception: Exception) -> bool:
        """Check if an exception should count toward the failure threshold."""
        # If it's in the excluded list, don't count it
        if any(isinstance(exception, exc_type) for exc_type in self.config.excluded_exceptions):
            return False
        
        # If it's in the included list, count it
        if any(isinstance(exception, exc_type) for exc_type in self.config.included_exceptions):
            return True
        
        # Default to not counting if it doesn't match either list
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "consecutive_failures": self.metrics.consecutive_failures,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "open_circuits_count": self.metrics.open_circuits_count,
            "total_state_changes": self.metrics.total_state_changes,
            "recent_state_changes": self.metrics.state_change_history[-5:] if self.metrics.state_change_history else []
        }
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.metrics.consecutive_failures = 0
        self.half_open_calls = 0
        logger.info(f"Circuit {self.name} manually reset to CLOSED state")


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open."""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    _instance = None
    _circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    @classmethod
    def get_instance(cls) -> 'CircuitBreakerRegistry':
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = CircuitBreakerRegistry()
        return cls._instance
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]
    
    def get_all_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        return self._circuit_breakers.copy()
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._circuit_breakers.values():
            cb.reset()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get a circuit breaker by name, creating it if it doesn't exist.
    
    Args:
        name: Name of the circuit breaker
        config: Optional configuration for new circuit breakers
        
    Returns:
        CircuitBreaker instance
    """
    registry = CircuitBreakerRegistry.get_instance()
    return registry.get_or_create(name, config)


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        name: Name of the circuit breaker
        config: Optional configuration for the circuit breaker
        
    Returns:
        Decorated function
    """
    def decorator(func):
        cb = get_circuit_breaker(name, config)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await cb.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we need to run the async execute in an event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cb.execute(func, *args, **kwargs))
        
        # Choose the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
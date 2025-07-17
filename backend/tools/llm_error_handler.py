"""
LLM API error handling and recovery system.

This module provides specialized error handling for LLM API calls,
including retry logic, fallback mechanisms, and error classification.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, cast
from functools import wraps

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
from .error_handler import ErrorHandlerTool, ErrorSeverity, RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)

# Type for function return value
T = TypeVar('T')


class LLMErrorType(Enum):
    """Types of LLM API errors."""
    RATE_LIMIT = "rate_limit"
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class LLMProviderType(Enum):
    """Types of LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class LLMErrorInfo:
    """Information about an LLM API error."""
    error_type: LLMErrorType
    provider: LLMProviderType
    model: str
    error_message: str
    status_code: Optional[int] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


class LLMErrorHandler:
    """
    Specialized error handler for LLM API calls.
    
    Provides error classification, retry logic with exponential backoff,
    fallback to alternative models, and error reporting.
    """
    
    def __init__(self):
        """Initialize the LLM error handler."""
        self.error_handler = ErrorHandlerTool()
        self.error_history: List[LLMErrorInfo] = []
        self.max_history_size = 100
        
        # Configure circuit breakers for different providers
        self.circuit_breakers = {
            LLMProviderType.OPENAI: get_circuit_breaker("openai-llm", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)),
            LLMProviderType.ANTHROPIC: get_circuit_breaker("anthropic-llm", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)),
            LLMProviderType.GOOGLE: get_circuit_breaker("google-llm", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)),
            LLMProviderType.AZURE: get_circuit_breaker("azure-llm", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)),
            LLMProviderType.HUGGINGFACE: get_circuit_breaker("huggingface-llm", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)),
        }
    
    async def handle_with_retry(self, 
                               func: Callable[..., T], 
                               provider: LLMProviderType,
                               model: str,
                               *args: Any,
                               max_retries: int = 3,
                               retry_delay: float = 1.0,
                               fallback_models: Optional[List[str]] = None,
                               **kwargs: Any) -> T:
        """
        Execute an LLM API call with retry logic and fallback.
        
        Args:
            func: LLM API function to call
            provider: LLM provider type
            model: Model name
            *args: Function arguments
            max_retries: Maximum number of retries
            retry_delay: Base delay for retries (will be multiplied for backoff)
            fallback_models: List of models to try if the primary model fails
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries and fallbacks fail
        """
        # Get the circuit breaker for this provider
        circuit_breaker = self.circuit_breakers.get(provider)
        if not circuit_breaker:
            # Use a default circuit breaker for unknown providers
            circuit_breaker = get_circuit_breaker(f"{provider.value}-llm")
        
        # Try with the primary model
        try:
            return await self._execute_with_circuit_breaker(
                circuit_breaker, func, provider, model, *args, 
                max_retries=max_retries, retry_delay=retry_delay, **kwargs
            )
        except Exception as primary_error:
            logger.warning(f"Primary model {model} failed: {primary_error}")
            
            # Try fallback models if available
            if fallback_models:
                for fallback_model in fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        # Replace the model in kwargs if it exists there
                        if 'model' in kwargs:
                            kwargs['model'] = fallback_model
                        
                        return await self._execute_with_circuit_breaker(
                            circuit_breaker, func, provider, fallback_model, *args, 
                            max_retries=1, retry_delay=retry_delay, **kwargs
                        )
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
                        continue
            
            # If we get here, all models failed
            logger.error(f"All models failed for {provider.value} call")
            raise primary_error
    
    async def _execute_with_circuit_breaker(self,
                                          circuit_breaker: CircuitBreaker,
                                          func: Callable[..., T],
                                          provider: LLMProviderType,
                                          model: str,
                                          *args: Any,
                                          max_retries: int = 3,
                                          retry_delay: float = 1.0,
                                          **kwargs: Any) -> T:
        """Execute with circuit breaker and retry logic."""
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Use circuit breaker to protect against cascading failures
                return await circuit_breaker.execute(func, *args, **kwargs)
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                # Classify the error
                error_info = self._classify_llm_error(e, provider, model)
                error_info.retry_count = retry_count
                self._add_to_history(error_info)
                
                # Check if we should retry based on error type
                if not self._should_retry(error_info.error_type) or retry_count > max_retries:
                    logger.error(f"LLM error not retryable or max retries reached: {error_info.error_type}")
                    raise
                
                # Calculate backoff delay
                backoff_delay = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                
                # Add jitter to prevent thundering herd
                import random
                jitter = random.uniform(0, 0.1 * backoff_delay)
                total_delay = backoff_delay + jitter
                
                logger.info(f"Retrying LLM call after {total_delay:.2f}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(total_delay)
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        
        # This should never happen, but to satisfy type checking
        raise RuntimeError("Unexpected error in LLM retry logic")
    
    def _classify_llm_error(self, error: Exception, provider: LLMProviderType, model: str) -> LLMErrorInfo:
        """Classify an LLM API error by type."""
        error_message = str(error).lower()
        error_type = LLMErrorType.UNKNOWN
        status_code = None
        request_id = None
        
        # Extract status code if available
        if hasattr(error, 'status_code'):
            status_code = getattr(error, 'status_code')
        
        # Extract request ID if available
        if hasattr(error, 'request_id'):
            request_id = getattr(error, 'request_id')
        
        # Classify based on error message and status code
        if 'rate limit' in error_message or 'too many requests' in error_message or status_code == 429:
            error_type = LLMErrorType.RATE_LIMIT
        elif 'context length' in error_message or 'maximum context' in error_message:
            error_type = LLMErrorType.CONTEXT_LENGTH
        elif 'content filter' in error_message or 'content policy' in error_message:
            error_type = LLMErrorType.CONTENT_FILTER
        elif 'invalid request' in error_message or 'bad request' in error_message or status_code == 400:
            error_type = LLMErrorType.INVALID_REQUEST
        elif 'authentication' in error_message or 'unauthorized' in error_message or status_code in (401, 403):
            error_type = LLMErrorType.AUTHENTICATION
        elif 'server error' in error_message or status_code in (500, 502, 503, 504):
            error_type = LLMErrorType.SERVER
        elif 'timeout' in error_message or 'timed out' in error_message:
            error_type = LLMErrorType.TIMEOUT
        elif 'connection' in error_message or 'network' in error_message:
            error_type = LLMErrorType.CONNECTION
        
        return LLMErrorInfo(
            error_type=error_type,
            provider=provider,
            model=model,
            error_message=str(error),
            status_code=status_code,
            request_id=request_id
        )
    
    def _should_retry(self, error_type: LLMErrorType) -> bool:
        """Determine if an error type should be retried."""
        # These error types are typically transient and can be retried
        retryable_types = {
            LLMErrorType.RATE_LIMIT,
            LLMErrorType.SERVER,
            LLMErrorType.TIMEOUT,
            LLMErrorType.CONNECTION
        }
        
        return error_type in retryable_types
    
    def _add_to_history(self, error_info: LLMErrorInfo) -> None:
        """Add error to history with size management."""
        self.error_history.append(error_info)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about LLM errors."""
        if not self.error_history:
            return {
                "total_errors": 0,
                "error_types": {},
                "providers": {},
                "models": {}
            }
        
        # Count errors by type, provider, and model
        error_types = {}
        providers = {}
        models = {}
        
        for error in self.error_history:
            # Count by error type
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count by provider
            provider = error.provider.value
            providers[provider] = providers.get(provider, 0) + 1
            
            # Count by model
            model = error.model
            models[model] = models.get(model, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "providers": providers,
            "models": models,
            "recent_errors": [
                {
                    "error_type": e.error_type.value,
                    "provider": e.provider.value,
                    "model": e.model,
                    "message": e.error_message,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in self.error_history[-5:]  # Last 5 errors
            ]
        }


# Singleton instance
_llm_error_handler = None


def get_llm_error_handler() -> LLMErrorHandler:
    """
    Get the singleton LLM error handler instance.
    
    Returns:
        LLMErrorHandler instance
    """
    global _llm_error_handler
    if _llm_error_handler is None:
        _llm_error_handler = LLMErrorHandler()
    return _llm_error_handler


def with_llm_error_handling(provider: Union[LLMProviderType, str], model: str, 
                           max_retries: int = 3, retry_delay: float = 1.0,
                           fallback_models: Optional[List[str]] = None):
    """
    Decorator for handling LLM API errors with retry and fallback.
    
    Args:
        provider: LLM provider type or string name
        model: Model name
        max_retries: Maximum number of retries
        retry_delay: Base delay for retries
        fallback_models: List of models to try if the primary model fails
        
    Returns:
        Decorated function
    """
    # Convert string to enum if needed
    if isinstance(provider, str):
        try:
            provider = LLMProviderType(provider)
        except ValueError:
            provider = LLMProviderType.CUSTOM
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = get_llm_error_handler()
            return await handler.handle_with_retry(
                func, provider, model, *args,
                max_retries=max_retries,
                retry_delay=retry_delay,
                fallback_models=fallback_models,
                **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we need to run the async handle_with_retry in an event loop
            handler = get_llm_error_handler()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(handler.handle_with_retry(
                func, provider, model, *args,
                max_retries=max_retries,
                retry_delay=retry_delay,
                fallback_models=fallback_models,
                **kwargs
            ))
        
        # Choose the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
"""
Error inspection and retry mechanisms tool.
"""

import time
import traceback
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .base import BaseTool, ToolResult


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategy types."""
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class ErrorInfo:
    """Information about an error."""
    error_type: str
    error_message: str
    traceback: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolved: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[type] = field(default_factory=list)


class ErrorHandlerTool(BaseTool):
    """
    Tool for error inspection, handling, and retry mechanisms.
    
    Provides capabilities to analyze errors, implement retry logic,
    and maintain error history for debugging and improvement.
    """
    
    def __init__(self):
        """Initialize the error handler tool."""
        super().__init__("ErrorHandler")
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 1000
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute error handling operation.
        
        Args:
            action (str): Action to perform - 'inspect', 'retry', 'history', 'clear_history'
            error (Exception): Error to inspect (for 'inspect' action)
            function (callable): Function to retry (for 'retry' action)
            retry_config (dict): Retry configuration (for 'retry' action)
            context (dict): Additional context information
            
        Returns:
            ToolResult with operation outcome
        """
        start_time = time.time()
        
        # Validate required parameters
        error = self._validate_required_params(kwargs, ['action'])
        if error:
            return self._create_error_result(error)
        
        action = kwargs['action'].lower()
        
        try:
            if action == 'inspect':
                return await self._inspect_error(kwargs, start_time)
            elif action == 'retry':
                return await self._retry_with_backoff(kwargs, start_time)
            elif action == 'history':
                return await self._get_error_history(kwargs, start_time)
            elif action == 'clear_history':
                return await self._clear_error_history(start_time)
            elif action == 'analyze':
                return await self._analyze_errors(kwargs, start_time)
            else:
                return self._create_error_result(f"Unsupported action: {action}")
                
        except Exception as e:
            return self._create_error_result(f"Unexpected error in error handler: {e}")
    
    async def _inspect_error(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Inspect and categorize an error."""
        error = kwargs.get('error')
        context = kwargs.get('context', {})
        
        if not error:
            return self._create_error_result("No error provided for inspection")
        
        try:
            # Create error info
            error_info = ErrorInfo(
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exception(type(error), error, error.__traceback__),
                severity=self._determine_severity(error),
                context=context
            )
            
            # Add to history
            self._add_to_history(error_info)
            
            # Analyze error
            analysis = self._analyze_single_error(error_info)
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'error_type': error_info.error_type,
                    'error_message': error_info.error_message,
                    'severity': error_info.severity.value,
                    'timestamp': error_info.timestamp.isoformat(),
                    'analysis': analysis,
                    'suggestions': self._get_error_suggestions(error_info)
                },
                metadata={
                    'execution_time': execution_time,
                    'context': context
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Error inspecting error: {e}")
    
    async def _retry_with_backoff(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Retry a function with configurable backoff strategy."""
        function = kwargs.get('function')
        args = kwargs.get('args', ())
        kwargs_func = kwargs.get('kwargs', {})
        retry_config_dict = kwargs.get('retry_config', {})
        
        if not function:
            return self._create_error_result("No function provided for retry")
        
        # Create retry config
        retry_config = RetryConfig(**retry_config_dict)
        
        attempt = 0
        last_error = None
        delay = retry_config.base_delay
        
        while attempt < retry_config.max_attempts:
            try:
                # Execute function
                if asyncio.iscoroutinefunction(function):
                    result = await function(*args, **kwargs_func)
                else:
                    result = function(*args, **kwargs_func)
                
                execution_time = time.time() - start_time
                
                # Success
                return self._create_success_result(
                    data={
                        'result': result,
                        'attempts': attempt + 1,
                        'success': True
                    },
                    metadata={
                        'execution_time': execution_time,
                        'retry_config': retry_config_dict
                    }
                )
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                # Check if we should stop retrying
                if any(isinstance(e, exc_type) for exc_type in retry_config.stop_on_exceptions):
                    break
                
                # Check if we should retry this exception
                if not any(isinstance(e, exc_type) for exc_type in retry_config.retry_on_exceptions):
                    break
                
                # Log the attempt
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                
                # If not the last attempt, wait before retrying
                if attempt < retry_config.max_attempts:
                    await asyncio.sleep(delay)
                    
                    # Calculate next delay based on strategy
                    if retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
                        delay += retry_config.base_delay
                    elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        delay *= retry_config.backoff_multiplier
                    elif retry_config.strategy == RetryStrategy.FIXED_DELAY:
                        delay = retry_config.base_delay
                    # IMMEDIATE strategy keeps delay at 0
                    
                    # Cap the delay
                    delay = min(delay, retry_config.max_delay)
        
        # All attempts failed
        execution_time = time.time() - start_time
        
        # Record the final error
        if last_error:
            error_info = ErrorInfo(
                error_type=type(last_error).__name__,
                error_message=str(last_error),
                traceback=traceback.format_exception(type(last_error), last_error, last_error.__traceback__),
                retry_count=attempt,
                context={'retry_config': retry_config_dict}
            )
            self._add_to_history(error_info)
        
        return self._create_error_result(
            f"All {attempt} retry attempts failed. Last error: {last_error}",
            metadata={
                'execution_time': execution_time,
                'attempts': attempt,
                'last_error_type': type(last_error).__name__ if last_error else None,
                'retry_config': retry_config_dict
            }
        )
    
    async def _get_error_history(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Get error history with optional filtering."""
        limit = kwargs.get('limit', 50)
        severity_filter = kwargs.get('severity')
        error_type_filter = kwargs.get('error_type')
        
        try:
            # Filter errors
            filtered_errors = self.error_history.copy()
            
            if severity_filter:
                try:
                    severity = ErrorSeverity(severity_filter.lower())
                    filtered_errors = [e for e in filtered_errors if e.severity == severity]
                except ValueError:
                    return self._create_error_result(f"Invalid severity filter: {severity_filter}")
            
            if error_type_filter:
                filtered_errors = [e for e in filtered_errors if e.error_type == error_type_filter]
            
            # Sort by timestamp (most recent first) and limit
            filtered_errors.sort(key=lambda x: x.timestamp, reverse=True)
            filtered_errors = filtered_errors[:limit]
            
            # Format for response
            error_data = []
            for error_info in filtered_errors:
                error_data.append({
                    'error_type': error_info.error_type,
                    'error_message': error_info.error_message,
                    'severity': error_info.severity.value,
                    'timestamp': error_info.timestamp.isoformat(),
                    'retry_count': error_info.retry_count,
                    'resolved': error_info.resolved,
                    'context': error_info.context
                })
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'errors': error_data,
                    'total_count': len(error_data),
                    'total_in_history': len(self.error_history),
                    'filters': {
                        'severity': severity_filter,
                        'error_type': error_type_filter,
                        'limit': limit
                    }
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Error getting error history: {e}")
    
    async def _clear_error_history(self, start_time: float) -> ToolResult:
        """Clear the error history."""
        try:
            cleared_count = len(self.error_history)
            self.error_history.clear()
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'cleared_count': cleared_count,
                    'remaining_count': 0
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Error clearing error history: {e}")
    
    async def _analyze_errors(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Analyze error patterns and trends."""
        try:
            if not self.error_history:
                return self._create_success_result(
                    data={
                        'analysis': 'No errors in history to analyze',
                        'patterns': [],
                        'recommendations': ['System appears to be running without errors']
                    }
                )
            
            # Analyze error patterns
            error_types = {}
            severity_counts = {}
            recent_errors = []
            
            for error_info in self.error_history:
                # Count error types
                error_types[error_info.error_type] = error_types.get(error_info.error_type, 0) + 1
                
                # Count severities
                severity_counts[error_info.severity.value] = severity_counts.get(error_info.severity.value, 0) + 1
                
                # Collect recent errors (last 24 hours)
                if (datetime.now(timezone.utc) - error_info.timestamp).total_seconds() < 86400:
                    recent_errors.append(error_info)
            
            # Find most common error types
            most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Generate recommendations
            recommendations = self._generate_error_recommendations(
                most_common_errors, severity_counts, recent_errors
            )
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'total_errors': len(self.error_history),
                    'error_types': error_types,
                    'severity_distribution': severity_counts,
                    'most_common_errors': most_common_errors,
                    'recent_errors_24h': len(recent_errors),
                    'recommendations': recommendations
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Error analyzing errors: {e}")
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine the severity of an error."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['security', 'permission denied', 'unauthorized']):
            return ErrorSeverity.CRITICAL
        
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'MemoryError']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['ConnectionError', 'TimeoutError', 'DatabaseError']:
            return ErrorSeverity.HIGH
        
        if any(keyword in error_message for keyword in ['failed to connect', 'timeout', 'database']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError', 'KeyError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if error_type in ['FileNotFoundError', 'ImportError']:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _analyze_single_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Analyze a single error and provide insights."""
        analysis = {
            'category': self._categorize_error(error_info),
            'likely_causes': self._get_likely_causes(error_info),
            'impact_assessment': self._assess_impact(error_info)
        }
        
        return analysis
    
    def _categorize_error(self, error_info: ErrorInfo) -> str:
        """Categorize an error by type."""
        error_type = error_info.error_type.lower()
        error_message = error_info.error_message.lower()
        
        if 'connection' in error_message or 'network' in error_message:
            return 'Network/Connection'
        elif 'permission' in error_message or 'access' in error_message:
            return 'Security/Permissions'
        elif 'timeout' in error_message:
            return 'Performance/Timeout'
        elif error_type in ['valueerror', 'typeerror', 'attributeerror']:
            return 'Data/Logic'
        elif 'file' in error_message or 'directory' in error_message:
            return 'File System'
        else:
            return 'General'
    
    def _get_likely_causes(self, error_info: ErrorInfo) -> List[str]:
        """Get likely causes for an error."""
        causes = []
        error_type = error_info.error_type.lower()
        error_message = error_info.error_message.lower()
        
        if 'connection' in error_message:
            causes.extend([
                'Network connectivity issues',
                'Service unavailable',
                'Firewall blocking connection'
            ])
        elif 'timeout' in error_message:
            causes.extend([
                'Slow network or service response',
                'Resource contention',
                'Insufficient timeout configuration'
            ])
        elif error_type == 'valueerror':
            causes.extend([
                'Invalid input data format',
                'Data validation failure',
                'Unexpected data type'
            ])
        elif 'permission' in error_message:
            causes.extend([
                'Insufficient file/directory permissions',
                'Authentication failure',
                'Security policy restriction'
            ])
        
        return causes or ['Unknown cause - requires further investigation']
    
    def _assess_impact(self, error_info: ErrorInfo) -> str:
        """Assess the impact of an error."""
        if error_info.severity == ErrorSeverity.CRITICAL:
            return 'High - May cause system instability or security issues'
        elif error_info.severity == ErrorSeverity.HIGH:
            return 'Medium-High - May affect core functionality'
        elif error_info.severity == ErrorSeverity.MEDIUM:
            return 'Medium - May affect specific features'
        else:
            return 'Low - Minimal impact on system operation'
    
    def _get_error_suggestions(self, error_info: ErrorInfo) -> List[str]:
        """Get suggestions for resolving an error."""
        suggestions = []
        error_type = error_info.error_type.lower()
        error_message = error_info.error_message.lower()
        
        if 'connection' in error_message:
            suggestions.extend([
                'Check network connectivity',
                'Verify service availability',
                'Review firewall settings'
            ])
        elif 'timeout' in error_message:
            suggestions.extend([
                'Increase timeout values',
                'Optimize query/request performance',
                'Check system resources'
            ])
        elif error_type == 'valueerror':
            suggestions.extend([
                'Validate input data format',
                'Add data type checking',
                'Review data transformation logic'
            ])
        elif 'permission' in error_message:
            suggestions.extend([
                'Check file/directory permissions',
                'Verify authentication credentials',
                'Review security policies'
            ])
        
        return suggestions or ['Review error details and context for resolution']
    
    def _generate_error_recommendations(self, most_common_errors: List[tuple], 
                                      severity_counts: Dict[str, int],
                                      recent_errors: List[ErrorInfo]) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        # Check for frequent errors
        if most_common_errors and most_common_errors[0][1] > 5:
            error_type, count = most_common_errors[0]
            recommendations.append(f'Address frequent {error_type} errors ({count} occurrences)')
        
        # Check for critical errors
        if severity_counts.get('critical', 0) > 0:
            recommendations.append('Investigate and resolve critical errors immediately')
        
        # Check for recent error spike
        if len(recent_errors) > 10:
            recommendations.append('High error rate in last 24 hours - investigate system health')
        
        # General recommendations
        if not recommendations:
            recommendations.append('Error levels appear normal - continue monitoring')
        
        return recommendations
    
    def _add_to_history(self, error_info: ErrorInfo) -> None:
        """Add error to history with size management."""
        self.error_history.append(error_info)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            # Remove oldest errors
            self.error_history = self.error_history[-self.max_history_size:]
    
    async def handle_with_retry(self, func: Callable, *args, 
                               retry_config: Optional[RetryConfig] = None, **kwargs) -> Any:
        """
        Convenience method to handle a function with automatic retry.
        
        Args:
            func: Function to execute with retry
            *args: Function arguments
            retry_config: Retry configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises last exception
        """
        result = await self.execute(
            action='retry',
            function=func,
            args=args,
            kwargs=kwargs,
            retry_config=retry_config.__dict__ if retry_config else {}
        )
        
        if result.success:
            return result.data['result']
        else:
            raise Exception(result.error)
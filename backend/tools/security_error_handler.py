"""
Security error containment and alerting system.

This module provides specialized handling for security-related errors,
including containment strategies, alerting mechanisms, and audit logging.
"""

import asyncio
import logging
import time
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from functools import wraps

from .error_handler import ErrorHandlerTool, ErrorSeverity

logger = logging.getLogger(__name__)


class SecurityErrorType(Enum):
    """Types of security errors."""
    SANDBOX_ESCAPE = "sandbox_escape"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_CODE = "malicious_code"
    DATA_EXFILTRATION = "data_exfiltration"
    RATE_LIMIT_BREACH = "rate_limit_breach"
    INJECTION_ATTEMPT = "injection_attempt"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    POLICY_VIOLATION = "policy_violation"


class SecuritySeverity(Enum):
    """Security error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAction(Enum):
    """Actions to take in response to security errors."""
    LOG = "log"                  # Just log the error
    ALERT = "alert"              # Log and send alerts
    BLOCK = "block"              # Block the operation
    QUARANTINE = "quarantine"    # Isolate the affected component
    TERMINATE = "terminate"      # Terminate the affected process


@dataclass
class SecurityAlert:
    """Security alert information."""
    alert_id: str
    error_type: SecurityErrorType
    severity: SecuritySeverity
    description: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[SecurityAction] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class SecurityErrorHandler:
    """
    Security error containment and alerting system.
    
    Provides specialized handling for security-related errors,
    including containment strategies, alerting mechanisms, and audit logging.
    """
    
    def __init__(self, audit_log_path: Optional[str] = None):
        """
        Initialize the security error handler.
        
        Args:
            audit_log_path: Path to the security audit log file
        """
        self.error_handler = ErrorHandlerTool()
        self.alerts: List[SecurityAlert] = []
        self.max_alerts = 1000
        self.alert_callbacks: List[Callable[[SecurityAlert], None]] = []
        
        # Set up audit logging
        self.audit_log_path = audit_log_path or os.path.expanduser("~/.james/logs/security_audit.log")
        self._ensure_log_directory()
        
        # Configure security policies
        self.policies = self._load_default_policies()
    
    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists."""
        log_dir = os.path.dirname(self.audit_log_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _load_default_policies(self) -> Dict[SecurityErrorType, Dict[str, Any]]:
        """Load default security policies."""
        return {
            SecurityErrorType.SANDBOX_ESCAPE: {
                "severity": SecuritySeverity.CRITICAL,
                "actions": [SecurityAction.ALERT, SecurityAction.TERMINATE],
                "description": "Attempt to escape sandbox containment"
            },
            SecurityErrorType.UNAUTHORIZED_ACCESS: {
                "severity": SecuritySeverity.HIGH,
                "actions": [SecurityAction.ALERT, SecurityAction.BLOCK],
                "description": "Unauthorized access attempt"
            },
            SecurityErrorType.MALICIOUS_CODE: {
                "severity": SecuritySeverity.CRITICAL,
                "actions": [SecurityAction.ALERT, SecurityAction.QUARANTINE],
                "description": "Malicious code detected"
            },
            SecurityErrorType.DATA_EXFILTRATION: {
                "severity": SecuritySeverity.HIGH,
                "actions": [SecurityAction.ALERT, SecurityAction.BLOCK],
                "description": "Potential data exfiltration attempt"
            },
            SecurityErrorType.RATE_LIMIT_BREACH: {
                "severity": SecuritySeverity.MEDIUM,
                "actions": [SecurityAction.ALERT, SecurityAction.BLOCK],
                "description": "Rate limit breach detected"
            },
            SecurityErrorType.INJECTION_ATTEMPT: {
                "severity": SecuritySeverity.HIGH,
                "actions": [SecurityAction.ALERT, SecurityAction.BLOCK],
                "description": "Injection attack attempt"
            },
            SecurityErrorType.SUSPICIOUS_PATTERN: {
                "severity": SecuritySeverity.MEDIUM,
                "actions": [SecurityAction.ALERT],
                "description": "Suspicious activity pattern detected"
            },
            SecurityErrorType.POLICY_VIOLATION: {
                "severity": SecuritySeverity.MEDIUM,
                "actions": [SecurityAction.LOG, SecurityAction.BLOCK],
                "description": "Security policy violation"
            }
        }
    
    def handle_security_error(self, 
                             error_type: Union[SecurityErrorType, str], 
                             description: str,
                             source: str,
                             context: Optional[Dict[str, Any]] = None,
                             override_actions: Optional[List[SecurityAction]] = None) -> SecurityAlert:
        """
        Handle a security error.
        
        Args:
            error_type: Type of security error
            description: Description of the error
            source: Source of the error (component, module, etc.)
            context: Additional context information
            override_actions: Override default actions for this error
            
        Returns:
            SecurityAlert object with information about the handled error
        """
        # Convert string to enum if needed
        if isinstance(error_type, str):
            try:
                error_type = SecurityErrorType(error_type)
            except ValueError:
                error_type = SecurityErrorType.SUSPICIOUS_PATTERN
        
        # Get policy for this error type
        policy = self.policies.get(error_type, {
            "severity": SecuritySeverity.MEDIUM,
            "actions": [SecurityAction.ALERT],
            "description": "Security issue detected"
        })
        
        # Create alert
        alert = SecurityAlert(
            alert_id=f"sec-{int(time.time())}-{hash(description) % 10000:04d}",
            error_type=error_type,
            severity=policy["severity"],
            description=description,
            source=source,
            context=context or {},
            actions_taken=override_actions or policy["actions"]
        )
        
        # Add to alerts list
        self._add_alert(alert)
        
        # Log to audit log
        self._audit_log(alert)
        
        # Execute actions
        self._execute_actions(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in security alert callback: {e}")
        
        return alert
    
    def _add_alert(self, alert: SecurityAlert) -> None:
        """Add alert to the list with size management."""
        self.alerts.append(alert)
        
        # Maintain size limit
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
    
    def _audit_log(self, alert: SecurityAlert) -> None:
        """Log alert to the security audit log."""
        try:
            log_entry = {
                "timestamp": alert.timestamp.isoformat(),
                "alert_id": alert.alert_id,
                "error_type": alert.error_type.value,
                "severity": alert.severity.value,
                "description": alert.description,
                "source": alert.source,
                "actions": [action.value for action in alert.actions_taken],
                "context": alert.context
            }
            
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write to security audit log: {e}")
    
    def _execute_actions(self, alert: SecurityAlert) -> None:
        """Execute the actions specified for an alert."""
        for action in alert.actions_taken:
            try:
                if action == SecurityAction.LOG:
                    # Already logged in _audit_log
                    pass
                    
                elif action == SecurityAction.ALERT:
                    # Log at appropriate level based on severity
                    if alert.severity == SecuritySeverity.CRITICAL:
                        logger.critical(f"SECURITY ALERT: {alert.description} [{alert.error_type.value}]")
                    elif alert.severity == SecuritySeverity.HIGH:
                        logger.error(f"SECURITY ALERT: {alert.description} [{alert.error_type.value}]")
                    elif alert.severity == SecuritySeverity.MEDIUM:
                        logger.warning(f"SECURITY ALERT: {alert.description} [{alert.error_type.value}]")
                    else:
                        logger.info(f"SECURITY ALERT: {alert.description} [{alert.error_type.value}]")
                    
                elif action == SecurityAction.BLOCK:
                    # Blocking is handled by the caller
                    logger.warning(f"SECURITY ACTION: Blocking operation - {alert.description}")
                    
                elif action == SecurityAction.QUARANTINE:
                    logger.warning(f"SECURITY ACTION: Quarantining - {alert.description}")
                    # Quarantine logic would be implemented by the caller
                    
                elif action == SecurityAction.TERMINATE:
                    logger.critical(f"SECURITY ACTION: Terminating - {alert.description}")
                    # Termination logic would be implemented by the caller
                    
            except Exception as e:
                logger.error(f"Error executing security action {action.value}: {e}")
    
    def register_alert_callback(self, callback: Callable[[SecurityAlert], None]) -> None:
        """
        Register a callback to be notified of security alerts.
        
        Args:
            callback: Function to call with the SecurityAlert
        """
        self.alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str, resolution_notes: Optional[str] = None) -> bool:
        """
        Mark a security alert as resolved.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            True if the alert was found and resolved, False otherwise
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now(timezone.utc)
                alert.resolution_notes = resolution_notes
                
                # Log the resolution
                self._audit_log_resolution(alert)
                return True
        
        return False
    
    def _audit_log_resolution(self, alert: SecurityAlert) -> None:
        """Log alert resolution to the security audit log."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": alert.alert_id,
                "event": "resolution",
                "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None,
                "resolution_notes": alert.resolution_notes
            }
            
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write resolution to security audit log: {e}")
    
    def get_active_alerts(self, severity_filter: Optional[SecuritySeverity] = None) -> List[SecurityAlert]:
        """
        Get active (unresolved) security alerts.
        
        Args:
            severity_filter: Optional filter for alert severity
            
        Returns:
            List of active SecurityAlert objects
        """
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if severity_filter:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity_filter]
        
        return active_alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about security alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        if not self.alerts:
            return {
                "total_alerts": 0,
                "active_alerts": 0,
                "resolved_alerts": 0,
                "by_severity": {},
                "by_type": {}
            }
        
        # Count alerts by status, severity, and type
        total = len(self.alerts)
        active = len([a for a in self.alerts if not a.resolved])
        resolved = total - active
        
        by_severity = {}
        by_type = {}
        
        for alert in self.alerts:
            # Count by severity
            severity = alert.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by type
            error_type = alert.error_type.value
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        return {
            "total_alerts": total,
            "active_alerts": active,
            "resolved_alerts": resolved,
            "by_severity": by_severity,
            "by_type": by_type,
            "recent_alerts": [
                {
                    "alert_id": a.alert_id,
                    "error_type": a.error_type.value,
                    "severity": a.severity.value,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved
                }
                for a in sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:5]  # Last 5 alerts
            ]
        }


# Singleton instance
_security_error_handler = None


def get_security_error_handler() -> SecurityErrorHandler:
    """
    Get the singleton security error handler instance.
    
    Returns:
        SecurityErrorHandler instance
    """
    global _security_error_handler
    if _security_error_handler is None:
        _security_error_handler = SecurityErrorHandler()
    return _security_error_handler


def with_security_error_handling(source: str):
    """
    Decorator for handling security errors.
    
    Args:
        source: Source identifier for the security error
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if this is a security-related exception
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in 
                      ['security', 'unauthorized', 'permission', 'sandbox', 'injection', 'malicious']):
                    
                    # Determine error type based on message
                    error_type = SecurityErrorType.SUSPICIOUS_PATTERN
                    
                    if 'sandbox' in error_msg:
                        error_type = SecurityErrorType.SANDBOX_ESCAPE
                    elif 'unauthorized' in error_msg or 'permission' in error_msg:
                        error_type = SecurityErrorType.UNAUTHORIZED_ACCESS
                    elif 'malicious' in error_msg:
                        error_type = SecurityErrorType.MALICIOUS_CODE
                    elif 'injection' in error_msg:
                        error_type = SecurityErrorType.INJECTION_ATTEMPT
                    elif 'rate limit' in error_msg:
                        error_type = SecurityErrorType.RATE_LIMIT_BREACH
                    elif 'data' in error_msg and ('leak' in error_msg or 'exfil' in error_msg):
                        error_type = SecurityErrorType.DATA_EXFILTRATION
                    elif 'policy' in error_msg:
                        error_type = SecurityErrorType.POLICY_VIOLATION
                    
                    # Handle the security error
                    handler = get_security_error_handler()
                    alert = handler.handle_security_error(
                        error_type=error_type,
                        description=str(e),
                        source=source,
                        context={"function": func.__name__}
                    )
                    
                    # Re-raise with additional context
                    raise type(e)(f"Security error: {e} (Alert ID: {alert.alert_id})") from e
                
                # Not a security error, re-raise as is
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if this is a security-related exception
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in 
                      ['security', 'unauthorized', 'permission', 'sandbox', 'injection', 'malicious']):
                    
                    # Determine error type based on message
                    error_type = SecurityErrorType.SUSPICIOUS_PATTERN
                    
                    if 'sandbox' in error_msg:
                        error_type = SecurityErrorType.SANDBOX_ESCAPE
                    elif 'unauthorized' in error_msg or 'permission' in error_msg:
                        error_type = SecurityErrorType.UNAUTHORIZED_ACCESS
                    elif 'malicious' in error_msg:
                        error_type = SecurityErrorType.MALICIOUS_CODE
                    elif 'injection' in error_msg:
                        error_type = SecurityErrorType.INJECTION_ATTEMPT
                    elif 'rate limit' in error_msg:
                        error_type = SecurityErrorType.RATE_LIMIT_BREACH
                    elif 'data' in error_msg and ('leak' in error_msg or 'exfil' in error_msg):
                        error_type = SecurityErrorType.DATA_EXFILTRATION
                    elif 'policy' in error_msg:
                        error_type = SecurityErrorType.POLICY_VIOLATION
                    
                    # Handle the security error
                    handler = get_security_error_handler()
                    alert = handler.handle_security_error(
                        error_type=error_type,
                        description=str(e),
                        source=source,
                        context={"function": func.__name__}
                    )
                    
                    # Re-raise with additional context
                    raise type(e)(f"Security error: {e} (Alert ID: {alert.alert_id})") from e
                
                # Not a security error, re-raise as is
                raise
        
        # Choose the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
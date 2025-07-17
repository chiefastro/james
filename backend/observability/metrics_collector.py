"""
Custom metrics collection for agent performance and decision tracking.

This module provides comprehensive metrics collection for the conscious agent system,
including performance metrics, decision quality tracking, and system health monitoring.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricCategory(Enum):
    """Categories of metrics for organization."""
    PERFORMANCE = "performance"
    DECISION_QUALITY = "decision_quality"
    SYSTEM_HEALTH = "system_health"
    USER_INTERACTION = "user_interaction"
    SUBAGENT_ACTIVITY = "subagent_activity"
    ERROR_TRACKING = "error_tracking"


@dataclass
class MetricValue:
    """Represents a single metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Comprehensive metrics for agent operations."""
    # Performance metrics
    message_processing_time_ms: List[float] = field(default_factory=list)
    classification_accuracy: List[float] = field(default_factory=list)
    delegation_success_rate: List[float] = field(default_factory=list)
    
    # Decision quality metrics
    classification_confidence: List[float] = field(default_factory=list)
    subagent_selection_quality: List[float] = field(default_factory=list)
    task_completion_rate: List[float] = field(default_factory=list)
    
    # System health metrics
    memory_usage_mb: List[float] = field(default_factory=list)
    active_connections: List[int] = field(default_factory=list)
    error_rate: List[float] = field(default_factory=list)
    
    # Activity counters
    messages_processed: int = 0
    tasks_delegated: int = 0
    subagents_activated: int = 0
    errors_encountered: int = 0
    
    # Timestamps
    collection_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """
    Comprehensive metrics collection system for agent monitoring.
    
    Collects, aggregates, and analyzes metrics from all agent operations
    to provide insights into system performance and decision quality.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history_size: Maximum number of metric values to keep in memory
        """
        self.max_history_size = max_history_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Agent-specific metrics
        self.agent_metrics = AgentMetrics()
        
        # Metric metadata
        self._metric_definitions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("MetricsCollector initialized")
    
    def define_metric(
        self,
        name: str,
        metric_type: MetricType,
        category: MetricCategory,
        description: str,
        unit: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Define a new metric with metadata.
        
        Args:
            name: Unique name for the metric
            metric_type: Type of metric (counter, gauge, etc.)
            category: Category for organization
            description: Human-readable description
            unit: Unit of measurement
            tags: Default tags for the metric
        """
        self._metric_definitions[name] = {
            "type": metric_type.value,
            "category": category.value,
            "description": description,
            "unit": unit,
            "default_tags": tags or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.debug(f"Defined metric: {name} ({metric_type.value})")
    
    def record_counter(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a counter metric (cumulative value).
        
        Args:
            name: Name of the counter metric
            value: Value to add to the counter
            tags: Optional tags for this measurement
        """
        self._counters[name] += value
        
        metric_value = MetricValue(
            name=name,
            value=self._counters[name],
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._metrics[name].append(metric_value)
        logger.debug(f"Counter {name}: {self._counters[name]} (+{value})")
    
    def record_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a gauge metric (point-in-time value).
        
        Args:
            name: Name of the gauge metric
            value: Current value of the gauge
            tags: Optional tags for this measurement
        """
        self._gauges[name] = value
        
        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._metrics[name].append(metric_value)
        logger.debug(f"Gauge {name}: {value}")
    
    def record_histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a histogram metric (distribution of values).
        
        Args:
            name: Name of the histogram metric
            value: Value to add to the histogram
            tags: Optional tags for this measurement
        """
        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._metrics[name].append(metric_value)
        logger.debug(f"Histogram {name}: {value}")
    
    def start_timer(self, name: str) -> str:
        """
        Start a timer for measuring operation duration.
        
        Args:
            name: Name of the timer metric
            
        Returns:
            Timer ID for stopping the timer
        """
        from uuid import uuid4
        timer_id = str(uuid4())
        
        if name not in self._timers:
            self._timers[name] = []
        
        # Store start time with timer ID
        start_time = datetime.now(timezone.utc)
        self._timers[f"{name}_starts"] = self._timers.get(f"{name}_starts", {})
        self._timers[f"{name}_starts"][timer_id] = start_time
        
        logger.debug(f"Started timer {name} (ID: {timer_id})")
        return timer_id
    
    def stop_timer(
        self,
        name: str,
        timer_id: str,
        tags: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            name: Name of the timer metric
            timer_id: ID returned from start_timer
            tags: Optional tags for this measurement
            
        Returns:
            Duration in milliseconds
        """
        end_time = datetime.now(timezone.utc)
        
        # Get start time
        starts_key = f"{name}_starts"
        if starts_key not in self._timers or timer_id not in self._timers[starts_key]:
            logger.warning(f"Timer {name} (ID: {timer_id}) not found")
            return 0.0
        
        start_time = self._timers[starts_key].pop(timer_id)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Record the duration
        self.record_histogram(name, duration_ms, tags)
        
        logger.debug(f"Stopped timer {name}: {duration_ms:.2f}ms")
        return duration_ms
    
    # Agent-specific metric recording methods
    
    def record_message_processing_time(self, duration_ms: float) -> None:
        """Record message processing time."""
        self.agent_metrics.message_processing_time_ms.append(duration_ms)
        self.record_histogram("message_processing_time_ms", duration_ms)
        self.agent_metrics.messages_processed += 1
        self.record_counter("messages_processed")
        self._update_last_updated()
    
    def record_classification_result(
        self,
        accuracy: float,
        confidence: float,
        classification_type: str
    ) -> None:
        """Record message classification results."""
        self.agent_metrics.classification_accuracy.append(accuracy)
        self.agent_metrics.classification_confidence.append(confidence)
        
        self.record_histogram("classification_accuracy", accuracy)
        self.record_histogram("classification_confidence", confidence)
        self.record_counter("classifications_made", tags={"type": classification_type})
        self._update_last_updated()
    
    def record_delegation_result(
        self,
        success: bool,
        subagents_selected: int,
        selection_quality: float
    ) -> None:
        """Record task delegation results."""
        success_rate = 1.0 if success else 0.0
        self.agent_metrics.delegation_success_rate.append(success_rate)
        self.agent_metrics.subagent_selection_quality.append(selection_quality)
        
        self.record_histogram("delegation_success_rate", success_rate)
        self.record_histogram("subagent_selection_quality", selection_quality)
        self.record_gauge("subagents_selected", subagents_selected)
        
        self.agent_metrics.tasks_delegated += 1
        self.agent_metrics.subagents_activated += subagents_selected
        
        self.record_counter("tasks_delegated")
        self.record_counter("subagents_activated", subagents_selected)
        self._update_last_updated()
    
    def record_task_completion(self, success: bool, duration_ms: float) -> None:
        """Record task completion results."""
        completion_rate = 1.0 if success else 0.0
        self.agent_metrics.task_completion_rate.append(completion_rate)
        
        self.record_histogram("task_completion_rate", completion_rate)
        self.record_histogram("task_duration_ms", duration_ms)
        self.record_counter("tasks_completed" if success else "tasks_failed")
        self._update_last_updated()
    
    def record_system_health(
        self,
        memory_usage_mb: float,
        active_connections: int,
        error_rate: float
    ) -> None:
        """Record system health metrics."""
        self.agent_metrics.memory_usage_mb.append(memory_usage_mb)
        self.agent_metrics.active_connections.append(active_connections)
        self.agent_metrics.error_rate.append(error_rate)
        
        self.record_gauge("memory_usage_mb", memory_usage_mb)
        self.record_gauge("active_connections", active_connections)
        self.record_gauge("error_rate", error_rate)
        self._update_last_updated()
    
    def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = "error"
    ) -> None:
        """Record error occurrence."""
        self.agent_metrics.errors_encountered += 1
        
        self.record_counter("errors_total")
        self.record_counter(
            "errors_by_type",
            tags={"type": error_type, "component": component, "severity": severity}
        )
        self._update_last_updated()
    
    def _update_last_updated(self) -> None:
        """Update the last updated timestamp."""
        self.agent_metrics.last_updated = datetime.now(timezone.utc)
    
    # Analysis and reporting methods
    
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get summary statistics for a metric within a time window.
        
        Args:
            name: Name of the metric
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with summary statistics
        """
        if name not in self._metrics:
            return {"error": f"Metric {name} not found"}
        
        # Filter values within time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_values = [
            mv.value for mv in self._metrics[name]
            if mv.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {"error": f"No recent values for metric {name}"}
        
        return {
            "metric_name": name,
            "window_minutes": window_minutes,
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "std_dev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            "latest_value": recent_values[-1],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent performance summary."""
        def safe_mean(values: List[float]) -> float:
            return statistics.mean(values) if values else 0.0
        
        def safe_latest(values: List[float]) -> float:
            return values[-1] if values else 0.0
        
        return {
            "performance": {
                "avg_message_processing_time_ms": safe_mean(
                    self.agent_metrics.message_processing_time_ms
                ),
                "avg_classification_accuracy": safe_mean(
                    self.agent_metrics.classification_accuracy
                ),
                "avg_delegation_success_rate": safe_mean(
                    self.agent_metrics.delegation_success_rate
                ),
                "avg_task_completion_rate": safe_mean(
                    self.agent_metrics.task_completion_rate
                )
            },
            "decision_quality": {
                "avg_classification_confidence": safe_mean(
                    self.agent_metrics.classification_confidence
                ),
                "avg_subagent_selection_quality": safe_mean(
                    self.agent_metrics.subagent_selection_quality
                )
            },
            "system_health": {
                "current_memory_usage_mb": safe_latest(
                    self.agent_metrics.memory_usage_mb
                ),
                "current_active_connections": safe_latest(
                    self.agent_metrics.active_connections
                ),
                "current_error_rate": safe_latest(
                    self.agent_metrics.error_rate
                )
            },
            "activity_counters": {
                "messages_processed": self.agent_metrics.messages_processed,
                "tasks_delegated": self.agent_metrics.tasks_delegated,
                "subagents_activated": self.agent_metrics.subagents_activated,
                "errors_encountered": self.agent_metrics.errors_encountered
            },
            "collection_period": {
                "start": self.agent_metrics.collection_start.isoformat(),
                "last_updated": self.agent_metrics.last_updated.isoformat(),
                "duration_hours": (
                    self.agent_metrics.last_updated - self.agent_metrics.collection_start
                ).total_seconds() / 3600
            }
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics and their current values."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "metric_definitions": self._metric_definitions,
            "agent_metrics": self.get_agent_performance_summary(),
            "collection_stats": {
                "total_metrics": len(self._metrics),
                "max_history_size": self.max_history_size,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._timers.clear()
        self.agent_metrics = AgentMetrics()
        
        logger.info("All metrics reset")
    
    def export_metrics_for_langsmith(self) -> Dict[str, Any]:
        """Export metrics in a format suitable for LangSmith integration."""
        summary = self.get_agent_performance_summary()
        
        return {
            "conscious_agent_metrics": {
                "performance_metrics": summary["performance"],
                "decision_quality_metrics": summary["decision_quality"],
                "system_health_metrics": summary["system_health"],
                "activity_counters": summary["activity_counters"],
                "collection_metadata": summary["collection_period"]
            },
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "export_version": "1.0"
        }


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
        
        # Define standard metrics
        _define_standard_metrics(_global_collector)
    
    return _global_collector


def _define_standard_metrics(collector: MetricsCollector) -> None:
    """Define standard metrics for the conscious agent system."""
    # Performance metrics
    collector.define_metric(
        "message_processing_time_ms",
        MetricType.HISTOGRAM,
        MetricCategory.PERFORMANCE,
        "Time taken to process a message from start to finish"
    )
    
    collector.define_metric(
        "classification_accuracy",
        MetricType.HISTOGRAM,
        MetricCategory.DECISION_QUALITY,
        "Accuracy of message classification decisions"
    )
    
    collector.define_metric(
        "delegation_success_rate",
        MetricType.HISTOGRAM,
        MetricCategory.DECISION_QUALITY,
        "Success rate of task delegation to subagents"
    )
    
    # System health metrics
    collector.define_metric(
        "memory_usage_mb",
        MetricType.GAUGE,
        MetricCategory.SYSTEM_HEALTH,
        "Current memory usage in megabytes"
    )
    
    collector.define_metric(
        "active_connections",
        MetricType.GAUGE,
        MetricCategory.SYSTEM_HEALTH,
        "Number of active WebSocket connections"
    )
    
    collector.define_metric(
        "error_rate",
        MetricType.GAUGE,
        MetricCategory.ERROR_TRACKING,
        "Current error rate (errors per minute)"
    )
    
    # Activity counters
    collector.define_metric(
        "messages_processed",
        MetricType.COUNTER,
        MetricCategory.USER_INTERACTION,
        "Total number of messages processed"
    )
    
    collector.define_metric(
        "tasks_delegated",
        MetricType.COUNTER,
        MetricCategory.SUBAGENT_ACTIVITY,
        "Total number of tasks delegated to subagents"
    )
    
    collector.define_metric(
        "errors_total",
        MetricType.COUNTER,
        MetricCategory.ERROR_TRACKING,
        "Total number of errors encountered"
    )
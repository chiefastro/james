"""
Anomaly detection system for identifying unusual agent behavior.

This module provides real-time anomaly detection capabilities to identify
performance issues, decision quality problems, and system health concerns.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

from .metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DECISION_QUALITY_DROP = "decision_quality_drop"
    ERROR_SPIKE = "error_spike"
    MEMORY_LEAK = "memory_leak"
    CONNECTION_ISSUES = "connection_issues"
    PROCESSING_DELAY = "processing_delay"
    CLASSIFICATION_DRIFT = "classification_drift"
    DELEGATION_FAILURE = "delegation_failure"


@dataclass
class AnomalyAlert:
    """Represents an anomaly detection alert."""
    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    deviation_score: float
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class AnomalyThreshold:
    """Configuration for anomaly detection thresholds."""
    metric_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    threshold_type: str  # "absolute", "relative", "statistical"
    threshold_value: float
    window_minutes: int = 30
    min_samples: int = 10
    enabled: bool = True


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using z-score and moving averages."""
    
    def __init__(self, z_score_threshold: float = 2.0):
        """
        Initialize statistical anomaly detector.
        
        Args:
            z_score_threshold: Z-score threshold for anomaly detection
        """
        self.z_score_threshold = z_score_threshold
    
    def detect_anomalies(
        self,
        values: List[float],
        window_size: int = 50
    ) -> List[Tuple[int, float, str]]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            values: List of metric values
            window_size: Size of the sliding window for analysis
            
        Returns:
            List of (index, z_score, description) tuples for anomalies
        """
        if len(values) < window_size:
            return []
        
        anomalies = []
        
        for i in range(window_size, len(values)):
            # Get window of previous values
            window = values[i - window_size:i]
            current_value = values[i]
            
            # Calculate statistics
            mean = statistics.mean(window)
            std_dev = statistics.stdev(window) if len(window) > 1 else 0.0
            
            if std_dev == 0:
                continue
            
            # Calculate z-score
            z_score = abs(current_value - mean) / std_dev
            
            if z_score > self.z_score_threshold:
                description = f"Value {current_value:.2f} deviates from mean {mean:.2f} by {z_score:.2f} standard deviations"
                anomalies.append((i, z_score, description))
        
        return anomalies
    
    def detect_trend_anomalies(
        self,
        values: List[float],
        trend_window: int = 20
    ) -> List[Tuple[int, float, str]]:
        """
        Detect trend-based anomalies (sudden changes in direction).
        
        Args:
            values: List of metric values
            trend_window: Window size for trend calculation
            
        Returns:
            List of (index, change_magnitude, description) tuples
        """
        if len(values) < trend_window * 2:
            return []
        
        anomalies = []
        
        for i in range(trend_window, len(values) - trend_window):
            # Calculate trends before and after current point
            before_window = values[i - trend_window:i]
            after_window = values[i:i + trend_window]
            
            # Calculate slopes (simple linear trend)
            before_trend = self._calculate_trend(before_window)
            after_trend = self._calculate_trend(after_window)
            
            # Detect significant trend changes
            trend_change = abs(after_trend - before_trend)
            
            if trend_change > 0.5:  # Threshold for significant trend change
                description = f"Trend change detected: {before_trend:.3f} to {after_trend:.3f}"
                anomalies.append((i, trend_change, description))
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend (slope) for a list of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class AnomalyDetector:
    """
    Comprehensive anomaly detection system for the conscious agent system.
    
    Monitors various metrics and detects anomalous behavior patterns
    that might indicate performance issues or system problems.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        check_interval_seconds: int = 60
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            metrics_collector: Metrics collector instance
            check_interval_seconds: How often to check for anomalies
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.check_interval_seconds = check_interval_seconds
        
        # Anomaly detection components
        self.statistical_detector = StatisticalAnomalyDetector()
        
        # Alert management
        self.active_alerts: Dict[str, AnomalyAlert] = {}
        self.alert_history: List[AnomalyAlert] = []
        self.alert_callbacks: List[Callable[[AnomalyAlert], None]] = []
        
        # Detection thresholds
        self.thresholds: List[AnomalyThreshold] = []
        self._setup_default_thresholds()
        
        # Detection state
        self._detection_running = False
        self._detection_task: Optional[asyncio.Task] = None
        
        logger.info("AnomalyDetector initialized")
    
    def _setup_default_thresholds(self) -> None:
        """Set up default anomaly detection thresholds."""
        self.thresholds = [
            # Performance thresholds
            AnomalyThreshold(
                metric_name="message_processing_time_ms",
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=AnomalySeverity.HIGH,
                threshold_type="statistical",
                threshold_value=3.0,  # 3 standard deviations
                window_minutes=30
            ),
            
            # Decision quality thresholds
            AnomalyThreshold(
                metric_name="classification_accuracy",
                anomaly_type=AnomalyType.DECISION_QUALITY_DROP,
                severity=AnomalySeverity.MEDIUM,
                threshold_type="absolute",
                threshold_value=0.7,  # Below 70% accuracy
                window_minutes=15
            ),
            
            # Error rate thresholds
            AnomalyThreshold(
                metric_name="error_rate",
                anomaly_type=AnomalyType.ERROR_SPIKE,
                severity=AnomalySeverity.HIGH,
                threshold_type="absolute",
                threshold_value=5.0,  # More than 5 errors per minute
                window_minutes=10
            ),
            
            # Memory usage thresholds
            AnomalyThreshold(
                metric_name="memory_usage_mb",
                anomaly_type=AnomalyType.MEMORY_LEAK,
                severity=AnomalySeverity.CRITICAL,
                threshold_type="relative",
                threshold_value=0.5,  # 50% increase
                window_minutes=60
            ),
            
            # Connection issues
            AnomalyThreshold(
                metric_name="active_connections",
                anomaly_type=AnomalyType.CONNECTION_ISSUES,
                severity=AnomalySeverity.MEDIUM,
                threshold_type="statistical",
                threshold_value=2.5,
                window_minutes=20
            )
        ]
    
    def add_threshold(self, threshold: AnomalyThreshold) -> None:
        """Add a custom anomaly detection threshold."""
        self.thresholds.append(threshold)
        logger.info(f"Added anomaly threshold for {threshold.metric_name}")
    
    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]) -> None:
        """Add a callback function to be called when anomalies are detected."""
        self.alert_callbacks.append(callback)
        logger.info("Added anomaly alert callback")
    
    async def start_detection(self) -> None:
        """Start the anomaly detection loop."""
        if self._detection_running:
            logger.warning("Anomaly detection is already running")
            return
        
        self._detection_running = True
        self._detection_task = asyncio.create_task(self._detection_loop())
        logger.info("Started anomaly detection")
    
    async def stop_detection(self) -> None:
        """Stop the anomaly detection loop."""
        if not self._detection_running:
            return
        
        self._detection_running = False
        
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped anomaly detection")
    
    async def _detection_loop(self) -> None:
        """Main anomaly detection loop."""
        while self._detection_running:
            try:
                await self._check_for_anomalies()
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _check_for_anomalies(self) -> None:
        """Check all configured thresholds for anomalies."""
        current_time = datetime.now(timezone.utc)
        
        for threshold in self.thresholds:
            if not threshold.enabled:
                continue
            
            try:
                anomaly = await self._check_threshold(threshold, current_time)
                if anomaly:
                    await self._handle_anomaly(anomaly)
            except Exception as e:
                logger.error(f"Error checking threshold {threshold.metric_name}: {e}")
    
    async def _check_threshold(
        self,
        threshold: AnomalyThreshold,
        current_time: datetime
    ) -> Optional[AnomalyAlert]:
        """Check a specific threshold for anomalies."""
        # Get metric summary for the specified window
        summary = self.metrics_collector.get_metric_summary(
            threshold.metric_name,
            threshold.window_minutes
        )
        
        if "error" in summary:
            return None
        
        if summary["count"] < threshold.min_samples:
            return None
        
        current_value = summary["latest_value"]
        
        # Check different threshold types
        if threshold.threshold_type == "absolute":
            is_anomaly, description = self._check_absolute_threshold(
                current_value, threshold.threshold_value, threshold.anomaly_type
            )
        elif threshold.threshold_type == "relative":
            is_anomaly, description = self._check_relative_threshold(
                current_value, summary["mean"], threshold.threshold_value
            )
        elif threshold.threshold_type == "statistical":
            is_anomaly, description = self._check_statistical_threshold(
                current_value, summary["mean"], summary["std_dev"], threshold.threshold_value
            )
        else:
            logger.warning(f"Unknown threshold type: {threshold.threshold_type}")
            return None
        
        if not is_anomaly:
            return None
        
        # Calculate deviation score
        deviation_score = self._calculate_deviation_score(
            current_value, summary["mean"], summary["std_dev"]
        )
        
        # Create anomaly alert
        from uuid import uuid4
        alert = AnomalyAlert(
            id=str(uuid4()),
            anomaly_type=threshold.anomaly_type,
            severity=threshold.severity,
            metric_name=threshold.metric_name,
            current_value=current_value,
            expected_range=(summary["min"], summary["max"]),
            deviation_score=deviation_score,
            description=description,
            timestamp=current_time,
            metadata={
                "threshold_config": {
                    "type": threshold.threshold_type,
                    "value": threshold.threshold_value,
                    "window_minutes": threshold.window_minutes
                },
                "metric_summary": summary
            }
        )
        
        return alert
    
    def _check_absolute_threshold(
        self,
        current_value: float,
        threshold_value: float,
        anomaly_type: AnomalyType
    ) -> Tuple[bool, str]:
        """Check absolute threshold (value above/below a fixed limit)."""
        if anomaly_type in [AnomalyType.ERROR_SPIKE, AnomalyType.MEMORY_LEAK]:
            # Higher values are bad
            is_anomaly = current_value > threshold_value
            description = f"Value {current_value:.2f} exceeds threshold {threshold_value:.2f}"
        else:
            # Lower values are bad (e.g., accuracy)
            is_anomaly = current_value < threshold_value
            description = f"Value {current_value:.2f} below threshold {threshold_value:.2f}"
        
        return is_anomaly, description
    
    def _check_relative_threshold(
        self,
        current_value: float,
        baseline_value: float,
        threshold_percentage: float
    ) -> Tuple[bool, str]:
        """Check relative threshold (percentage change from baseline)."""
        if baseline_value == 0:
            return False, "Cannot calculate relative change from zero baseline"
        
        relative_change = abs(current_value - baseline_value) / baseline_value
        is_anomaly = relative_change > threshold_percentage
        
        description = f"Value {current_value:.2f} changed {relative_change:.1%} from baseline {baseline_value:.2f}"
        return is_anomaly, description
    
    def _check_statistical_threshold(
        self,
        current_value: float,
        mean_value: float,
        std_dev: float,
        z_score_threshold: float
    ) -> Tuple[bool, str]:
        """Check statistical threshold (z-score based)."""
        if std_dev == 0:
            return False, "Cannot calculate z-score with zero standard deviation"
        
        z_score = abs(current_value - mean_value) / std_dev
        is_anomaly = z_score > z_score_threshold
        
        description = f"Value {current_value:.2f} has z-score {z_score:.2f} (threshold: {z_score_threshold:.2f})"
        return is_anomaly, description
    
    def _calculate_deviation_score(
        self,
        current_value: float,
        mean_value: float,
        std_dev: float
    ) -> float:
        """Calculate a normalized deviation score."""
        if std_dev == 0:
            return 0.0
        
        return abs(current_value - mean_value) / std_dev
    
    async def _handle_anomaly(self, alert: AnomalyAlert) -> None:
        """Handle a detected anomaly."""
        # Check if we already have an active alert for this metric
        existing_alert_key = f"{alert.metric_name}_{alert.anomaly_type.value}"
        
        if existing_alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[existing_alert_key]
            existing_alert.current_value = alert.current_value
            existing_alert.deviation_score = alert.deviation_score
            existing_alert.timestamp = alert.timestamp
            logger.debug(f"Updated existing anomaly alert: {existing_alert_key}")
        else:
            # Create new alert
            self.active_alerts[existing_alert_key] = alert
            self.alert_history.append(alert)
            
            logger.warning(
                f"ANOMALY DETECTED: {alert.anomaly_type.value} "
                f"({alert.severity.value}) - {alert.description}"
            )
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in anomaly alert callback: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert_key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now(timezone.utc)
                del self.active_alerts[alert_key]
                logger.info(f"Resolved anomaly alert: {alert_id}")
                return True
        
        return False
    
    def get_active_alerts(self) -> List[AnomalyAlert]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(
        self,
        hours: int = 24,
        severity: Optional[AnomalySeverity] = None
    ) -> List[AnomalyAlert]:
        """
        Get alert history within a time window.
        
        Args:
            hours: Number of hours to look back
            severity: Optional severity filter
            
        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.severity == severity
            ]
        
        return sorted(filtered_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about anomaly detection."""
        active_count = len(self.active_alerts)
        total_alerts = len(self.alert_history)
        
        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
        
        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            type_counts[alert.anomaly_type.value] = type_counts.get(alert.anomaly_type.value, 0) + 1
        
        return {
            "detection_status": "running" if self._detection_running else "stopped",
            "check_interval_seconds": self.check_interval_seconds,
            "active_alerts": active_count,
            "total_alerts_generated": total_alerts,
            "alerts_by_severity": severity_counts,
            "alerts_by_type": type_counts,
            "configured_thresholds": len(self.thresholds),
            "enabled_thresholds": len([t for t in self.thresholds if t.enabled]),
            "alert_callbacks": len(self.alert_callbacks),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global anomaly detector instance
_global_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get the global anomaly detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = AnomalyDetector()
    return _global_detector


async def setup_anomaly_detection() -> AnomalyDetector:
    """Set up and start anomaly detection."""
    detector = get_anomaly_detector()
    await detector.start_detection()
    return detector
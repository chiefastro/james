"""
Tests for the observability and monitoring system.

This module tests LangSmith integration, metrics collection,
anomaly detection, and trace analysis functionality.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from backend.observability.langsmith_tracer import (
    LangSmithTracer, TraceConfig, AgentTrace, get_tracer, configure_tracer
)
from backend.observability.metrics_collector import (
    MetricsCollector, AgentMetrics, MetricType, MetricCategory, get_metrics_collector
)
from backend.observability.anomaly_detector import (
    AnomalyDetector, AnomalyThreshold, AnomalySeverity, AnomalyType, get_anomaly_detector
)
from backend.observability.trace_analyzer import (
    TraceAnalyzer, TraceInsight, InsightType, PerformanceMetrics
)


class TestLangSmithTracer:
    """Test cases for LangSmith tracer functionality."""
    
    def test_tracer_initialization(self):
        """Test tracer initialization with different configurations."""
        # Test default configuration
        tracer = LangSmithTracer()
        assert tracer.config.project_name == "conscious-agent-system"
        assert tracer.config.enabled == True
        assert tracer.config.sample_rate == 1.0
        
        # Test custom configuration
        config = TraceConfig(
            project_name="test-project",
            enabled=False,
            sample_rate=0.5
        )
        tracer = LangSmithTracer(config)
        assert tracer.config.project_name == "test-project"
        assert tracer.config.enabled == False
        assert tracer.config.sample_rate == 0.5
    
    def test_should_trace_sampling(self):
        """Test trace sampling logic."""
        # Test with 100% sampling
        config = TraceConfig(sample_rate=1.0)
        tracer = LangSmithTracer(config)
        assert tracer.should_trace() == True
        
        # Test with 0% sampling
        config = TraceConfig(sample_rate=0.0)
        tracer = LangSmithTracer(config)
        assert tracer.should_trace() == False
        
        # Test with disabled tracing
        config = TraceConfig(enabled=False)
        tracer = LangSmithTracer(config)
        assert tracer.should_trace() == False
    
    @pytest.mark.asyncio
    async def test_trace_lifecycle(self):
        """Test complete trace lifecycle from start to end."""
        tracer = LangSmithTracer()
        
        # Start trace
        trace_id = await tracer.start_trace(
            operation_name="test_operation",
            agent_type="TestAgent",
            inputs={"test_input": "value"},
            metadata={"test_meta": "data"}
        )
        
        assert trace_id != ""
        assert trace_id in tracer._active_traces
        
        active_trace = tracer._active_traces[trace_id]
        assert active_trace.operation_name == "test_operation"
        assert active_trace.agent_type == "TestAgent"
        assert active_trace.inputs["test_input"] == "value"
        assert active_trace.metadata["test_meta"] == "data"
        
        # End trace
        completed_trace = await tracer.end_trace(
            trace_id=trace_id,
            outputs={"test_output": "result"},
            success=True
        )
        
        assert completed_trace is not None
        assert completed_trace.outputs["test_output"] == "result"
        assert completed_trace.success == True
        assert completed_trace.duration_ms is not None
        assert completed_trace.duration_ms > 0
        assert trace_id not in tracer._active_traces
    
    @pytest.mark.asyncio
    async def test_trace_error_handling(self):
        """Test trace error handling."""
        tracer = LangSmithTracer()
        
        # Start trace
        trace_id = await tracer.start_trace(
            operation_name="failing_operation",
            agent_type="TestAgent"
        )
        
        # End trace with error
        completed_trace = await tracer.end_trace(
            trace_id=trace_id,
            error="Test error message",
            success=False
        )
        
        assert completed_trace is not None
        assert completed_trace.error == "Test error message"
        assert completed_trace.success == False
    
    def test_trace_statistics(self):
        """Test trace statistics collection."""
        tracer = LangSmithTracer()
        stats = tracer.get_trace_statistics()
        
        assert "total_traces_started" in stats
        assert "active_traces" in stats
        assert "config" in stats
        assert "client_status" in stats
        assert "timestamp" in stats


class TestMetricsCollector:
    """Test cases for metrics collection functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_history_size=100)
        assert collector.max_history_size == 100
        assert len(collector._metrics) == 0
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0
    
    def test_counter_metrics(self):
        """Test counter metric recording."""
        collector = MetricsCollector()
        
        # Record counter values
        collector.record_counter("test_counter", 1)
        collector.record_counter("test_counter", 5)
        collector.record_counter("test_counter", 2)
        
        assert collector._counters["test_counter"] == 8
        assert len(collector._metrics["test_counter"]) == 3
    
    def test_gauge_metrics(self):
        """Test gauge metric recording."""
        collector = MetricsCollector()
        
        # Record gauge values
        collector.record_gauge("test_gauge", 10.5)
        collector.record_gauge("test_gauge", 15.2)
        collector.record_gauge("test_gauge", 8.7)
        
        assert collector._gauges["test_gauge"] == 8.7  # Latest value
        assert len(collector._metrics["test_gauge"]) == 3
    
    def test_histogram_metrics(self):
        """Test histogram metric recording."""
        collector = MetricsCollector()
        
        # Record histogram values
        values = [1.0, 2.5, 3.2, 1.8, 4.1]
        for value in values:
            collector.record_histogram("test_histogram", value)
        
        assert len(collector._metrics["test_histogram"]) == 5
    
    def test_timer_metrics(self):
        """Test timer metric functionality."""
        collector = MetricsCollector()
        
        # Start timer
        timer_id = collector.start_timer("test_timer")
        assert timer_id is not None
        
        # Simulate some work
        import time
        time.sleep(0.01)  # 10ms
        
        # Stop timer
        duration = collector.stop_timer("test_timer", timer_id)
        assert duration > 0
        assert len(collector._metrics["test_timer"]) == 1
    
    def test_agent_metrics_recording(self):
        """Test agent-specific metrics recording."""
        collector = MetricsCollector()
        
        # Record message processing
        collector.record_message_processing_time(150.5)
        assert collector.agent_metrics.messages_processed == 1
        assert len(collector.agent_metrics.message_processing_time_ms) == 1
        
        # Record classification result
        collector.record_classification_result(0.85, 0.92, "ACT_NOW")
        assert len(collector.agent_metrics.classification_accuracy) == 1
        assert len(collector.agent_metrics.classification_confidence) == 1
        
        # Record delegation result
        collector.record_delegation_result(True, 3, 0.78)
        assert collector.agent_metrics.tasks_delegated == 1
        assert collector.agent_metrics.subagents_activated == 3
        assert len(collector.agent_metrics.delegation_success_rate) == 1
    
    def test_metric_summary(self):
        """Test metric summary generation."""
        collector = MetricsCollector()
        
        # Add some test data
        values = [10.0, 15.0, 12.0, 18.0, 14.0]
        for value in values:
            collector.record_histogram("test_metric", value)
        
        summary = collector.get_metric_summary("test_metric", window_minutes=60)
        
        assert summary["count"] == 5
        assert summary["min"] == 10.0
        assert summary["max"] == 18.0
        assert summary["mean"] == 13.8
        assert summary["median"] == 14.0
    
    def test_performance_summary(self):
        """Test agent performance summary."""
        collector = MetricsCollector()
        
        # Add some performance data
        collector.record_message_processing_time(100.0)
        collector.record_classification_result(0.9, 0.85, "ACT_NOW")
        collector.record_delegation_result(True, 2, 0.8)
        
        summary = collector.get_agent_performance_summary()
        
        assert "performance" in summary
        assert "decision_quality" in summary
        assert "system_health" in summary
        assert "activity_counters" in summary
        assert summary["activity_counters"]["messages_processed"] == 1


class TestAnomalyDetector:
    """Test cases for anomaly detection functionality."""
    
    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = AnomalyDetector(check_interval_seconds=30)
        assert detector.check_interval_seconds == 30
        assert len(detector.thresholds) > 0  # Should have default thresholds
        assert not detector._detection_running
    
    def test_threshold_management(self):
        """Test anomaly threshold management."""
        detector = AnomalyDetector()
        initial_count = len(detector.thresholds)
        
        # Add custom threshold
        threshold = AnomalyThreshold(
            metric_name="custom_metric",
            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
            severity=AnomalySeverity.HIGH,
            threshold_type="absolute",
            threshold_value=100.0
        )
        detector.add_threshold(threshold)
        
        assert len(detector.thresholds) == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_detection_lifecycle(self):
        """Test anomaly detection start/stop lifecycle."""
        detector = AnomalyDetector(check_interval_seconds=0.1)
        
        # Start detection
        await detector.start_detection()
        assert detector._detection_running == True
        assert detector._detection_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop detection
        await detector.stop_detection()
        assert detector._detection_running == False
    
    def test_alert_management(self):
        """Test anomaly alert management."""
        detector = AnomalyDetector()
        
        # Initially no alerts
        assert len(detector.get_active_alerts()) == 0
        
        # Simulate alert creation (would normally be done by detection loop)
        from backend.observability.anomaly_detector import AnomalyAlert
        alert = AnomalyAlert(
            id="test_alert",
            anomaly_type=AnomalyType.ERROR_SPIKE,
            severity=AnomalySeverity.HIGH,
            metric_name="error_rate",
            current_value=10.0,
            expected_range=(0.0, 5.0),
            deviation_score=2.5,
            description="Test alert",
            timestamp=datetime.now(timezone.utc)
        )
        
        detector.active_alerts["test_key"] = alert
        detector.alert_history.append(alert)
        
        # Check active alerts
        active_alerts = detector.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].id == "test_alert"
        
        # Resolve alert
        success = detector.resolve_alert("test_alert")
        assert success == True
        assert len(detector.get_active_alerts()) == 0
    
    def test_detection_statistics(self):
        """Test detection statistics."""
        detector = AnomalyDetector()
        stats = detector.get_detection_statistics()
        
        assert "detection_status" in stats
        assert "active_alerts" in stats
        assert "total_alerts_generated" in stats
        assert "configured_thresholds" in stats
        assert "timestamp" in stats


class TestTraceAnalyzer:
    """Test cases for trace analysis functionality."""
    
    def create_sample_traces(self) -> List[AgentTrace]:
        """Create sample traces for testing."""
        traces = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(10):
            trace = AgentTrace(
                trace_id=f"trace_{i}",
                operation_name="test_operation",
                agent_type="TestAgent",
                start_time=base_time + timedelta(seconds=i),
                end_time=base_time + timedelta(seconds=i, milliseconds=100 + i * 10),
                duration_ms=100 + i * 10,
                success=i < 8,  # 2 failures
                error="Test error" if i >= 8 else None
            )
            traces.append(trace)
        
        return traces
    
    def test_performance_analysis(self):
        """Test performance metrics analysis."""
        analyzer = TraceAnalyzer()
        traces = self.create_sample_traces()
        
        metrics = analyzer.analyze_performance(traces)
        
        assert metrics.total_operations == 10
        assert metrics.success_rate == 0.8  # 8 out of 10 successful
        assert metrics.error_rate == 0.2   # 2 out of 10 failed
        assert metrics.avg_duration_ms > 0
        assert metrics.min_duration_ms == 100
        assert metrics.max_duration_ms == 190
    
    def test_decision_pattern_analysis(self):
        """Test decision pattern analysis."""
        analyzer = TraceAnalyzer()
        traces = self.create_sample_traces()
        
        # Add classification metadata to traces
        for i, trace in enumerate(traces):
            trace.metadata["classification"] = {
                "type": "ACT_NOW" if i % 2 == 0 else "ARCHIVE",
                "confidence": 0.8 + (i * 0.02),
                "accuracy": 0.9 - (i * 0.01)
            }
        
        analysis = analyzer.analyze_decision_patterns(traces)
        
        assert len(analysis.decision_types) == 2
        assert "ACT_NOW" in analysis.decision_types
        assert "ARCHIVE" in analysis.decision_types
        assert len(analysis.confidence_distribution) == 10
        assert len(analysis.accuracy_metrics) > 0
    
    def test_insight_generation(self):
        """Test insight generation from traces."""
        analyzer = TraceAnalyzer()
        traces = self.create_sample_traces()
        
        # Make some traces slow to trigger performance insights
        for trace in traces[-3:]:
            trace.duration_ms = 8000  # 8 seconds
        
        insights = analyzer.generate_insights(traces, analysis_window_hours=1)
        
        # Should generate some insights
        assert len(insights) > 0
        
        # Check insight structure
        for insight in insights:
            assert hasattr(insight, 'insight_type')
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'confidence')
            assert hasattr(insight, 'recommendations')
    
    def test_trace_summary(self):
        """Test trace summary generation."""
        analyzer = TraceAnalyzer()
        traces = self.create_sample_traces()
        
        summary = analyzer.generate_trace_summary(traces, group_by="agent_type")
        
        assert "overall_summary" in summary
        assert "group_summaries" in summary
        assert summary["overall_summary"]["total_traces"] == 10
        assert "TestAgent" in summary["group_summaries"]
    
    def test_trace_export(self):
        """Test trace export functionality."""
        analyzer = TraceAnalyzer()
        traces = self.create_sample_traces()
        
        # Test JSON export
        json_export = analyzer.export_traces_for_visualization(traces, "json")
        assert json_export["format"] == "json"
        assert len(json_export["traces"]) == 10
        
        # Test timeline export
        timeline_export = analyzer.export_traces_for_visualization(traces, "timeline")
        assert timeline_export["format"] == "timeline"
        assert len(timeline_export["events"]) == 10
        
        # Test CSV export
        csv_export = analyzer.export_traces_for_visualization(traces, "csv")
        assert csv_export["format"] == "csv"
        assert len(csv_export["rows"]) == 10


class TestObservabilityIntegration:
    """Integration tests for the complete observability system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_observability(self):
        """Test end-to-end observability workflow."""
        # Initialize components
        tracer = LangSmithTracer()
        metrics = MetricsCollector()
        detector = AnomalyDetector(check_interval_seconds=0.1)
        analyzer = TraceAnalyzer(tracer)
        
        # Start a trace
        trace_id = await tracer.start_trace(
            operation_name="integration_test",
            agent_type="TestAgent",
            inputs={"test": "data"}
        )
        
        # Record some metrics
        timer_id = metrics.start_timer("test_operation_time")
        
        # Simulate some work
        await asyncio.sleep(0.01)
        
        # Complete the trace and metrics
        duration = metrics.stop_timer("test_operation_time", timer_id)
        await tracer.end_trace(
            trace_id=trace_id,
            outputs={"result": "success"},
            success=True
        )
        
        # Record additional metrics
        metrics.record_message_processing_time(duration)
        metrics.record_classification_result(0.9, 0.85, "ACT_NOW")
        
        # Check that everything was recorded
        assert len(tracer.get_active_traces()) == 0  # Trace completed
        assert metrics.agent_metrics.messages_processed == 1
        
        # Get performance summary
        summary = metrics.get_agent_performance_summary()
        assert summary["activity_counters"]["messages_processed"] == 1
        
        # Test anomaly detection setup
        await detector.start_detection()
        await asyncio.sleep(0.2)  # Let it run briefly
        await detector.stop_detection()
        
        # Verify no anomalies detected (normal operation)
        assert len(detector.get_active_alerts()) == 0
    
    def test_global_instances(self):
        """Test global instance management."""
        # Test that global instances work correctly
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2  # Should be same instance
        
        metrics1 = get_metrics_collector()
        metrics2 = get_metrics_collector()
        assert metrics1 is metrics2  # Should be same instance
        
        detector1 = get_anomaly_detector()
        detector2 = get_anomaly_detector()
        assert detector1 is detector2  # Should be same instance
    
    def test_configuration_management(self):
        """Test observability configuration management."""
        # Test tracer configuration
        config = TraceConfig(
            project_name="test-config",
            enabled=True,
            sample_rate=0.5
        )
        configure_tracer(config)
        
        tracer = get_tracer()
        assert tracer.config.project_name == "test-config"
        assert tracer.config.sample_rate == 0.5


@pytest.fixture
def sample_traces():
    """Fixture providing sample traces for testing."""
    traces = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(5):
        trace = AgentTrace(
            trace_id=f"fixture_trace_{i}",
            operation_name=f"operation_{i % 3}",
            agent_type="FixtureAgent",
            start_time=base_time + timedelta(seconds=i),
            end_time=base_time + timedelta(seconds=i, milliseconds=50 + i * 5),
            duration_ms=50 + i * 5,
            success=True,
            inputs={"input": f"value_{i}"},
            outputs={"output": f"result_{i}"}
        )
        traces.append(trace)
    
    return traces


def test_with_sample_traces(sample_traces):
    """Test using the sample traces fixture."""
    assert len(sample_traces) == 5
    assert all(trace.success for trace in sample_traces)
    assert sample_traces[0].operation_name == "operation_0"
    assert sample_traces[1].operation_name == "operation_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
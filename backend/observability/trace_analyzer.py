"""
Trace visualization and analysis tools for agent operations.

This module provides comprehensive analysis of agent traces, including
performance insights, decision pattern analysis, and visualization tools.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, Counter

from .langsmith_tracer import AgentTrace, LangSmithTracer

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be generated from traces."""
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    DECISION_PATTERN = "decision_pattern"
    ERROR_CORRELATION = "error_correlation"
    EFFICIENCY_OPPORTUNITY = "efficiency_opportunity"
    RESOURCE_USAGE = "resource_usage"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"


@dataclass
class TraceInsight:
    """Represents an insight derived from trace analysis."""
    insight_type: InsightType
    title: str
    description: str
    severity: str  # "info", "warning", "critical"
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PerformanceMetrics:
    """Performance metrics derived from traces."""
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    total_operations: int
    success_rate: float
    error_rate: float


@dataclass
class DecisionAnalysis:
    """Analysis of decision-making patterns."""
    decision_types: Dict[str, int]
    confidence_distribution: List[float]
    accuracy_metrics: Dict[str, float]
    pattern_frequency: Dict[str, int]
    decision_latency_ms: List[float]


class TraceAnalyzer:
    """
    Comprehensive trace analysis system for agent operations.
    
    Analyzes collected traces to provide insights into performance,
    decision quality, and optimization opportunities.
    """
    
    def __init__(self, tracer: Optional[LangSmithTracer] = None):
        """
        Initialize the trace analyzer.
        
        Args:
            tracer: LangSmith tracer instance for accessing traces
        """
        self.tracer = tracer
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_ttl_minutes = 15
        
        logger.info("TraceAnalyzer initialized")
    
    def analyze_performance(
        self,
        traces: List[AgentTrace],
        operation_filter: Optional[str] = None,
        agent_filter: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Analyze performance metrics from traces.
        
        Args:
            traces: List of traces to analyze
            operation_filter: Optional filter for specific operations
            agent_filter: Optional filter for specific agent types
            
        Returns:
            PerformanceMetrics with comprehensive performance data
        """
        # Filter traces based on criteria
        filtered_traces = self._filter_traces(traces, operation_filter, agent_filter)
        
        if not filtered_traces:
            return PerformanceMetrics(
                avg_duration_ms=0.0,
                min_duration_ms=0.0,
                max_duration_ms=0.0,
                p50_duration_ms=0.0,
                p95_duration_ms=0.0,
                p99_duration_ms=0.0,
                total_operations=0,
                success_rate=0.0,
                error_rate=0.0
            )
        
        # Extract durations and success status
        durations = [trace.duration_ms for trace in filtered_traces if trace.duration_ms is not None]
        successes = [trace.success for trace in filtered_traces]
        
        if not durations:
            durations = [0.0]
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        n = len(sorted_durations)
        
        def percentile(p: float) -> float:
            index = int(p * n / 100)
            if index >= n:
                index = n - 1
            return sorted_durations[index]
        
        # Calculate metrics
        success_count = sum(successes)
        total_count = len(successes)
        
        return PerformanceMetrics(
            avg_duration_ms=statistics.mean(durations),
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            p50_duration_ms=percentile(50),
            p95_duration_ms=percentile(95),
            p99_duration_ms=percentile(99),
            total_operations=total_count,
            success_rate=success_count / total_count if total_count > 0 else 0.0,
            error_rate=(total_count - success_count) / total_count if total_count > 0 else 0.0
        )
    
    def analyze_decision_patterns(
        self,
        traces: List[AgentTrace],
        decision_key: str = "classification"
    ) -> DecisionAnalysis:
        """
        Analyze decision-making patterns from traces.
        
        Args:
            traces: List of traces to analyze
            decision_key: Key in trace metadata containing decision information
            
        Returns:
            DecisionAnalysis with decision pattern insights
        """
        decision_types = Counter()
        confidence_values = []
        accuracy_values = []
        decision_latencies = []
        pattern_frequency = Counter()
        
        for trace in traces:
            # Extract decision information from metadata
            if decision_key in trace.metadata:
                decision_info = trace.metadata[decision_key]
                
                if isinstance(decision_info, dict):
                    # Extract decision type
                    if "type" in decision_info:
                        decision_types[decision_info["type"]] += 1
                    
                    # Extract confidence
                    if "confidence" in decision_info:
                        confidence_values.append(float(decision_info["confidence"]))
                    
                    # Extract accuracy (if available)
                    if "accuracy" in decision_info:
                        accuracy_values.append(float(decision_info["accuracy"]))
                    
                    # Extract pattern information
                    if "pattern" in decision_info:
                        pattern_frequency[decision_info["pattern"]] += 1
            
            # Decision latency (time to make decision)
            if trace.duration_ms is not None:
                decision_latencies.append(trace.duration_ms)
        
        # Calculate accuracy metrics
        accuracy_metrics = {}
        if accuracy_values:
            accuracy_metrics = {
                "mean_accuracy": statistics.mean(accuracy_values),
                "min_accuracy": min(accuracy_values),
                "max_accuracy": max(accuracy_values),
                "accuracy_std": statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0
            }
        
        return DecisionAnalysis(
            decision_types=dict(decision_types),
            confidence_distribution=confidence_values,
            accuracy_metrics=accuracy_metrics,
            pattern_frequency=dict(pattern_frequency),
            decision_latency_ms=decision_latencies
        )
    
    def generate_insights(
        self,
        traces: List[AgentTrace],
        analysis_window_hours: int = 24
    ) -> List[TraceInsight]:
        """
        Generate actionable insights from trace analysis.
        
        Args:
            traces: List of traces to analyze
            analysis_window_hours: Time window for analysis
            
        Returns:
            List of TraceInsight objects with recommendations
        """
        insights = []
        
        # Filter traces to analysis window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=analysis_window_hours)
        recent_traces = [
            trace for trace in traces
            if trace.start_time >= cutoff_time
        ]
        
        if not recent_traces:
            return insights
        
        # Performance analysis
        perf_metrics = self.analyze_performance(recent_traces)
        insights.extend(self._generate_performance_insights(perf_metrics, recent_traces))
        
        # Decision pattern analysis
        decision_analysis = self.analyze_decision_patterns(recent_traces)
        insights.extend(self._generate_decision_insights(decision_analysis, recent_traces))
        
        # Error correlation analysis
        insights.extend(self._generate_error_insights(recent_traces))
        
        # Resource usage analysis
        insights.extend(self._generate_resource_insights(recent_traces))
        
        return sorted(insights, key=lambda x: x.confidence, reverse=True)
    
    def _generate_performance_insights(
        self,
        metrics: PerformanceMetrics,
        traces: List[AgentTrace]
    ) -> List[TraceInsight]:
        """Generate performance-related insights."""
        insights = []
        
        # High latency insight
        if metrics.p95_duration_ms > 5000:  # 5 seconds
            insights.append(TraceInsight(
                insight_type=InsightType.PERFORMANCE_BOTTLENECK,
                title="High Latency Detected",
                description=f"95th percentile latency is {metrics.p95_duration_ms:.0f}ms, indicating performance bottlenecks",
                severity="warning" if metrics.p95_duration_ms < 10000 else "critical",
                confidence=0.9,
                supporting_data={
                    "p95_latency_ms": metrics.p95_duration_ms,
                    "avg_latency_ms": metrics.avg_duration_ms,
                    "max_latency_ms": metrics.max_duration_ms
                },
                recommendations=[
                    "Investigate slow operations using trace details",
                    "Consider optimizing LLM calls or database queries",
                    "Implement caching for frequently accessed data"
                ]
            ))
        
        # High error rate insight
        if metrics.error_rate > 0.1:  # 10% error rate
            insights.append(TraceInsight(
                insight_type=InsightType.ERROR_CORRELATION,
                title="High Error Rate",
                description=f"Error rate is {metrics.error_rate:.1%}, indicating system reliability issues",
                severity="critical" if metrics.error_rate > 0.2 else "warning",
                confidence=0.95,
                supporting_data={
                    "error_rate": metrics.error_rate,
                    "total_operations": metrics.total_operations,
                    "failed_operations": int(metrics.total_operations * metrics.error_rate)
                },
                recommendations=[
                    "Review error logs for common failure patterns",
                    "Implement better error handling and retry logic",
                    "Add circuit breakers for external service calls"
                ]
            ))
        
        return insights
    
    def _generate_decision_insights(
        self,
        analysis: DecisionAnalysis,
        traces: List[AgentTrace]
    ) -> List[TraceInsight]:
        """Generate decision-related insights."""
        insights = []
        
        # Low confidence decisions
        if analysis.confidence_distribution:
            avg_confidence = statistics.mean(analysis.confidence_distribution)
            if avg_confidence < 0.7:
                insights.append(TraceInsight(
                    insight_type=InsightType.DECISION_PATTERN,
                    title="Low Decision Confidence",
                    description=f"Average decision confidence is {avg_confidence:.2f}, indicating uncertainty in decision-making",
                    severity="warning",
                    confidence=0.8,
                    supporting_data={
                        "avg_confidence": avg_confidence,
                        "min_confidence": min(analysis.confidence_distribution),
                        "confidence_std": statistics.stdev(analysis.confidence_distribution) if len(analysis.confidence_distribution) > 1 else 0.0
                    },
                    recommendations=[
                        "Review and improve decision-making prompts",
                        "Add more training examples for edge cases",
                        "Consider ensemble methods for critical decisions"
                    ]
                ))
        
        # Decision pattern imbalance
        if analysis.decision_types:
            total_decisions = sum(analysis.decision_types.values())
            most_common = max(analysis.decision_types.values())
            if most_common / total_decisions > 0.8:  # 80% of decisions are the same type
                dominant_type = max(analysis.decision_types, key=analysis.decision_types.get)
                insights.append(TraceInsight(
                    insight_type=InsightType.DECISION_PATTERN,
                    title="Decision Pattern Imbalance",
                    description=f"80%+ of decisions are '{dominant_type}', indicating potential bias or limited input variety",
                    severity="info",
                    confidence=0.7,
                    supporting_data={
                        "decision_distribution": analysis.decision_types,
                        "dominant_type": dominant_type,
                        "dominance_percentage": most_common / total_decisions
                    },
                    recommendations=[
                        "Review input data for diversity",
                        "Check for bias in classification logic",
                        "Consider adjusting decision thresholds"
                    ]
                ))
        
        return insights
    
    def _generate_error_insights(self, traces: List[AgentTrace]) -> List[TraceInsight]:
        """Generate error correlation insights."""
        insights = []
        
        # Find error patterns
        error_traces = [trace for trace in traces if not trace.success and trace.error]
        
        if not error_traces:
            return insights
        
        # Group errors by type
        error_types = Counter()
        error_operations = Counter()
        
        for trace in error_traces:
            # Extract error type from error message
            error_msg = trace.error.lower()
            if "timeout" in error_msg:
                error_types["timeout"] += 1
            elif "connection" in error_msg:
                error_types["connection"] += 1
            elif "authentication" in error_msg:
                error_types["authentication"] += 1
            elif "rate limit" in error_msg:
                error_types["rate_limit"] += 1
            else:
                error_types["other"] += 1
            
            error_operations[trace.operation_name] += 1
        
        # Generate insights for common error patterns
        if error_types:
            most_common_error = max(error_types, key=error_types.get)
            error_count = error_types[most_common_error]
            
            if error_count >= 5:  # At least 5 occurrences
                insights.append(TraceInsight(
                    insight_type=InsightType.ERROR_CORRELATION,
                    title=f"Recurring {most_common_error.title()} Errors",
                    description=f"Detected {error_count} {most_common_error} errors, indicating a systematic issue",
                    severity="warning" if error_count < 10 else "critical",
                    confidence=0.85,
                    supporting_data={
                        "error_type": most_common_error,
                        "error_count": error_count,
                        "affected_operations": dict(error_operations),
                        "total_errors": len(error_traces)
                    },
                    recommendations=[
                        f"Investigate root cause of {most_common_error} errors",
                        "Implement specific error handling for this error type",
                        "Add monitoring alerts for this error pattern"
                    ]
                ))
        
        return insights
    
    def _generate_resource_insights(self, traces: List[AgentTrace]) -> List[TraceInsight]:
        """Generate resource usage insights."""
        insights = []
        
        # Analyze operation frequency
        operation_counts = Counter(trace.operation_name for trace in traces)
        
        if operation_counts:
            total_ops = sum(operation_counts.values())
            most_frequent_op = max(operation_counts, key=operation_counts.get)
            frequency = operation_counts[most_frequent_op]
            
            if frequency / total_ops > 0.5:  # More than 50% of operations
                insights.append(TraceInsight(
                    insight_type=InsightType.EFFICIENCY_OPPORTUNITY,
                    title="High Frequency Operation Detected",
                    description=f"Operation '{most_frequent_op}' accounts for {frequency/total_ops:.1%} of all operations",
                    severity="info",
                    confidence=0.8,
                    supporting_data={
                        "operation_name": most_frequent_op,
                        "frequency": frequency,
                        "percentage": frequency / total_ops,
                        "operation_distribution": dict(operation_counts)
                    },
                    recommendations=[
                        f"Consider optimizing '{most_frequent_op}' operation",
                        "Implement caching if operation is read-heavy",
                        "Review if operation frequency is expected"
                    ]
                ))
        
        return insights
    
    def _filter_traces(
        self,
        traces: List[AgentTrace],
        operation_filter: Optional[str] = None,
        agent_filter: Optional[str] = None
    ) -> List[AgentTrace]:
        """Filter traces based on criteria."""
        filtered = traces
        
        if operation_filter:
            filtered = [t for t in filtered if operation_filter in t.operation_name]
        
        if agent_filter:
            filtered = [t for t in filtered if agent_filter in t.agent_type]
        
        return filtered
    
    def generate_trace_summary(
        self,
        traces: List[AgentTrace],
        group_by: str = "operation_name"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of traces.
        
        Args:
            traces: List of traces to summarize
            group_by: Field to group traces by
            
        Returns:
            Dictionary with trace summary statistics
        """
        if not traces:
            return {"error": "No traces provided"}
        
        # Group traces
        groups = defaultdict(list)
        for trace in traces:
            key = getattr(trace, group_by, "unknown")
            groups[key].append(trace)
        
        # Generate summary for each group
        group_summaries = {}
        for group_name, group_traces in groups.items():
            perf_metrics = self.analyze_performance(group_traces)
            
            group_summaries[group_name] = {
                "count": len(group_traces),
                "success_rate": perf_metrics.success_rate,
                "avg_duration_ms": perf_metrics.avg_duration_ms,
                "p95_duration_ms": perf_metrics.p95_duration_ms,
                "error_count": len([t for t in group_traces if not t.success])
            }
        
        # Overall summary
        overall_perf = self.analyze_performance(traces)
        
        return {
            "overall_summary": {
                "total_traces": len(traces),
                "success_rate": overall_perf.success_rate,
                "avg_duration_ms": overall_perf.avg_duration_ms,
                "p95_duration_ms": overall_perf.p95_duration_ms,
                "time_range": {
                    "start": min(t.start_time for t in traces).isoformat(),
                    "end": max(t.end_time for t in traces if t.end_time).isoformat()
                }
            },
            "group_summaries": group_summaries,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def export_traces_for_visualization(
        self,
        traces: List[AgentTrace],
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Export traces in a format suitable for visualization tools.
        
        Args:
            traces: List of traces to export
            format_type: Export format ("json", "csv", "timeline")
            
        Returns:
            Formatted trace data for visualization
        """
        if format_type == "timeline":
            return self._export_timeline_format(traces)
        elif format_type == "csv":
            return self._export_csv_format(traces)
        else:  # Default to JSON
            return self._export_json_format(traces)
    
    def _export_timeline_format(self, traces: List[AgentTrace]) -> Dict[str, Any]:
        """Export traces in timeline format for visualization."""
        timeline_events = []
        
        for trace in traces:
            event = {
                "id": trace.trace_id,
                "name": f"{trace.agent_type}.{trace.operation_name}",
                "start": trace.start_time.isoformat(),
                "end": trace.end_time.isoformat() if trace.end_time else None,
                "duration_ms": trace.duration_ms,
                "success": trace.success,
                "agent_type": trace.agent_type,
                "operation": trace.operation_name,
                "error": trace.error
            }
            timeline_events.append(event)
        
        return {
            "format": "timeline",
            "events": timeline_events,
            "metadata": {
                "total_events": len(timeline_events),
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def _export_csv_format(self, traces: List[AgentTrace]) -> Dict[str, Any]:
        """Export traces in CSV-compatible format."""
        csv_rows = []
        
        for trace in traces:
            row = {
                "trace_id": trace.trace_id,
                "agent_type": trace.agent_type,
                "operation_name": trace.operation_name,
                "start_time": trace.start_time.isoformat(),
                "end_time": trace.end_time.isoformat() if trace.end_time else "",
                "duration_ms": trace.duration_ms or 0,
                "success": trace.success,
                "error": trace.error or "",
                "parent_trace_id": trace.parent_trace_id or ""
            }
            csv_rows.append(row)
        
        return {
            "format": "csv",
            "headers": list(csv_rows[0].keys()) if csv_rows else [],
            "rows": csv_rows,
            "metadata": {
                "total_rows": len(csv_rows),
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def _export_json_format(self, traces: List[AgentTrace]) -> Dict[str, Any]:
        """Export traces in JSON format."""
        json_traces = []
        
        for trace in traces:
            trace_dict = {
                "trace_id": trace.trace_id,
                "operation_name": trace.operation_name,
                "agent_type": trace.agent_type,
                "start_time": trace.start_time.isoformat(),
                "end_time": trace.end_time.isoformat() if trace.end_time else None,
                "duration_ms": trace.duration_ms,
                "inputs": trace.inputs,
                "outputs": trace.outputs,
                "metadata": trace.metadata,
                "error": trace.error,
                "success": trace.success,
                "parent_trace_id": trace.parent_trace_id,
                "child_traces": trace.child_traces
            }
            json_traces.append(trace_dict)
        
        return {
            "format": "json",
            "traces": json_traces,
            "metadata": {
                "total_traces": len(json_traces),
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
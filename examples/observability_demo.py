#!/usr/bin/env python3
"""
Observability and Error Handling System Demo

This script demonstrates the LangSmith observability, tracing capabilities,
and error handling systems of the conscious agent system, including metrics collection,
anomaly detection, trace analysis, circuit breakers, LLM error handling, and security error handling.
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import observability components
from backend.observability.langsmith_tracer import (
    LangSmithTracer, TraceConfig, get_tracer, configure_tracer
)
from backend.observability.metrics_collector import (
    MetricsCollector, get_metrics_collector
)
from backend.observability.anomaly_detector import (
    AnomalyDetector, get_anomaly_detector, setup_anomaly_detection
)
from backend.observability.trace_analyzer import TraceAnalyzer

# Import error handling components
from backend.tools.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError,
    get_circuit_breaker, circuit_breaker
)
from backend.tools.llm_error_handler import (
    LLMErrorHandler, LLMErrorType, LLMProviderType,
    get_llm_error_handler, with_llm_error_handling
)
from backend.tools.security_error_handler import (
    SecurityErrorHandler, SecurityErrorType, SecuritySeverity, SecurityAction,
    get_security_error_handler, with_security_error_handling
)


async def demo_tracing():
    """Demonstrate LangSmith tracing capabilities."""
    print("\n=== LangSmith Tracing Demo ===")
    
    # Configure tracer
    config = TraceConfig(
        project_name="observability-demo",
        enabled=True,
        sample_rate=1.0
    )
    configure_tracer(config)
    tracer = get_tracer()
    
    print(f"Tracer configured: {tracer.config.project_name}")
    
    # Simulate agent operations with tracing
    operations = [
        ("message_classification", "Observer", {"message": "Hello world"}),
        ("task_delegation", "Delegator", {"task": "Process user request"}),
        ("subagent_execution", "SubAgent", {"action": "analyze_sentiment"}),
    ]
    
    for op_name, agent_type, inputs in operations:
        print(f"Starting trace for {agent_type}.{op_name}")
        
        # Start trace
        trace_id = await tracer.start_trace(
            operation_name=op_name,
            agent_type=agent_type,
            inputs=inputs,
            metadata={"demo": True, "timestamp": datetime.now().isoformat()}
        )
        
        # Simulate work
        await asyncio.sleep(0.1)
        
        # End trace
        await tracer.end_trace(
            trace_id=trace_id,
            outputs={"result": f"Completed {op_name}"},
            success=True,
            additional_metadata={"processing_time": "100ms"}
        )
        
        print(f"Completed trace {trace_id}")
    
    # Show trace statistics
    stats = tracer.get_trace_statistics()
    print(f"Trace statistics: {stats}")


def demo_metrics():
    """Demonstrate metrics collection capabilities."""
    print("\n=== Metrics Collection Demo ===")
    
    collector = get_metrics_collector()
    
    # Simulate various metrics
    print("Recording performance metrics...")
    
    # Message processing times
    processing_times = [120, 95, 180, 110, 200, 85, 150]
    for time_ms in processing_times:
        collector.record_message_processing_time(time_ms)
    
    # Classification results
    classifications = [
        (0.95, 0.88, "ACT_NOW"),
        (0.82, 0.75, "ARCHIVE"),
        (0.91, 0.85, "ACT_NOW"),
        (0.78, 0.70, "DELAY"),
        (0.96, 0.92, "ACT_NOW")
    ]
    
    for accuracy, confidence, class_type in classifications:
        collector.record_classification_result(accuracy, confidence, class_type)
    
    # Delegation results
    delegations = [
        (True, 3, 0.85),
        (True, 2, 0.78),
        (False, 0, 0.0),
        (True, 1, 0.92),
        (True, 4, 0.80)
    ]
    
    for success, subagents, quality in delegations:
        collector.record_delegation_result(success, subagents, quality)
    
    # System health metrics
    collector.record_system_health(
        memory_usage_mb=256.5,
        active_connections=12,
        error_rate=0.02
    )
    
    # Show performance summary
    summary = collector.get_agent_performance_summary()
    print("Performance Summary:")
    print(f"  Messages processed: {summary['activity_counters']['messages_processed']}")
    print(f"  Avg processing time: {summary['performance']['avg_message_processing_time_ms']:.1f}ms")
    print(f"  Avg classification accuracy: {summary['decision_quality']['avg_classification_confidence']:.2f}")
    print(f"  Tasks delegated: {summary['activity_counters']['tasks_delegated']}")


async def demo_anomaly_detection():
    """Demonstrate anomaly detection capabilities."""
    print("\n=== Anomaly Detection Demo ===")
    
    detector = get_anomaly_detector()
    collector = get_metrics_collector()
    
    print("Setting up anomaly detection...")
    
    # Add some normal metrics first
    for i in range(20):
        collector.record_gauge("response_time_ms", 100 + (i % 10) * 5)
        await asyncio.sleep(0.01)
    
    print("Recording normal metrics...")
    
    # Now add some anomalous values
    print("Introducing anomalies...")
    collector.record_gauge("response_time_ms", 500)  # Spike
    collector.record_gauge("response_time_ms", 600)  # Another spike
    collector.record_gauge("error_rate", 10.0)       # High error rate
    
    # Start detection briefly
    await detector.start_detection()
    await asyncio.sleep(0.5)  # Let it run briefly
    await detector.stop_detection()
    
    # Check for alerts
    active_alerts = detector.get_active_alerts()
    print(f"Active anomaly alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        print(f"  ALERT: {alert.anomaly_type.value} - {alert.description}")
    
    # Show detection statistics
    stats = detector.get_detection_statistics()
    print(f"Detection statistics: {stats}")


def demo_trace_analysis():
    """Demonstrate trace analysis capabilities."""
    print("\n=== Trace Analysis Demo ===")
    
    tracer = get_tracer()
    analyzer = TraceAnalyzer(tracer)
    
    # Create some sample traces for analysis
    from backend.observability.langsmith_tracer import AgentTrace
    
    sample_traces = []
    base_time = datetime.now(timezone.utc)
    
    # Create traces with varying performance
    trace_data = [
        ("classify_message", "Observer", 120, True),
        ("classify_message", "Observer", 95, True),
        ("classify_message", "Observer", 180, True),
        ("delegate_task", "Delegator", 250, True),
        ("delegate_task", "Delegator", 300, False),  # Failed
        ("execute_task", "SubAgent", 150, True),
        ("execute_task", "SubAgent", 2000, True),    # Slow
        ("execute_task", "SubAgent", 180, True),
    ]
    
    for i, (op_name, agent_type, duration, success) in enumerate(trace_data):
        trace = AgentTrace(
            trace_id=f"demo_trace_{i}",
            operation_name=op_name,
            agent_type=agent_type,
            start_time=base_time,
            end_time=base_time,
            duration_ms=duration,
            success=success,
            error="Simulated error" if not success else None,
            metadata={
                "classification": {
                    "type": "ACT_NOW" if i % 2 == 0 else "ARCHIVE",
                    "confidence": 0.8 + (i * 0.02),
                    "accuracy": 0.9 - (i * 0.01)
                }
            }
        )
        sample_traces.append(trace)
    
    # Analyze performance
    print("Analyzing performance metrics...")
    perf_metrics = analyzer.analyze_performance(sample_traces)
    print(f"  Total operations: {perf_metrics.total_operations}")
    print(f"  Success rate: {perf_metrics.success_rate:.1%}")
    print(f"  Avg duration: {perf_metrics.avg_duration_ms:.1f}ms")
    print(f"  P95 duration: {perf_metrics.p95_duration_ms:.1f}ms")
    
    # Analyze decision patterns
    print("Analyzing decision patterns...")
    decision_analysis = analyzer.analyze_decision_patterns(sample_traces)
    print(f"  Decision types: {decision_analysis.decision_types}")
    print(f"  Avg confidence: {sum(decision_analysis.confidence_distribution) / len(decision_analysis.confidence_distribution):.2f}")
    
    # Generate insights
    print("Generating insights...")
    insights = analyzer.generate_insights(sample_traces, analysis_window_hours=1)
    print(f"Generated {len(insights)} insights:")
    
    for insight in insights[:3]:  # Show top 3
        print(f"  {insight.title} ({insight.severity})")
        print(f"    {insight.description}")
        if insight.recommendations:
            print(f"    Recommendation: {insight.recommendations[0]}")
    
    # Generate trace summary
    summary = analyzer.generate_trace_summary(sample_traces, group_by="agent_type")
    print(f"Trace summary by agent type:")
    for agent_type, stats in summary["group_summaries"].items():
        print(f"  {agent_type}: {stats['count']} traces, {stats['success_rate']:.1%} success rate")


async def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern for external service calls."""
    print("\n=== Circuit Breaker Demo ===")
    
    # Create a circuit breaker with a low threshold for demo purposes
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=2.0,
        reset_timeout=5.0
    )
    cb = CircuitBreaker("demo-service", config)
    
    print("Testing circuit breaker with simulated service calls...")
    
    # Define test functions
    async def stable_service(param):
        """Simulated stable service that always succeeds."""
        await asyncio.sleep(0.1)
        return f"Success: {param}"
    
    async def unstable_service(param):
        """Simulated unstable service that sometimes fails."""
        await asyncio.sleep(0.1)
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Service temporarily unavailable")
        return f"Success: {param}"
    
    # Test with stable service
    print("Calling stable service...")
    for i in range(3):
        try:
            result = await cb.execute(stable_service, f"request-{i}")
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: Error - {e}")
    
    # Reset circuit breaker state
    cb.reset()
    
    # Test with unstable service
    print("\nCalling unstable service...")
    for i in range(6):
        try:
            result = await cb.execute(unstable_service, f"request-{i}")
            print(f"  Call {i+1}: {result}")
        except CircuitOpenError as e:
            print(f"  Call {i+1}: Circuit open - {e}")
        except Exception as e:
            print(f"  Call {i+1}: Error - {e}")
    
    # Show circuit breaker metrics
    print("\nCircuit breaker metrics:")
    metrics = cb.get_metrics()
    print(f"  State: {metrics['state']}")
    print(f"  Total calls: {metrics['total_calls']}")
    print(f"  Successful calls: {metrics['successful_calls']}")
    print(f"  Failed calls: {metrics['failed_calls']}")
    print(f"  Consecutive failures: {metrics['consecutive_failures']}")
    
    # Wait for recovery timeout and try again
    if cb.state == CircuitState.OPEN:
        print("\nWaiting for recovery timeout (2 seconds)...")
        await asyncio.sleep(2.1)
        
        print("Circuit should now be half-open, trying again...")
        try:
            result = await cb.execute(stable_service, "recovery-test")
            print(f"  Recovery call: {result}")
            print(f"  New circuit state: {cb.state.value}")
        except Exception as e:
            print(f"  Recovery call failed: {e}")
            print(f"  New circuit state: {cb.state.value}")


async def demo_llm_error_handler():
    """Demonstrate LLM API error handling with retry and fallback."""
    print("\n=== LLM Error Handler Demo ===")
    
    # Get the LLM error handler
    handler = get_llm_error_handler()
    
    print("Testing LLM error handling with simulated API calls...")
    
    # Define simulated LLM API functions
    async def stable_llm_call(prompt, **kwargs):
        """Simulated stable LLM that always succeeds."""
        await asyncio.sleep(0.1)
        return f"LLM response to: {prompt}"
    
    async def rate_limited_llm_call(prompt, **kwargs):
        """Simulated LLM that throws rate limit errors."""
        await asyncio.sleep(0.1)
        if random.random() < 0.7:  # 70% rate limit errors
            raise Exception("Rate limit exceeded. Please try again later.")
        return f"LLM response to: {prompt}"
    
    async def context_length_llm_call(prompt, **kwargs):
        """Simulated LLM that throws context length errors."""
        await asyncio.sleep(0.1)
        raise Exception("Maximum context length exceeded")
    
    # Test with stable LLM
    print("Calling stable LLM...")
    try:
        result = await handler.handle_with_retry(
            stable_llm_call, 
            LLMProviderType.OPENAI, 
            "gpt-4", 
            prompt="Hello, world!",
            max_retries=2
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test with rate limited LLM
    print("\nCalling rate-limited LLM (should retry)...")
    try:
        result = await handler.handle_with_retry(
            rate_limited_llm_call, 
            LLMProviderType.OPENAI, 
            "gpt-4", 
            prompt="Test with retries",
            max_retries=3,
            retry_delay=0.2
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  All retries failed: {e}")
    
    # Test with context length error and fallback
    print("\nCalling LLM with context length error (should try fallback)...")
    try:
        result = await handler.handle_with_retry(
            context_length_llm_call, 
            LLMProviderType.OPENAI, 
            "gpt-4", 
            prompt="Very long prompt",
            max_retries=1,
            fallback_models=["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  All models failed: {e}")
    
    # Show error statistics
    print("\nLLM error statistics:")
    stats = handler.get_error_statistics()
    print(f"  Total errors: {stats['total_errors']}")
    if stats['total_errors'] > 0:
        print(f"  Error types: {stats['error_types']}")
        print(f"  Providers: {stats['providers']}")
        print(f"  Models: {stats['models']}")
    
    # Demonstrate the decorator
    @with_llm_error_handling(LLMProviderType.OPENAI, "gpt-4", max_retries=2, retry_delay=0.1)
    async def decorated_llm_function(prompt):
        if "fail" in prompt:
            raise Exception("Rate limit exceeded")
        return f"Decorated response to: {prompt}"
    
    print("\nTesting decorator with successful call...")
    result = await decorated_llm_function("Hello from decorator")
    print(f"  Result: {result}")
    
    print("\nTesting decorator with failing call...")
    try:
        result = await decorated_llm_function("Please fail this request")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")


async def demo_security_error_handler():
    """Demonstrate security error containment and alerting system."""
    print("\n=== Security Error Handler Demo ===")
    
    # Get the security error handler
    handler = get_security_error_handler()
    
    print("Testing security error handling...")
    
    # Handle different types of security errors
    print("Handling different security error types:")
    
    # Sandbox escape attempt
    sandbox_alert = handler.handle_security_error(
        error_type=SecurityErrorType.SANDBOX_ESCAPE,
        description="Attempted to access file outside sandbox directory",
        source="sandbox_module"
    )
    print(f"  Created alert: {sandbox_alert.alert_id} - {sandbox_alert.error_type.value} ({sandbox_alert.severity.value})")
    
    # Unauthorized access
    auth_alert = handler.handle_security_error(
        error_type=SecurityErrorType.UNAUTHORIZED_ACCESS,
        description="Attempted to access restricted API endpoint",
        source="api_gateway"
    )
    print(f"  Created alert: {auth_alert.alert_id} - {auth_alert.error_type.value} ({auth_alert.severity.value})")
    
    # Malicious code
    code_alert = handler.handle_security_error(
        error_type=SecurityErrorType.MALICIOUS_CODE,
        description="Potentially harmful code pattern detected in user input",
        source="code_validator",
        context={"code_snippet": "import os; os.system('rm -rf /')"}
    )
    print(f"  Created alert: {code_alert.alert_id} - {code_alert.error_type.value} ({code_alert.severity.value})")
    
    # Resolve an alert
    print("\nResolving security alert...")
    handler.resolve_alert(
        alert_id=auth_alert.alert_id,
        resolution_notes="Updated access control rules"
    )
    print(f"  Resolved alert: {auth_alert.alert_id}")
    
    # Show active alerts
    active_alerts = handler.get_active_alerts()
    print(f"\nActive security alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  {alert.alert_id}: {alert.error_type.value} - {alert.description}")
    
    # Show alert statistics
    print("\nSecurity alert statistics:")
    stats = handler.get_alert_statistics()
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Active alerts: {stats['active_alerts']}")
    print(f"  Resolved alerts: {stats['resolved_alerts']}")
    print(f"  By severity: {stats['by_severity']}")
    print(f"  By type: {stats['by_type']}")
    
    # Demonstrate the decorator
    @with_security_error_handling("demo_function")
    async def secure_function(param):
        if param == "malicious":
            raise ValueError("Malicious code detected in input")
        elif param == "unauthorized":
            raise PermissionError("Unauthorized access attempt")
        return f"Secure result: {param}"
    
    print("\nTesting security decorator with safe input...")
    result = await secure_function("safe_input")
    print(f"  Result: {result}")
    
    print("\nTesting security decorator with malicious input...")
    try:
        result = await secure_function("malicious")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")


async def main():
    """Run the complete observability and error handling demo."""
    print("🔍 Conscious Agent System - Observability and Error Handling Demo")
    print("=" * 70)
    
    try:
        # Run observability demos
        await demo_tracing()
        demo_metrics()
        await demo_anomaly_detection()
        demo_trace_analysis()
        
        # Run error handling demos
        await demo_circuit_breaker()
        await demo_llm_error_handler()
        await demo_security_error_handler()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • LangSmith trace collection and monitoring")
        print("  • Custom metrics for performance and decision tracking")
        print("  • Real-time anomaly detection with alerting")
        print("  • Trace analysis with actionable insights")
        print("  • Circuit breaker pattern for external service protection")
        print("  • LLM API error handling with retry and fallback")
        print("  • Security error containment and alerting system")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    asyncio.run(main())
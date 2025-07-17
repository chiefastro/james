"""
Tests for the circuit breaker pattern implementation.

This module tests the circuit breaker functionality for protecting
against cascading failures when external services are unavailable.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta

from backend.tools.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError,
    CircuitBreakerRegistry, get_circuit_breaker, circuit_breaker
)


class TestCircuitBreaker:
    """Test cases for the CircuitBreaker class."""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful function execution."""
        cb = CircuitBreaker("test")
        
        # Mock function that succeeds
        mock_func = AsyncMock(return_value="success")
        
        # Execute with circuit breaker
        result = await cb.execute(mock_func, "arg1", kwarg1="value1")
        
        # Check result
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
        
        # Check metrics
        assert cb.metrics.total_calls == 1
        assert cb.metrics.successful_calls == 1
        assert cb.metrics.failed_calls == 0
        assert cb.metrics.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_failure_below_threshold(self):
        """Test handling of failures below the threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        # Mock function that fails
        mock_func = AsyncMock(side_effect=ValueError("test error"))
        
        # Execute with circuit breaker - should fail but keep circuit closed
        with pytest.raises(ValueError):
            await cb.execute(mock_func)
        
        # Check metrics
        assert cb.metrics.total_calls == 1
        assert cb.metrics.successful_calls == 0
        assert cb.metrics.failed_calls == 1
        assert cb.metrics.consecutive_failures == 1
        assert cb.state == CircuitState.CLOSED  # Still closed
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test circuit opens after threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        # Mock function that fails
        mock_func = AsyncMock(side_effect=ValueError("test error"))
        
        # Fail 3 times (threshold)
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.execute(mock_func)
        
        # Check circuit is now open
        assert cb.state == CircuitState.OPEN
        assert cb.metrics.consecutive_failures == 3
        
        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await cb.execute(mock_func)
    
    @pytest.mark.asyncio
    async def test_half_open_state_success(self):
        """Test transition from open to half-open to closed on success."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1  # Short timeout for testing
        )
        cb = CircuitBreaker("test", config)
        
        # Mock functions
        fail_func = AsyncMock(side_effect=ValueError("test error"))
        success_func = AsyncMock(return_value="success")
        
        # Fail enough to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should now be half-open and allow one call
        result = await cb.execute(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED  # Success should close circuit
    
    @pytest.mark.asyncio
    async def test_half_open_state_failure(self):
        """Test transition from open to half-open back to open on failure."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1  # Short timeout for testing
        )
        cb = CircuitBreaker("test", config)
        
        # Mock function that fails
        fail_func = AsyncMock(side_effect=ValueError("test error"))
        
        # Fail enough to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should now be half-open and allow one call, but it will fail
        with pytest.raises(ValueError):
            await cb.execute(fail_func)
        
        assert cb.state == CircuitState.OPEN  # Failure should reopen circuit
    
    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test excluded exceptions don't count toward failure threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions={KeyError}
        )
        cb = CircuitBreaker("test", config)
        
        # Mock functions for different exceptions
        value_error_func = AsyncMock(side_effect=ValueError("counted failure"))
        key_error_func = AsyncMock(side_effect=KeyError("excluded failure"))
        
        # KeyError should not count toward threshold
        with pytest.raises(KeyError):
            await cb.execute(key_error_func)
        
        assert cb.metrics.consecutive_failures == 0
        
        # ValueError should count
        with pytest.raises(ValueError):
            await cb.execute(value_error_func)
        
        assert cb.metrics.consecutive_failures == 1
        assert cb.state == CircuitState.CLOSED
        
        # Another ValueError should open circuit
        with pytest.raises(ValueError):
            await cb.execute(value_error_func)
        
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_included_exceptions(self):
        """Test only included exceptions count toward failure threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            included_exceptions={ValueError}  # Only ValueError counts
        )
        cb = CircuitBreaker("test", config)
        
        # Mock functions for different exceptions
        value_error_func = AsyncMock(side_effect=ValueError("counted failure"))
        key_error_func = AsyncMock(side_effect=KeyError("not counted failure"))
        
        # KeyError should not count toward threshold
        with pytest.raises(KeyError):
            await cb.execute(key_error_func)
        
        assert cb.metrics.consecutive_failures == 0
        
        # ValueError should count
        with pytest.raises(ValueError):
            await cb.execute(value_error_func)
        
        assert cb.metrics.consecutive_failures == 1
        
        # Another ValueError should open circuit
        with pytest.raises(ValueError):
            await cb.execute(value_error_func)
        
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_reset_method(self):
        """Test manual reset of circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        
        # Mock function that fails
        mock_func = AsyncMock(side_effect=ValueError("test error"))
        
        # Fail enough to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(mock_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Reset circuit breaker
        cb.reset()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.metrics.consecutive_failures == 0
        
        # Should allow calls again
        with pytest.raises(ValueError):
            await cb.execute(mock_func)
        
        assert cb.metrics.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_half_open_max_calls(self):
        """Test half-open state respects max calls limit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1  # Only allow 1 call in half-open state
        )
        cb = CircuitBreaker("test", config)
        
        # Mock functions
        fail_func = AsyncMock(side_effect=ValueError("test error"))
        success_func = AsyncMock(return_value="success")
        
        # Fail enough to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should now be half-open and allow one call
        result1 = await cb.execute(success_func)
        assert result1 == "success"
        
        # Second call should be allowed since the first one succeeded and closed the circuit
        result2 = await cb.execute(success_func)
        assert result2 == "success"
        
        # Circuit should now be closed
        assert cb.state == CircuitState.CLOSED
    
    def test_get_metrics(self):
        """Test getting circuit breaker metrics."""
        cb = CircuitBreaker("test_metrics")
        
        # Add some test data to metrics
        cb.metrics.total_calls = 10
        cb.metrics.successful_calls = 8
        cb.metrics.failed_calls = 2
        cb.metrics.consecutive_failures = 0
        cb.metrics.last_failure_time = datetime.now(timezone.utc)
        cb.metrics.open_circuits_count = 1
        
        metrics = cb.get_metrics()
        
        assert metrics["name"] == "test_metrics"
        assert metrics["state"] == "closed"
        assert metrics["total_calls"] == 10
        assert metrics["successful_calls"] == 8
        assert metrics["failed_calls"] == 2
        assert metrics["consecutive_failures"] == 0
        assert metrics["open_circuits_count"] == 1
        assert "last_failure_time" in metrics
        assert "last_success_time" in metrics


class TestCircuitBreakerRegistry:
    """Test cases for the CircuitBreakerRegistry class."""
    
    def test_singleton_pattern(self):
        """Test registry uses singleton pattern."""
        registry1 = CircuitBreakerRegistry.get_instance()
        registry2 = CircuitBreakerRegistry.get_instance()
        
        assert registry1 is registry2
    
    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        
        # Create new circuit breaker
        cb1 = registry.get_or_create("service1")
        assert cb1.name == "service1"
        
        # Get existing circuit breaker
        cb2 = registry.get_or_create("service1")
        assert cb1 is cb2  # Same instance
        
        # Create another circuit breaker
        cb3 = registry.get_or_create("service2")
        assert cb3.name == "service2"
        assert cb1 is not cb3  # Different instance
    
    def test_get_all_circuit_breakers(self):
        """Test getting all circuit breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        
        # Create some circuit breakers
        cb1 = registry.get_or_create("test_service1")
        cb2 = registry.get_or_create("test_service2")
        
        all_cbs = registry.get_all_circuit_breakers()
        
        assert "test_service1" in all_cbs
        assert "test_service2" in all_cbs
        assert all_cbs["test_service1"] is cb1
        assert all_cbs["test_service2"] is cb2
    
    def test_get_metrics(self):
        """Test getting metrics for all circuit breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        
        # Create some circuit breakers
        registry.get_or_create("metrics_test1")
        registry.get_or_create("metrics_test2")
        
        metrics = registry.get_metrics()
        
        assert "metrics_test1" in metrics
        assert "metrics_test2" in metrics
        assert metrics["metrics_test1"]["name"] == "metrics_test1"
        assert metrics["metrics_test2"]["name"] == "metrics_test2"
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        
        # Create circuit breakers and open them
        cb1 = registry.get_or_create("reset_test1", 
                                    CircuitBreakerConfig(failure_threshold=1))
        cb2 = registry.get_or_create("reset_test2", 
                                    CircuitBreakerConfig(failure_threshold=1))
        
        # Simulate failures to open circuits
        cb1.metrics.consecutive_failures = 1
        cb1.state = CircuitState.OPEN
        
        cb2.metrics.consecutive_failures = 1
        cb2.state = CircuitState.OPEN
        
        # Reset all
        registry.reset_all()
        
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
        assert cb1.metrics.consecutive_failures == 0
        assert cb2.metrics.consecutive_failures == 0


class TestCircuitBreakerHelpers:
    """Test cases for circuit breaker helper functions."""
    
    def test_get_circuit_breaker(self):
        """Test get_circuit_breaker helper function."""
        cb1 = get_circuit_breaker("helper_test1")
        cb2 = get_circuit_breaker("helper_test1")
        
        assert cb1 is cb2  # Same instance
        assert cb1.name == "helper_test1"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_async(self):
        """Test circuit_breaker decorator with async function."""
        # Create test async function
        @circuit_breaker("decorator_test_async")
        async def test_func(value):
            if value == "fail":
                raise ValueError("test error")
            return f"success-{value}"
        
        # Test successful call
        result = await test_func("pass")
        assert result == "success-pass"
        
        # Test failing calls
        for _ in range(5):  # Default threshold is 5
            with pytest.raises(ValueError):
                await test_func("fail")
        
        # Circuit should now be open
        with pytest.raises(CircuitOpenError):
            await test_func("pass")
    
    def test_circuit_breaker_decorator_sync(self):
        """Test circuit_breaker decorator with sync function."""
        # Create test sync function
        @circuit_breaker("decorator_test_sync")
        def test_func(value):
            if value == "fail":
                raise ValueError("test error")
            return f"success-{value}"
        
        # Test successful call
        result = test_func("pass")
        assert result == "success-pass"
        
        # Test failing calls
        for _ in range(5):  # Default threshold is 5
            with pytest.raises(ValueError):
                test_func("fail")
        
        # Circuit should now be open
        with pytest.raises(CircuitOpenError):
            test_func("pass")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
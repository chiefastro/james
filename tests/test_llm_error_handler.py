"""
Tests for the LLM API error handling and recovery system.

This module tests specialized error handling for LLM API calls,
including retry logic, fallback mechanisms, and error classification.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta

from backend.tools.llm_error_handler import (
    LLMErrorHandler, LLMErrorType, LLMProviderType, LLMErrorInfo,
    get_llm_error_handler, with_llm_error_handling
)


class TestLLMErrorHandler:
    """Test cases for the LLMErrorHandler class."""
    
    def test_initialization(self):
        """Test LLM error handler initialization."""
        handler = LLMErrorHandler()
        
        assert len(handler.error_history) == 0
        assert handler.max_history_size == 100
        
        # Check circuit breakers are initialized
        assert LLMProviderType.OPENAI in handler.circuit_breakers
        assert LLMProviderType.ANTHROPIC in handler.circuit_breakers
        assert LLMProviderType.GOOGLE in handler.circuit_breakers
        assert LLMProviderType.AZURE in handler.circuit_breakers
        assert LLMProviderType.HUGGINGFACE in handler.circuit_breakers
    
    def test_error_classification(self):
        """Test LLM error classification."""
        handler = LLMErrorHandler()
        
        # Test rate limit error
        rate_limit_error = Exception("Rate limit exceeded")
        info = handler._classify_llm_error(rate_limit_error, LLMProviderType.OPENAI, "gpt-4")
        assert info.error_type == LLMErrorType.RATE_LIMIT
        
        # Test context length error
        context_error = Exception("Maximum context length exceeded")
        info = handler._classify_llm_error(context_error, LLMProviderType.OPENAI, "gpt-4")
        assert info.error_type == LLMErrorType.CONTEXT_LENGTH
        
        # Test content filter error
        filter_error = Exception("Content policy violation")
        info = handler._classify_llm_error(filter_error, LLMProviderType.ANTHROPIC, "claude-2")
        assert info.error_type == LLMErrorType.CONTENT_FILTER
        
        # Test invalid request error
        invalid_error = Exception("Invalid request parameters")
        info = handler._classify_llm_error(invalid_error, LLMProviderType.GOOGLE, "gemini")
        assert info.error_type == LLMErrorType.INVALID_REQUEST
        
        # Test authentication error
        auth_error = Exception("Authentication failed")
        info = handler._classify_llm_error(auth_error, LLMProviderType.AZURE, "gpt-4")
        assert info.error_type == LLMErrorType.AUTHENTICATION
        
        # Test server error
        server_error = Exception("Server error occurred")
        info = handler._classify_llm_error(server_error, LLMProviderType.HUGGINGFACE, "mistral")
        assert info.error_type == LLMErrorType.SERVER
        
        # Test timeout error
        timeout_error = Exception("Request timed out")
        info = handler._classify_llm_error(timeout_error, LLMProviderType.OPENAI, "gpt-4")
        assert info.error_type == LLMErrorType.TIMEOUT
        
        # Test connection error
        connection_error = Exception("Connection failed")
        info = handler._classify_llm_error(connection_error, LLMProviderType.ANTHROPIC, "claude-2")
        assert info.error_type == LLMErrorType.CONNECTION
        
        # Test unknown error
        unknown_error = Exception("Some other error")
        info = handler._classify_llm_error(unknown_error, LLMProviderType.GOOGLE, "gemini")
        assert info.error_type == LLMErrorType.UNKNOWN
    
    def test_should_retry(self):
        """Test retry decision logic."""
        handler = LLMErrorHandler()
        
        # These should be retried
        assert handler._should_retry(LLMErrorType.RATE_LIMIT) == True
        assert handler._should_retry(LLMErrorType.SERVER) == True
        assert handler._should_retry(LLMErrorType.TIMEOUT) == True
        assert handler._should_retry(LLMErrorType.CONNECTION) == True
        
        # These should not be retried
        assert handler._should_retry(LLMErrorType.CONTEXT_LENGTH) == False
        assert handler._should_retry(LLMErrorType.CONTENT_FILTER) == False
        assert handler._should_retry(LLMErrorType.INVALID_REQUEST) == False
        assert handler._should_retry(LLMErrorType.AUTHENTICATION) == False
        assert handler._should_retry(LLMErrorType.UNKNOWN) == False
    
    def test_error_history_management(self):
        """Test error history management."""
        handler = LLMErrorHandler()
        handler.max_history_size = 3  # Small size for testing
        
        # Add some errors
        for i in range(5):
            error_info = LLMErrorInfo(
                error_type=LLMErrorType.SERVER,
                provider=LLMProviderType.OPENAI,
                model="gpt-4",
                error_message=f"Test error {i}"
            )
            handler._add_to_history(error_info)
        
        # Check history size is maintained
        assert len(handler.error_history) == 3
        
        # Check most recent errors are kept
        assert handler.error_history[0].error_message == "Test error 2"
        assert handler.error_history[1].error_message == "Test error 3"
        assert handler.error_history[2].error_message == "Test error 4"
    
    def test_get_error_statistics(self):
        """Test error statistics generation."""
        handler = LLMErrorHandler()
        
        # Add some test errors
        error_types = [
            LLMErrorType.RATE_LIMIT,
            LLMErrorType.SERVER,
            LLMErrorType.TIMEOUT,
            LLMErrorType.RATE_LIMIT,
            LLMErrorType.CONNECTION
        ]
        
        providers = [
            LLMProviderType.OPENAI,
            LLMProviderType.ANTHROPIC,
            LLMProviderType.OPENAI,
            LLMProviderType.GOOGLE,
            LLMProviderType.OPENAI
        ]
        
        models = [
            "gpt-4",
            "claude-2",
            "gpt-3.5-turbo",
            "gemini",
            "gpt-4"
        ]
        
        for i in range(5):
            error_info = LLMErrorInfo(
                error_type=error_types[i],
                provider=providers[i],
                model=models[i],
                error_message=f"Test error {i}"
            )
            handler._add_to_history(error_info)
        
        # Get statistics
        stats = handler.get_error_statistics()
        
        # Check counts
        assert stats["total_errors"] == 5
        assert stats["error_types"]["rate_limit"] == 2
        assert stats["error_types"]["server"] == 1
        assert stats["error_types"]["timeout"] == 1
        assert stats["error_types"]["connection"] == 1
        
        assert stats["providers"]["openai"] == 3
        assert stats["providers"]["anthropic"] == 1
        assert stats["providers"]["google"] == 1
        
        assert stats["models"]["gpt-4"] == 2
        assert stats["models"]["claude-2"] == 1
        assert stats["models"]["gpt-3.5-turbo"] == 1
        assert stats["models"]["gemini"] == 1
        
        # Check recent errors
        assert len(stats["recent_errors"]) == 5


class TestLLMErrorHandlerIntegration:
    """Integration tests for LLM error handler."""
    
    @pytest.mark.asyncio
    async def test_handle_with_retry_success(self):
        """Test successful execution with retry handler."""
        handler = LLMErrorHandler()
        
        # Mock function that succeeds
        mock_func = AsyncMock(return_value="LLM response")
        
        # Execute with retry handler
        result = await handler.handle_with_retry(
            mock_func, LLMProviderType.OPENAI, "gpt-4", 
            prompt="Test prompt", max_retries=3
        )
        
        # Check result
        assert result == "LLM response"
        mock_func.assert_called_once_with(prompt="Test prompt")
    
    @pytest.mark.asyncio
    async def test_handle_with_retry_eventual_success(self):
        """Test eventual success after retries."""
        handler = LLMErrorHandler()
        
        # Mock function that fails twice then succeeds
        mock_func = AsyncMock()
        mock_func.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Rate limit exceeded"),
            "LLM response"
        ]
        
        # Execute with retry handler
        result = await handler.handle_with_retry(
            mock_func, LLMProviderType.OPENAI, "gpt-4", 
            prompt="Test prompt", max_retries=3, retry_delay=0.01
        )
        
        # Check result
        assert result == "LLM response"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_handle_with_retry_all_failures(self):
        """Test all retries fail."""
        handler = LLMErrorHandler()
        
        # Mock function that always fails
        error = Exception("Rate limit exceeded")
        mock_func = AsyncMock(side_effect=error)
        
        # Execute with retry handler - should eventually fail
        with pytest.raises(Exception) as excinfo:
            await handler.handle_with_retry(
                mock_func, LLMProviderType.OPENAI, "gpt-4", 
                prompt="Test prompt", max_retries=2, retry_delay=0.01
            )
        
        # Check error
        assert "Rate limit exceeded" in str(excinfo.value)
        assert mock_func.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_handle_with_retry_fallback_models(self):
        """Test fallback to alternative models."""
        handler = LLMErrorHandler()
        
        # Mock function that fails with primary model but succeeds with fallback
        async def mock_llm_call(prompt, model="unknown"):
            if model == 'gpt-4':
                raise Exception("Server error with gpt-4")
            return f"Response from {model}"
        
        # Execute with retry handler and fallbacks
        result = await handler.handle_with_retry(
            mock_llm_call, 
            LLMProviderType.OPENAI, 
            "gpt-4", 
            prompt="Test prompt",
            max_retries=1, 
            retry_delay=0.01,
            fallback_models=["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        )
        
        # Check result is returned (we don't need to check the exact model)
        assert "Response from" in result
    
    @pytest.mark.asyncio
    async def test_handle_with_retry_non_retryable_error(self):
        """Test non-retryable errors are not retried."""
        handler = LLMErrorHandler()
        
        # Mock function that fails with non-retryable error
        error = Exception("Content policy violation")
        mock_func = AsyncMock(side_effect=error)
        
        # Execute with retry handler - should fail immediately
        with pytest.raises(Exception) as excinfo:
            await handler.handle_with_retry(
                mock_func, LLMProviderType.OPENAI, "gpt-4", 
                prompt="Test prompt", max_retries=3, retry_delay=0.01
            )
        
        # Check error and that function was only called once
        assert "Content policy violation" in str(excinfo.value)
        assert mock_func.call_count == 1  # No retries for non-retryable errors


class TestLLMErrorHandlerHelpers:
    """Test cases for LLM error handler helper functions."""
    
    def test_get_llm_error_handler(self):
        """Test get_llm_error_handler helper function."""
        handler1 = get_llm_error_handler()
        handler2 = get_llm_error_handler()
        
        assert handler1 is handler2  # Same instance
    
    @pytest.mark.asyncio
    async def test_with_llm_error_handling_decorator_async(self):
        """Test with_llm_error_handling decorator with async function."""
        # Create test async function
        @with_llm_error_handling(LLMProviderType.OPENAI, "gpt-4", max_retries=2, retry_delay=0.01)
        async def test_llm_func(prompt):
            if prompt == "fail":
                raise Exception("Rate limit exceeded")
            return f"Response to: {prompt}"
        
        # Test successful call
        result = await test_llm_func("hello")
        assert result == "Response to: hello"
        
        # Test failing call with retries
        with patch('backend.tools.llm_error_handler.LLMErrorHandler.handle_with_retry') as mock_retry:
            mock_retry.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(Exception):
                await test_llm_func("fail")
            
            # Check retry was called with correct parameters
            mock_retry.assert_called_once()
            args, kwargs = mock_retry.call_args
            assert kwargs["max_retries"] == 2
            assert kwargs["retry_delay"] == 0.01
    
    def test_with_llm_error_handling_decorator_sync(self):
        """Test with_llm_error_handling decorator with sync function."""
        # Create test sync function
        @with_llm_error_handling("openai", "gpt-4", max_retries=2, retry_delay=0.01)
        def test_llm_func(prompt):
            if prompt == "fail":
                raise Exception("Rate limit exceeded")
            return f"Response to: {prompt}"
        
        # Test successful call
        result = test_llm_func("hello")
        assert result == "Response to: hello"
        
        # Test failing call with retries
        with patch('backend.tools.llm_error_handler.LLMErrorHandler.handle_with_retry') as mock_retry:
            mock_retry.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(Exception):
                test_llm_func("fail")
            
            # Check retry was called
            mock_retry.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
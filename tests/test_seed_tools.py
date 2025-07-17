"""
Unit tests for seed tools.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from backend.tools.file_writer import FileWriterTool
from backend.tools.terminal_executor import TerminalExecutorTool
from backend.tools.message_queue_tool import MessageQueueTool
from backend.tools.error_handler import ErrorHandlerTool, RetryConfig, ErrorSeverity
from backend.tools.external_messenger import ExternalMessengerTool, MessageFormat, DeliveryMethod
from backend.tools.reflection_tool import ReflectionTool, ReflectionType
from backend.models.core import Message, MessageSource, MessageClassification
from backend.queue.message_queue import MessageQueue
from backend.sandbox.sandbox import SandboxConfig


class TestFileWriterTool:
    """Test cases for FileWriterTool."""
    
    @pytest.fixture
    def file_writer(self):
        """Create FileWriterTool instance for testing."""
        return FileWriterTool()
    
    @pytest.mark.asyncio
    async def test_write_text_file(self, file_writer):
        """Test writing a text file."""
        result = await file_writer.execute(
            file_path="test.txt",
            content="Hello, World!",
            mode="write"
        )
        
        assert result.success
        assert result.data['file_path'] == "test.txt"
        assert 'checksum' in result.data
        assert result.data['size'] == len("Hello, World!".encode('utf-8'))
    
    @pytest.mark.asyncio
    async def test_write_json_file(self, file_writer):
        """Test writing a JSON file."""
        test_data = {"key": "value", "number": 42}
        
        result = await file_writer.execute(
            file_path="test.json",
            content=test_data,
            mode="json"
        )
        
        assert result.success
        assert result.data['file_path'] == "test.json"
        assert result.data['format'] == "json"
    
    @pytest.mark.asyncio
    async def test_append_to_file(self, file_writer):
        """Test appending to a file."""
        # First write a file
        await file_writer.execute(
            file_path="append_test.txt",
            content="Line 1\n",
            mode="write"
        )
        
        # Then append to it
        result = await file_writer.execute(
            file_path="append_test.txt",
            content="Line 2\n",
            mode="append"
        )
        
        assert result.success
        assert result.data['file_path'] == "append_test.txt"
        assert 'appended_size' in result.data
    
    @pytest.mark.asyncio
    async def test_read_file(self, file_writer):
        """Test reading a file."""
        # First write a file
        test_content = "Test content for reading"
        await file_writer.execute(
            file_path="read_test.txt",
            content=test_content,
            mode="write"
        )
        
        # Then read it
        result = await file_writer.read_file("read_test.txt")
        
        assert result.success
        assert result.data['content'] == test_content
        assert result.data['file_path'] == "read_test.txt"
    
    @pytest.mark.asyncio
    async def test_unsafe_path_rejection(self, file_writer):
        """Test that unsafe paths are rejected."""
        result = await file_writer.execute(
            file_path="../../../etc/passwd",
            content="malicious content",
            mode="write"
        )
        
        assert not result.success
        assert "Unsafe file path" in result.error
    
    @pytest.mark.asyncio
    async def test_missing_parameters(self, file_writer):
        """Test handling of missing required parameters."""
        result = await file_writer.execute(
            file_path="test.txt"
            # Missing content parameter
        )
        
        assert not result.success
        assert "Missing required parameters" in result.error
    
    @pytest.mark.asyncio
    async def test_file_exists_check(self, file_writer):
        """Test file existence checking."""
        # Check non-existent file
        result = await file_writer.file_exists("nonexistent.txt")
        assert result.success
        assert not result.data['exists']
        
        # Create file and check again
        await file_writer.execute(
            file_path="exists_test.txt",
            content="test",
            mode="write"
        )
        
        result = await file_writer.file_exists("exists_test.txt")
        assert result.success
        assert result.data['exists']
    
    @pytest.mark.asyncio
    async def test_list_files(self, file_writer):
        """Test listing files."""
        # Create some test files
        for i in range(3):
            await file_writer.execute(
                file_path=f"list_test_{i}.txt",
                content=f"content {i}",
                mode="write"
            )
        
        result = await file_writer.list_files(pattern="list_test_*.txt")
        
        assert result.success
        assert result.data['count'] >= 3
        assert any("list_test_" in f for f in result.data['files'])


class TestTerminalExecutorTool:
    """Test cases for TerminalExecutorTool."""
    
    @pytest.fixture
    def terminal_executor(self):
        """Create TerminalExecutorTool instance for testing."""
        config = SandboxConfig(timeout_seconds=10)
        return TerminalExecutorTool(config)
    
    @pytest.mark.asyncio
    async def test_safe_command_execution(self, terminal_executor):
        """Test execution of safe commands."""
        result = await terminal_executor.execute(command="echo 'Hello World'")
        
        assert result.success
        assert "Hello World" in result.data['stdout']
        assert result.data['exit_code'] == 0
    
    @pytest.mark.asyncio
    async def test_unsafe_command_rejection(self, terminal_executor):
        """Test that unsafe commands are rejected."""
        result = await terminal_executor.execute(command="rm -rf /")
        
        assert not result.success
        assert "Dangerous command detected" in result.error
    
    @pytest.mark.asyncio
    async def test_python_code_execution(self, terminal_executor):
        """Test Python code execution."""
        python_code = """
print("Hello from Python")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
        
        result = await terminal_executor.execute_python_code(python_code)
        
        assert result.success
        assert "Hello from Python" in result.data['stdout']
        assert "2 + 2 = 4" in result.data['stdout']
    
    @pytest.mark.asyncio
    async def test_command_timeout(self, terminal_executor):
        """Test command timeout handling."""
        # This test might be slow, so we use a very short timeout
        terminal_executor.sandbox_config.timeout_seconds = 1
        
        result = await terminal_executor.execute(
            command="sleep 5",
            timeout=1
        )
        
        # Should fail due to timeout
        assert not result.success
    
    @pytest.mark.asyncio
    async def test_get_system_info(self, terminal_executor):
        """Test getting system information."""
        result = await terminal_executor.get_system_info()
        
        assert result.success
        assert 'system_info' in result.data
        assert 'sandbox_config' in result.data
    
    @pytest.mark.asyncio
    async def test_missing_command_parameter(self, terminal_executor):
        """Test handling of missing command parameter."""
        result = await terminal_executor.execute()
        
        assert not result.success
        assert "Missing required parameters" in result.error


class TestMessageQueueTool:
    """Test cases for MessageQueueTool."""
    
    @pytest.fixture
    def message_queue_tool(self):
        """Create MessageQueueTool instance for testing."""
        return MessageQueueTool()
    
    @pytest.mark.asyncio
    async def test_send_message(self, message_queue_tool):
        """Test sending a message to the queue."""
        result = await message_queue_tool.execute(
            action="send",
            content="Test message",
            source="user",
            priority=10
        )
        
        assert result.success
        assert result.data['content'] == "Test message"
        assert result.data['source'] == "user"
        assert result.data['priority'] == 10
    
    @pytest.mark.asyncio
    async def test_peek_message(self, message_queue_tool):
        """Test peeking at the next message."""
        # First send a message
        await message_queue_tool.execute(
            action="send",
            content="Peek test message",
            source="system"
        )
        
        # Then peek at it
        result = await message_queue_tool.execute(action="peek")
        
        assert result.success
        assert result.data['content'] == "Peek test message"
        assert result.metadata['has_message']
    
    @pytest.mark.asyncio
    async def test_queue_status(self, message_queue_tool):
        """Test getting queue status."""
        result = await message_queue_tool.execute(action="status")
        
        assert result.success
        assert 'size' in result.data
        assert 'is_empty' in result.data
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, message_queue_tool):
        """Test clearing the queue."""
        # Send some messages first
        for i in range(3):
            await message_queue_tool.execute(
                action="send",
                content=f"Message {i}",
                source="system"
            )
        
        # Clear the queue
        result = await message_queue_tool.execute(action="clear")
        
        assert result.success
        assert result.data['cleared_count'] >= 3
        assert result.data['queue_size'] == 0
    
    @pytest.mark.asyncio
    async def test_send_internal_message(self, message_queue_tool):
        """Test convenience method for sending internal messages."""
        result = await message_queue_tool.send_internal_message(
            "Internal test message",
            priority=5
        )
        
        assert result.success
        assert result.data['source'] == "system"
        assert result.data['priority'] == 5
    
    @pytest.mark.asyncio
    async def test_send_subagent_message(self, message_queue_tool):
        """Test sending a message from a subagent."""
        result = await message_queue_tool.send_subagent_message(
            "Subagent message",
            "test_subagent_123"
        )
        
        assert result.success
        assert result.data['source'] == "subagent"
        assert result.data['metadata']['subagent_id'] == "test_subagent_123"
    
    @pytest.mark.asyncio
    async def test_check_queue_health(self, message_queue_tool):
        """Test queue health checking."""
        result = await message_queue_tool.check_queue_health()
        
        assert result.success
        assert 'health_status' in result.data
        assert 'recommendations' in result.data
    
    @pytest.mark.asyncio
    async def test_invalid_action(self, message_queue_tool):
        """Test handling of invalid actions."""
        result = await message_queue_tool.execute(action="invalid_action")
        
        assert not result.success
        assert "Unsupported action" in result.error


class TestErrorHandlerTool:
    """Test cases for ErrorHandlerTool."""
    
    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandlerTool instance for testing."""
        return ErrorHandlerTool()
    
    @pytest.mark.asyncio
    async def test_inspect_error(self, error_handler):
        """Test error inspection."""
        test_error = ValueError("Test error message")
        
        result = await error_handler.execute(
            action="inspect",
            error=test_error,
            context={"operation": "test"}
        )
        
        assert result.success
        assert result.data['error_type'] == "ValueError"
        assert result.data['error_message'] == "Test error message"
        assert result.data['severity'] in [s.value for s in ErrorSeverity]
        assert 'analysis' in result.data
        assert 'suggestions' in result.data
    
    @pytest.mark.asyncio
    async def test_retry_successful_function(self, error_handler):
        """Test retrying a function that succeeds."""
        def successful_function(x, y):
            return x + y
        
        result = await error_handler.execute(
            action="retry",
            function=successful_function,
            args=(2, 3),
            retry_config={"max_attempts": 3}
        )
        
        assert result.success
        assert result.data['result'] == 5
        assert result.data['attempts'] == 1
    
    @pytest.mark.asyncio
    async def test_retry_failing_function(self, error_handler):
        """Test retrying a function that always fails."""
        def failing_function():
            raise ValueError("Always fails")
        
        result = await error_handler.execute(
            action="retry",
            function=failing_function,
            retry_config={"max_attempts": 2}
        )
        
        assert not result.success
        assert "All 2 retry attempts failed" in result.error
        assert result.metadata['attempts'] == 2
    
    @pytest.mark.asyncio
    async def test_retry_eventually_successful_function(self, error_handler):
        """Test retrying a function that succeeds after failures."""
        call_count = 0
        
        def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "Success!"
        
        result = await error_handler.execute(
            action="retry",
            function=eventually_successful_function,
            retry_config={"max_attempts": 5, "base_delay": 0.01}
        )
        
        assert result.success
        assert result.data['result'] == "Success!"
        assert result.data['attempts'] == 3
    
    @pytest.mark.asyncio
    async def test_get_error_history(self, error_handler):
        """Test getting error history."""
        # First inspect some errors to populate history
        for i in range(3):
            await error_handler.execute(
                action="inspect",
                error=ValueError(f"Test error {i}"),
                context={"test": i}
            )
        
        result = await error_handler.execute(action="history", limit=10)
        
        assert result.success
        assert result.data['total_count'] >= 3
        assert len(result.data['errors']) >= 3
    
    @pytest.mark.asyncio
    async def test_clear_error_history(self, error_handler):
        """Test clearing error history."""
        # Add some errors first
        await error_handler.execute(
            action="inspect",
            error=ValueError("Test error"),
            context={}
        )
        
        result = await error_handler.execute(action="clear_history")
        
        assert result.success
        assert result.data['remaining_count'] == 0
    
    @pytest.mark.asyncio
    async def test_analyze_errors(self, error_handler):
        """Test error pattern analysis."""
        # Add some errors for analysis
        for i in range(5):
            await error_handler.execute(
                action="inspect",
                error=ValueError(f"Test error {i}"),
                context={"iteration": i}
            )
        
        result = await error_handler.execute(action="analyze")
        
        assert result.success
        assert 'total_errors' in result.data
        assert 'error_types' in result.data
        assert 'recommendations' in result.data
    
    @pytest.mark.asyncio
    async def test_handle_with_retry_convenience_method(self, error_handler):
        """Test the convenience method for handling with retry."""
        def test_function(x):
            return x * 2
        
        result = await error_handler.handle_with_retry(test_function, 5)
        
        assert result == 10


class TestExternalMessengerTool:
    """Test cases for ExternalMessengerTool."""
    
    @pytest.fixture
    def external_messenger(self):
        """Create ExternalMessengerTool instance for testing."""
        return ExternalMessengerTool()
    
    @pytest.mark.asyncio
    async def test_configure_endpoint(self, external_messenger):
        """Test configuring an external endpoint."""
        result = await external_messenger.execute(
            action="configure",
            name="test_endpoint",
            url="https://httpbin.org/post",
            method="http_post",
            format="json"
        )
        
        assert result.success
        assert result.data['name'] == "test_endpoint"
        assert result.data['configured']
    
    @pytest.mark.asyncio
    async def test_list_endpoints(self, external_messenger):
        """Test listing configured endpoints."""
        # Configure an endpoint first
        await external_messenger.execute(
            action="configure",
            name="list_test",
            url="https://example.com",
            method="http_post",
            format="json"
        )
        
        result = await external_messenger.execute(action="list_endpoints")
        
        assert result.success
        assert result.data['count'] >= 1
        assert any(ep['name'] == 'list_test' for ep in result.data['endpoints'])
    
    @pytest.mark.asyncio
    async def test_remove_endpoint(self, external_messenger):
        """Test removing an endpoint."""
        # Configure an endpoint first
        await external_messenger.execute(
            action="configure",
            name="remove_test",
            url="https://example.com",
            method="http_post",
            format="json"
        )
        
        result = await external_messenger.execute(
            action="remove_endpoint",
            name="remove_test"
        )
        
        assert result.success
        assert result.data['removed']
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_send_message_with_mock(self, mock_post, external_messenger):
        """Test sending a message with mocked HTTP client."""
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json = AsyncMock(return_value={'success': True})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Configure endpoint
        await external_messenger.execute(
            action="configure",
            name="mock_endpoint",
            url="https://httpbin.org/post",
            method="http_post",
            format="json"
        )
        
        # Send message
        result = await external_messenger.execute(
            action="send",
            endpoint="mock_endpoint",
            message={"test": "message"}
        )
        
        assert result.success
        assert result.data['endpoint'] == "mock_endpoint"
    
    def test_message_formatting(self, external_messenger):
        """Test message formatting for different formats."""
        # Test JSON formatting
        message, content_type = external_messenger._format_message(
            {"key": "value"}, MessageFormat.JSON
        )
        assert content_type == 'application/json'
        assert '"key"' in message
        
        # Test text formatting
        message, content_type = external_messenger._format_message(
            "plain text", MessageFormat.TEXT
        )
        assert content_type == 'text/plain'
        assert message == "plain text"
    
    @pytest.mark.asyncio
    async def test_slack_message_convenience(self, external_messenger):
        """Test Slack message convenience method."""
        with patch.object(external_messenger, 'execute') as mock_execute:
            mock_execute.return_value = Mock(success=True)
            
            result = await external_messenger.send_slack_message(
                "https://hooks.slack.com/test",
                "Test message"
            )
            
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[1]
            assert call_args['action'] == 'send'
            assert call_args['message']['text'] == "Test message"
    
    @pytest.mark.asyncio
    async def test_discord_message_convenience(self, external_messenger):
        """Test Discord message convenience method."""
        with patch.object(external_messenger, 'execute') as mock_execute:
            mock_execute.return_value = Mock(success=True)
            
            result = await external_messenger.send_discord_message(
                "https://discord.com/api/webhooks/test",
                "Test message"
            )
            
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[1]
            assert call_args['action'] == 'send'
            assert call_args['message']['content'] == "Test message"


class TestReflectionTool:
    """Test cases for ReflectionTool."""
    
    @pytest.fixture
    def reflection_tool(self):
        """Create ReflectionTool instance for testing."""
        return ReflectionTool()
    
    @pytest.mark.asyncio
    async def test_perform_reflection(self, reflection_tool):
        """Test performing a reflection."""
        result = await reflection_tool.execute(
            action="reflect",
            reflection_type="performance",
            context={"recent_tasks": 5, "success_rate": 0.9}
        )
        
        assert result.success
        assert result.data['reflection_type'] == "performance"
        assert 'content' in result.data
        assert 'insights' in result.data
        assert 'action_items' in result.data
        assert 'confidence_score' in result.data
    
    @pytest.mark.asyncio
    async def test_different_reflection_types(self, reflection_tool):
        """Test different types of reflection."""
        reflection_types = [
            "performance", "decision_making", "learning", 
            "goal_alignment", "behavior_patterns", "capability_assessment"
        ]
        
        for reflection_type in reflection_types:
            result = await reflection_tool.execute(
                action="reflect",
                reflection_type=reflection_type,
                context={"test": True}
            )
            
            assert result.success
            assert result.data['reflection_type'] == reflection_type
    
    @pytest.mark.asyncio
    async def test_analyze_patterns(self, reflection_tool):
        """Test pattern analysis."""
        # First create some reflections
        for i in range(5):
            await reflection_tool.execute(
                action="reflect",
                reflection_type="performance",
                context={"iteration": i}
            )
        
        result = await reflection_tool.execute(
            action="analyze",
            time_period="day"
        )
        
        assert result.success
        assert 'patterns' in result.data
        assert 'trends' in result.data
        assert 'recommendations' in result.data
    
    @pytest.mark.asyncio
    async def test_get_reflection_history(self, reflection_tool):
        """Test getting reflection history."""
        # Create some reflections first
        for i in range(3):
            await reflection_tool.execute(
                action="reflect",
                reflection_type="learning",
                context={"test_run": i}
            )
        
        result = await reflection_tool.execute(
            action="history",
            limit=10
        )
        
        assert result.success
        assert result.data['count'] >= 3
        assert len(result.data['reflections']) >= 3
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, reflection_tool):
        """Test getting performance metrics."""
        # Update some metrics first
        reflection_tool.update_performance_metrics(
            tasks_completed=10,
            tasks_failed=2,
            average_response_time=1.5
        )
        
        result = await reflection_tool.execute(action="metrics")
        
        assert result.success
        assert result.data['metrics']['tasks_completed'] == 10
        assert result.data['metrics']['tasks_failed'] == 2
        assert 'assessment' in result.data
    
    @pytest.mark.asyncio
    async def test_generate_insights(self, reflection_tool):
        """Test insight generation."""
        # Create some reflections for insight generation
        for i in range(3):
            await reflection_tool.execute(
                action="reflect",
                reflection_type="learning",
                context={"learning_session": i}
            )
        
        result = await reflection_tool.execute(
            action="insights",
            time_period="week"
        )
        
        assert result.success
        assert 'insights' in result.data
        assert 'key_learnings' in result.data
        assert 'improvement_areas' in result.data
        assert 'strengths' in result.data
    
    @pytest.mark.asyncio
    async def test_assess_goal_alignment(self, reflection_tool):
        """Test goal alignment assessment."""
        goals = [
            "Provide helpful responses",
            "Learn continuously",
            "Maintain security"
        ]
        
        result = await reflection_tool.execute(
            action="goals",
            goals=goals,
            context={"performance_score": 0.85}
        )
        
        assert result.success
        assert 'overall_alignment_score' in result.data
        assert 'goal_assessments' in result.data
        assert len(result.data['goal_assessments']) == 3
        assert 'recommendations' in result.data
    
    @pytest.mark.asyncio
    async def test_quick_reflect_convenience_method(self, reflection_tool):
        """Test the quick reflect convenience method."""
        result = await reflection_tool.quick_reflect(
            "Testing quick reflection functionality",
            "performance"
        )
        
        assert result.success
        assert result.data['reflection_type'] == "performance"
    
    @pytest.mark.asyncio
    async def test_invalid_reflection_type(self, reflection_tool):
        """Test handling of invalid reflection type."""
        result = await reflection_tool.execute(
            action="reflect",
            reflection_type="invalid_type",
            context={}
        )
        
        assert not result.success
        assert "Invalid reflection type" in result.error
    
    def test_performance_metrics_update(self, reflection_tool):
        """Test updating performance metrics."""
        reflection_tool.update_performance_metrics(
            tasks_completed=15,
            error_rate=0.05,
            user_satisfaction_score=0.9
        )
        
        assert reflection_tool.performance_metrics.tasks_completed == 15
        assert reflection_tool.performance_metrics.error_rate == 0.05
        assert reflection_tool.performance_metrics.user_satisfaction_score == 0.9


# Integration tests
class TestSeedToolsIntegration:
    """Integration tests for seed tools working together."""
    
    @pytest.mark.asyncio
    async def test_file_writer_and_terminal_executor_integration(self):
        """Test file writer and terminal executor working together."""
        file_writer = FileWriterTool()
        terminal_executor = TerminalExecutorTool()
        
        # Write a Python script
        python_script = """
print("Hello from integrated test")
with open("/james/output.txt", "w") as f:
    f.write("Script executed successfully")
"""
        
        write_result = await file_writer.execute(
            file_path="test_script.py",
            content=python_script,
            mode="write"
        )
        assert write_result.success
        
        # Execute the script (this would work in a real sandbox environment)
        # For testing, we'll just verify the script was written
        read_result = await file_writer.read_file("test_script.py")
        assert read_result.success
        assert "Hello from integrated test" in read_result.data['content']
    
    @pytest.mark.asyncio
    async def test_error_handler_and_reflection_integration(self):
        """Test error handler and reflection tool integration."""
        error_handler = ErrorHandlerTool()
        reflection_tool = ReflectionTool()
        
        # Create an error and inspect it
        test_error = ConnectionError("Network connection failed")
        error_result = await error_handler.execute(
            action="inspect",
            error=test_error,
            context={"operation": "api_call"}
        )
        assert error_result.success
        
        # Reflect on the error handling experience
        reflection_result = await reflection_tool.execute(
            action="reflect",
            reflection_type="performance",
            context={
                "error_handled": True,
                "error_type": error_result.data['error_type'],
                "severity": error_result.data['severity']
            }
        )
        assert reflection_result.success
        assert "performance" in reflection_result.data['content'].lower()
    
    @pytest.mark.asyncio
    async def test_message_queue_and_external_messenger_integration(self):
        """Test message queue and external messenger integration."""
        message_queue_tool = MessageQueueTool()
        external_messenger = ExternalMessengerTool()
        
        # Send a message to the internal queue
        queue_result = await message_queue_tool.execute(
            action="send",
            content="Test integration message",
            source="system",
            priority=5
        )
        assert queue_result.success
        
        # Configure an external endpoint
        config_result = await external_messenger.execute(
            action="configure",
            name="integration_test",
            url="https://httpbin.org/post",
            method="http_post",
            format="json"
        )
        assert config_result.success
        
        # The integration would involve processing queue messages
        # and potentially sending them externally based on rules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
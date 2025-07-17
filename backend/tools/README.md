# James Seed Tools

This directory contains the core seed tools that enable James to operate autonomously and bootstrap his capabilities. These tools provide the fundamental operations needed for file management, code execution, error handling, communication, and self-reflection.

## Overview

The seed tools are designed to give James the essential capabilities needed to:

- **Manage files securely** within the ~/.james directory
- **Execute code safely** in sandboxed environments  
- **Handle errors intelligently** with retry mechanisms
- **Communicate internally** via priority-based message queues
- **Send external messages** to various platforms and services
- **Reflect on performance** and continuously improve

## Tools

### 1. FileWriterTool (`file_writer.py`)

Provides secure file operations within the ~/.james directory with security validation and backup capabilities.

**Key Features:**
- Secure path validation (prevents directory traversal)
- Multiple file formats (text, JSON, binary)
- Automatic backups before modifications
- File existence checking and listing
- Comprehensive error handling

**Usage Example:**
```python
file_writer = FileWriterTool()

# Write a JSON configuration file
result = await file_writer.execute(
    file_path="config/settings.json",
    content={"key": "value"},
    mode="json"
)

# Read a text file
result = await file_writer.read_file("logs/app.log")
```

### 2. TerminalExecutorTool (`terminal_executor.py`)

Executes terminal commands and code in a secure Docker-based sandbox environment.

**Key Features:**
- Docker-based sandboxing for security
- Command safety validation
- Python and bash code execution
- Resource limits and timeouts
- Controlled file system access

**Usage Example:**
```python
terminal_executor = TerminalExecutorTool()

# Execute a safe command
result = await terminal_executor.execute(command="ls -la")

# Execute Python code
result = await terminal_executor.execute_python_code("""
print("Hello from Python!")
import json
data = {"result": 42}
print(json.dumps(data))
""")
```

### 3. MessageQueueTool (`message_queue_tool.py`)

Manages internal communication via a priority-based message queue system.

**Key Features:**
- Priority-based message ordering
- Message classification and routing
- Queue health monitoring
- Bulk operations and filtering
- Integration with core message models

**Usage Example:**
```python
message_queue = MessageQueueTool()

# Send a high-priority system message
result = await message_queue.execute(
    action="send",
    content="System alert: High CPU usage detected",
    source="system",
    priority=1,
    classification="act_now"
)

# Check queue status
result = await message_queue.execute(action="status")
```

### 4. ErrorHandlerTool (`error_handler.py`)

Provides intelligent error handling, analysis, and retry mechanisms.

**Key Features:**
- Error severity classification
- Retry strategies (exponential backoff, linear, fixed delay)
- Error pattern analysis
- Historical error tracking
- Actionable recommendations

**Usage Example:**
```python
error_handler = ErrorHandlerTool()

# Analyze an error
result = await error_handler.execute(
    action="inspect",
    error=ConnectionError("Network timeout"),
    context={"operation": "api_call"}
)

# Retry a function with backoff
result = await error_handler.execute(
    action="retry",
    function=flaky_network_call,
    retry_config={
        "max_attempts": 3,
        "strategy": "exponential_backoff"
    }
)
```

### 5. ExternalMessengerTool (`external_messenger.py`)

Sends messages to external systems and services via HTTP, webhooks, and messaging platforms.

**Key Features:**
- Multiple delivery methods (HTTP POST/GET, webhooks)
- Various message formats (JSON, text, XML, form data)
- Authentication support (Bearer, Basic, API key)
- Endpoint configuration management
- Platform-specific integrations (Slack, Discord)

**Usage Example:**
```python
external_messenger = ExternalMessengerTool()

# Configure an endpoint
result = await external_messenger.execute(
    action="configure",
    name="slack_alerts",
    url="https://hooks.slack.com/services/...",
    method="http_post",
    format="json"
)

# Send a message
result = await external_messenger.execute(
    action="send",
    endpoint="slack_alerts",
    message={"text": "James is online and operational!"}
)
```

### 6. ReflectionTool (`reflection_tool.py`)

Enables self-reflection, performance analysis, and continuous improvement.

**Key Features:**
- Multiple reflection types (performance, learning, decision-making)
- Pattern recognition and trend analysis
- Goal alignment assessment
- Performance metrics tracking
- Insight generation and recommendations

**Usage Example:**
```python
reflection_tool = ReflectionTool()

# Perform performance reflection
result = await reflection_tool.execute(
    action="reflect",
    reflection_type="performance",
    context={"tasks_completed": 50, "success_rate": 0.94}
)

# Analyze behavioral patterns
result = await reflection_tool.execute(
    action="analyze",
    time_period="week"
)

# Assess goal alignment
result = await reflection_tool.execute(
    action="goals",
    goals=["Be helpful", "Learn continuously", "Operate safely"]
)
```

## Base Tool Architecture

All seed tools inherit from `BaseTool` which provides:

- **Consistent interface** with `execute()` method
- **Standardized result format** via `ToolResult` class
- **Common error handling** and validation
- **Logging integration** for debugging and monitoring

```python
@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: Optional[float] = None
```

## Integration Patterns

The seed tools are designed to work together seamlessly:

### File + Terminal Integration
```python
# Write a script and execute it
await file_writer.execute(
    file_path="scripts/analysis.py",
    content=python_script,
    mode="write"
)

result = await terminal_executor.execute_python_code(
    open("~/.james/scripts/analysis.py").read(),
    allow_file_ops=True
)
```

### Error Handling + Reflection Integration
```python
# Handle errors and reflect on them
try:
    result = await some_operation()
except Exception as e:
    error_result = await error_handler.execute(
        action="inspect", 
        error=e
    )
    
    await reflection_tool.execute(
        action="reflect",
        reflection_type="performance",
        context={"error_handled": True, "severity": error_result.data['severity']}
    )
```

### Message Queue + External Messenger Integration
```python
# Process internal messages and send external notifications
message = await message_queue.execute(action="peek")
if message.data and message.data['priority'] <= 5:  # High priority
    await external_messenger.execute(
        action="send",
        endpoint="alerts",
        message={"alert": message.data['content']}
    )
```

## Security Considerations

The seed tools implement multiple security layers:

1. **Path Validation**: File operations are restricted to ~/.james directory
2. **Command Filtering**: Dangerous commands are blocked before execution
3. **Sandboxing**: Code execution happens in isolated Docker containers
4. **Input Validation**: All parameters are validated before processing
5. **Resource Limits**: Memory, CPU, and time limits prevent resource exhaustion

## Testing

Comprehensive unit tests are provided in `tests/test_seed_tools.py`:

```bash
# Run all seed tool tests
pytest tests/test_seed_tools.py -v

# Run specific tool tests
pytest tests/test_seed_tools.py::TestFileWriterTool -v
```

## Demo

A complete demonstration of all seed tools working together is available:

```bash
python examples/seed_tools_demo.py
```

This demo shows:
- File operations (config files, logs)
- Terminal execution (system commands, Python code)
- Message queue operations (sending, prioritizing, monitoring)
- Error handling (inspection, retry mechanisms)
- External communication (endpoint configuration, message sending)
- Self-reflection (performance analysis, goal alignment)

## Performance Metrics

The seed tools track various performance metrics:

- **Execution time** for each operation
- **Success/failure rates** across tool usage
- **Error patterns** and recovery statistics
- **Resource usage** in sandboxed environments
- **Queue throughput** and message processing rates

These metrics feed into the reflection system for continuous improvement.

## Future Enhancements

Planned improvements include:

1. **Advanced Security**: Enhanced sandboxing with additional isolation layers
2. **Performance Optimization**: Caching, connection pooling, batch operations
3. **Extended Integrations**: More external platforms and services
4. **Machine Learning**: Pattern recognition for better error prediction
5. **Distributed Operations**: Multi-node coordination and load balancing

## Contributing

When extending the seed tools:

1. **Inherit from BaseTool** for consistency
2. **Follow security best practices** for all operations
3. **Add comprehensive tests** for new functionality
4. **Update documentation** with usage examples
5. **Consider integration patterns** with existing tools

The seed tools form the foundation of James's autonomous capabilities, enabling sophisticated operations while maintaining security and reliability.
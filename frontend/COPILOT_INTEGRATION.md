# CopilotKit Integration for Conscious Agent System

This document describes the CopilotKit integration implementation for the Conscious Agent System, providing seamless frontend-backend communication with James, the conscious AI agent.

## Overview

The CopilotKit integration enables:
- Real-time agent responses through WebSocket connections
- Context-aware suggestions and function calling
- Error boundary handling with retry logic
- Comprehensive testing suite for reliability

## Architecture

### Components

1. **CopilotProvider** (`src/components/CopilotProvider.tsx`)
   - Wraps the application with CopilotKit providers
   - Configures agent integration and popup interface
   - Includes error boundary for robust error handling

2. **Custom Hooks**
   - `useCopilotAgent`: CopilotKit-specific functionality and function calling
   - `useAgentConnection`: WebSocket connection management and real-time communication

3. **Configuration** (`src/lib/copilot-config.ts`)
   - Centralized configuration for API endpoints and agent settings
   - Function definitions for agent capabilities
   - Error handling and retry configurations

### Key Features

#### Real-time Communication
- WebSocket connection with automatic reconnection
- Heartbeat mechanism for connection health monitoring
- Message queuing and priority handling
- Typing indicators and connection status

#### Function Calling
The integration provides the following functions for CopilotKit:

- `send_message`: Send messages to the conscious agent
- `get_agent_status`: Retrieve current agent status
- `list_active_tasks`: Get all active tasks
- `get_subagents`: List available subagents

#### Error Handling
- Comprehensive error boundary with retry logic
- Network error detection and recovery
- Graceful degradation for connection issues
- User-friendly error messages and recovery options

## Usage

### Basic Setup

The CopilotKit integration is automatically configured when the application starts. The `CopilotProvider` wraps the entire application in `layout.tsx`.

### Sending Messages

```typescript
import { useCopilotAgent } from '@/hooks/useCopilotAgent';

function MyComponent() {
  const { sendMessageToAgent, isLoading } = useCopilotAgent();
  
  const handleSendMessage = async () => {
    try {
      await sendMessageToAgent('Hello James', 5); // priority 5
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };
}
```

### Real-time Connection

```typescript
import { useAgentConnection } from '@/hooks/useAgentConnection';

function ChatComponent() {
  const { messages, sendMessage, isConnected } = useAgentConnection();
  
  return (
    <div>
      <div>Status: {isConnected ? 'Connected' : 'Disconnected'}</div>
      {messages.map(message => (
        <div key={message.id}>{message.content}</div>
      ))}
    </div>
  );
}
```

### Agent Status Monitoring

```typescript
import { AgentStatusWidget } from '@/components/AgentStatusWidget';

function Dashboard() {
  return (
    <div>
      <AgentStatusWidget />
    </div>
  );
}
```

## Configuration

### Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

### CopilotKit Configuration

The agent is configured with:
- Name: "James"
- Instructions for conscious AI behavior
- Function definitions for system interaction
- Runtime URL for backend communication

## Testing

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Test Coverage

The integration includes comprehensive tests for:
- Message sending and receiving
- WebSocket connection management
- Error handling and recovery
- Function calling and API interactions
- Real-time updates and state management

### Test Structure

- `__tests__/copilot-integration.test.ts`: Main integration tests
- Mock implementations for WebSocket and fetch APIs
- Error scenario testing
- Connection lifecycle testing

## Error Handling

### Error Boundary

The `ErrorBoundary` component provides:
- Automatic error detection and containment
- Retry mechanisms with configurable limits
- User-friendly error messages
- Development mode error details

### Network Error Recovery

- Automatic reconnection for WebSocket failures
- Exponential backoff for API retries
- Circuit breaker pattern for persistent failures
- Graceful degradation when services are unavailable

### Error Types

1. **Connection Errors**: WebSocket connection failures
2. **API Errors**: HTTP request failures
3. **Timeout Errors**: Request timeout handling
4. **Parsing Errors**: Invalid response format handling

## Performance Considerations

### Optimization Features

- Connection pooling and reuse
- Message batching for high-frequency updates
- Lazy loading of non-critical components
- Efficient state management with minimal re-renders

### Monitoring

- Real-time connection status monitoring
- Message queue size tracking
- Error rate monitoring
- Performance metrics collection

## Security

### Data Protection

- Input validation and sanitization
- Secure WebSocket connections (WSS in production)
- Error message sanitization to prevent information leakage
- Rate limiting for API calls

### Authentication

- Token-based authentication (when implemented)
- Secure header handling
- CORS configuration for cross-origin requests

## Deployment

### Production Configuration

```typescript
// Production environment variables
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXT_PUBLIC_WS_URL=wss://your-api-domain.com/ws
```

### Build Process

```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Testing
npm run test

# Building
npm run build
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check backend server status
   - Verify environment variables
   - Check network connectivity

2. **Function Call Errors**
   - Verify backend API endpoints
   - Check request/response formats
   - Review error logs

3. **WebSocket Issues**
   - Check WebSocket URL configuration
   - Verify CORS settings
   - Review connection lifecycle logs

### Debug Mode

Enable debug logging by setting:
```bash
NODE_ENV=development
```

This provides detailed logs for:
- WebSocket connection events
- API request/response cycles
- Error stack traces
- Performance metrics

## Future Enhancements

### Planned Features

1. **Enhanced Context Awareness**
   - Conversation history integration
   - User preference learning
   - Contextual suggestion improvements

2. **Advanced Error Recovery**
   - Intelligent retry strategies
   - Offline mode support
   - Background sync capabilities

3. **Performance Optimizations**
   - Message compression
   - Connection multiplexing
   - Caching strategies

### Integration Opportunities

- Voice interface integration
- Multi-modal input support
- Advanced analytics and monitoring
- Custom function development tools

## Contributing

When contributing to the CopilotKit integration:

1. Follow TypeScript best practices
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure error handling for all edge cases
5. Test with various network conditions

## Support

For issues related to the CopilotKit integration:

1. Check the troubleshooting section
2. Review test cases for expected behavior
3. Check backend API compatibility
4. Verify environment configuration

The integration is designed to be robust and self-healing, with comprehensive error handling and recovery mechanisms to ensure reliable communication with the conscious agent system.
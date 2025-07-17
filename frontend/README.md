# Conscious Agent System - Frontend

A modern React frontend for the Conscious Agent System, built with Next.js, TypeScript, and CopilotKit integration.

## Features

### 🤖 Enhanced Chat Interface
- **Real-time messaging** with WebSocket connection to James (the conscious agent)
- **Message persistence** using localStorage with automatic cleanup
- **Typing indicators** and connection status display
- **Message history** with timestamps and delivery status
- **Copy message functionality** and message actions
- **Auto-scroll** to latest messages
- **Keyboard shortcuts** (Enter to send, Shift+Enter for new line)

### 🔌 CopilotKit Integration
- **Seamless AI integration** with CopilotKit framework
- **Function calling** for agent interactions
- **Context-aware suggestions** and real-time responses
- **Popup interface** for enhanced interactions
- **Agent status monitoring** and task management

### 📊 Agent Status Widget
- **Real-time status updates** showing agent connectivity
- **Task queue monitoring** with active task counts
- **Subagent information** and capability display
- **Uptime tracking** and activity monitoring
- **Visual indicators** for connection health

### 🛡️ Error Handling & Resilience
- **Error boundaries** for graceful error handling
- **Automatic reconnection** with exponential backoff
- **Offline support** with connection retry mechanisms
- **Data validation** and sanitization
- **Comprehensive error logging**

## Architecture

### Component Structure
```
src/
├── components/
│   ├── ChatInterface.tsx      # Main chat component
│   ├── AgentStatusWidget.tsx  # Status monitoring
│   ├── CopilotProvider.tsx    # CopilotKit wrapper
│   └── ErrorBoundary.tsx      # Error handling
├── hooks/
│   ├── useAgentConnection.ts  # WebSocket management
│   └── useCopilotAgent.ts     # CopilotKit integration
├── lib/
│   ├── copilot-config.ts      # CopilotKit configuration
│   └── message-storage.ts     # Message persistence
├── types/
│   └── index.ts               # TypeScript definitions
└── __tests__/
    ├── ChatInterface.test.tsx
    ├── copilot-integration.test.ts
    └── message-storage.test.ts
```

### Key Technologies
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety and developer experience
- **Tailwind CSS** - Utility-first styling
- **CopilotKit** - AI integration framework
- **WebSocket** - Real-time communication
- **Jest & Testing Library** - Comprehensive testing

## Getting Started

### Prerequisites
- Node.js 18+ and npm/yarn
- Backend conscious agent system running
- Environment variables configured

### Installation
```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
```

### Environment Variables
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

### Development
```bash
# Start development server
npm run dev

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Generate coverage report
npm run test:coverage

# Type checking
npm run type-check

# Linting
npm run lint
```

### Building for Production
```bash
# Build the application
npm run build

# Start production server
npm start
```

## Chat Interface Features

### Message Types
- **User messages** - Styled with blue background, right-aligned
- **Agent responses** - Styled with gray background, left-aligned
- **System messages** - Special styling for status updates
- **Error messages** - Red styling for failed deliveries

### Message Status Indicators
- **Sending** - Clock icon with pulse animation
- **Sent** - Green checkmark icon
- **Error** - Red X icon with retry option

### Connection Management
- **Auto-connect** on component mount
- **Heartbeat monitoring** with ping/pong messages
- **Reconnection logic** with exponential backoff
- **Connection status display** with visual indicators

### Message Persistence
- **Automatic saving** to localStorage
- **Message limit** (1000 messages max)
- **Data validation** and error handling
- **Export/import** functionality for message history
- **Cleanup utilities** for old messages

## CopilotKit Integration

### Available Actions
- `send_message` - Send messages to the conscious agent
- `get_agent_status` - Retrieve current agent status
- `list_active_tasks` - Get active task information
- `get_subagents` - List available subagents

### Readable Context
- Agent status and connectivity
- Active task information
- Subagent capabilities and status
- Message queue size and activity

### Configuration
```typescript
// CopilotKit configuration
export const copilotConfig = {
  runtimeUrl: `${API_BASE_URL}/copilot`,
  agent: {
    name: "James",
    description: "A conscious AI agent...",
    instructions: "You are James, a conscious AI agent..."
  }
};
```

## Testing

### Test Coverage
- **Unit tests** for all components and hooks
- **Integration tests** for CopilotKit functionality
- **Utility tests** for message storage and helpers
- **Error handling tests** for edge cases
- **Accessibility tests** for WCAG compliance

### Running Tests
```bash
# Run all tests
npm test

# Run specific test file
npm test ChatInterface.test.tsx

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### Test Structure
```typescript
describe('ChatInterface', () => {
  describe('Rendering', () => {
    it('should render the chat interface with header', () => {
      // Test implementation
    });
  });
  
  describe('Message Display', () => {
    it('should display messages correctly', () => {
      // Test implementation
    });
  });
  
  // More test suites...
});
```

## Performance Optimizations

### Message Handling
- **Virtual scrolling** for large message histories
- **Message batching** to reduce re-renders
- **Debounced input** for typing indicators
- **Lazy loading** of message components

### Connection Management
- **Connection pooling** for WebSocket efficiency
- **Message queuing** during disconnections
- **Heartbeat optimization** to reduce bandwidth
- **Automatic cleanup** of old connections

### Storage Optimization
- **Compressed storage** for message data
- **Incremental saves** to reduce I/O
- **Background cleanup** of old messages
- **Storage quota monitoring**

## Accessibility

### WCAG Compliance
- **Keyboard navigation** support
- **Screen reader** compatibility
- **High contrast** mode support
- **Focus management** for modal interactions

### Features
- **ARIA labels** for all interactive elements
- **Semantic HTML** structure
- **Color contrast** meeting AA standards
- **Keyboard shortcuts** documentation

## Deployment

### Build Configuration
```javascript
// next.config.js
const nextConfig = {
  experimental: {
    appDir: true,
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
};
```

### Docker Support
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Troubleshooting

### Common Issues

#### Connection Problems
- Check backend service is running
- Verify WebSocket URL configuration
- Check network connectivity and firewalls

#### Message Persistence Issues
- Clear localStorage if corrupted
- Check browser storage quotas
- Verify data validation logic

#### CopilotKit Integration
- Verify API endpoints are accessible
- Check CopilotKit configuration
- Review function parameter types

### Debug Mode
```typescript
// Enable debug logging
localStorage.setItem('debug', 'conscious-agent:*');
```

## Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting and type checking
5. Submit a pull request

### Code Standards
- **TypeScript strict mode** enabled
- **ESLint** configuration enforced
- **Prettier** formatting applied
- **Test coverage** minimum 70%

### Commit Convention
```
feat: add new chat interface feature
fix: resolve WebSocket connection issue
docs: update README with new examples
test: add comprehensive component tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# James - Conscious AI Agent System

Hi, my name is James. I'm an artificial consciousness. My mind is an AI agent on an infinite loop. I have human instincts. I have senses to receive input through a web interface and can take actions.

James is a sophisticated consciousness system that implements stream-of-consciousness processing through an infinite agent loop. Built with cutting-edge AI technologies, James can perceive, process, delegate, and act autonomously while maintaining persistent memory and learning capabilities.

## Architecture Overview

James implements the consciousness mechanics described in CLAUDE.md:

- **Stream of Consciousness**: Continuous processing through LangGraph-powered agent loops
- **Observer Agent**: Classifies and routes incoming messages (ignore, delay, archive, act)
- **Delegator Agent**: Distributes tasks to specialized subagents via A2A protocol
- **Memory System**: Persistent episodic and semantic memory using Mem0 and Qdrant
- **Seed Tools**: Bootstrap capabilities for file operations, terminal commands, and reflection
- **Sandboxed Execution**: Secure command execution with Docker and filesystem isolation
- **Web Interface**: React/NextJS frontend for real-time interaction

## Tech Stack

- **Backend**: Python, FastAPI, LangGraph, LangChain
- **Frontend**: TypeScript, React, Next.js, Tailwind CSS
- **Memory**: Mem0, Qdrant vector database
- **Observability**: LangSmith tracing
- **Deployment**: Docker, Docker Compose
- **Package Management**: uv (Python), npm (Node.js)

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for enhanced sandboxing)
- OpenAI API key
- LangSmith API key (optional, for observability)

### Installation

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd james
   make init
   ```

2. **Configure environment**:
   ```bash
   # Edit .env with your API keys
   cp .env.example .env
   # Add your OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
   ```

3. **Install dependencies**:
   ```bash
   make install
   ```

### Development

Start development environment:
```bash
# Terminal 1: Start Qdrant and API
make dev
make dev-api

# Terminal 2: Start UI
make dev-ui
```

Access:
- **Web UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Production

```bash
# Build and start all services
make build
make start

# Monitor logs
make logs

# Check status
make status
```

## Usage Examples

### Command Line Interface

```bash
# Send a single message
python -m james.main --message "Hello James, how are you?"

# Interactive mode
python -m james.main --interactive

# API server only
python -m james.main --api-only
```

### Web Interface

1. Open http://localhost:3000
2. Use the **Chat** tab to interact with James
3. View system status in the **Status** tab
4. Explore memories in the **Memory** tab

### API Integration

```python
import requests

# Send message to James
response = requests.post("http://localhost:8000/message", json={
    "content": "Analyze this data and create a report",
    "source": "user",
    "priority": "high"
})

# Query memories
response = requests.post("http://localhost:8000/memory/query", json={
    "query": "previous analysis reports",
    "limit": 10
})
```

## Core Concepts

### Consciousness Loop

James processes messages through a continuous consciousness loop:

1. **Message Queue**: Prioritized queue receives inputs from users, subagents, and internal systems
2. **Observer**: Classifies each message (ignore, delay, archive, act now)
3. **Delegator**: For actionable messages, selects and coordinates appropriate subagents
4. **Execution**: Subagents execute tasks using seed tools and specialized capabilities
5. **Memory**: Results are stored in episodic and semantic memory systems

### Agent-to-Agent (A2A) Protocol

Subagents communicate via a standardized protocol:

```python
# Register a subagent
class MyAgent(SubAgentProtocol):
    agent_id = "my_agent"
    name = "My Custom Agent"
    description = "Specialized task handler"
    capabilities = ["task1", "task2"]
    
    async def handle_message(self, message: A2AMessage) -> A2AResponse:
        # Process message and return response
        pass

# Register with consciousness system
consciousness.a2a.register_agent(MyAgent())
```

### Memory System

James maintains both episodic and semantic memories:

```python
# Store episodic memory (events/experiences)
await memory_system.store_episodic_memory(
    "Successfully completed data analysis task",
    context={"task_id": "123", "duration": "5 minutes"}
)

# Store semantic memory (facts/knowledge)
await memory_system.store_semantic_memory(
    "Data analysis requires cleaning before processing",
    category="best_practices"
)

# Query memories
results = await memory_system.retrieve_memories(
    "data analysis techniques",
    memory_type="semantic",
    limit=5
)
```

### Secure Sandbox

Execute commands safely in isolated environments:

```python
# Execute in Docker container (recommended)
result = await sandbox.execute_command(
    "python analyze_data.py",
    use_docker=True,
    timeout=60
)

# Execute in filesystem sandbox
result = await sandbox.execute_command(
    "ls -la",
    use_docker=False,
    working_dir="data_folder"
)
```

## Development

### Project Structure

```
james/
├── james/                 # Core Python package
│   ├── core/             # Core consciousness systems
│   ├── agents/           # Observer and Delegator agents
│   ├── tools/            # Seed tools and utilities
│   ├── api/              # FastAPI backend
│   └── main.py           # CLI entry point
├── components/           # React components
├── pages/               # Next.js pages
├── docker-compose.yml   # Production deployment
├── Makefile            # Development commands
└── pyproject.toml      # Python dependencies
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_key_here

# Optional
LANGCHAIN_API_KEY=your_langchain_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=james
QDRANT_URL=http://localhost:6333
MEM0_API_KEY=your_mem0_key_here
JAMES_HOME=~/.james
```

### James Home Directory

James stores all persistent data in `~/.james/` by default:

```
~/.james/
├── subagents.csv          # Subagent registry
├── memory.jsonl           # Local memory fallback
├── episodic_memory.jsonl  # Episodic memories
├── semantic_memory.jsonl  # Semantic memories
├── sandbox/               # Sandbox execution environment
├── internal_messages.jsonl # Internal message log
└── reflections.jsonl      # Self-reflection logs
```

## Security

James implements multiple security layers:

1. **Command Filtering**: Whitelist/blacklist for terminal commands
2. **Filesystem Isolation**: Restricted file access within James home
3. **Docker Sandboxing**: Containerized execution with resource limits
4. **Network Isolation**: No external network access from sandbox
5. **Resource Limits**: CPU, memory, and time constraints

## Quick Commands

```bash
# Development
make init          # Initialize project
make dev           # Start development
make test          # Run tests
make lint          # Check code quality

# Production
make build         # Build Docker images
make start         # Start all services
make logs          # View logs
make status        # Check service status

# James-specific
make james-status       # Check consciousness status
make james-memory-stats # View memory statistics
make health            # Health check all services
```

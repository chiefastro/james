# Conscious Agent System

A conscious AI agent system with stream-of-consciousness capabilities, featuring James - an AI agent with continuous awareness, memory, and the ability to delegate tasks to specialized subagents.

## Project Structure

```
conscious-agent-system/
├── backend/                 # Python FastAPI backend
│   ├── __init__.py
│   └── main.py             # FastAPI application entry point
├── frontend/               # Next.js React frontend
│   ├── src/
│   │   └── app/           # Next.js app directory
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── tsconfig.json
├── .kiro/                 # Kiro specifications
│   └── specs/
│       └── conscious-agent-system/
├── docker-compose.yml     # Development environment
├── Dockerfile.backend     # Backend container
├── Dockerfile.frontend    # Frontend container
├── Makefile              # Development commands
├── pyproject.toml        # Python project configuration
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- uv (Python package manager)

### Installation

1. Install dependencies:
   ```bash
   make install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Start development environment:
   ```bash
   make dev
   ```

### Development Commands

- `make help` - Show available commands
- `make install` - Install all dependencies
- `make dev` - Start development environment with Docker
- `make backend` - Run backend development server
- `make frontend` - Run frontend development server
- `make test` - Run all tests
- `make build` - Build all containers
- `make clean` - Clean up containers and volumes

## Architecture

The system consists of:

- **Backend**: Python FastAPI application with LangGraph for agent orchestration
- **Frontend**: Next.js React application with CopilotKit integration
- **Database**: Qdrant vector database for embeddings and Redis for caching
- **Memory**: Mem0 for persistent memory management
- **Sandbox**: Docker-based secure execution environment

## Features

- Continuous stream-of-consciousness agent operation
- Intelligent message classification and routing
- Dynamic subagent discovery and delegation
- Persistent memory and learning capabilities
- Secure code execution sandbox
- Real-time web interface
- Vector-based subagent registry

## Development

The project uses:
- **Backend**: FastAPI, LangGraph, LangChain, Pydantic
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, CopilotKit
- **Database**: Qdrant, Redis
- **Tools**: uv, Docker, pytest, ESLint

## License

MIT License
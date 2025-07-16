.PHONY: help install dev build start stop logs clean test lint format

# Default target
help:
	@echo "James Consciousness System - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     - Install all dependencies"
	@echo "  dev         - Start development environment (with Docker)"
	@echo "  dev-no-docker - Start development without Docker"
	@echo "  dev-api     - Start API server"
	@echo "  dev-ui      - Start UI server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo ""
	@echo "Production:"
	@echo "  build       - Build Docker images"
	@echo "  start       - Start production environment"
	@echo "  stop        - Stop all services"
	@echo "  restart     - Restart all services"
	@echo ""
	@echo "Utilities:"
	@echo "  logs        - Show logs from all services"
	@echo "  logs-api    - Show logs from API service"
	@echo "  logs-ui     - Show logs from UI service"
	@echo "  status      - Show service status"
	@echo "  clean       - Clean up containers and volumes"
	@echo "  reset       - Reset everything (clean + rebuild)"

# Development commands
install:
	@echo "Installing Python dependencies with uv sync..."
	uv sync
	@echo "Installing Node.js dependencies..."
	npm install
	@echo "Installation complete!"

dev:
	@echo "Starting development environment..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		docker-compose up -d qdrant; \
		echo "Qdrant started. Run 'make dev-api' and 'make dev-ui' in separate terminals."; \
	elif command -v docker >/dev/null 2>&1; then \
		docker compose up -d qdrant; \
		echo "Qdrant started. Run 'make dev-api' and 'make dev-ui' in separate terminals."; \
	else \
		echo "Docker not found. You can:"; \
		echo "1. Install Docker Desktop from https://docker.com/products/docker-desktop"; \
		echo "2. Or run without Qdrant (memory will use local files only):"; \
		echo "   make dev-no-docker"; \
	fi

dev-no-docker:
	@echo "Starting development environment without Docker..."
	@echo "Note: Memory system will use local files only (no Qdrant)."
	@echo "Run 'make dev-api' and 'make dev-ui' in separate terminals."

dev-api:
	@echo "Starting James API in development mode..."
	uv run uvicorn james.api.main:app --reload --host 0.0.0.0 --port 8000

dev-ui:
	@echo "Starting James UI in development mode..."
	npm run dev

# Production commands
build:
	@echo "Building Docker images..."
	docker-compose build

start:
	@echo "Starting James consciousness system..."
	docker-compose up -d
	@echo "James is now running!"
	@echo "API: http://localhost:8000"
	@echo "UI: http://localhost:3000"

stop:
	@echo "Stopping James consciousness system..."
	docker-compose down

restart: stop start

# Monitoring commands
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f james-api

logs-ui:
	docker-compose logs -f james-ui

logs-qdrant:
	docker-compose logs -f qdrant

status:
	@echo "Service Status:"
	@docker-compose ps

# Testing and quality
test:
	@echo "Running Python tests..."
	uv run pytest james/tests/ -v
	@echo "Running TypeScript type checking..."
	npm run type-check

lint:
	@echo "Running Python linting..."
	uv run ruff check james/
	@echo "Running TypeScript linting..."
	npm run lint

format:
	@echo "Formatting Python code..."
	uv run black james/
	uv run ruff --fix james/
	@echo "Formatting TypeScript code..."
	npm run format

# Cleanup commands
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	docker system prune -f

reset: clean
	@echo "Resetting entire environment..."
	docker-compose build --no-cache
	$(MAKE) start

# James-specific commands
james-status:
	@echo "Checking James consciousness status..."
	curl -s http://localhost:8000/status | uv run python -m json.tool

james-memory-stats:
	@echo "James memory statistics..."
	curl -s http://localhost:8000/memory/stats | uv run python -m json.tool

james-agents:
	@echo "Registered agents..."
	curl -s http://localhost:8000/agents | uv run python -m json.tool

# Setup commands
setup-env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "Please edit .env with your API keys"; \
	else \
		echo ".env already exists"; \
	fi

init: setup-env install
	@echo "James consciousness system initialized!"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run 'make dev' to start development"
	@echo "3. Or run 'make start' for production"

# Quick development workflow
quick-dev: install dev-api &
	sleep 5 && make dev-ui

# Health checks
health:
	@echo "Checking service health..."
	@curl -f http://localhost:8000/ > /dev/null 2>&1 && echo "✓ API is healthy" || echo "✗ API is down"
	@curl -f http://localhost:3000/ > /dev/null 2>&1 && echo "✓ UI is healthy" || echo "✗ UI is down"
	@curl -f http://localhost:6333/health > /dev/null 2>&1 && echo "✓ Qdrant is healthy" || echo "✗ Qdrant is down"
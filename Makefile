.PHONY: help build test run clean install dev stop logs

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install all dependencies"
	@echo "  dev         - Start development environment"
	@echo "  build       - Build all containers"
	@echo "  run         - Run the application in production mode"
	@echo "  test        - Run all tests"
	@echo "  clean       - Clean up containers and volumes"
	@echo "  stop        - Stop all running containers"
	@echo "  logs        - Show logs from all services"
	@echo "  backend     - Run backend development server"
	@echo "  frontend    - Run frontend development server"

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	uv pip install -e .
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose up --build

# Build containers
build:
	@echo "Building containers..."
	docker-compose build

# Run in production mode
run:
	@echo "Starting production environment..."
	docker-compose up -d

# Run tests
test:
	@echo "Running backend tests..."
	pytest
	@echo "Running frontend tests..."
	cd frontend && npm run lint && npm run type-check

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	docker system prune -f

# Stop containers
stop:
	@echo "Stopping containers..."
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Backend development
backend:
	@echo "Starting backend development server..."
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Frontend development
frontend:
	@echo "Starting frontend development server..."
	cd frontend && npm run dev
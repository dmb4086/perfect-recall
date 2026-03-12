# Perfect Recall - Development Commands

.PHONY: help install db-up db-down db-reset test lint format clean

help:
	@echo "Perfect Recall - Available Commands:"
	@echo ""
	@echo "  make install    - Install Python dependencies"
	@echo "  make db-up      - Start PostgreSQL with pgvector"
	@echo "  make db-down    - Stop PostgreSQL"
	@echo "  make db-reset   - Reset database (delete all data)"
	@echo "  make test       - Run tests"
	@echo "  make example    - Run basic usage example"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean generated files"

install:
	pip install -r requirements.txt

db-up:
	docker-compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@echo "Database ready! Connection: postgresql://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall"

db-down:
	docker-compose down

db-reset:
	docker-compose down -v
	docker-compose up -d
	@echo "Database reset complete"

test:
	pytest tests/ -v

example:
	python examples/basic_usage.py

lint:
	flake8 src/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ examples/ --line-length=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

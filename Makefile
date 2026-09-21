.PHONY: help install lint format test test-cov clean check

help:  ## Show this help message
	@echo "Available commands:"
	@echo "  make install   Install dependencies and set up venv with uv"
	@echo "  make lint      Run static analysis (ruff + mypy)"
	@echo "  make format    Format codebase with ruff"
	@echo "  make test      Run unit and integration test suite"
	@echo "  make test-cov  Run tests with terminal and XML coverage report"
	@echo "  make check     Run all checks (format check, lint, mypy, test)"
	@echo "  make clean     Clean caches and build artifacts"

install:  ## Install dependencies using uv
	uv sync --all-extras

lint:  ## Run linter and type checker
	uv run ruff check src tests
	uv run mypy src tests

format:  ## Automatically format and fix lint issues
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:  ## Run pytest
	uv run pytest

test-cov:  ## Run pytest with coverage
	uv run pytest --cov=src --cov-report=term-missing --cov-report=xml

check:  ## Run formatting check, linting, typing, and tests
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run mypy src tests
	uv run pytest

clean:  ## Clean temporary files and caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage coverage.xml htmlcov dist build *.egg-info

# REV Project Makefile

.PHONY: help install test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-integration Run integration tests"
	@echo "  make test-performance Run performance benchmarks"
	@echo "  make test-adversarial Run adversarial tests"
	@echo "  make lint         Run linting"
	@echo "  make format       Format code with black"
	@echo "  make clean        Clean cache files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-unit:
	pytest tests/test_core_sequential.py tests/test_hdc_components.py -v -m unit

test-integration:
	pytest tests/test_integration.py -v -m integration

test-performance:
	pytest tests/test_performance.py -v --benchmark-only -m benchmark

test-adversarial:
	pytest tests/test_adversarial.py -v -m adversarial

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ --line-length 100
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
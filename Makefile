# Makefile for word-count-investigation project

# Virtual environment settings
VENV_NAME = .venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python
UV = uv

.PHONY: help venv install test test-verbose clean lint format setup-dev

# Default target
help:
	@echo "Available commands:"
	@echo "  help         Show this help message"
	@echo "  venv         Create virtual environment"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  test-verbose Run tests with verbose output"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code"
	@echo "  clean        Clean up temporary files"
	@echo "  setup-dev    Set up development environment"

# Check if uv is installed
check-uv:
	@which uv > /dev/null || (echo "uv not found. Please install it: https://docs.astral.sh/uv/getting-started/installation/" && exit 1)

# Create virtual environment
venv: check-uv
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "Creating virtual environment with uv..."; \
		$(UV) venv $(VENV_NAME); \
	else \
		echo "Virtual environment already exists"; \
	fi

# Install dependencies
install: venv
	@echo "Installing dependencies with uv..."
	$(UV) pip install -r tests/requirements.txt
	$(UV) pip install -e .

# Run tests
test: venv
	@echo "Running tests in virtual environment..."
	$(PYTHON) -m pytest tests/ -q

# Run tests with verbose output
test-verbose: venv
	@echo "Running tests with verbose output..."
	$(PYTHON) -m pytest tests/ -v

# Run tests with coverage
test-coverage: venv
	@echo "Running tests with coverage..."
	$(UV) pip install pytest-cov
	$(PYTHON) -m pytest tests/ --cov=experiments --cov-report=html --cov-report=term

# Run specific test file
test-file: venv
	@echo "Usage: make test-file FILE=test_filename.py"
	@if [ -z "$(FILE)" ]; then echo "Please specify FILE=test_filename.py"; exit 1; fi
	$(PYTHON) -m pytest tests/$(FILE) -v

# Run linting
lint: venv
	@echo "Running flake8..."
	@$(UV) pip install flake8 > /dev/null 2>&1 || true
	@$(VENV_NAME)/bin/flake8 experiments/ tests/ 2>/dev/null || echo "flake8 check completed"
	@echo "Running pylint..."
	@$(UV) pip install pylint > /dev/null 2>&1 || true
	@$(VENV_NAME)/bin/pylint experiments/ 2>/dev/null || echo "pylint check completed"

# Format code
format: venv
	@echo "Running black..."
	@$(UV) pip install black > /dev/null 2>&1 || true
	@$(VENV_NAME)/bin/black experiments/ tests/ 2>/dev/null || echo "black formatting completed"
	@echo "Running isort..."
	@$(UV) pip install isort > /dev/null 2>&1 || true
	@$(VENV_NAME)/bin/isort experiments/ tests/ 2>/dev/null || echo "isort formatting completed"

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf $(VENV_NAME)

# Set up development environment
setup-dev: venv
	@echo "Setting up development environment..."
	$(UV) pip install pytest pytest-cov pytest-mock black flake8 pylint isort
	$(UV) pip install -r tests/requirements.txt
	$(UV) pip install -e .

# Check if experiments module can be imported
check-import: venv
	$(PYTHON) -c "import experiments; print('✓ experiments module imported successfully')"

# Run a quick smoke test
smoke-test: venv
	$(PYTHON) -c "from experiments import run_experiment, Metric; print('✓ Core imports work')"
	$(PYTHON) -c "from experiments.custom_decoding import GracefulWordCountLogitsProcessor; print('✓ Custom decoding imports work')"
	$(PYTHON) -c "from experiments.fine_tuning import WordCountLossTrainer; print('✓ Fine-tuning imports work')"
	$(PYTHON) -c "from experiments.prompt_templates import create_fine_tuning_prompt; print('✓ Prompt templates imports work')"

# Run tests for a specific module
test-experiments: venv
	$(PYTHON) -m pytest tests/test_experiments.py -v

test-custom-decoding: venv
	$(PYTHON) -m pytest tests/test_custom_decoding.py -v

test-fine-tuning: venv
	$(PYTHON) -m pytest tests/test_fine_tuning.py -v

test-prompt-templates: venv
	$(PYTHON) -m pytest tests/test_prompt_templates.py -v

test-metrics-tasks: venv
	$(PYTHON) -m pytest tests/test_metrics_and_tasks.py -v

# Debug failing tests
debug-test: venv
	@echo "Usage: make debug-test TEST=test_function_name"
	@if [ -z "$(TEST)" ]; then echo "Please specify TEST=test_function_name"; exit 1; fi
	$(PYTHON) -m pytest tests/ -k "$(TEST)" -v -s --tb=long

# Show virtual environment info
venv-info: venv
	@echo "Virtual environment: $(VENV_NAME)"
	@echo "Python executable: $(PYTHON)"
	@$(PYTHON) --version
	@echo "UV version:"
	@$(UV) --version
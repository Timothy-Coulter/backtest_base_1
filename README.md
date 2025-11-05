# quant-bench

A quantitative backtesting framework for financial analysis.

## Features

- Market data handling and validation
- Portfolio management and backtesting
- Strategy development and optimization
- Performance analysis and reporting

## Installation

```bash
uv sync
```

## Usage

```python
from backtester.main import run_backtest

# Run a basic backtest
results = run_backtest()

## Development Commands

### Code Formatting and Linting

**Windows one-liners:**
```cmd
rem Format code (ruff format, black, isort)
uv run ruff format . && uv run black . && uv run isort .

rem Lint code (ruff check)
uv run ruff check .

rem Lint and fix (ruff check --fix)
uv run ruff check --fix .

rem Type checking (mypy)
uv run mypy .

rem combined command
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check --fix . && uv run mypy .

```

**Linux/macOS one-liners:**
```bash
# Format code (ruff format, black, isort)
uv run ruff format . && uv run black . && uv run isort .

# Lint code (ruff check)
uv run ruff check .

# Lint and fix (ruff check --fix)
uv run ruff check --fix .

# Type checking (mypy)
uv run mypy .
```

### Testing

```bash
# Run tests (uses pyproject addopts: -n auto --reruns 3)
uv run pytest

# Run tests with coverage
uv run pytest --cov=backtester --cov-report=term-missing --cov-report=html
```

### Complete Development Workflow

**Windows:**
```cmd
rem Format, lint, and typecheck
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```

**Linux/macOS:**
```bash
# Format, lint, and typecheck
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```
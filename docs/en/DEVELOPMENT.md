# Development Guide

## Installation

To set up the development environment:

```bash
pip install .[dev,pt]
```

## Running Tests

We use `pytest` for testing. Run the following command to execute tests:

```bash
pytest
```

## Code Style

We use `ruff` and `black` for code formatting and linting.

```bash
ruff check .
black .
```

## Project Structure

- `rica/`: Source code
  - `core/`: Core application logic
  - `adapters/`: LLM adapters (e.g., Transformers)
  - `utils/`: Utility functions
- `tests/`: Unit and integration tests
- `demo/`: Example scripts and notebooks
- `docs/`: Documentation

# Development Environment and Testing Guide

This document outlines the standardized approach for development and testing in the KastenRAG project.

## Development Environment

We provide two options for development, with Docker being the preferred and recommended approach:

### Option 1: Docker Environment (Recommended)

The Docker environment ensures consistency across all development machines and CI/CD pipelines:

```bash
# Build and start all services
docker-compose up -d

# Run commands inside the container
docker-compose exec app bash

# Run tests
docker-compose exec app pytest

# View logs
docker-compose logs -f app

# Stop all services
docker-compose down
```

The Docker setup includes:
- Python environment with all dependencies
- Neo4j database for graph storage
- Volume mounts for code changes without rebuilding

### Option 2: Local Virtual Environment

For lightweight development, you can use a local virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate
# Or on Windows
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest
```

Note: This approach requires manually setting up dependencies like Neo4j.

## Testing Strategy

### Test Structure

Tests are organized following pytest's recommended structure:

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_transcribers.py
│   ├── test_chunkers.py
│   └── ...
├── integration/          # Tests for component interactions
│   ├── test_transcription_to_chunking.py
│   └── ...
├── e2e/                  # End-to-end pipeline tests
│   ├── test_pipeline_flow.py
│   └── ...
├── conftest.py           # Shared pytest fixtures
└── test_data/            # Test data files
```

### Running Tests

To run tests in the Docker environment:
```bash
docker-compose exec app pytest
```

To run tests in a virtual environment:
```bash
pytest
```

Specific test options:
- Run specific test file: `pytest tests/unit/test_transcribers.py`
- Run specific test function: `pytest tests/unit/test_transcribers.py::test_whisper_transcriber`
- Run with coverage: `pytest --cov=kastenrag`
- Generate coverage report: `pytest --cov=kastenrag --cov-report=html`

### Test-Driven Development

For each phase of implementation:
1. Write tests first (test-driven development)
2. Implement the feature
3. Verify tests pass in the Docker environment
4. Run performance benchmarks where applicable

### Continuous Integration

Our CI pipeline automatically runs tests on all pull requests and merges to the main branch using the same Docker configuration as the development environment.

## Code Organization

Follow these guidelines when developing new components:

1. Create a new module in the appropriate package (e.g., `kastenrag/transcribers/whisper.py`)
2. Implement the component class and a factory function with the `@register_component` decorator
3. Create unit tests in the corresponding test file (e.g., `tests/unit/test_transcribers.py`)
4. Create integration tests if the component interacts with others
5. Add the component to the pipeline orchestrator if needed

## Testing Utilities

The test framework provides several utilities:

- Mock components for testing in isolation
- Fixtures for common test data and configurations
- Helpers for creating test audio files and transcripts
- Metrics for evaluating system performance

These utilities should be used to ensure thorough testing without depending on external services.
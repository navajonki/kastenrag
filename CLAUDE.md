# KastenRAG Development Guide

## Environment Setup
- Python 3.11.3
- `pip install -r requirements.txt`

## Build & Test Commands
- Run tests: `pytest`
- Run specific test: `pytest tests/path/to/test_file.py::test_function`
- Run test with coverage: `pytest --cov=.`

## Code Style Guidelines
- **Imports**: Group in order: standard lib, third party, local modules; alphabetize within groups
- **Formatting**: Black with 88 character line length; isort for import sorting
- **Types**: Use type hints throughout; Pydantic for model validation
- **Naming**: snake_case for functions/variables; PascalCase for classes; UPPER_CASE for constants
- **Documentation**: Docstrings for all public functions/classes; doctest examples where appropriate
- **Error Handling**: Use specific exceptions; create custom exception classes when needed

## Project Structure
- Modular architecture with component registry system
- Configuration-driven behavior using Pydantic models
- Use dependency injection for component management
- Structured logging for all LLM interactions and performance metrics

## Development Workflow
- Maintain test coverage above 90% for critical components
- Document all API endpoints and configuration options
- Use BatchTool for parallel processing when possible
- Implement comprehensive logging for debugging and performance tracking
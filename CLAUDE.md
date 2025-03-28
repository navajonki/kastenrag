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

## Hybrid Retrieval Approaches

The project implements four hybrid retrieval approaches:

### 1. Sparse + Dense Hybrid Search
- Combines BM25 keyword search with vector embeddings
- Merges results using weighted score fusion or rank fusion
- Implementation: `retrievers/hybrid/sparse_dense.py`

### 2. Vector + Graph Hybrid (GraphRAG)
- Integrates vector store with graph database
- Sequential: vector search first, then graph expansion
- Parallel: query both independently and combine evidence
- Implementation: `retrievers/hybrid/graph_rag.py`

### 3. Hierarchical Multi-step Retrieval
- Two-stage process: high-recall method followed by high-precision
- Example: BM25 → neural reranker or dense → graph filtering
- Implementation: `retrievers/hybrid/multi_step.py`

### 4. Hierarchical Chunking with Multi-level Indexing
- Indexes chunks at multiple granularity levels
- Retrieves parent chunks first, then refines within context
- Implementation: `retrievers/hybrid/hierarchical.py`

## Development Workflow
- Maintain test coverage above 90% for critical components
- Document all API endpoints and configuration options
- Use BatchTool for parallel processing when possible
- Implement comprehensive logging for debugging and performance tracking

## Testing Commands
- Run unit tests: `pytest`
- Run specific test: `pytest tests/path/to/test_file.py::test_function`
- Test atomic chunking: `python scripts/test_atomic_chunking.py`
- Run with production config: `python app.py --config config/default_config.yaml`
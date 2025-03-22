# KastenRAG: Voice Note Taking with Atomic Chunking

A system for processing audio recordings into retrievable atomic knowledge units.

## Overview

KastenRAG processes voice recordings into atomic, self-contained facts that can be searched and retrieved with high precision. The system follows a pipeline approach from text chunking through storage, retrieval, and response generation, with audio transcription capabilities integrated later in development.

## Documentation

- [Implementation Plan](implementation_plan.md) - Detailed phased approach
- [Development Guide](DEVELOPMENT.md) - Environment setup and testing
- [PRD Overview](langchain-prd-overview.md) - Product requirements

## Setup Instructions

For detailed setup and development instructions, please refer to the [Development Guide](DEVELOPMENT.md).

### Quick Start: Docker Environment (Recommended)

1. Clone the repository and build the environment:
   ```bash
   git clone https://github.com/yourusername/kastenrag.git
   cd kastenrag
   docker-compose build
   docker-compose up -d
   ```

2. Test the Phase 1 implementation:
   ```bash
   docker-compose exec app python test_output.py
   ```

### Alternative: Local Virtual Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kastenrag.git
   cd kastenrag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. Test the Phase 1 implementation:
   ```bash
   python test_output.py
   ```

## Project Structure

```
kastenrag/
├── kastenrag/                # Main package
│   ├── chunkers/             # Text chunking components
│   ├── enrichers/            # Metadata enrichment components
│   ├── validators/           # Validation components
│   ├── storage/              # Storage components (vector, graph)
│   ├── retrieval/            # Retrieval components
│   ├── generators/           # Response generation components
│   ├── transcribers/         # Audio transcription components (later phase)
│   ├── api/                  # API endpoints
│   ├── config/               # Configuration system
│   ├── utils/                # Utility functions
│   └── pipeline/             # Pipeline orchestration
├── tests/                    # Test suite
├── docs/                     # Documentation
├── data/                     # Data directory
├── scripts/                  # Utility scripts
├── Dockerfile                # Docker definition
└── docker-compose.yml        # Docker composition
```

## Implementation Status

The system is being built in phases:

- [x] Phase 1: Development Environment and Framework Setup
- [ ] Phase 2: Atomic Chunking Implementation
- [ ] Phase 3: Vector Database Integration
- [ ] Phase 4: Graph Database Integration
- [ ] Phase 5: Hybrid Search Implementation
- [ ] Phase 6: Response Generation
- [ ] Phase 7: Audio Transcription Integration
- [ ] Phase 8: Performance Optimization and Scaling
- [ ] Phase 9: User Interface and API Development
- [ ] Phase 10: Testing, Documentation, and Deployment
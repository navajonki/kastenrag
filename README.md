# KastenRAG: Voice Note Taking with Atomic Chunking

A system for processing audio recordings into retrievable atomic knowledge units.

## Overview

KastenRAG processes voice recordings into atomic, self-contained facts that can be searched and retrieved with high precision. The system follows a pipeline approach from audio transcription through atomic fact extraction, enrichment, storage, and retrieval.

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
│   ├── transcribers/         # Audio transcription components
│   ├── chunkers/             # Text chunking components
│   ├── llm/                  # LLM integration components
│   ├── enrichers/            # Metadata enrichment components
│   ├── validators/           # Validation components
│   ├── storage/              # Storage components (vector, graph)
│   ├── retrieval/            # Retrieval components
│   ├── generators/           # Response generation components
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

The system is being built in phases following the data flow:

- [x] Phase 1: Development Environment and Framework Setup
- [ ] Phase 2: Audio Transcription Implementation
- [ ] Phase 3: Text Chunking Pipeline
- [ ] Phase 4: LLM-Powered Atomic Fact Extraction
- [ ] Phase 5: Metadata Enrichment and Topic Classification
- [ ] Phase 6: Quality Assurance and Validation
- [ ] Phase 7: Database Integration and Storage
- [ ] Phase 8: Retrieval System Implementation
- [ ] Phase 9: Response Generation and LLM Integration
- [ ] Phase 10: Performance Optimization and Scaling
- [ ] Phase 11: User Interface and API Development
- [ ] Phase 12: Testing, Documentation, and Deployment
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

## Testing with Real LLMs

The system supports multiple LLM providers for atomic chunking:

1. **OpenAI models**:
   - gpt-4o
   - gpt-4o-mini-2024-07-18
   - o3-mini-2025-01-31

2. **Replicate models**:
   - meta/meta-llama-3-8b-instruct
   - meta/meta-llama-3-70b-instruct

To test with a specific LLM:

```bash
# Option 1: Set API keys in environment
export OPENAI_API_KEY="your_openai_key"
export REPLICATE_API_TOKEN="your_replicate_token"

# Option 2: Create a .env file (recommended)
cp .env.template .env
# Edit .env with your API keys and preferred settings

# Test a specific model
python scripts/test_real_llm.py --provider openai --model gpt-4o-mini-2024-07-18 --input-file test_inputs/JCS_transcript_short.txt

# For quick testing
./scripts/test_quick_llm.sh openai gpt-4o-mini-2024-07-18

# Test all configured models
./scripts/test_all_llms.sh
```

## Implementation Status

The system is being built in phases:

- [x] Phase 1: Development Environment and Framework Setup
- [x] Phase 2: Atomic Chunking Implementation
- [ ] Phase 3: Vector Database Integration (In Progress)
- [ ] Phase 4: Graph Database Integration
- [ ] Phase 5: Retrieval Engine and API
- [ ] Phase 6: Generation Layer
- [ ] Phase 7: User Interface
- [ ] Phase 8: Deployment and Scaling
- [ ] Phase 9: Security and Compliance
- [ ] Phase 10: Evaluation and Fine-tuning

### Phase 2 Completion
The atomic chunking system has been successfully implemented, featuring:
- Two-pass LLM-based fact extraction and refinement
- Entity resolution for contextual independence
- Topic tagging for improved retrieval
- Relationship extraction for entity connections
- Quality validation metrics for chunks
- Multiple LLM provider support (OpenAI, Replicate, Mock)
- Structured logging system

### Phase 3 Next Steps
The next phase focuses on vector database integration:
- Embedding generation for atomic chunks
- ChromaDB integration for vector storage
- Efficient similarity search mechanisms
- Metadata management for improved retrieval
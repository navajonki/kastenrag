# KastenRAG: Voice Note Taking with Atomic Chunking

A system for processing audio recordings into retrievable atomic knowledge units.

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for Neo4j and containerized development)
- FFmpeg (for audio processing)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/kastenrag.git
   cd kastenrag
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

4. Start the Neo4j database using Docker Compose:
   ```
   docker-compose up -d neo4j
   ```

### Configuration

- Copy and modify the default configuration file:
  ```
  cp config/default_config.yaml config/my_config.yaml
  ```

- Edit `config/my_config.yaml` with your preferred settings.

### Preparing Test Data

- Run the test data preparation script:
  ```
  python -m scripts.prepare_test_data
  ```

### Running the System

- Process an audio file:
  ```
  python -m kastenrag.app --config config/my_config.yaml --audio data/sample_audio/sample1.mp3
  ```

- Results will be stored in the configured output directory.

## Development

### Running Tests

- Run the test suite:
  ```
  pytest
  ```

- Run with coverage:
  ```
  pytest --cov=kastenrag
  ```

### Docker Development Environment

- Build and start the development container:
  ```
  docker-compose up -d app
  ```

- Run commands inside the container:
  ```
  docker-compose exec app bash
  ```

## Implementation Plan

The system is built in phases following the data flow from audio input to retrieval:

1. Development Environment and Framework Setup
2. Audio Transcription Implementation
3. Text Chunking Pipeline
4. LLM-Powered Atomic Fact Extraction
5. Metadata Enrichment and Topic Classification
6. Quality Assurance and Validation
7. Database Integration and Storage
8. Retrieval System Implementation
9. Response Generation and LLM Integration
10. Performance Optimization and Scaling
11. User Interface and API Development
12. Testing, Documentation, and Deployment

See the [Implementation Plan](implementation_plan.md) for details.
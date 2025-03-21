"""Pytest fixtures and setup for KastenRAG tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator, Optional

import pytest

from kastenrag.config.models import SystemConfig
from kastenrag.utils.logging import setup_logging
from kastenrag.utils.registry import registry


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config() -> SystemConfig:
    """Create a sample configuration for testing."""
    return SystemConfig(
        transcriber={
            "type": "whisper_local",
            "model_name": "tiny",
            "language": "en",
            "device": "cpu"
        },
        chunker={
            "window_size": 500,
            "overlap": 50,
            "boundary_rules": ["sentence"]
        },
        embedding={
            "model_name": "test-embeddings",
            "dimensions": 128,
            "batch_size": 4
        },
        vector_store={
            "type": "memory",
            "persist_directory": "./test_data/vector_store",
            "collection_name": "test_collection"
        },
        graph_store={
            "type": "memory",
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database": "test"
        },
        llm={
            "provider": "mock",
            "model_name": "test-model",
            "temperature": 0.0,
            "max_tokens": 100,
            "streaming": False
        },
        logging={
            "level": "DEBUG",
            "log_dir": "./test_logs",
            "log_llm_interactions": True,
            "log_performance": True
        },
        output_dir="./test_output",
        run_id="test_run"
    )


@pytest.fixture
def setup_test_logging(sample_config, temp_dir):
    """Set up logging for tests."""
    # Override log directory
    sample_config.logging.log_dir = str(temp_dir / "logs")
    
    # Set up logging
    logger, llm_logger, perf_logger = setup_logging(
        sample_config.logging, 
        sample_config.run_id
    )
    
    return logger, llm_logger, perf_logger


@pytest.fixture
def mock_registry():
    """Set up a registry with mock components for testing."""
    # Register mock transcriber
    @registry.register("transcriber", "mock")
    def create_mock_transcriber(**kwargs):
        return MockTranscriber(**kwargs)
    
    # Register mock chunker
    @registry.register("chunker", "mock")
    def create_mock_chunker(**kwargs):
        return MockChunker(**kwargs)
    
    # Add more mock components as needed...
    
    return registry


class MockTranscriber:
    """Mock transcriber for testing."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def transcribe(self, audio_path):
        return {
            "text": "This is a mock transcription.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "This is a mock transcription."
                }
            ]
        }


class MockChunker:
    """Mock chunker for testing."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def chunk(self, text):
        return [
            {"text": "This is chunk 1.", "metadata": {}},
            {"text": "This is chunk 2.", "metadata": {}}
        ]
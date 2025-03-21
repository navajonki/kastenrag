#!/usr/bin/env python3
"""Test script for validating Phase 1 implementation."""

import os
import sys
from pathlib import Path

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath("."))

from kastenrag.config.loader import load_and_validate_config
from kastenrag.config.models import SystemConfig
from kastenrag.pipeline.orchestrator import create_pipeline
from kastenrag.utils.logging import setup_logging
from kastenrag.utils.registry import registry

# Import component implementations to register them
from kastenrag.transcribers.mock import create_mock_transcriber
from kastenrag.chunkers.basic import create_sliding_window_chunker


def main():
    """Run a test of the Phase 1 implementation."""
    print("Testing KastenRAG Phase 1 implementation...")
    
    # Create test directories if they don't exist
    os.makedirs("./data/sample_audio", exist_ok=True)
    
    # Create a test audio file if it doesn't exist
    test_audio_path = "./data/sample_audio/test_sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"Creating test audio file at {test_audio_path}...")
        # Just create an empty file for testing
        with open(test_audio_path, "w") as f:
            f.write("This is a placeholder for a WAV file")
    
    # Create a test configuration
    config = SystemConfig(
        transcriber={
            "type": "mock",
            "model_name": "test-model",
            "language": "en"
        },
        chunker={
            "window_size": 200,
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
            "persist_directory": "./test_output/vector_store",
            "collection_name": "test"
        },
        graph_store={
            "type": "neo4j",
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j"
        },
        llm={
            "provider": "mock",
            "model_name": "test-model",
            "temperature": 0.0,
            "max_tokens": 100
        },
        logging={
            "level": "INFO",
            "log_dir": "./test_output/logs",
            "log_llm_interactions": True,
            "log_performance": True
        },
        output_dir="./test_output",
        run_id="test_run_123"
    )
    
    # Set up logging
    print("Setting up logging system...")
    logger, llm_logger, perf_logger = setup_logging(config.logging, config.run_id)
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline(config)
    
    # Check registered components
    print("\nRegistered component types:")
    for component_type in registry.list_component_types():
        implementations = registry.list_implementations(component_type)
        print(f"  {component_type}: {implementations}")
    
    # Set audio path in context
    pipeline.context.set("audio_path", test_audio_path)
    pipeline.context.set("llm_logger", llm_logger)
    pipeline.context.set("performance_logger", perf_logger)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    pipeline.execute_all()
    
    # Print results
    print("\nPipeline execution complete!")
    print("\nTranscript:")
    transcript = pipeline.context.get("transcript")
    print(f"  {transcript['text']}")
    
    print("\nChunks:")
    chunks = pipeline.context.get("chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['text'][:50]}...")
        print(f"    Metadata: {chunk['metadata']}")
        print()
    
    print(f"All logs written to: {config.logging.log_dir}/{config.run_id}")
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
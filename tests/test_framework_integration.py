"""Tests for the framework integration."""

import os
from pathlib import Path

import pytest

from kastenrag.config.loader import load_and_validate_config
from kastenrag.pipeline.orchestrator import create_pipeline


def test_config_loading(sample_config):
    """Test that configuration can be loaded and validated."""
    assert sample_config.transcriber.type == "whisper_local"
    assert sample_config.chunker.window_size == 500
    assert sample_config.llm.provider == "mock"


def test_pipeline_creation(sample_config):
    """Test that a pipeline can be created from configuration."""
    pipeline = create_pipeline(sample_config)
    assert len(pipeline.steps) > 0
    

def test_context_operations(sample_config):
    """Test operations on the pipeline context."""
    pipeline = create_pipeline(sample_config)
    
    # Set a value in the context
    pipeline.context.set("test_key", "test_value")
    
    # Get the value back
    value = pipeline.context.get("test_key")
    assert value == "test_value"
    
    # Check that non-existent keys return None
    value = pipeline.context.get("non_existent_key")
    assert value is None
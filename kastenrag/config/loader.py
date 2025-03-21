"""Configuration loading and validation utilities."""

import os
import yaml
import uuid
from datetime import datetime
from typing import Dict, Optional, Union

from pydantic import ValidationError
from .models import SystemConfig


def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_config_from_env() -> Dict:
    """Load configuration from environment variables."""
    config = {}
    
    # Example of environment variable mapping
    if os.environ.get('KASTENRAG_TRANSCRIBER_TYPE'):
        config['transcriber'] = {
            'type': os.environ.get('KASTENRAG_TRANSCRIBER_TYPE'),
            'model_name': os.environ.get('KASTENRAG_TRANSCRIBER_MODEL'),
            'api_key': os.environ.get('KASTENRAG_TRANSCRIBER_API_KEY'),
            'language': os.environ.get('KASTENRAG_TRANSCRIBER_LANGUAGE', 'en'),
        }
    
    # Add more mappings for other configuration sections...
    
    return config


def generate_run_id() -> str:
    """Generate a unique run ID with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{unique_id}"


def load_and_validate_config(
    config_path: Optional[str] = None,
    env_override: bool = True
) -> SystemConfig:
    """
    Load configuration from a file and/or environment variables, and validate it.
    
    Args:
        config_path: Path to config YAML file. If None, will use environment variables only.
        env_override: Whether to allow environment variables to override file config.
        
    Returns:
        Validated SystemConfig object
    """
    # Start with empty config
    config_dict = {}
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        config_dict = load_config_from_yaml(config_path)
    
    # Override with environment variables if requested
    if env_override:
        env_config = load_config_from_env()
        deep_update(config_dict, env_config)
    
    # Generate run ID if not provided
    if 'run_id' not in config_dict or not config_dict['run_id']:
        config_dict['run_id'] = generate_run_id()
    
    # Create output and log directories if they don't exist
    if 'output_dir' in config_dict:
        os.makedirs(config_dict['output_dir'], exist_ok=True)
    
    if 'logging' in config_dict and 'log_dir' in config_dict['logging']:
        os.makedirs(config_dict['logging']['log_dir'], exist_ok=True)
    
    # Validate against Pydantic model
    try:
        return SystemConfig(**config_dict)
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        raise


def deep_update(original: Dict, update: Dict) -> Dict:
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        original: Original dictionary to update
        update: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update(original[key], value)
        elif value is not None:
            original[key] = value
    return original
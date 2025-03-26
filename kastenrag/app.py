"""Main application entry point for KastenRAG."""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

from .config.loader import load_and_validate_config
from .config.models import SystemConfig
from .pipeline.orchestrator import create_pipeline
from .utils.logging import setup_logging
from .prompts.config import initialize_templates
from .prompts.registry import register_default_templates


def run_pipeline(config: SystemConfig, audio_path: str) -> Dict:
    """
    Run the full processing pipeline on an audio file.
    
    Args:
        config: System configuration
        audio_path: Path to the audio file to process
        
    Returns:
        Processing results
    """
    # Set up logging
    logger, llm_logger, perf_logger = setup_logging(config.logging, config.run_id)
    logger.info(f"Starting pipeline for run {config.run_id}")
    logger.info(f"Processing audio file: {audio_path}")
    
    # Create and execute pipeline
    pipeline = create_pipeline(config)
    
    # Add audio path to pipeline context
    pipeline.context.set("audio_path", audio_path)
    pipeline.context.set("llm_logger", llm_logger)
    pipeline.context.set("performance_logger", perf_logger)
    
    # Execute pipeline
    pipeline.execute_all()
    
    # Retrieve results from context
    results = {
        "run_id": config.run_id,
        "transcript": pipeline.context.get("transcript"),
        "chunks": pipeline.context.get("chunks"),
        # More results will be added in future phases
    }
    
    logger.info(f"Pipeline execution complete for run {config.run_id}")
    return results


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="KastenRAG Voice Note Processing System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file to process")
    parser.add_argument("--output-dir", type=str, help="Directory for outputs")
    parser.add_argument("--template-dir", type=str, help="Directory for custom prompt templates")
    
    args = parser.parse_args()
    
    # Initialize prompt templates
    register_default_templates()  # Register built-in templates
    initialize_templates()        # Load templates from system and user directories
    
    # Load additional templates from user-specified directory if provided
    if args.template_dir:
        from .prompts.config import load_user_templates
        load_user_templates(args.template_dir)
    
    # Load configuration
    config = load_and_validate_config(config_path=args.config)
    
    # Override output directory if provided
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run pipeline
    results = run_pipeline(config, args.audio)
    
    # Output results
    print(f"Processing complete. Results stored in {config.output_dir}/{config.run_id}")


if __name__ == "__main__":
    main()
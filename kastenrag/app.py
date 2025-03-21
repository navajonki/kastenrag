"""Main application entry point for KastenRAG."""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

from .config.loader import load_and_validate_config
from .config.models import SystemConfig
from .pipeline.orchestrator import create_pipeline
from .utils.logging import setup_logging


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
    
    args = parser.parse_args()
    
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
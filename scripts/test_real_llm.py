#!/usr/bin/env python
"""Test script for real LLM integration."""

import os
import sys
import time
import argparse
import json
import datetime
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")

# Add the parent directory to the path to import kastenrag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kastenrag.config.models import SystemConfig, ChunkerConfig, LLMConfig, LoggingConfig
from kastenrag.pipeline.orchestrator import Pipeline, create_pipeline
from kastenrag.llm import set_llm_client, create_llm_client
from kastenrag.utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test real LLM integration")
    
    # LLM provider
    parser.add_argument(
        "--provider", 
        type=str, 
        default=os.environ.get("LLM_PROVIDER", "mock"), 
        choices=["openai", "replicate", "mock"],
        help="LLM provider to use (openai, replicate, mock)"
    )
    
    # Model name
    parser.add_argument(
        "--model", 
        type=str, 
        default=os.environ.get("LLM_MODEL"),
        help="Model name to use"
    )
    
    # API keys/tokens
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for OpenAI (falls back to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-token", 
        type=str, 
        default=os.environ.get("REPLICATE_API_TOKEN"),
        help="API token for Replicate (falls back to REPLICATE_API_TOKEN env var)"
    )
    
    # Input file
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="/Users/zbodnar/kastenrag/test_inputs/JCS_transcript.txt",
        help="Input file to process"
    )

    # Chunker parameters
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=int(os.environ.get("WINDOW_SIZE", 1000)),
        help="Window size for chunking"
    )
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=int(os.environ.get("OVERLAP", 200)),
        help="Overlap for chunking"
    )
    
    # LLM parameters
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=float(os.environ.get("TEMPERATURE", 0.0)),
        help="Temperature for LLM"
    )
    
    # Handle MAX_TOKENS environment variable if it exists
    default_max_tokens = None
    if "MAX_TOKENS" in os.environ:
        default_max_tokens = int(os.environ.get("MAX_TOKENS"))
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=default_max_tokens,
        help="Maximum tokens for LLM responses"
    )
    
    return parser.parse_args()


def get_default_model(provider):
    """Get default model name for provider."""
    if provider == "openai":
        return "gpt-4o"
    elif provider == "replicate":
        return "meta/meta-llama-3-8b-instruct"
    else:
        return "mock-model"


def main():
    """Run the test script."""
    args = parse_args()
    
    # Create a run ID
    run_id = f"{args.provider}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Make sure our chunkers are registered
    import kastenrag.chunkers
    
    # Create output directory which will also contain logs
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'test_output', run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Create log directory within the output directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up model name
    model_name = args.model or get_default_model(args.provider)
    print(f"Using provider: {args.provider}, model: {model_name}")
    
    # Create LLM client
    llm_client = create_llm_client(
        provider=args.provider,
        model_name=model_name,
        api_key=args.api_key,
        api_token=args.api_token,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    set_llm_client(llm_client)
    
    # Create configuration
    # Handle max_tokens which can be None but the model doesn't accept None
    llm_config = {
        "provider": args.provider,
        "model_name": model_name,
        "temperature": args.temperature
    }
    # Only add max_tokens if it's not None
    if args.max_tokens is not None:
        llm_config["max_tokens"] = args.max_tokens
        
    config = SystemConfig(
        transcriber={"type": "mock", "language": "en"},
        chunker=ChunkerConfig(
            type="atomic",
            window_size=args.window_size,
            overlap=args.overlap,
            boundary_rules=["sentence", "paragraph"],
            entity_resolution=True,
            topic_tagging=True,
            relationship_extraction=True,
        ),
        embedding={"model_name": "text-embedding-ada-002", "dimensions": 1536, "batch_size": 8},
        vector_store={"type": "chroma", "persist_directory": "./data/vector_store", "collection_name": "kastenrag"},
        graph_store={"type": "neo4j", "uri": "bolt://localhost:7687", "username": "neo4j", "password": "password", "database": "neo4j"},
        llm=LLMConfig(**llm_config),
        logging=LoggingConfig(level="INFO", log_dir=log_dir, log_llm_interactions=True, log_performance=True),
        output_dir=output_dir
    )
    
    # Create and set up the pipeline
    pipeline = create_pipeline(config)
    
    # Load input text
    with open(args.input_file, 'r') as f:
        input_text = f.read()
    
    # Set up the pipeline context
    pipeline.context.set("text_input", input_text)
    
    # Set up logging
    logging_config = config.logging
    logger, llm_logger, perf_logger = setup_logging(logging_config, run_id)
    logger.info(f"Starting pipeline for run {run_id}")
    logger.info(f"Using provider: {args.provider}, model: {model_name}")
    
    # Add loggers to pipeline context
    pipeline.context.set("llm_logger", llm_logger)
    pipeline.context.set("performance_logger", perf_logger)
    
    # Track execution time
    start_time = time.time()
    
    # Execute the pipeline
    print("Executing pipeline...")
    logger.info("Starting pipeline execution")
    context = pipeline.execute_all()
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"Pipeline execution completed in {execution_time:.2f} seconds")
    
    # Print results
    print("\nProcessing Results:")
    print("===================")
    
    # Get the validated chunks
    validated_chunks = context.get("validated_chunks")
    if validated_chunks:
        print(f"\nGenerated {len(validated_chunks)} atomic chunks.\n")
        
        # Create markdown directory for chunks
        markdown_dir = os.path.join(output_dir, "markdown")
        os.makedirs(markdown_dir, exist_ok=True)
        
        # Create a summary file
        summary_file = os.path.join(output_dir, "summary.md")
        with open(summary_file, 'w') as summary:
            summary.write(f"# {args.provider.capitalize()} ({model_name}) Processing Summary\n\n")
            summary.write(f"Generated {len(validated_chunks)} atomic chunks on {datetime.datetime.now().isoformat()}\n")
            summary.write(f"Execution time: {execution_time:.2f} seconds\n\n")
            summary.write("## Chunks Overview\n\n")
            
            # Save each chunk as a markdown file with YAML front matter
            for i, chunk in enumerate(validated_chunks):
                chunk_id = chunk['metadata']['chunk_id']
                
                # Add to summary
                summary.write(f"### Chunk {i+1}: {chunk_id}\n")
                summary.write(f"**Text:** {chunk['text']}\n")
                summary.write(f"**Topics:** {', '.join(chunk['metadata'].get('topics', []))}\n")
                summary.write(f"**Quality:** {chunk['metadata'].get('quality_metrics', {}).get('overall_quality', 0):.2f}\n\n")
                
                # Create the markdown file with YAML front matter
                filename = os.path.join(markdown_dir, f"{chunk_id}.md")
                with open(filename, 'w') as f:
                    # Write YAML front matter
                    f.write("---\n")
                    f.write(f"chunk_id: {chunk_id}\n")
                    f.write(f"entities: {json.dumps(chunk['metadata'].get('entities', []))}\n")
                    f.write(f"topics: {json.dumps(chunk['metadata'].get('topics', []))}\n")
                    
                    # Add relationships if present
                    relationships = chunk['metadata'].get('relationships', [])
                    if relationships:
                        f.write(f"relationships: {json.dumps(relationships)}\n")
                    
                    # Add quality metrics
                    quality_metrics = chunk['metadata'].get('quality_metrics', {})
                    f.write(f"quality:\n")
                    for metric, value in quality_metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    
                    f.write(f"created_at: {datetime.datetime.now().isoformat()}\n")
                    f.write("---\n\n")
                    
                    # Write the chunk text
                    f.write(chunk['text'])
                
                # Print short info about the chunk
                if i < 5 or i >= len(validated_chunks) - 5:  # Only print first and last 5 chunks
                    print(f"Chunk {i+1}: {chunk['text'][:100]}...")
            
            if len(validated_chunks) > 10:
                print(f"... {len(validated_chunks) - 10} more chunks not shown ...")
    
    # Get the overall quality metrics
    metrics = context.get("quality_metrics")
    if metrics:
        print("\nOverall Quality Metrics:")
        print("------------------------")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Add metrics to summary
        with open(summary_file, 'a') as summary:
            summary.write("## Quality Metrics\n\n")
            for key, value in metrics.items():
                summary.write(f"- **{key}**: {value:.4f}\n")
        
        # Save metrics to a JSON file
        metrics_file = os.path.join(output_dir, "quality_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            # Add execution metadata
            metrics_data = {
                **metrics,
                "execution_time": execution_time,
                "provider": args.provider,
                "model": model_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
            json.dump(metrics_data, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_file}")
        
    # Save input text for reference
    input_file = os.path.join(output_dir, "input.txt")
    with open(input_file, 'w') as f:
        f.write(input_text)
    print(f"Input text saved to: {input_file}")
    print(f"Summary saved to: {summary_file}")
    
    print(f"\nAll output saved to: {output_dir}")


if __name__ == "__main__":
    main()
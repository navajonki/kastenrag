"""Logging utilities for KastenRAG."""

import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Tuple

from loguru import logger

from ..config.models import LoggingConfig


class LLMLogger:
    """Logger for LLM interactions."""
    
    def __init__(self, config: LoggingConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.log_dir = Path(config.log_dir) / run_id / "llm"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{run_id}_llm.jsonl"
        
    def log_interaction(
        self, 
        model: str, 
        prompt: str, 
        response: str,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        latency: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """Log an LLM interaction."""
        if not self.config.log_llm_interactions:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "model": model,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "latency": latency,
            "metadata": metadata or {}
        }
        
        # Print to console for real-time feedback
        print(f"LLM Interaction: {model} - Latency: {latency:.2f}s, Tokens: {prompt_tokens}/{response_tokens}")
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()  # Ensure logs are written immediately


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, config: LoggingConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.log_dir = Path(config.log_dir) / run_id / "performance"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{run_id}_performance.jsonl"
    
    def log_metric(self, component: str, operation: str, elapsed_time: float, metadata: Optional[Dict] = None):
        """Log a performance metric."""
        if not self.config.log_performance:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "component": component,
            "operation": operation,
            "elapsed_time": elapsed_time,
            "metadata": metadata or {}
        }
        
        # Print for real-time feedback
        print(f"Performance: {component}.{operation} - Elapsed time: {elapsed_time:.4f}s")
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()  # Ensure logs are written immediately


def performance_timer(component: str, operation: str):
    """Decorator to time function execution and log performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # Try to get the logger from the instance if it's a method
            logger_instance = None
            if args and hasattr(args[0], 'performance_logger'):
                logger_instance = args[0].performance_logger
            
            if logger_instance:
                logger_instance.log_metric(
                    component=component,
                    operation=operation,
                    elapsed_time=elapsed_time,
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
            
            return result
        return wrapper
    return decorator


def setup_logging(config: LoggingConfig, run_id: str) -> Tuple[object, LLMLogger, PerformanceLogger]:
    """Set up the logging system."""
    # Create log directories
    log_dir = Path(config.log_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=config.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler for general logs
    logger.add(
        sink=log_dir / f"{run_id}_app.log",
        level=config.level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="1 week"
    )
    
    # Create specialized loggers
    llm_logger = LLMLogger(config, run_id)
    performance_logger = PerformanceLogger(config, run_id)
    
    return logger, llm_logger, performance_logger
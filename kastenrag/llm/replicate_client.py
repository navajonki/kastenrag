"""Replicate LLM client implementation."""

import os
import time
from typing import Dict, Any, Optional, List
import json

import replicate

from ..utils.logging import performance_timer


class ReplicateClient:
    """Replicate API client for LLM interactions."""
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        model: str = "meta/meta-llama-3-8b-instruct",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Replicate client.
        
        Args:
            api_token: Replicate API token (falls back to REPLICATE_API_TOKEN env variable)
            model: Model to use (e.g., "meta/meta-llama-3-8b-instruct")
            temperature: Temperature parameter controlling randomness
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Replicate API token must be provided either in the constructor "
                "or as the REPLICATE_API_TOKEN environment variable"
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Set API token for replicate
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
    
    @performance_timer("llm", "invoke")
    def invoke(
        self, 
        prompt: str, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Invoke the Replicate API.
        
        Args:
            prompt: Prompt text to send to LLM
            temperature: Temperature parameter (overrides instance default)
            max_tokens: Maximum number of tokens (overrides instance default)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Use provided parameters or fall back to instance defaults
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Merge instance kwargs with call-specific kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        
        # Get the logger from the context if available and store it separately
        # It should not be passed to the Replicate API
        llm_logger = None
        if "llm_logger" in call_kwargs:
            llm_logger = call_kwargs.pop("llm_logger")
        
        # Prepare API call parameters
        api_params = {
            "temperature": temperature,
            **call_kwargs
        }
        
        # Add max_tokens if specified (called max_new_tokens in Replicate)
        if max_tokens is not None:
            api_params["max_new_tokens"] = max_tokens
        
        # Convert prompt to Replicate's expected format
        # Different models might have different input formats
        if "meta-llama-3" in self.model:
            # Llama 3 typically expects a specific format
            api_params["prompt"] = prompt
        else:
            # General fallback
            api_params["prompt"] = prompt
        
        # Measure response time
        start_time = time.time()
        
        # Make API call
        try:
            output = replicate.run(
                self.model,
                input=api_params
            )
            
            # Replicate returns a generator, so we join the results
            response_text = ""
            if hasattr(output, "__iter__") and not isinstance(output, str):
                response_text = "".join(output)
            else:
                response_text = str(output)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Log the interaction if a logger is available
            if llm_logger and hasattr(llm_logger, 'log_interaction'):
                # Replicate doesn't provide token counts directly
                # Make sure metadata is JSON serializable
                clean_kwargs = {}
                for k, v in call_kwargs.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_kwargs[k] = v
                    else:
                        clean_kwargs[k] = str(v)
                
                llm_logger.log_interaction(
                    model=self.model,
                    prompt=prompt,
                    response=response_text,
                    prompt_tokens=None,  # Replicate doesn't provide token counts
                    response_tokens=None,
                    latency=latency,
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "provider": "replicate",
                        **clean_kwargs
                    }
                )
            
            return response_text
            
        except Exception as e:
            print(f"Error invoking Replicate API: {e}")
            
            # Log error if logger is available
            if llm_logger and hasattr(llm_logger, 'log_interaction'):
                # Make sure metadata is JSON serializable
                clean_kwargs = {}
                for k, v in call_kwargs.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_kwargs[k] = v
                    else:
                        clean_kwargs[k] = str(v)
                
                llm_logger.log_interaction(
                    model=self.model,
                    prompt=prompt,
                    response=f"ERROR: {str(e)}",
                    latency=time.time() - start_time,
                    metadata={
                        "error": str(e),
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "provider": "replicate",
                        **clean_kwargs
                    }
                )
            
            raise
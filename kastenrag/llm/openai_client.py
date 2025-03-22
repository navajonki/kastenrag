"""OpenAI LLM client implementation."""

import os
import time
from typing import Dict, Any, Optional, List
import json

import openai
from openai import OpenAI

from ..utils.logging import performance_timer


class OpenAIClient:
    """OpenAI API client for LLM interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env variable)
            model: Model to use (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Temperature parameter controlling randomness
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either in the constructor "
                "or as the OPENAI_API_KEY environment variable"
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
    
    @performance_timer("llm", "invoke")
    def invoke(
        self, 
        prompt: str, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Invoke the OpenAI API.
        
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
        # It should not be passed to the OpenAI API
        llm_logger = None
        if "llm_logger" in call_kwargs:
            llm_logger = call_kwargs.pop("llm_logger")
        
        # Remove None values
        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}
        
        # Create a message for the chat API
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **call_kwargs
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        # Measure response time
        start_time = time.time()
        
        # Make API call
        try:
            # Call the API
            response = self.client.chat.completions.create(**api_params)
            response_text = response.choices[0].message.content or ""
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Log the interaction if a logger is available
            if llm_logger and hasattr(llm_logger, 'log_interaction'):
                # Get token counts if available
                usage = getattr(response, 'usage', None)
                prompt_tokens = usage.prompt_tokens if usage else None
                completion_tokens = usage.completion_tokens if usage else None
                
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
                    prompt_tokens=prompt_tokens,
                    response_tokens=completion_tokens,
                    latency=latency,
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **clean_kwargs
                    }
                )
            
            return response_text
            
        except Exception as e:
            print(f"Error invoking OpenAI API: {e}")
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
                        **clean_kwargs
                    }
                )
            raise
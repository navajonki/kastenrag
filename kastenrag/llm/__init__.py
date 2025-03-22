"""LLM client implementations for KastenRAG."""

from typing import Dict, Any, Optional, Union
import os
import json
import importlib
from ..utils.logging import performance_timer

# Import clients (with error handling for missing dependencies)
try:
    from .openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from .replicate_client import ReplicateClient
except ImportError:
    ReplicateClient = None


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, **kwargs):
        """Initialize mock LLM client."""
        self.kwargs = kwargs
        self.model = kwargs.get("model", "mock-model")
    
    @performance_timer("llm", "invoke")
    def invoke(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Invoke the mock LLM client.
        
        Args:
            prompt: Prompt text to send to LLM
            temperature: Temperature parameter controlling randomness
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Simulate processing delay
        import time
        start_time = time.time()
        time.sleep(0.1)  # Add a small delay for realism
        
        # Check if the JCS transcript is being processed (look for a snippet in the prompt)
        is_jcs_transcript = "Jennifer" in prompt and "grew up in Colorado" in prompt
        
        # For mock implementation, generate a response based on the prompt
        if "extract atomic facts" in prompt.lower():
            if is_jcs_transcript:
                # Return mock facts for JCS transcript
                response = """
                FACT 1: Jennifer grew up in Colorado and graduated from a small high school with 79 people in her graduating class.
                FACT 2: Jennifer joined the military at 17 years old to get away from her small town and see what she could make of herself.
                FACT 3: Jennifer served in the military for five years.
                FACT 4: Jennifer described her childhood as rough with money being tight and education in short supply.
                FACT 5: Jennifer's mother always encouraged her in whatever she wanted to do.
                FACT 6: Jennifer got married right after leaving the military.
                FACT 7: Jennifer got into acting as a way of dealing with personal issues she was experiencing after the military.
                FACT 8: Jennifer's hair started falling out during the COVID-19 pandemic in 2020 and she was not feeling well.
                FACT 9: Jennifer's doctor initially couldn't see her due to COVID restrictions unless she was a cancer patient.
                FACT 10: Jennifer's doctor found a spot on her foot that turned out to be melanoma.
                """
            else:
                # Return standard mock atomic facts for AI
                response = """
                FACT 1: Artificial Intelligence (AI) is the simulation of human intelligence by machines.
                FACT 2: AI encompasses many subfields including machine learning and natural language processing.
                FACT 3: The goal of AI is to create systems that can perform tasks requiring human intelligence.
                FACT 4: Machine Learning is a subset of Artificial Intelligence that involves algorithms that improve through experience.
                FACT 5: Machine Learning is based on the idea that systems can learn from data and make decisions with minimal human intervention.
                """
        elif "improve them to be completely self-contained" in prompt.lower():
            if "Jennifer grew up in Colorado" in prompt:
                # Return refined facts for JCS transcript
                response = """
                FACT 1: Jennifer grew up in Colorado and graduated from a tiny high school with only 79 people in her graduating class.
                FACT 2: Jennifer joined the United States military at 17 years old as a way to escape her small hometown and discover her potential.
                FACT 3: Jennifer served in the United States military for five years, describing it as "the best of times, worst of times."
                FACT 4: Jennifer described her childhood as rough, with financial hardships, limited educational opportunities, and a "hard scrabble" existence.
                FACT 5: Jennifer's mother was a supportive figure who always encouraged Jennifer in whatever she wanted to pursue.
                FACT 6: Jennifer got married immediately after her five-year service in the military ended.
                FACT 7: Jennifer pursued acting as a therapeutic way to address personal struggles she experienced after leaving the military.
                FACT 8: Jennifer began experiencing health problems during the COVID-19 pandemic in 2020, including hair loss and general unwellness.
                FACT 9: Jennifer's doctor was initially unable to see her for medical care due to COVID-19 restrictions unless she was already a cancer patient.
                FACT 10: A doctor discovered a spot on Jennifer's foot that was biopsied and diagnosed as melanoma, despite not having the typical appearance of skin cancer.
                FACT 11: Jennifer was diagnosed with Stage 3 melanoma when doctors found that the cancer had spread to her lymph nodes.
                FACT 12: Jennifer kept her cancer diagnosis secret while working on the show "Stranger Things" because she feared being written out of the show.
                FACT 13: Jennifer underwent eighteen months of cancer treatment which caused significant side effects including severe illness and cognitive issues.
                FACT 14: Jennifer has been cancer-free for four months but remains classified as Stage 3 with No Evidence of Disease (NED).
                FACT 15: Jennifer considers cancer to have been the greatest blessing in her life besides her children because it completely changed her perspective.
                """
            else:
                # Return standard mock refined facts
                response = """
                FACT 1: Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems and machines.
                FACT 2: Artificial Intelligence (AI) encompasses many subfields including machine learning, natural language processing, and computer vision.
                FACT 3: The primary goal of Artificial Intelligence (AI) is to create computer systems that can perform tasks that would normally require human intelligence.
                FACT 4: Machine Learning is a subset of Artificial Intelligence that involves computer algorithms that improve automatically through experience and data analysis.
                FACT 5: Machine Learning systems can analyze data, identify patterns, and make decisions with minimal human intervention by building mathematical models based on sample data.
                """
        else:
            # Generic mock response
            response = "This is a mock response from the LLM client."
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log the interaction if a logger is available
        llm_logger = kwargs.get("llm_logger")
        if llm_logger and hasattr(llm_logger, 'log_interaction'):
            # Mock token counts
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response.split())
            
            llm_logger.log_interaction(
                model=self.model,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_tokens,
                response_tokens=completion_tokens,
                latency=latency,
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "mock": True,
                    **{k: v for k, v in kwargs.items() if k != "llm_logger"}
                }
            )
        
        return response


# Global LLM client instance
_llm_client = None


def create_llm_client(
    provider: str = "mock",
    model_name: str = "mock-model",
    api_key: Optional[str] = None,
    api_token: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create an LLM client based on the specified provider.
    
    Args:
        provider: LLM provider (mock, openai, replicate)
        model_name: Name of the model to use
        api_key: API key for the provider (for OpenAI)
        api_token: API token for the provider (for Replicate)
        **kwargs: Additional parameters for the client
        
    Returns:
        LLM client instance
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == "mock":
        return MockLLMClient(**kwargs)
    
    elif provider == "openai":
        if OpenAIClient is None:
            raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
        return OpenAIClient(api_key=api_key, model=model_name, **kwargs)
    
    elif provider == "replicate":
        if ReplicateClient is None:
            raise ImportError("Replicate package not installed. Install with 'pip install replicate'")
        return ReplicateClient(api_token=api_token, model=model_name, **kwargs)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_client(**kwargs) -> Any:
    """
    Get or create an LLM client.
    
    Args:
        **kwargs: Configuration parameters for the LLM client
        
    Returns:
        LLM client instance
    """
    global _llm_client
    
    if _llm_client is None:
        # Get provider from kwargs or use mock as default
        provider = kwargs.pop("provider", "mock")
        model_name = kwargs.pop("model_name", "mock-model")
        api_key = kwargs.pop("api_key", None)
        api_token = kwargs.pop("api_token", None)
        
        _llm_client = create_llm_client(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            api_token=api_token,
            **kwargs
        )
    
    return _llm_client


def set_llm_client(client: Any) -> None:
    """
    Set the global LLM client.
    
    Args:
        client: LLM client instance
    """
    global _llm_client
    _llm_client = client
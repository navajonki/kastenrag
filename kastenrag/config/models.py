"""Pydantic models for system configuration."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class TranscriberConfig(BaseModel):
    """Configuration for audio transcription."""
    type: str = Field(..., description="Type of transcriber to use")
    model_name: Optional[str] = Field(None, description="Model name for transcriber")
    api_key: Optional[str] = Field(None, description="API key for cloud services")
    language: str = Field("en", description="Language code for transcription")
    device: str = Field("cpu", description="Device to use for local inference")
    
    @validator('type')
    def valid_transcriber_type(cls, v):
        valid_types = ["whisper_local", "whisper_api", "deepgram", "custom", "mock"]
        if v not in valid_types:
            raise ValueError(f"Transcriber type must be one of {valid_types}")
        return v


class ChunkerConfig(BaseModel):
    """Configuration for text chunking."""
    type: str = Field("sliding_window", description="Type of chunker to use")
    window_size: int = Field(1000, description="Token window size for chunking")
    overlap: int = Field(100, description="Token overlap between chunks")
    boundary_rules: List[str] = Field(
        ["sentence", "paragraph"], 
        description="Natural boundary rules to respect"
    )
    # Prompt template configuration
    first_pass_template_name: Optional[str] = Field(
        None, 
        description="Name of the template to use for first pass extraction"
    )
    second_pass_template_name: Optional[str] = Field(
        None, 
        description="Name of the template to use for second pass refinement"
    )
    # Legacy prompt templates (deprecated)
    first_pass_prompt_template: Optional[str] = Field(
        None, 
        description="(Deprecated) Direct prompt template for first pass extraction"
    )
    second_pass_prompt_template: Optional[str] = Field(
        None, 
        description="(Deprecated) Direct prompt template for second pass refinement"
    )
    # Metadata enrichment flags
    entity_resolution: bool = Field(True, description="Whether to resolve entity references")
    topic_tagging: bool = Field(True, description="Whether to tag topics in chunks")
    relationship_extraction: bool = Field(True, description="Whether to extract entity relationships")
    
    @validator('type')
    def valid_chunker_type(cls, v):
        valid_types = ["sliding_window", "atomic"]
        if v not in valid_types:
            raise ValueError(f"Chunker type must be one of {valid_types}")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model_name: str = Field("text-embedding-ada-002", description="Embedding model to use")
    dimensions: int = Field(1536, description="Embedding dimensions")
    batch_size: int = Field(8, description="Batch size for embedding generation")


class VectorStoreConfig(BaseModel):
    """Configuration for vector database storage."""
    type: str = Field("chroma", description="Vector store type")
    persist_directory: str = Field("./data/vector_store", description="Persistence directory")
    collection_name: str = Field("kastenrag", description="Collection name")


class GraphStoreConfig(BaseModel):
    """Configuration for graph database storage."""
    type: str = Field("neo4j", description="Graph store type")
    uri: str = Field("bolt://localhost:7687", description="Database URI")
    username: str = Field("neo4j", description="Database username")
    password: str = Field("password", description="Database password")
    database: str = Field("neo4j", description="Database name")


class LLMConfig(BaseModel):
    """Configuration for language model interaction."""
    provider: str = Field("openai", description="LLM provider")
    model_name: str = Field("gpt-4", description="Model name")
    temperature: float = Field(0.0, description="Temperature for generation")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    streaming: bool = Field(False, description="Whether to stream responses")
    api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    level: str = Field("INFO", description="Logging level")
    log_dir: str = Field("./logs", description="Directory for log files")
    log_llm_interactions: bool = Field(True, description="Whether to log LLM interactions")
    log_performance: bool = Field(True, description="Whether to log performance metrics")


class SystemConfig(BaseModel):
    """Complete system configuration."""
    transcriber: TranscriberConfig
    chunker: ChunkerConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    graph_store: GraphStoreConfig
    llm: LLMConfig
    logging: LoggingConfig
    output_dir: str = Field("./output", description="Directory for output files")
    run_id: Optional[str] = None
"""Pipeline orchestration for processing audio recordings."""

from typing import Any, Dict, List, Optional, Type

from ..config.models import SystemConfig
from ..utils.logging import performance_timer
from ..utils.registry import registry


class PipelineContext:
    """Context object for sharing data between pipeline steps."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data: Dict[str, Any] = {}
        
    def set(self, key: str, value: Any):
        """Set a value in the context."""
        self.data[key] = value
        
    def get(self, key: str) -> Any:
        """Get a value from the context."""
        return self.data.get(key)


class PipelineStep:
    """Base class for pipeline steps."""
    
    def __init__(self, context: PipelineContext):
        self.context = context
    
    def execute(self) -> None:
        """Execute this pipeline step."""
        raise NotImplementedError("Subclasses must implement execute()")


class Pipeline:
    """Orchestrator for the processing pipeline."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.context = PipelineContext(config)
        self.steps: List[PipelineStep] = []
        
    def add_step(self, step: Type[PipelineStep]) -> 'Pipeline':
        """Add a step to the pipeline."""
        self.steps.append(step(self.context))
        return self
    
    @performance_timer("pipeline", "execute_all")
    def execute_all(self) -> PipelineContext:
        """Execute all steps in the pipeline."""
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            print(f"Executing step {i+1}/{len(self.steps)}: {step_name}")
            step.execute()
        return self.context


# Example pipeline steps (to be expanded in future phases)

class TranscriptionStep(PipelineStep):
    """Pipeline step for audio transcription."""
    
    @performance_timer("pipeline", "transcription")
    def execute(self) -> None:
        transcriber_config = self.context.config.transcriber
        
        transcriber = registry.create(
            component_type="transcriber",
            implementation_name=transcriber_config.type,
            **transcriber_config.dict(exclude={"type"})
        )
        
        audio_path = self.context.get("audio_path")
        if not audio_path:
            raise ValueError("No audio path provided in context")
        
        result = transcriber.transcribe(audio_path)
        self.context.set("transcript", result)


class ChunkingStep(PipelineStep):
    """Pipeline step for text chunking."""
    
    @performance_timer("pipeline", "chunking")
    def execute(self) -> None:
        chunker_config = self.context.config.chunker
        
        chunker = registry.create(
            component_type="chunker",
            implementation_name=chunker_config.type,
            **chunker_config.dict(exclude={"type"})
        )
        
        # Give the chunker access to the pipeline context
        chunker.context = self.context
        
        # Get transcript from context (could be from transcription or direct input)
        transcript = self.context.get("transcript")
        if not transcript:
            # Check if there's direct text input instead
            text_input = self.context.get("text_input")
            if not text_input:
                raise ValueError("No transcript or text input provided in context")
            text_to_chunk = text_input
        else:
            text_to_chunk = transcript["text"]
        
        chunks = chunker.chunk(text_to_chunk)
        self.context.set("chunks", chunks)


class ValidationStep(PipelineStep):
    """Pipeline step for validating chunk quality."""
    
    @performance_timer("pipeline", "validation")
    def execute(self) -> None:
        from ..validators.chunk_validator import ChunkValidator
        
        chunks = self.context.get("chunks")
        if not chunks:
            raise ValueError("No chunks provided in context")
        
        validator = ChunkValidator()
        validated_chunks, metrics = validator.validate_chunk_batch(chunks)
        
        self.context.set("validated_chunks", validated_chunks)
        self.context.set("quality_metrics", metrics)
        
        # Log the quality metrics
        print(f"Chunk Quality Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


# More pipeline steps to be implemented in future phases...


def create_pipeline(config: SystemConfig) -> Pipeline:
    """Create a pipeline with all necessary steps."""
    pipeline = Pipeline(config)
    
    # Configure pipeline based on input type
    if config.transcriber.type != "mock":
        # Full audio processing pipeline
        pipeline.add_step(TranscriptionStep)
    
    # Always include chunking and validation
    pipeline.add_step(ChunkingStep)
    pipeline.add_step(ValidationStep)
    
    return pipeline
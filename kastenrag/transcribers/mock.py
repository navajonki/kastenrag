"""Mock transcriber implementation for testing."""

from pathlib import Path
from typing import Dict, List, Optional

from ..utils.registry import register_component


class MockTranscriber:
    """
    Mock transcriber for testing purposes.
    
    This transcriber doesn't actually process audio files but returns
    predefined transcripts for testing the pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "mock",
        language: str = "en",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize mock transcriber.
        
        Args:
            model_name: Mock model name
            language: Language code
            device: Device to use (cpu/gpu)
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        self.kwargs = kwargs
    
    def transcribe(self, audio_path: str) -> Dict:
        """
        Return a mock transcription for the given audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Mock transcription result
        """
        # Get the file name without extension
        file_name = Path(audio_path).stem
        
        # Return a different transcription based on the file name
        if "synthetic" in file_name:
            return {
                "text": "This is a synthetic audio sample created for testing purposes.",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 2.5,
                        "text": "This is a synthetic audio sample",
                        "speaker": "speaker_0"
                    },
                    {
                        "id": 1,
                        "start": 2.5,
                        "end": 5.0,
                        "text": "created for testing purposes.",
                        "speaker": "speaker_0"
                    }
                ],
                "speakers": [
                    {
                        "id": "speaker_0",
                        "name": "Speaker 1"
                    }
                ]
            }
        else:
            return {
                "text": "This is a mock transcription for testing the pipeline. It contains some example text that can be processed by the chunkers.",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 3.0,
                        "text": "This is a mock transcription for testing the pipeline.",
                        "speaker": "speaker_0"
                    },
                    {
                        "id": 1,
                        "start": 3.0,
                        "end": 6.0,
                        "text": "It contains some example text that can be processed by the chunkers.",
                        "speaker": "speaker_0"
                    }
                ],
                "speakers": [
                    {
                        "id": "speaker_0",
                        "name": "Speaker 1"
                    }
                ]
            }


@register_component("transcriber", "mock")
def create_mock_transcriber(**kwargs):
    """Factory function for creating a mock transcriber."""
    return MockTranscriber(**kwargs)
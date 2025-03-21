#!/usr/bin/env python3
"""Script to prepare test data for KastenRAG development."""

import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from pydub import AudioSegment

# Define paths
DATA_DIR = Path("./data")
SAMPLE_AUDIO_DIR = DATA_DIR / "sample_audio"
SAMPLE_TRANSCRIPT_DIR = DATA_DIR / "sample_transcripts"

# Create directories
SAMPLE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# URLs for sample audio files
SAMPLE_AUDIO_URLS = [
    "https://example.com/sample1.mp3",  # Replace with actual URLs
    "https://example.com/sample2.wav",
]

# Download sample audio files
for url in SAMPLE_AUDIO_URLS:
    filename = os.path.basename(url)
    output_path = SAMPLE_AUDIO_DIR / filename
    
    if not output_path.exists():
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded to {output_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"File {output_path} already exists, skipping download")


# If no sample audio files are available online, generate synthetic audio
def generate_synthetic_audio(duration_ms=5000, output_path="synthetic_sample.wav"):
    """Generate a synthetic audio file with a sine wave."""
    sample_rate = 44100
    t = np.linspace(0, duration_ms/1000, int(sample_rate * duration_ms/1000), endpoint=False)
    
    # Generate a sine wave at 440 Hz (A4 note)
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1  # mono
    )
    
    # Export to file
    audio_segment.export(output_path, format="wav")
    return output_path


# Generate synthetic audio if no samples were downloaded
if not any(SAMPLE_AUDIO_DIR.iterdir()):
    print("No sample audio files downloaded, generating synthetic audio...")
    for i in range(3):
        output_path = SAMPLE_AUDIO_DIR / f"synthetic_sample_{i+1}.wav"
        generate_synthetic_audio(duration_ms=5000, output_path=output_path)
        print(f"Generated synthetic audio: {output_path}")


# Create sample transcripts of varying complexity
SAMPLE_TRANSCRIPTS = [
    # Simple transcript with a single speaker
    {
        "filename": "simple_transcript.txt",
        "content": """
This is a simple transcript with a single speaker. It contains some basic information about the voice note taking system.
The system is designed to process audio recordings into atomic self-contained chunks of information.
Each chunk is enriched with metadata, including entity and topic information, and stored in vector and graph databases for efficient retrieval.
        """.strip()
    },
    
    # Complex transcript with multiple speakers
    {
        "filename": "multi_speaker_transcript.txt",
        "content": """
Speaker 1: Hi everyone, welcome to our meeting about the new voice note taking system.
Speaker 2: Thanks for having me. I'm excited to discuss the technical implementation.
Speaker 1: Great! Let's start with the transcription component. How does it work?
Speaker 2: The system uses Whisper for transcription, with support for both local and API-based processing.
Speaker 1: And what about speaker diarization?
Speaker 2: Yes, we support that too. It helps identify who said what in multi-person recordings.
Speaker 1: That sounds useful. What about the chunking strategy?
Speaker 2: We use a two-pass approach. First, we split by sliding window, then we refine into atomic facts.
Speaker 1: Can you explain what you mean by "atomic facts"?
Speaker 2: Sure! An atomic fact is a self-contained piece of information that makes sense on its own, without needing additional context.
        """.strip()
    },
    
    # Technical transcript with specialized terminology
    {
        "filename": "technical_transcript.txt",
        "content": """
The RAG system architecture consists of several key components. First, the audio transcription module processes input files using state-of-the-art ASR techniques.
The resulting transcript then undergoes semantic segmentation, where we identify natural boundaries in the content.
Next, the LLM-powered atomic fact extraction identifies self-contained units of information, replacing pronoun references with full entity names.
Each atomic fact is enriched with metadata, including named entities, topic classifications, and relationship identifiers.
The enriched chunks are then stored in both vector and graph databases to support different retrieval strategies.
For retrieval, we implement a hybrid approach that combines vector similarity search with graph traversal, providing comprehensive coverage of the knowledge base.
The system uses a Chain-of-Thought approach for complex queries, breaking them down into sub-questions that can be answered more precisely.
        """.strip()
    },
    
    # Narrative transcript with storytelling elements
    {
        "filename": "narrative_transcript.txt",
        "content": """
When I first started working on this project, I didn't realize how complex it would become. The initial idea was simple: create a system that could take voice notes and make them searchable. But as we explored the requirements, we realized we needed something much more sophisticated.

John suggested we look into LLM-based chunking, which seemed promising. He had worked on a similar project at his previous company and had some insights about the challenges we might face.

We spent about two weeks experimenting with different approaches. Sarah's background in computational linguistics proved invaluable during this phase. She pointed out that we needed to maintain context across chunks, which led us to develop the entity resolution system.

The breakthrough came when we tried the two-pass extraction method. The first pass identified potential atomic facts, and the second pass refined them and resolved entity references. This dramatically improved the quality of our results.

We presented our findings to the management team last Thursday, and they were impressed enough to increase our budget. Now we're planning to extend the system with multi-modal capabilities in the next quarter.
        """.strip()
    }
]

# Write sample transcripts to files
for sample in SAMPLE_TRANSCRIPTS:
    output_path = SAMPLE_TRANSCRIPT_DIR / sample["filename"]
    with open(output_path, "w") as f:
        f.write(sample["content"])
    print(f"Created sample transcript: {output_path}")

print("Test data preparation complete!")
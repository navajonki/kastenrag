"""Chunking components for KastenRAG."""

# Import chunker implementations to register them
from .basic import SlidingWindowChunker
from .atomic import AtomicChunker

"""Basic chunking implementations."""

import re
from typing import Dict, List, Optional

from ..utils.registry import register_component


@register_component("chunker", "sliding_window")
class SlidingWindowChunker:
    """
    Basic sliding window text chunker.
    
    This chunker splits text into overlapping chunks of a specified size.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        overlap: int = 100,
        boundary_rules: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize sliding window chunker.
        
        Args:
            window_size: Number of characters in each chunk
            overlap: Number of characters to overlap between chunks
            boundary_rules: Rules for respecting natural boundaries
            **kwargs: Additional arguments
        """
        self.window_size = window_size
        self.overlap = overlap
        self.boundary_rules = boundary_rules or ["sentence"]
        self.kwargs = kwargs
    
    def chunk(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries
        """
        # If text is smaller than window size, return as a single chunk
        if len(text) <= self.window_size:
            return [{
                "text": text,
                "metadata": {
                    "chunk_id": "chunk-0",
                    "start_char": 0,
                    "end_char": len(text),
                    "word_count": len(text.split())
                }
            }]
        
        chunks = []
        
        # Simple implementation - just split by window size with overlap
        start = 0
        while start < len(text):
            # Get chunk with window_size or remaining text
            end = min(start + self.window_size, len(text))
            
            # If we're not at the end of the text and we want to respect sentence boundaries
            if end < len(text) and "sentence" in self.boundary_rules:
                # Find the last sentence boundary in the current window
                # Look for ., !, ? followed by a space or newline
                sentence_boundaries = [
                    m.end() for m in re.finditer(r'[.!?][\s\n]', text[start:end])
                ]
                
                if sentence_boundaries:
                    # Use the last sentence boundary we found
                    end = start + sentence_boundaries[-1]
            
            # Extract the chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "start_char": start,
                        "end_char": end,
                        "chunk_id": f"chunk-{len(chunks)}",
                        "word_count": len(chunk_text.split())
                    }
                })
            
            # Move start position for next chunk (with overlap)
            start = max(end - self.overlap, start + 1)  # Ensure we make progress
        
        return chunks


# Legacy factory function - class is now registered directly with decorator
def create_sliding_window_chunker(**kwargs):
    """Factory function for creating a sliding window chunker."""
    return SlidingWindowChunker(**kwargs)
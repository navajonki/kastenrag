"""Atomic chunking implementations for KastenRAG."""

import re
from typing import Dict, List, Optional, Any

from ..utils.registry import register_component
from ..llm import get_llm_client


class AtomicChunker:
    """
    Atomic chunker that uses LLM to extract self-contained facts.
    
    This chunker processes text in two passes:
    1. First pass: Split text into segments and extract candidate atomic facts
    2. Second pass: Refine facts to ensure they are self-contained
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        overlap: int = 100,
        boundary_rules: Optional[List[str]] = None,
        first_pass_prompt_template: Optional[str] = None,
        second_pass_prompt_template: Optional[str] = None,
        entity_resolution: bool = True,
        topic_tagging: bool = True,
        relationship_extraction: bool = True,
        **kwargs
    ):
        """
        Initialize the atomic chunker.
        
        Args:
            window_size: Number of characters in initial sliding window
            overlap: Number of characters to overlap between windows
            boundary_rules: Rules for respecting natural boundaries
            first_pass_prompt_template: Prompt template for initial fact extraction
            second_pass_prompt_template: Prompt template for ensuring facts are self-contained
            entity_resolution: Whether to resolve entity references
            topic_tagging: Whether to add topic tags to chunks
            relationship_extraction: Whether to extract entity relationships
            **kwargs: Additional arguments
        """
        self.window_size = window_size
        self.overlap = overlap
        self.boundary_rules = boundary_rules or ["sentence", "paragraph"]
        
        # Set up prompt templates
        self.first_pass_prompt_template = first_pass_prompt_template or self._default_first_pass_prompt()
        self.second_pass_prompt_template = second_pass_prompt_template or self._default_second_pass_prompt()
        
        # Metadata enrichment flags
        self.entity_resolution = entity_resolution
        self.topic_tagging = topic_tagging
        self.relationship_extraction = relationship_extraction
        
        # Other configuration
        self.kwargs = kwargs
        self.llm_client = get_llm_client()
    
    def _default_first_pass_prompt(self) -> str:
        """Default prompt template for first pass extraction."""
        return """
        You are an expert at extracting atomic facts from text. An atomic fact is a single, self-contained 
        piece of information that can stand on its own without requiring additional context.
        
        Extract atomic facts from the following text. For each fact:
        1. Make it stand alone (a reader should understand it with no other context)
        2. Include full entity names and remove pronouns (replace "he", "she", "it", "they" with the actual entity)
        3. Express exactly one idea per fact
        4. Maintain factual accuracy
        
        Text: {text}
        
        Extract atomic facts in this format:
        FACT 1: [The atomic fact with all pronouns resolved]
        FACT 2: [Another atomic fact with all pronouns resolved]
        ...
        """
    
    def _default_second_pass_prompt(self) -> str:
        """Default prompt template for second pass refinement."""
        return """
        You are an expert at ensuring facts are truly atomic and self-contained.
        
        Review these extracted facts and improve them to be completely self-contained.
        For each fact:
        1. Verify it expresses exactly ONE idea
        2. Ensure ALL pronouns are replaced with explicit entities
        3. Add any missing context that's needed to understand the fact
        4. If a fact contains multiple ideas, split it into multiple facts
        5. Preserve the original meaning while making the fact stand alone
        
        Original facts:
        {facts}
        
        Improved atomic facts:
        """
    
    def _extract_initial_facts(self, text_segment: str) -> List[str]:
        """
        Extract initial atomic facts from a text segment.
        
        Args:
            text_segment: A segment of text to process
            
        Returns:
            List of extracted atomic facts
        """
        prompt = self.first_pass_prompt_template.format(text=text_segment)
        
        # Get the loggers from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to extract facts
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        # Parse the response to extract facts
        facts = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("FACT ") and ":" in line:
                # Extract the fact part after the colon
                fact_text = line.split(":", 1)[1].strip()
                facts.append(fact_text)
        
        return facts
    
    def _refine_facts(self, facts: List[str]) -> List[str]:
        """
        Refine extracted facts to ensure they are truly atomic and self-contained.
        
        Args:
            facts: List of initial facts
            
        Returns:
            List of refined atomic facts
        """
        if not facts:
            return []
            
        facts_text = "\n".join(f"FACT {i+1}: {fact}" for i, fact in enumerate(facts))
        prompt = self.second_pass_prompt_template.format(facts=facts_text)
        
        # Get the loggers from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to refine facts
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        # Parse the response to extract refined facts
        refined_facts = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("FACT ") and ":" in line:
                # Extract the fact part after the colon
                fact_text = line.split(":", 1)[1].strip()
                refined_facts.append(fact_text)
        
        return refined_facts
    
    def _extract_entities(self, fact: str) -> List[str]:
        """
        Extract named entities from a fact.
        
        Args:
            fact: The fact text
            
        Returns:
            List of named entities
        """
        if not self.entity_resolution:
            return []
            
        # Simple entity extraction - this would be enhanced with a proper NER model
        # For now, extract multi-word capitalized phrases and single capitalized words
        
        # First, try to extract multi-word entities (e.g., "Artificial Intelligence")
        multi_word_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z]?[a-zA-Z]*)+\b'
        multi_word_entities = re.findall(multi_word_pattern, fact)
        
        # Then extract single-word entities
        single_word_pattern = r'\b[A-Z][a-zA-Z]{1,}\b'
        single_word_entities = re.findall(single_word_pattern, fact)
        
        # Combine and remove duplicates
        all_entities = multi_word_entities + single_word_entities
        
        # Clean entities by removing punctuation
        cleaned_entities = [entity.strip(',.;:()[]{}') for entity in all_entities]
        
        # Filter out common words that might be capitalized at the start of sentences
        common_words = ["The", "A", "An", "This", "That", "These", "Those", "It", "They"]
        filtered_entities = [entity for entity in cleaned_entities if entity not in common_words]
        
        return list(set(filtered_entities))
    
    def _assign_topics(self, fact: str) -> List[str]:
        """
        Assign topic tags to a fact.
        
        Args:
            fact: The fact text
            
        Returns:
            List of topic tags
        """
        if not self.topic_tagging:
            return []
            
        # This would be enhanced with a proper topic classification model
        # For now, use simple keyword matching
        topics = []
        keywords = {
            "AI": ["artificial intelligence", "AI", "machine intelligence"],
            "Machine Learning": ["machine learning", "ML", "deep learning", "neural network"],
            "NLP": ["natural language processing", "NLP", "language model", "text analysis"],
            "Data Science": ["data science", "statistics", "data analysis", "big data"],
            "Ethics": ["ethics", "ethical", "bias", "privacy", "responsible"]
        }
        
        fact_lower = fact.lower()
        for topic, words in keywords.items():
            if any(kw.lower() in fact_lower for kw in words):
                topics.append(topic)
        
        return topics
    
    def _extract_relationships(self, fact: str, entities: List[str]) -> List[Dict]:
        """
        Extract relationships between entities.
        
        Args:
            fact: The fact text
            entities: List of entities in the fact
            
        Returns:
            List of relationships
        """
        if not self.relationship_extraction or len(entities) < 2:
            return []
            
        # Simple relationship extraction - would be enhanced with proper relation extraction
        # For now, just connect entities that appear in the same fact
        relationships = []
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationships.append({
                    "source": entity1,
                    "target": entity2,
                    "type": "co-occurrence",
                    "fact": fact
                })
        
        return relationships
    
    def _create_chunk_metadata(self, fact: str, index: int) -> Dict[str, Any]:
        """
        Create metadata for a chunk.
        
        Args:
            fact: The atomic fact
            index: Chunk index
            
        Returns:
            Metadata dictionary
        """
        # Extract entities for the fact
        entities = self._extract_entities(fact)
        
        # Assign topics to the fact
        topics = self._assign_topics(fact)
        
        # Extract relationships between entities
        relationships = self._extract_relationships(fact, entities)
        
        return {
            "chunk_id": f"atomic_{index}",
            "entities": entities,
            "topics": topics,
            "relationships": relationships,
            "is_atomic": True,
            "word_count": len(fact.split())
        }
    
    def chunk(self, text: str) -> List[Dict]:
        """
        Process text to extract atomic fact chunks.
        
        Args:
            text: Input text to process
            
        Returns:
            List of chunk dictionaries with atomic facts and metadata
        """
        # Get logger if available
        logger = None
        if hasattr(self, 'context') and self.context:
            logger = self.context.get("llm_logger")
        
        # Step 1: Split text into overlapping segments
        print(f"Splitting text into segments (length: {len(text)}, window: {self.window_size}, overlap: {self.overlap})")
        segments = []
        start = 0
        while start < len(text):
            # Get segment with window_size or remaining text
            end = min(start + self.window_size, len(text))
            
            # Adjust boundaries based on boundary rules
            if end < len(text) and "paragraph" in self.boundary_rules:
                paragraph_boundaries = [
                    m.end() for m in re.finditer(r'\n\s*\n', text[start:end])
                ]
                if paragraph_boundaries:
                    end = start + paragraph_boundaries[-1]
            elif end < len(text) and "sentence" in self.boundary_rules:
                sentence_boundaries = [
                    m.end() for m in re.finditer(r'[.!?][\s\n]', text[start:end])
                ]
                if sentence_boundaries:
                    end = start + sentence_boundaries[-1]
            
            segment_text = text[start:end].strip()
            if segment_text:
                segments.append(segment_text)
            
            # Move start position for next segment (with overlap)
            start = max(end - self.overlap, start + 1)  # Ensure we make progress
        
        print(f"Created {len(segments)} text segments")
        
        # Step 2: Extract initial facts from each segment
        print("Extracting initial facts from segments...")
        all_initial_facts = []
        for i, segment in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)} ({len(segment)} chars)")
            segment_facts = self._extract_initial_facts(segment)
            all_initial_facts.extend(segment_facts)
            print(f"  Extracted {len(segment_facts)} facts from segment {i+1}")
        
        print(f"Total initial facts extracted: {len(all_initial_facts)}")
        
        # Step 3: Refine facts to ensure they are atomic and self-contained
        print("Refining facts to ensure they are self-contained...")
        refined_facts = self._refine_facts(all_initial_facts)
        print(f"Refined facts: {len(refined_facts)}")
        
        # Step 4: Create chunks with metadata
        print("Creating chunks with metadata...")
        chunks = []
        for i, fact in enumerate(refined_facts):
            chunk = {
                "text": fact,
                "metadata": self._create_chunk_metadata(fact, i)
            }
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} atomic chunks with metadata")
        return chunks


@register_component("chunker", "atomic")
def create_atomic_chunker(**kwargs):
    """Factory function for creating an atomic chunker."""
    return AtomicChunker(**kwargs)
"""Atomic chunking implementations for KastenRAG."""

import re
from typing import Dict, List, Optional, Any

from ..utils.registry import register_component
from ..llm import get_llm_client
from ..prompts import get_prompt_template, PromptTemplate


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
        first_pass_template_name: Optional[str] = None,
        second_pass_template_name: Optional[str] = None,
        first_pass_prompt_template: Optional[str] = None,  # Kept for backward compatibility
        second_pass_prompt_template: Optional[str] = None,  # Kept for backward compatibility
        entity_resolution: bool = True,
        topic_tagging: bool = True,
        relationship_extraction: bool = True,
        entity_template_name: Optional[str] = None,
        topic_template_name: Optional[str] = None,
        relationship_template_name: Optional[str] = None,
        use_llm_for_metadata: bool = False,  # Legacy flag for backward compatibility
        use_llm_for_entities: bool = False,
        use_llm_for_topics: bool = False, 
        use_llm_for_relationships: bool = False,
        **kwargs
    ):
        """
        Initialize the atomic chunker.
        
        Args:
            window_size: Number of characters in initial sliding window
            overlap: Number of characters to overlap between windows
            boundary_rules: Rules for respecting natural boundaries
            first_pass_template_name: Name of the template to use for first pass extraction
            second_pass_template_name: Name of the template to use for second pass refinement
            first_pass_prompt_template: (Deprecated) Prompt template string for initial fact extraction
            second_pass_prompt_template: (Deprecated) Prompt template string for ensuring facts are self-contained
            entity_resolution: Whether to resolve entity references
            topic_tagging: Whether to add topic tags to chunks
            relationship_extraction: Whether to extract entity relationships
            entity_template_name: Name of the template to use for entity extraction (if using LLM)
            topic_template_name: Name of the template to use for topic tagging (if using LLM)
            relationship_template_name: Name of the template to use for relationship extraction (if using LLM)
            use_llm_for_metadata: Whether to use LLM for metadata generation (instead of rule-based approach)
            **kwargs: Additional arguments
        """
        self.window_size = window_size
        self.overlap = overlap
        self.boundary_rules = boundary_rules or ["sentence", "paragraph"]
        
        # Set up prompt templates
        self.first_pass_template_name = first_pass_template_name
        self.second_pass_template_name = second_pass_template_name
        
        # For backward compatibility, store any directly provided template strings
        self._first_pass_prompt_template_str = first_pass_prompt_template
        self._second_pass_prompt_template_str = second_pass_prompt_template
        
        # Actual template objects - will be loaded on first use
        self._first_pass_template = None
        self._second_pass_template = None
        
        # Metadata enrichment flags
        self.entity_resolution = entity_resolution
        self.topic_tagging = topic_tagging
        self.relationship_extraction = relationship_extraction
        
        # Metadata template names
        self.entity_template_name = entity_template_name or "llm_entities"
        self.topic_template_name = topic_template_name or "llm_topics"
        self.relationship_template_name = relationship_template_name or "llm_relationships"
        
        # Metadata template objects - will be loaded on first use
        self._entity_template = None
        self._topic_template = None
        self._relationship_template = None
        
        # Handle metadata LLM flags with backward compatibility
        # If the legacy flag is set, use it for all three types
        if use_llm_for_metadata:
            self.use_llm_for_entities = True
            self.use_llm_for_topics = True
            self.use_llm_for_relationships = True
        else:
            # Otherwise use the individual flags
            self.use_llm_for_entities = use_llm_for_entities
            self.use_llm_for_topics = use_llm_for_topics
            self.use_llm_for_relationships = use_llm_for_relationships
        
        # Keep the legacy flag for backward compatibility
        self.use_llm_for_metadata = (
            self.use_llm_for_entities and 
            self.use_llm_for_topics and 
            self.use_llm_for_relationships
        )
        
        # Other configuration
        self.kwargs = kwargs
        self.llm_client = get_llm_client()
    
    def _get_first_pass_template(self) -> PromptTemplate:
        """Get the first pass prompt template."""
        if self._first_pass_template is None:
            try:
                # Try to get the template from the registry
                if self.first_pass_template_name:
                    self._first_pass_template = get_prompt_template(
                        "chunker", "first_pass", self.first_pass_template_name
                    )
                else:
                    # If no specific template name was provided, use the default
                    self._first_pass_template = get_prompt_template("chunker", "first_pass")
            except KeyError:
                # If the template is not found in the registry, use the provided string
                # or create a default template
                if self._first_pass_prompt_template_str:
                    template_str = self._first_pass_prompt_template_str
                else:
                    template_str = self._default_first_pass_prompt()
                
                # Create a template object (not registered, just for local use)
                self._first_pass_template = PromptTemplate(
                    name="local_first_pass",
                    description="Local first pass template",
                    version="1.0.0",
                    component_type="chunker",
                    task="first_pass",
                    template=template_str
                )
        
        return self._first_pass_template
    
    def _get_second_pass_template(self) -> PromptTemplate:
        """Get the second pass prompt template."""
        if self._second_pass_template is None:
            try:
                # Try to get the template from the registry
                if self.second_pass_template_name:
                    self._second_pass_template = get_prompt_template(
                        "chunker", "second_pass", self.second_pass_template_name
                    )
                else:
                    # If no specific template name was provided, use the default
                    self._second_pass_template = get_prompt_template("chunker", "second_pass")
            except KeyError:
                # If the template is not found in the registry, use the provided string
                # or create a default template
                if self._second_pass_prompt_template_str:
                    template_str = self._second_pass_prompt_template_str
                else:
                    template_str = self._default_second_pass_prompt()
                
                # Create a template object (not registered, just for local use)
                self._second_pass_template = PromptTemplate(
                    name="local_second_pass",
                    description="Local second pass template",
                    version="1.0.0",
                    component_type="chunker",
                    task="second_pass",
                    template=template_str
                )
        
        return self._second_pass_template
        
    def _get_entity_template(self) -> PromptTemplate:
        """Get the entity extraction prompt template."""
        if self._entity_template is None:
            try:
                # Try to get the template from the registry
                if self.entity_template_name:
                    self._entity_template = get_prompt_template(
                        "chunker", "metadata_entity", self.entity_template_name
                    )
                else:
                    # If no specific template name was provided, use the default
                    self._entity_template = get_prompt_template("chunker", "metadata_entity")
            except KeyError:
                # If the template is not found in the registry, create a default template
                self._entity_template = PromptTemplate(
                    name="default_entity_extraction",
                    description="Default entity extraction template",
                    version="1.0.0",
                    component_type="chunker",
                    task="metadata_entity",
                    template="""
                    Extract all named entities from the following fact. A named entity is a real-world object, such as a person, location, organization, product, event, date, or other proper noun.
                    
                    For each entity:
                    1. Identify the full name/phrase of the entity
                    2. Categorize it as one of: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT, or OTHER
                    3. Include compound entities (e.g., "John Smith" as one entity, not "John" and "Smith" separately)
                    4. Do not include common nouns - only named entities and proper nouns
                    
                    Fact: {fact}
                    
                    Return ONLY a JSON array of extracted entities with their categories in this exact format:
                    [
                      {"entity": "entity name", "category": "CATEGORY"},
                      {"entity": "another entity", "category": "CATEGORY"}
                    ]
                    """
                )
        
        return self._entity_template
    
    def _get_topic_template(self) -> PromptTemplate:
        """Get the topic tagging prompt template."""
        if self._topic_template is None:
            try:
                # Try to get the template from the registry
                if self.topic_template_name:
                    self._topic_template = get_prompt_template(
                        "chunker", "metadata_topic", self.topic_template_name
                    )
                else:
                    # If no specific template name was provided, use the default
                    self._topic_template = get_prompt_template("chunker", "metadata_topic")
            except KeyError:
                # If the template is not found in the registry, create a default template
                self._topic_template = PromptTemplate(
                    name="default_topic_tagging",
                    description="Default topic tagging template",
                    version="1.0.0",
                    component_type="chunker",
                    task="metadata_topic",
                    template="""
                    Identify the most relevant topics in the following fact. Topics should be general fields or subjects that the fact relates to.
                    
                    Some example topics: Politics, Economics, Technology, Healthcare, Environment, Education, Science, History, Culture, Sports, Entertainment.
                    
                    Select 1-3 most relevant topics that accurately categorize this fact. Be precise and specific where possible.
                    
                    Fact: {fact}
                    
                    Return ONLY a JSON array of topics in this exact format:
                    ["Topic1", "Topic2", "Topic3"]
                    """
                )
        
        return self._topic_template
    
    def _get_relationship_template(self) -> PromptTemplate:
        """Get the relationship extraction prompt template."""
        if self._relationship_template is None:
            try:
                # Try to get the template from the registry
                if self.relationship_template_name:
                    self._relationship_template = get_prompt_template(
                        "chunker", "metadata_relationship", self.relationship_template_name
                    )
                else:
                    # If no specific template name was provided, use the default
                    self._relationship_template = get_prompt_template("chunker", "metadata_relationship")
            except KeyError:
                # If the template is not found in the registry, create a default template
                self._relationship_template = PromptTemplate(
                    name="default_relationship_extraction",
                    description="Default relationship extraction template",
                    version="1.0.0",
                    component_type="chunker",
                    task="metadata_relationship",
                    template="""
                    Identify relationships between entities in the following fact.
                    
                    First, identify the named entities in the fact. Then, determine if there are any meaningful relationships between these entities.
                    
                    Fact: {fact}
                    
                    Return ONLY a JSON array of relationships in this exact format:
                    [
                      {
                        "source": "entity1", 
                        "target": "entity2", 
                        "type": "relationship type",
                        "description": "brief description of how they are related"
                      }
                    ]
                    
                    If there are no relationships or fewer than two entities, return an empty array: []
                    """
                )
        
        return self._relationship_template
    
    def _default_first_pass_prompt(self) -> str:
        """Default prompt template string for first pass extraction."""
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
        """Default prompt template string for second pass refinement."""
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
        
        Improved atomic facts (use the exact format below with FACT followed by a number and colon):
        FACT 1: [Improved atomic fact with all references resolved]
        FACT 2: [Another improved atomic fact with all references resolved]
        ...
        """
    
    def _extract_initial_facts(self, text_segment: str) -> List[str]:
        """
        Extract initial atomic facts from a text segment.
        
        Args:
            text_segment: A segment of text to process
            
        Returns:
            List of extracted atomic facts
        """
        # Get the chunker logger from the context if available
        chunker_logger = None
        if hasattr(self, 'context') and self.context:
            chunker_logger = self.context.get("logger")
            
        # Log the fact extraction start
        if chunker_logger:
            chunker_logger.log(
                step="first_pass",
                message="Starting first pass fact extraction",
                data={"segment_length": len(text_segment)}
            )
            
        # Get the template and format it with the text segment
        template = self._get_first_pass_template()
        prompt = template.format(text=text_segment)
        
        # Log the template being used
        if chunker_logger:
            chunker_logger.log(
                step="first_pass",
                message=f"Using template: {template.name}",
                data={"template_name": template.name}
            )
        
        # Get the LLM logger from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to extract facts
        if chunker_logger:
            chunker_logger.log(step="first_pass", message="Calling LLM for fact extraction")
            
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        # Parse the response to extract facts
        facts = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("FACT ") and ":" in line:
                # Extract the fact part after the colon
                fact_text = line.split(":", 1)[1].strip()
                facts.append(fact_text)
        
        # Log the extraction results
        if chunker_logger:
            chunker_logger.log(
                step="first_pass",
                message=f"Extracted {len(facts)} initial facts",
                data={"fact_count": len(facts), "facts": facts}
            )
            
        return facts
    
    def _refine_facts(self, facts: List[str]) -> List[str]:
        """
        Refine extracted facts to ensure they are truly atomic and self-contained.
        
        Args:
            facts: List of initial facts
            
        Returns:
            List of refined atomic facts
        """
        # Get the chunker logger from the context if available
        chunker_logger = None
        if hasattr(self, 'context') and self.context:
            chunker_logger = self.context.get("logger")
            
        # Log the refining process start
        if chunker_logger:
            chunker_logger.log(
                step="second_pass",
                message=f"Starting second pass fact refinement with {len(facts)} facts",
                data={"fact_count": len(facts)}
            )
        
        if not facts:
            if chunker_logger:
                chunker_logger.log(
                    step="second_pass",
                    message="No facts to refine, skipping second pass",
                    data={"warning": "Empty facts list"}
                )
            return []
            
        facts_text = "\n".join(f"FACT {i+1}: {fact}" for i, fact in enumerate(facts))
        
        # Get the template and format it with the facts
        template = self._get_second_pass_template()
        prompt = template.format(facts=facts_text)
        
        # Log the template being used
        if chunker_logger:
            chunker_logger.log(
                step="second_pass",
                message=f"Using template: {template.name}",
                data={"template_name": template.name, "input_fact_count": len(facts)}
            )
        
        # Get the LLM logger from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to refine facts
        if chunker_logger:
            chunker_logger.log(
                step="second_pass", 
                message="Calling LLM for fact refinement",
                data={"input_facts": facts}
            )
            
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        # Parse the response to extract refined facts
        refined_facts = []
        
        # Log the raw response for debugging
        if chunker_logger:
            chunker_logger.log(
                step="second_pass_response",
                message="Raw response from LLM",
                data={
                    "raw_response": response
                }
            )
        
        # First try the primary format (FACT #:)
        primary_format_found = False
        for line in response.splitlines():
            line = line.strip()
            if line and line.upper().startswith("FACT ") and ":" in line:
                primary_format_found = True
                # Extract the fact part after the colon
                fact_text = line.split(":", 1)[1].strip()
                if fact_text:  # Only add non-empty facts
                    refined_facts.append(fact_text)
        
        # If primary format didn't work, try fallback patterns
        if not primary_format_found:
            # Look for numbered facts (1. fact text)
            numbered_facts_pattern = r'^\s*(\d+)[\.\)]\s+(.+)$'
            for line in response.splitlines():
                line = line.strip()
                if line:
                    numbered_match = re.match(numbered_facts_pattern, line)
                    if numbered_match:
                        fact_text = numbered_match.group(2).strip()
                        if fact_text:
                            refined_facts.append(fact_text)
        
        # Log the refinement results
        if chunker_logger:
            chunker_logger.log(
                step="second_pass",
                message=f"Refined into {len(refined_facts)} facts",
                data={
                    "refined_fact_count": len(refined_facts),
                    "input_fact_count": len(facts),
                    "refined_facts": refined_facts,
                    "primary_format_found": primary_format_found
                }
            )
            
            # Check if the number of facts changed dramatically, indicating a potential issue
            if len(refined_facts) == 0 and len(facts) > 0:
                chunker_logger.log(
                    step="second_pass_warning",
                    message="Second pass produced no facts from non-empty input",
                    data={
                        "warning": "No output facts",
                        "input_facts": facts,
                        "raw_response": response,
                        "potential_cause": "Response format may not match expected format"
                    }
                )
        
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
            
        # Get the chunker logger from the context if available
        chunker_logger = None
        if hasattr(self, 'context') and self.context:
            chunker_logger = self.context.get("logger")
        
        # Use the LLM approach if specified for entities
        if self.use_llm_for_entities:
            if chunker_logger:
                chunker_logger.log(
                    step="metadata_entity", 
                    message=f"Using LLM for entity extraction"
                )
            
            try:
                return self._extract_entities_llm(fact)
            except Exception as e:
                if chunker_logger:
                    chunker_logger.log(
                        step="metadata_entity_error", 
                        message=f"Error using LLM for entity extraction: {str(e)}",
                        data={"error": str(e), "fact": fact}
                    )
                # Fall back to rule-based approach on error
                return self._extract_entities_rule_based(fact)
        else:
            # Use rule-based approach
            return self._extract_entities_rule_based(fact)
    
    def _extract_entities_rule_based(self, fact: str) -> List[str]:
        """
        Extract named entities from a fact using rule-based approach.
        
        Args:
            fact: The fact text
            
        Returns:
            List of named entities
        """
        # Simple entity extraction with regex patterns
        
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
    
    def _extract_entities_llm(self, fact: str) -> List[str]:
        """
        Extract named entities from a fact using LLM.
        
        Args:
            fact: The fact text
            
        Returns:
            List of named entities
        """
        # Get the entity template
        template = self._get_entity_template()
        prompt = template.format(fact=fact)
        
        # Get the LLM logger from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to extract entities
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        try:
            # Parse JSON response
            import json
            entities_data = json.loads(response)
            
            # Extract entity names
            entities = [item.get("entity") for item in entities_data if item.get("entity")]
            return entities
        except Exception as e:
            # If JSON parsing fails, fall back to returning an empty list
            return []
    
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
        
        # Get the chunker logger from the context if available
        chunker_logger = None
        if hasattr(self, 'context') and self.context:
            chunker_logger = self.context.get("logger")
        
        # Use the LLM approach if specified for topics
        if self.use_llm_for_topics:
            if chunker_logger:
                chunker_logger.log(
                    step="metadata_topic", 
                    message=f"Using LLM for topic tagging"
                )
            
            try:
                return self._assign_topics_llm(fact)
            except Exception as e:
                if chunker_logger:
                    chunker_logger.log(
                        step="metadata_topic_error", 
                        message=f"Error using LLM for topic tagging: {str(e)}",
                        data={"error": str(e), "fact": fact}
                    )
                # Fall back to rule-based approach on error
                return self._assign_topics_rule_based(fact)
        else:
            # Use rule-based approach
            return self._assign_topics_rule_based(fact)
    
    def _assign_topics_rule_based(self, fact: str) -> List[str]:
        """
        Assign topic tags to a fact using rule-based approach.
        
        Args:
            fact: The fact text
            
        Returns:
            List of topic tags
        """
        # Simple keyword matching
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
    
    def _assign_topics_llm(self, fact: str) -> List[str]:
        """
        Assign topic tags to a fact using LLM.
        
        Args:
            fact: The fact text
            
        Returns:
            List of topic tags
        """
        # Get the topic template
        template = self._get_topic_template()
        prompt = template.format(fact=fact)
        
        # Get the LLM logger from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to assign topics
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        try:
            # Parse JSON response
            import json
            topics = json.loads(response)
            return topics
        except Exception as e:
            # If JSON parsing fails, fall back to returning an empty list
            return []
    
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
        
        # Get the chunker logger from the context if available
        chunker_logger = None
        if hasattr(self, 'context') and self.context:
            chunker_logger = self.context.get("logger")
        
        # Use the LLM approach if specified for relationships
        if self.use_llm_for_relationships:
            if chunker_logger:
                chunker_logger.log(
                    step="metadata_relationship", 
                    message=f"Using LLM for relationship extraction"
                )
            
            try:
                return self._extract_relationships_llm(fact)
            except Exception as e:
                if chunker_logger:
                    chunker_logger.log(
                        step="metadata_relationship_error", 
                        message=f"Error using LLM for relationship extraction: {str(e)}",
                        data={"error": str(e), "fact": fact}
                    )
                # Fall back to rule-based approach on error
                return self._extract_relationships_rule_based(fact, entities)
        else:
            # Use rule-based approach
            return self._extract_relationships_rule_based(fact, entities)
    
    def _extract_relationships_rule_based(self, fact: str, entities: List[str]) -> List[Dict]:
        """
        Extract relationships between entities using rule-based approach.
        
        Args:
            fact: The fact text
            entities: List of entities in the fact
            
        Returns:
            List of relationships
        """
        # Simple co-occurrence based relationship extraction
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
    
    def _extract_relationships_llm(self, fact: str) -> List[Dict]:
        """
        Extract relationships between entities using LLM.
        
        Args:
            fact: The fact text
            
        Returns:
            List of relationships
        """
        # Get the relationship template
        template = self._get_relationship_template()
        prompt = template.format(fact=fact)
        
        # Get the LLM logger from the context if available
        kwargs = {}
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            if llm_logger:
                kwargs["llm_logger"] = llm_logger
        
        # Call LLM to extract relationships
        response = self.llm_client.invoke(prompt, temperature=0.0, **kwargs)
        
        try:
            # Parse JSON response
            import json
            relationships = json.loads(response)
            return relationships
        except Exception as e:
            # If JSON parsing fails, fall back to returning an empty list
            return []
    
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
        # Get loggers if available
        llm_logger = None
        chunker_logger = None
        if hasattr(self, 'context') and self.context:
            llm_logger = self.context.get("llm_logger")
            chunker_logger = self.context.get("logger")
        
        # Start logging the chunking process
        if chunker_logger:
            chunker_logger.log(
                step="initialization",
                message="Starting text chunking process",
                data={
                    "text_length": len(text),
                    "window_size": self.window_size,
                    "overlap": self.overlap,
                    "boundary_rules": self.boundary_rules,
                    "first_pass_template": self.first_pass_template_name,
                    "second_pass_template": self.second_pass_template_name
                }
            )
        
        # Step 1: Determine if we should process the text as a single segment
        # For small inputs, don't split into segments
        if len(text) <= self.window_size:
            if chunker_logger:
                chunker_logger.log(
                    step="segmentation",
                    message=f"Text is small enough ({len(text)} chars) to process as a single segment"
                )
            print(f"Text is small enough ({len(text)} chars) to process as a single segment")
            segments = [text]
        else:
            # Split into overlapping segments for larger texts
            expected_segments = max(1, int((len(text) - self.overlap) / (self.window_size - self.overlap)))
            if chunker_logger:
                chunker_logger.log(
                    step="segmentation",
                    message=f"Splitting text into segments",
                    data={
                        "text_length": len(text),
                        "window_size": self.window_size,
                        "overlap": self.overlap,
                        "expected_segments": expected_segments
                    }
                )
            print(f"Splitting text into segments (length: {len(text)}, window: {self.window_size}, overlap: {self.overlap})")
            print(f"Expected number of segments: ~{expected_segments} (with ideal sliding window)")
            segments = []
            start = 0
            while start < len(text):
                # Get segment with window_size or remaining text
                end = min(start + self.window_size, len(text))
                
                # Log segment boundary calculation
                if chunker_logger:
                    chunker_logger.log(
                        step="segmentation",
                        message=f"Calculating segment boundaries",
                        data={
                            "start": start,
                            "initial_end": end,
                            "segment_length": end - start
                        }
                    )
                
                # Adjust boundaries based on boundary rules
                if end < len(text) and "paragraph" in self.boundary_rules:
                    paragraph_boundaries = [
                        m.end() for m in re.finditer(r'\n\s*\n', text[start:end])
                    ]
                    if paragraph_boundaries:
                        end = start + paragraph_boundaries[-1]
                        if chunker_logger:
                            chunker_logger.log(
                                step="segmentation",
                                message=f"Adjusted segment to paragraph boundary",
                                data={"new_end": end, "boundary_type": "paragraph"}
                            )
                elif end < len(text) and "sentence" in self.boundary_rules:
                    sentence_boundaries = [
                        m.end() for m in re.finditer(r'[.!?][\s\n]', text[start:end])
                    ]
                    if sentence_boundaries:
                        end = start + sentence_boundaries[-1]
                        if chunker_logger:
                            chunker_logger.log(
                                step="segmentation",
                                message=f"Adjusted segment to sentence boundary",
                                data={"new_end": end, "boundary_type": "sentence"}
                            )
                
                segment_text = text[start:end].strip()
                if segment_text:
                    segments.append(segment_text)
                    if chunker_logger:
                        chunker_logger.log(
                            step="segmentation",
                            message=f"Created segment {len(segments)}",
                            data={"segment_index": len(segments)-1, "segment_length": len(segment_text)}
                        )
                
                # Move start position for next segment (with proper overlap)
                # We advance by (window_size - overlap) to create a sliding window with the desired overlap
                new_start = start + self.window_size - self.overlap
                
                # Log the window movement to help with debugging
                if chunker_logger:
                    chunker_logger.log(
                        step="segmentation",
                        message=f"Moving window",
                        data={
                            "current_start": start,
                            "current_end": end,
                            "new_start": new_start,
                            "advance_by": new_start - start,
                            "overlap": end - new_start
                        }
                    )
                
                # Ensure we make progress and maintain proper overlap
                start = max(end - self.overlap, new_start)
            
        if chunker_logger:
            chunker_logger.log(
                step="segmentation_complete",
                message=f"Created {len(segments)} text segments"
            )
        print(f"Processing {len(segments)} text segments")
        
        # Step 2: Extract initial facts from each segment
        if chunker_logger:
            chunker_logger.log(
                step="extraction",
                message=f"Starting fact extraction from {len(segments)} segments"
            )
        print("Extracting initial facts from segments...")
        all_initial_facts = []
        for i, segment in enumerate(segments):
            if chunker_logger:
                chunker_logger.log(
                    step="extraction",
                    message=f"Processing segment {i+1}/{len(segments)}",
                    data={"segment_index": i, "segment_length": len(segment)}
                )
            print(f"Processing segment {i+1}/{len(segments)} ({len(segment)} chars)")
            segment_facts = self._extract_initial_facts(segment)
            all_initial_facts.extend(segment_facts)
            if chunker_logger:
                chunker_logger.log(
                    step="extraction",
                    message=f"Extracted {len(segment_facts)} facts from segment {i+1}",
                    data={"segment_index": i, "fact_count": len(segment_facts)}
                )
            print(f"  Extracted {len(segment_facts)} facts from segment {i+1}")
        
        if chunker_logger:
            chunker_logger.log(
                step="extraction_complete",
                message=f"Total initial facts extracted: {len(all_initial_facts)}",
                data={"total_facts": len(all_initial_facts)}
            )
        print(f"Total initial facts extracted: {len(all_initial_facts)}")
        
        # Step 3: Refine facts to ensure they are atomic and self-contained
        if chunker_logger:
            chunker_logger.log(
                step="refinement",
                message=f"Starting fact refinement with {len(all_initial_facts)} initial facts"
            )
        print("Refining facts to ensure they are self-contained...")
        refined_facts = self._refine_facts(all_initial_facts)
        if chunker_logger:
            chunker_logger.log(
                step="refinement_complete",
                message=f"Refinement complete - {len(refined_facts)} facts created from {len(all_initial_facts)} initial facts",
                data={
                    "initial_facts": len(all_initial_facts),
                    "refined_facts": len(refined_facts),
                    "ratio": len(refined_facts) / max(1, len(all_initial_facts))
                }
            )
        print(f"Refined facts: {len(refined_facts)}")
        
        # Step 4: Create chunks with metadata
        if chunker_logger:
            chunker_logger.log(
                step="metadata",
                message=f"Creating metadata for {len(refined_facts)} facts"
            )
        print("Creating chunks with metadata...")
        chunks = []
        for i, fact in enumerate(refined_facts):
            chunk = {
                "text": fact,
                "metadata": self._create_chunk_metadata(fact, i)
            }
            chunks.append(chunk)
        
        if chunker_logger:
            chunker_logger.log(
                step="complete",
                message=f"Chunking complete - created {len(chunks)} chunks with metadata",
                data={"chunk_count": len(chunks)}
            )
        print(f"Created {len(chunks)} atomic chunks with metadata")
        return chunks


@register_component("chunker", "atomic")
def create_atomic_chunker(**kwargs):
    """Factory function for creating an atomic chunker."""
    return AtomicChunker(**kwargs)
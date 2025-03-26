"""Pipeline execution module for graph-based pipelines."""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import uuid

class PipelineNode:
    """Base class for pipeline nodes."""
    
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """Initialize a pipeline node.
        
        Args:
            node_id: Unique ID for the node
            node_type: Type of node (e.g., 'text-input', 'atomic-chunker')
            properties: Node-specific properties
        """
        self.id = node_id
        self.type = node_type
        self.properties = properties
        self.input_data = {}
        self.output_data = {}
        self.executed = False
        self.error = None
    
    def set_input(self, input_id: str, data: Any):
        """Set input data for the node.
        
        Args:
            input_id: ID of the input connector
            data: Input data
        """
        self.input_data[input_id] = data
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs.
        
        Returns:
            True if all required inputs are present, False otherwise
        """
        # To be implemented by subclasses
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute the node's operation.
        
        Returns:
            Dictionary of output data keyed by output connector ID
        """
        # To be implemented by subclasses
        raise NotImplementedError("execute method must be implemented by subclasses")


class TextInputNode(PipelineNode):
    """Node for providing text input."""
    
    def has_required_inputs(self) -> bool:
        """Text input nodes don't have any required inputs."""
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute the text input node."""
        try:
            text = self.properties.get('text', '')
            self.output_data = {'text': text}
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class FileUploadNode(PipelineNode):
    """Node for uploading file content."""
    
    def has_required_inputs(self) -> bool:
        """File upload nodes don't have any required inputs."""
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute the file upload node."""
        try:
            file_path = self.properties.get('file-path', '')
            if not file_path:
                raise ValueError("File path is required")
                
            try:
                with open(file_path, 'r') as f:
                    text = f.read()
            except Exception as e:
                raise ValueError(f"Could not read file: {str(e)}")
                
            self.output_data = {'text': text}
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class AtomicChunkerNode(PipelineNode):
    """Node for atomic chunking."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'text' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the atomic chunker node."""
        try:
            from ..chunkers.atomic import AtomicChunker
            
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'text'")
                
            text = self.input_data['text']
            
            # Get template names from properties
            first_pass_template = self.properties.get('template-first', 'detailed')
            second_pass_template = self.properties.get('template-second', 'default')
            
            # Create chunker
            chunker = AtomicChunker(
                first_pass_template_name=first_pass_template,
                second_pass_template_name=second_pass_template
            )
            
            # Process text
            chunks = chunker.chunk(text)
            
            self.output_data = {'chunks': chunks}
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class SlidingWindowNode(PipelineNode):
    """Node for sliding window chunking."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'text' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the sliding window chunker node."""
        try:
            from ..chunkers.basic import SlidingWindowChunker
            
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'text'")
                
            text = self.input_data['text']
            
            # Get properties
            window_size = int(self.properties.get('window-size', 1000))
            overlap = int(self.properties.get('overlap', 100))
            
            # Create chunker
            chunker = SlidingWindowChunker(
                window_size=window_size,
                overlap=overlap
            )
            
            # Process text
            chunks = chunker.chunk(text)
            
            self.output_data = {'chunks': chunks}
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class EntityExtractorNode(PipelineNode):
    """Node for entity extraction."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'chunks' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the entity extractor node."""
        try:
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'chunks'")
                
            chunks = self.input_data['chunks']
            
            # For now, this is a mock implementation
            # In a real implementation, we would use an entity extractor
            enriched_chunks = []
            for chunk in chunks:
                # Make a copy of the chunk to avoid modifying the original
                enriched_chunk = dict(chunk)
                
                # Add entity metadata if not present
                if 'metadata' not in enriched_chunk:
                    enriched_chunk['metadata'] = {}
                    
                if 'entities' not in enriched_chunk['metadata']:
                    # Mock entity extraction - in a real implementation,
                    # we would use an NLP model or LLM to extract entities
                    text = enriched_chunk.get('text', '')
                    words = text.split()
                    # Simple mock: consider capitalized words as entities
                    entities = [word for word in words if word and word[0].isupper()]
                    # Take only unique entities and limit to 10
                    entities = list(set(entities))[:10]
                    enriched_chunk['metadata']['entities'] = entities
                    
                enriched_chunks.append(enriched_chunk)
            
            self.output_data = {'enriched': enriched_chunks}
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class TopicTaggerNode(PipelineNode):
    """Node for topic tagging."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'chunks' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the topic tagger node."""
        try:
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'chunks'")
                
            chunks = self.input_data['chunks']
            
            # For now, this is a mock implementation
            # In a real implementation, we would use a topic tagger
            enriched_chunks = []
            for chunk in chunks:
                # Make a copy of the chunk to avoid modifying the original
                enriched_chunk = dict(chunk)
                
                # Add topic metadata if not present
                if 'metadata' not in enriched_chunk:
                    enriched_chunk['metadata'] = {}
                    
                if 'topics' not in enriched_chunk['metadata']:
                    # Mock topic tagging - in a real implementation,
                    # we would use an NLP model or LLM to tag topics
                    text = enriched_chunk.get('text', '')
                    # Simple mock: use some common topics
                    common_topics = [
                        "Technology", "Business", "Science", "Politics", 
                        "Health", "Education", "Environment", "Sports"
                    ]
                    # Randomly select 1-3 topics (based on text length)
                    import random
                    num_topics = min(3, max(1, len(text) // 500))
                    topics = random.sample(common_topics, num_topics)
                    enriched_chunk['metadata']['topics'] = topics
                    
                enriched_chunks.append(enriched_chunk)
            
            self.output_data = {'enriched': enriched_chunks}
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class VectorStoreNode(PipelineNode):
    """Node for vector storage."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'chunks' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the vector store node."""
        try:
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'chunks'")
                
            chunks = self.input_data['chunks']
            
            # For now, this is a mock implementation
            # In a real implementation, we would use a vector store
            
            # Generate unique IDs for each chunk
            stored_ids = []
            for _ in chunks:
                stored_ids.append(str(uuid.uuid4()))
                
            self.output_data = {
                'stored': stored_ids,
                # For demo purposes, keep the chunks in the output
                'chunks': chunks
            }
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class GraphStoreNode(PipelineNode):
    """Node for graph storage."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'chunks' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the graph store node."""
        try:
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'chunks'")
                
            chunks = self.input_data['chunks']
            
            # For now, this is a mock implementation
            # In a real implementation, we would use a graph store
            
            # Generate unique IDs for each chunk
            stored_ids = []
            for _ in chunks:
                stored_ids.append(str(uuid.uuid4()))
                
            self.output_data = {
                'stored': stored_ids,
                # For demo purposes, keep the chunks in the output
                'chunks': chunks
            }
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e


class VisualizerNode(PipelineNode):
    """Node for visualization."""
    
    def has_required_inputs(self) -> bool:
        """Check if the node has all required inputs."""
        return 'data' in self.input_data
    
    def execute(self) -> Dict[str, Any]:
        """Execute the visualizer node."""
        try:
            if not self.has_required_inputs():
                raise ValueError("Missing required input 'data'")
                
            data = self.input_data['data']
            visualization_type = self.properties.get('visualization-type', 'table')
            
            # Process the data based on visualization type
            visualization_data = self._prepare_visualization(data, visualization_type)
            
            self.output_data = {
                'visualization': {
                    'type': visualization_type, 
                    'data': data,
                    'processed_data': visualization_data
                }
            }
            self.executed = True
            return self.output_data
        except Exception as e:
            self.error = str(e)
            raise e
    
    def _prepare_visualization(self, data: Any, visualization_type: str) -> Dict[str, Any]:
        """
        Process input data for visualization.
        
        Args:
            data: The input data (chunks, stored IDs, etc.)
            visualization_type: The type of visualization (table, graph, json)
            
        Returns:
            Processed data ready for visualization
        """
        result = {
            'title': 'Visualization Results',
            'summary': '',
            'visualization_type': visualization_type,
            'elements': []
        }
        
        # Handle different input types
        if isinstance(data, list):
            # Assuming this is a chunks list
            if data and isinstance(data[0], dict) and 'text' in data[0]:
                # This is likely a list of chunks
                result['title'] = f'Chunk Visualization ({len(data)} chunks)'
                result['summary'] = f'Displaying {len(data)} chunks of text content'
                
                if visualization_type == 'table':
                    # Format data for table visualization
                    for i, chunk in enumerate(data):
                        metadata = chunk.get('metadata', {})
                        element = {
                            'id': metadata.get('chunk_id', f'chunk-{i}'),
                            'content': chunk.get('text', '')[:200] + ('...' if len(chunk.get('text', '')) > 200 else ''),
                            'full_content': chunk.get('text', ''),
                            'metadata': {
                                'Word Count': metadata.get('word_count', len(chunk.get('text', '').split())),
                                'Entities': ', '.join(metadata.get('entities', [])),
                                'Topics': ', '.join(metadata.get('topics', []))
                            }
                        }
                        result['elements'].append(element)
                
                elif visualization_type == 'graph':
                    # Create a graph visualization of relationships between chunks
                    nodes = []
                    edges = []
                    
                    for i, chunk in enumerate(data):
                        metadata = chunk.get('metadata', {})
                        chunk_id = metadata.get('chunk_id', f'chunk-{i}')
                        
                        # Add node
                        nodes.append({
                            'id': chunk_id,
                            'label': f"Chunk {i+1}",
                            'title': chunk.get('text', '')[:100] + '...',
                            'data': {
                                'text': chunk.get('text', ''),
                                'entities': metadata.get('entities', []),
                                'topics': metadata.get('topics', [])
                            }
                        })
                        
                        # Add edges from relationships
                        for rel in metadata.get('relationships', []):
                            if rel.get('target') and rel.get('type'):
                                edges.append({
                                    'from': chunk_id,
                                    'to': rel.get('target'),
                                    'label': rel.get('type')
                                })
                    
                    result['elements'] = {
                        'nodes': nodes,
                        'edges': edges
                    }
                
                elif visualization_type == 'json':
                    # Just format the raw JSON for inspection
                    result['elements'] = data
                
            else:
                # Generic list - could be stored IDs
                result['title'] = f'List Data ({len(data)} items)'
                result['summary'] = f'Displaying {len(data)} items'
                result['elements'] = data
                
        else:
            # Handle non-list data
            result['title'] = 'Raw Data Visualization'
            result['summary'] = 'Displaying raw data object'
            result['elements'] = data
        
        return result


class PipelineExecutor:
    """Executor for graph-based pipelines."""
    
    # Node type to class mapping
    NODE_TYPES = {
        'text-input': TextInputNode,
        'file-upload': FileUploadNode,
        'atomic-chunker': AtomicChunkerNode,
        'sliding-window': SlidingWindowNode,
        'entity-extractor': EntityExtractorNode,
        'topic-tagger': TopicTaggerNode,
        'vector-store': VectorStoreNode,
        'graph-store': GraphStoreNode,
        'visualizer': VisualizerNode
    }
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        """Initialize the pipeline executor.
        
        Args:
            pipeline_config: Pipeline configuration dictionary
        """
        self.config = pipeline_config
        self.nodes = {}
        self.edges = []
        self.execution_order = []
        self.results = {}
        self.errors = {}
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize the pipeline from the configuration."""
        try:
            # Extract nodes and edges from the configuration
            nodes_config = self.config.get('nodes', [])
            edges_config = self.config.get('edges', [])
            
            # Create node instances
            for node_config in nodes_config:
                node_id = node_config.get('id')
                node_type = node_config.get('type')
                properties = node_config.get('properties', {})
                
                if node_type not in self.NODE_TYPES:
                    raise ValueError(f"Unknown node type: {node_type}")
                
                node_class = self.NODE_TYPES[node_type]
                self.nodes[node_id] = node_class(node_id, node_type, properties)
            
            # Store edge configurations
            for edge_config in edges_config:
                self.edges.append({
                    'id': edge_config.get('id'),
                    'startNodeId': edge_config.get('startNodeId'),
                    'startConnectorId': edge_config.get('startConnectorId'),
                    'endNodeId': edge_config.get('endNodeId'),
                    'endConnectorId': edge_config.get('endConnectorId')
                })
            
            # Compute execution order (topological sort)
            self._compute_execution_order()
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            return False
    
    def _compute_execution_order(self):
        """Compute the execution order for nodes (topological sort)."""
        # Build a graph representation for topological sorting
        graph = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        # Add edges and count incoming edges for each node
        for edge in self.edges:
            start_node = edge['startNodeId']
            end_node = edge['endNodeId']
            
            if start_node in graph and end_node in graph:
                graph[start_node].append(end_node)
                in_degree[end_node] += 1
        
        # Nodes with no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        # Topological sort
        execution_order = []
        while queue:
            node_id = queue.pop(0)
            execution_order.append(node_id)
            
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(execution_order) != len(self.nodes):
            raise ValueError("Pipeline contains cycles, cannot determine execution order")
        
        self.execution_order = execution_order
    
    def execute(self) -> Dict[str, Any]:
        """Execute the pipeline.
        
        Returns:
            Dictionary with execution results and errors
        """
        self.results = {}
        self.errors = {}
        
        try:
            # Execute nodes in topological order
            for node_id in self.execution_order:
                node = self.nodes[node_id]
                
                try:
                    # Connect inputs to the node
                    self._connect_node_inputs(node)
                    
                    # Check if the node has all required inputs
                    if not node.has_required_inputs():
                        raise ValueError(f"Node {node_id} is missing required inputs")
                    
                    # Execute the node
                    outputs = node.execute()
                    
                    # Store outputs in results
                    self.results[node_id] = outputs
                except Exception as e:
                    self.errors[node_id] = str(e)
                    self.logger.error(f"Error executing node {node_id}: {str(e)}")
            
            return {
                'results': self.results,
                'errors': self.errors,
                'execution_order': self.execution_order
            }
        except Exception as e:
            self.logger.error(f"Error executing pipeline: {str(e)}")
            return {
                'results': self.results,
                'errors': {'pipeline': str(e)},
                'execution_order': self.execution_order
            }
    
    def _connect_node_inputs(self, node: PipelineNode):
        """Connect inputs to a node from previous node outputs.
        
        Args:
            node: The node to connect inputs to
        """
        # Find edges that end at this node
        incoming_edges = [
            edge for edge in self.edges 
            if edge['endNodeId'] == node.id
        ]
        
        # Connect each input
        for edge in incoming_edges:
            start_node_id = edge['startNodeId']
            start_connector_id = edge['startConnectorId']
            end_connector_id = edge['endConnectorId']
            
            # Get the output from the source node
            if start_node_id in self.results:
                source_outputs = self.results[start_node_id]
                if start_connector_id in source_outputs:
                    node.set_input(end_connector_id, source_outputs[start_connector_id])
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'PipelineExecutor':
        """Create a pipeline executor from a file.
        
        Args:
            file_path: Path to the pipeline configuration file
            
        Returns:
            A PipelineExecutor instance
        """
        with open(file_path, 'r') as f:
            pipeline_data = json.load(f)
            
        config = pipeline_data.get('config', {})
        return cls(config)
"""Component registry system for KastenRAG."""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

T = TypeVar('T')


class ComponentRegistry:
    """
    Registry for system components with factory pattern support.
    
    This allows for registration of different implementations of the same interface,
    which can be instantiated based on configuration.
    """
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Callable]] = {}
    
    def register(
        self,
        component_type: str,
        implementation_name: str,
        factory: Callable,
        replace: bool = False
    ):
        """
        Register a component implementation.
        
        Args:
            component_type: The type of component (e.g., "transcriber", "chunker")
            implementation_name: The name of this specific implementation
            factory: Factory function to create instances of this component
            replace: Whether to replace an existing registration with the same names
        """
        if component_type not in self._components:
            self._components[component_type] = {}
            
        if implementation_name in self._components[component_type] and not replace:
            raise ValueError(
                f"Implementation '{implementation_name}' already registered for "
                f"component type '{component_type}'"
            )
            
        self._components[component_type][implementation_name] = factory
    
    def get_factory(self, component_type: str, implementation_name: str) -> Callable:
        """
        Get the factory function for a specific component implementation.
        
        Args:
            component_type: The type of component
            implementation_name: The name of the implementation
            
        Returns:
            Factory function for the component
            
        Raises:
            ValueError: If the component type or implementation is not registered
        """
        if component_type not in self._components:
            raise ValueError(f"Component type '{component_type}' not registered")
            
        if implementation_name not in self._components[component_type]:
            raise ValueError(
                f"Implementation '{implementation_name}' not registered for "
                f"component type '{component_type}'"
            )
            
        return self._components[component_type][implementation_name]
    
    def create(self, component_type: str, implementation_name: str, **kwargs) -> Any:
        """
        Create an instance of a component using its factory.
        
        Args:
            component_type: The type of component
            implementation_name: The name of the implementation
            **kwargs: Arguments to pass to the factory
            
        Returns:
            Instance of the component
        """
        factory = self.get_factory(component_type, implementation_name)
        return factory(**kwargs)
    
    def list_implementations(self, component_type: str) -> List[str]:
        """
        List all registered implementations for a component type.
        
        Args:
            component_type: The type of component
            
        Returns:
            List of implementation names
        """
        if component_type not in self._components:
            return []
            
        return list(self._components[component_type].keys())
    
    def list_component_types(self) -> List[str]:
        """
        List all registered component types.
        
        Returns:
            List of component types
        """
        return list(self._components.keys())


# Create a global registry instance
registry = ComponentRegistry()


def register_component(
    component_type: str,
    implementation_name: str,
    replace: bool = False
):
    """
    Decorator to register a component factory function.
    
    Example:
        @register_component("transcriber", "whisper_local")
        def create_whisper_transcriber(**kwargs):
            return WhisperTranscriber(**kwargs)
    
    Args:
        component_type: The type of component
        implementation_name: The name of this specific implementation
        replace: Whether to replace an existing registration
        
    Returns:
        Decorator function
    """
    def decorator(factory_func: Callable) -> Callable:
        registry.register(
            component_type=component_type,
            implementation_name=implementation_name,
            factory=factory_func,
            replace=replace
        )
        return factory_func
    return decorator
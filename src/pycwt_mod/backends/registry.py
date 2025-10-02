"""
Backend registration system for Monte Carlo execution strategies.

This module provides a registry for discovering and managing available backends.
Backends can be registered, retrieved, and listed through this interface.
"""

from typing import Dict, List, Optional, Type
from .base import MonteCarloBackend


class BackendRegistry:
    """
    Registry for managing Monte Carlo backend implementations.
    
    This class maintains a registry of available backends and provides
    methods to register, retrieve, and list them.
    """
    
    _backends: Dict[str, Type[MonteCarloBackend]] = {}
    _default_backend: Optional[str] = None
    
    @classmethod
    def register(cls, name: str, backend_class: Type[MonteCarloBackend]) -> None:
        """
        Register a backend implementation.
        
        Parameters
        ----------
        name : str
            Unique name for the backend (e.g., 'sequential', 'joblib', 'dask')
        backend_class : type
            Backend class (must inherit from MonteCarloBackend)
            
        Raises
        ------
        TypeError
            If backend_class doesn't inherit from MonteCarloBackend
        ValueError
            If backend name is already registered
        """
        if not issubclass(backend_class, MonteCarloBackend):
            raise TypeError(
                f"Backend class must inherit from MonteCarloBackend, "
                f"got {backend_class}"
            )
        
        if name in cls._backends:
            raise ValueError(f"Backend '{name}' is already registered")
        
        cls._backends[name] = backend_class
        
        # Set first registered backend as default if none set
        if cls._default_backend is None:
            cls._default_backend = name
    
    @classmethod
    def get(cls, name: Optional[str] = None) -> MonteCarloBackend:
        """
        Get a backend instance by name.
        
        Parameters
        ----------
        name : str, optional
            Name of the backend to retrieve. If None, returns default backend.
            
        Returns
        -------
        backend : MonteCarloBackend
            Instance of the requested backend
            
        Raises
        ------
        ValueError
            If backend name is not registered or if no backends are registered
        """
        if name is None:
            name = cls._default_backend
        
        if name is None:
            raise ValueError("No backends registered")
        
        if name not in cls._backends:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Backend '{name}' not found. Available backends: {available}"
            )
        
        return cls._backends[name]()
    
    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered backend names.
        
        Returns
        -------
        backends : list of str
            List of registered backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List available (usable) backend names.
        
        Only returns backends whose dependencies are installed and that
        can run on the current system.
        
        Returns
        -------
        backends : list of str
            List of available backend names
        """
        available = []
        for name in cls._backends:
            try:
                backend = cls._backends[name]()
                if backend.is_available():
                    available.append(name)
            except Exception:
                # Skip backends that fail to instantiate
                pass
        return available
    
    @classmethod
    def get_info(cls) -> Dict[str, dict]:
        """
        Get information about all registered backends.
        
        Returns
        -------
        info : dict
            Dictionary mapping backend names to their info dictionaries
        """
        info = {}
        for name in cls._backends:
            try:
                backend = cls._backends[name]()
                info[name] = backend.get_info()
            except Exception as e:
                info[name] = {
                    "name": name,
                    "available": False,
                    "error": str(e)
                }
        return info
    
    @classmethod
    def set_default(cls, name: str) -> None:
        """
        Set the default backend.
        
        Parameters
        ----------
        name : str
            Name of the backend to set as default
            
        Raises
        ------
        ValueError
            If backend name is not registered
        """
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' is not registered")
        cls._default_backend = name
    
    @classmethod
    def get_default(cls) -> Optional[str]:
        """
        Get the name of the default backend.
        
        Returns
        -------
        name : str or None
            Name of the default backend, or None if no default is set
        """
        return cls._default_backend
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered backends.
        
        Warning: This is mainly for testing purposes.
        """
        cls._backends.clear()
        cls._default_backend = None


# Convenience functions for easier access
def register_backend(name: str, backend_class: Type[MonteCarloBackend]) -> None:
    """Register a backend (convenience wrapper)."""
    BackendRegistry.register(name, backend_class)


def get_backend(name: Optional[str] = None) -> MonteCarloBackend:
    """Get a backend instance (convenience wrapper)."""
    return BackendRegistry.get(name)


def list_backends(available_only: bool = False) -> List[str]:
    """
    List backend names (convenience wrapper).
    
    Parameters
    ----------
    available_only : bool, optional
        If True, only list backends that are available on this system
        
    Returns
    -------
    backends : list of str
        List of backend names
    """
    if available_only:
        return BackendRegistry.list_available()
    return BackendRegistry.list_all()

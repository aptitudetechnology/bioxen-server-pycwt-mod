"""
Backend system for Monte Carlo simulations in pycwt_mod.

This module provides a plugin architecture for different execution strategies
(sequential, parallel, distributed, GPU) while maintaining a consistent API.

Available backends:
- Sequential: Single-process execution (always available)
- Joblib: Multi-process parallelization (requires joblib)
- Dask: Distributed computing (future implementation)
- GPU: GPU acceleration (future implementation)

Basic usage:
    >>> from pycwt_mod.backends import get_backend, list_backends
    >>> 
    >>> # List available backends
    >>> print(list_backends(available_only=True))
    ['sequential', 'joblib']
    >>> 
    >>> # Get a backend
    >>> backend = get_backend('joblib')
    >>> 
    >>> # Run Monte Carlo simulation
    >>> results = backend.run_monte_carlo(worker_func, n_simulations=1000)

Advanced usage:
    >>> from pycwt_mod.backends import BackendRegistry, JoblibBackend
    >>> 
    >>> # Register custom backend
    >>> BackendRegistry.register('custom', MyCustomBackend)
    >>> 
    >>> # Configure backend
    >>> backend = JoblibBackend(n_jobs=4, prefer='processes')
    >>> 
    >>> # Get backend information
    >>> info = BackendRegistry.get_info()
"""

from .base import MonteCarloBackend
from .registry import (
    BackendRegistry,
    get_backend,
    list_backends,
    register_backend
)
from .sequential import SequentialBackend
from .joblib import JoblibBackend
from .dask import DaskBackend
from .gpu import GPUBackend
from .elm11 import ELM11Backend


# Auto-register built-in backends
def _register_builtin_backends():
    """Register all built-in backends."""
    BackendRegistry.register('sequential', SequentialBackend)
    BackendRegistry.register('joblib', JoblibBackend)
    BackendRegistry.register('dask', DaskBackend)
    BackendRegistry.register('gpu', GPUBackend)
    BackendRegistry.register('elm11', ELM11Backend)
    
    # Set default backend to sequential (always available)
    BackendRegistry.set_default('sequential')


# Register backends on module import
_register_builtin_backends()


__all__ = [
    # Base class
    'MonteCarloBackend',
    
    # Backend implementations
    'SequentialBackend',
    'JoblibBackend',
    'DaskBackend',
    'GPUBackend',
    'ELM11Backend',
    
    # Registry
    'BackendRegistry',
    'get_backend',
    'list_backends',
    'register_backend',
]


def get_recommended_backend(n_simulations: int = None) -> str:
    """
    Get recommended backend name based on workload and availability.
    
    Parameters
    ----------
    n_simulations : int, optional
        Number of simulations to run (helps choose optimal backend)
        
    Returns
    -------
    backend_name : str
        Name of recommended backend
        
    Notes
    -----
    Recommendation logic:
    - n_simulations < 10: Use 'sequential' (overhead not worth it)
    - n_simulations >= 10 and joblib available: Use 'joblib'
    - Otherwise: Use 'sequential'
    """
    available = list_backends(available_only=True)
    
    # For very small workloads, sequential is best
    if n_simulations is not None and n_simulations < 10:
        return 'sequential'
    
    # Prefer joblib for parallel workloads if available
    if 'joblib' in available:
        return 'joblib'
    
    # Fall back to sequential
    return 'sequential'


def print_backend_info():
    """Print information about all registered backends."""
    info = BackendRegistry.get_info()
    
    print("=" * 60)
    print("Monte Carlo Backend Information")
    print("=" * 60)
    
    for name, backend_info in info.items():
        available = "✓" if backend_info.get('available', False) else "✗"
        print(f"\n{available} {name}:")
        print(f"  Description: {backend_info.get('description', 'N/A')}")
        
        if 'capabilities' in backend_info:
            caps = ", ".join(backend_info['capabilities'])
            print(f"  Capabilities: {caps}")
        
        if 'error' in backend_info:
            print(f"  Error: {backend_info['error']}")
    
    default = BackendRegistry.get_default()
    print(f"\nDefault backend: {default}")
    print("=" * 60)

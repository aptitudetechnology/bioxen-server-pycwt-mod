"""
GPU-accelerated backend for Monte Carlo simulations (Future Implementation).

This is a placeholder/stub for future GPU/CUDA integration. GPU acceleration
can provide massive speedups for appropriate workloads.

Status: Not yet implemented
"""

from typing import Any, Callable, Dict, List, Optional
from .base import MonteCarloBackend


class GPUBackend(MonteCarloBackend):
    """
    GPU-accelerated backend (placeholder for future implementation).
    
    This backend will use GPU acceleration (CUDA/OpenCL) to execute
    simulations on graphics cards. Ideal for highly parallelizable
    workloads with simple operations.
    
    Status: Not yet implemented - this is a placeholder
    
    Future capabilities:
    - GPU-accelerated execution
    - Batch processing
    - Massive parallelism (thousands of concurrent simulations)
    - SIMD optimizations
    
    Note
    ----
    This is a stub implementation. GPU support may be added in a
    future release, potentially using CuPy, PyTorch, or custom kernels.
    """
    
    def __init__(self):
        """Initialize the GPU backend stub."""
        super().__init__()
        self.name = "GPU"
        self.description = "GPU-accelerated execution (not yet implemented)"
    
    def run_monte_carlo(
        self,
        worker_func: Callable,
        n_simulations: int,
        worker_args: tuple = (),
        worker_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        **backend_kwargs
    ) -> List[Any]:
        """
        Execute Monte Carlo simulations on GPU (not yet implemented).
        
        Raises
        ------
        NotImplementedError
            This backend is not yet implemented
        """
        raise NotImplementedError(
            "GPU backend is not yet implemented. "
            "Use 'sequential' or 'joblib' backend instead."
        )
    
    def is_available(self) -> bool:
        """
        Check if GPU backend is available.
        
        Returns
        -------
        available : bool
            Always False (not yet implemented)
        """
        return False
    
    def get_capabilities(self) -> List[str]:
        """
        Get backend capabilities.
        
        Returns
        -------
        capabilities : list of str
            Future capabilities list
        """
        return ["gpu", "massive_parallel", "simd", "not_implemented"]


# Placeholder for future implementation
# TODO: Implement GPUBackend with:
#   - GPU device detection and selection
#   - Kernel compilation (CUDA/OpenCL)
#   - Memory management (host <-> device transfers)
#   - Batch processing optimization
#   - Error handling for GPU-specific issues
#
# Potential approaches:
#   1. CuPy for NumPy-like GPU arrays
#   2. Numba CUDA for custom kernels
#   3. PyTorch for tensor operations
#   4. Custom CUDA/OpenCL kernels

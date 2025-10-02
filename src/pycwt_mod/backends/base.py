"""
Abstract base class for Monte Carlo backend execution strategies.

This module defines the interface that all backend implementations must follow.
Each backend provides a different execution strategy (sequential, parallel, GPU, etc.)
while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class MonteCarloBackend(ABC):
    """
    Abstract base class for Monte Carlo simulation backends.
    
    All backend implementations must inherit from this class and implement
    the required methods. This ensures a consistent interface across different
    execution strategies.
    
    Attributes
    ----------
    name : str
        Human-readable name of the backend
    description : str
        Brief description of the backend's execution strategy
    """
    
    def __init__(self):
        """Initialize the backend."""
        self.name = self.__class__.__name__
        self.description = "Base backend class"
    
    @abstractmethod
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
        Execute Monte Carlo simulations using this backend's strategy.
        
        Parameters
        ----------
        worker_func : callable
            Function to execute for each simulation. Must accept a seed as
            its first argument.
        n_simulations : int
            Number of Monte Carlo simulations to run
        worker_args : tuple, optional
            Additional positional arguments to pass to worker_func
        worker_kwargs : dict, optional
            Additional keyword arguments to pass to worker_func
        seed : int, optional
            Master random seed for reproducibility. If None, uses random seed.
        verbose : bool, optional
            Whether to display progress information
        **backend_kwargs : dict
            Backend-specific configuration options
            
        Returns
        -------
        results : list
            List of results from each simulation
            
        Notes
        -----
        Each backend must ensure:
        1. Deterministic results when seed is provided
        2. Proper error handling and reporting
        3. Progress monitoring (if verbose=True)
        4. Resource cleanup after execution
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available on the current system.
        
        Returns
        -------
        available : bool
            True if backend can be used, False otherwise
            
        Notes
        -----
        This method should check for required dependencies and system
        capabilities without raising exceptions.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this backend.
        
        Returns
        -------
        info : dict
            Dictionary containing backend metadata:
            - name: Backend name
            - description: Backend description
            - available: Whether backend is available
            - capabilities: List of special capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "available": self.is_available(),
            "capabilities": self.get_capabilities()
        }
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of backend capabilities.
        
        Returns
        -------
        capabilities : list of str
            List of capability strings (e.g., 'parallel', 'distributed', 'gpu')
        """
        return []
    
    def validate_config(self, **kwargs) -> bool:
        """
        Validate backend-specific configuration options.
        
        Parameters
        ----------
        **kwargs : dict
            Backend-specific configuration options
            
        Returns
        -------
        valid : bool
            True if configuration is valid
            
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        return True
    
    def __repr__(self) -> str:
        """String representation of the backend."""
        status = "available" if self.is_available() else "unavailable"
        return f"{self.name} ({status}): {self.description}"

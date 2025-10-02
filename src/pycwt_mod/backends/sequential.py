"""
Sequential backend for Monte Carlo simulations.

This is the default backend that executes simulations sequentially
in a single process. It's always available and serves as the fallback
when parallel backends are not available.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional
from .base import MonteCarloBackend


class SequentialBackend(MonteCarloBackend):
    """
    Sequential (single-process) backend for Monte Carlo simulations.
    
    This backend executes simulations one at a time in the current process.
    It has no external dependencies and is always available.
    
    Characteristics:
    - No parallelization overhead
    - Deterministic execution order
    - Minimal memory footprint
    - Best for small numbers of simulations or debugging
    
    Parameters
    ----------
    None
    
    Examples
    --------
    >>> from pycwt_mod.backends import SequentialBackend
    >>> backend = SequentialBackend()
    >>> def worker(seed, x):
    ...     rng = np.random.default_rng(seed)
    ...     return rng.normal() + x
    >>> results = backend.run_monte_carlo(worker, 100, worker_args=(5.0,), seed=42)
    """
    
    def __init__(self):
        """Initialize the sequential backend."""
        super().__init__()
        self.name = "Sequential"
        self.description = "Single-process sequential execution"
    
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
        Execute Monte Carlo simulations sequentially.
        
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
            Master random seed for reproducibility
        verbose : bool, optional
            Whether to display progress information
        **backend_kwargs : dict
            Backend-specific options (none used for sequential backend)
            
        Returns
        -------
        results : list
            List of results from each simulation
        """
        if worker_kwargs is None:
            worker_kwargs = {}
        
        # Generate independent seeds for each simulation
        if seed is not None:
            seed_sequence = np.random.SeedSequence(seed)
            child_seeds = seed_sequence.spawn(n_simulations)
            seeds = [s.generate_state(1)[0] for s in child_seeds]
        else:
            # Use random seeds
            seeds = [None] * n_simulations
        
        results = []
        
        if verbose:
            print(f"Running {n_simulations} simulations sequentially...")
        
        for i, sim_seed in enumerate(seeds):
            try:
                result = worker_func(sim_seed, *worker_args, **worker_kwargs)
                results.append(result)
                
                if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
                    progress = (i + 1) / n_simulations * 100
                    print(f"Progress: {i + 1}/{n_simulations} ({progress:.1f}%)")
                    
            except Exception as e:
                if verbose:
                    print(f"Error in simulation {i + 1}: {e}")
                raise
        
        if verbose:
            print(f"Completed {n_simulations} simulations")
        
        return results
    
    def is_available(self) -> bool:
        """
        Check if sequential backend is available.
        
        Returns
        -------
        available : bool
            Always True (sequential backend has no dependencies)
        """
        return True
    
    def get_capabilities(self) -> List[str]:
        """
        Get backend capabilities.
        
        Returns
        -------
        capabilities : list of str
            List of capabilities (empty for sequential)
        """
        return ["deterministic", "low_memory"]
    
    def validate_config(self, **kwargs) -> bool:
        """
        Validate configuration options.
        
        Sequential backend ignores all backend-specific kwargs,
        so validation always succeeds.
        
        Parameters
        ----------
        **kwargs : dict
            Backend-specific options (ignored)
            
        Returns
        -------
        valid : bool
            Always True
        """
        return True

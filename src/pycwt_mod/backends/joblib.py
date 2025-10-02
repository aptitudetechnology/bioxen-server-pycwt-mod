"""
Joblib parallel backend for Monte Carlo simulations.

This backend uses joblib to parallelize simulations across multiple CPU cores.
It provides a good balance between ease of use and performance for
embarrassingly parallel workloads.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional
from .base import MonteCarloBackend


class JoblibBackend(MonteCarloBackend):
    """
    Parallel backend using joblib for Monte Carlo simulations.
    
    This backend uses joblib.Parallel to distribute simulations across
    multiple CPU cores. It's ideal for CPU-bound Monte Carlo simulations.
    
    Characteristics:
    - Parallel execution across CPU cores
    - Low overhead for medium to large workloads
    - Automatic load balancing
    - Progress monitoring with tqdm
    - Deterministic with seed control
    
    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all CPUs (default: -1)
    prefer : str, optional
        Joblib execution mode: 'processes' or 'threads' (default: 'processes')
    
    Examples
    --------
    >>> from pycwt_mod.backends import JoblibBackend
    >>> backend = JoblibBackend(n_jobs=-1)
    >>> def worker(seed, x):
    ...     rng = np.random.default_rng(seed)
    ...     return rng.normal() + x
    >>> results = backend.run_monte_carlo(worker, 1000, worker_args=(5.0,), seed=42)
    """
    
    def __init__(self, n_jobs: int = -1, prefer: str = "processes"):
        """
        Initialize the joblib backend.
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs (-1 for all CPUs)
        prefer : str, optional
            Joblib backend: 'processes' or 'threads'
        """
        super().__init__()
        self.name = "Joblib"
        self.description = f"Parallel execution using joblib ({prefer})"
        self.n_jobs = n_jobs
        self.prefer = prefer
        self._joblib_available = None
    
    def run_monte_carlo(
        self,
        worker_func: Callable,
        n_simulations: int,
        worker_args: tuple = (),
        worker_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
        prefer: Optional[str] = None,
        **backend_kwargs
    ) -> List[Any]:
        """
        Execute Monte Carlo simulations in parallel using joblib.
        
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
        n_jobs : int, optional
            Number of parallel jobs (overrides instance default)
        prefer : str, optional
            Joblib backend preference (overrides instance default)
        **backend_kwargs : dict
            Additional joblib.Parallel options
            
        Returns
        -------
        results : list
            List of results from each simulation
            
        Raises
        ------
        ImportError
            If joblib is not installed
        """
        if not self.is_available():
            raise ImportError(
                "Joblib backend requires 'joblib' to be installed. "
                "Install it with: pip install joblib"
            )
        
        from joblib import Parallel, delayed
        
        if worker_kwargs is None:
            worker_kwargs = {}
        
        # Use provided values or fall back to instance defaults
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        prefer = prefer if prefer is not None else self.prefer
        
        # Generate independent seeds for each simulation
        if seed is not None:
            seed_sequence = np.random.SeedSequence(seed)
            child_seeds = seed_sequence.spawn(n_simulations)
            seeds = [s.generate_state(1)[0] for s in child_seeds]
        else:
            # Use random seeds
            seeds = [None] * n_simulations
        
        if verbose:
            print(f"Running {n_simulations} simulations with joblib "
                  f"(n_jobs={n_jobs}, prefer='{prefer}')...")
        
        # Execute in parallel
        verbose_level = 10 if verbose else 0
        
        results = Parallel(
            n_jobs=n_jobs,
            prefer=prefer,
            verbose=verbose_level,
            **backend_kwargs
        )(
            delayed(worker_func)(sim_seed, *worker_args, **worker_kwargs)
            for sim_seed in seeds
        )
        
        if verbose:
            print(f"Completed {n_simulations} simulations")
        
        return results
    
    def is_available(self) -> bool:
        """
        Check if joblib backend is available.
        
        Returns
        -------
        available : bool
            True if joblib is installed
        """
        if self._joblib_available is None:
            try:
                import joblib
                self._joblib_available = True
            except ImportError:
                self._joblib_available = False
        return self._joblib_available
    
    def get_capabilities(self) -> List[str]:
        """
        Get backend capabilities.
        
        Returns
        -------
        capabilities : list of str
            List of capabilities
        """
        capabilities = ["parallel", "deterministic", "progress_bar"]
        
        if self.is_available():
            import joblib
            capabilities.append(f"joblib_v{joblib.__version__}")
        
        return capabilities
    
    def validate_config(self, n_jobs: Optional[int] = None, 
                       prefer: Optional[str] = None, **kwargs) -> bool:
        """
        Validate configuration options.
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs
        prefer : str, optional
            Joblib backend preference
        **kwargs : dict
            Other options (passed through to joblib)
            
        Returns
        -------
        valid : bool
            True if configuration is valid
            
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        if prefer is not None and prefer not in ["processes", "threads"]:
            raise ValueError(
                f"prefer must be 'processes' or 'threads', got '{prefer}'"
            )
        
        if n_jobs is not None:
            if not isinstance(n_jobs, int):
                raise ValueError(f"n_jobs must be an integer, got {type(n_jobs)}")
            if n_jobs == 0 or n_jobs < -1:
                raise ValueError(f"n_jobs must be -1 or positive, got {n_jobs}")
        
        return True

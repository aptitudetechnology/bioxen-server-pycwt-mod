"""
Example: Using the backend system for Monte Carlo simulations.

This script demonstrates how to use different backends for Monte Carlo
simulations in pycwt_mod.
"""

import numpy as np
from pycwt_mod.backends import (
    get_backend,
    list_backends,
    print_backend_info,
    get_recommended_backend
)


def simple_worker(seed, mean=0.0, std=1.0):
    """
    Simple worker function for Monte Carlo simulation.
    
    Parameters
    ----------
    seed : int
        Random seed for this simulation
    mean : float
        Mean of the normal distribution
    std : float
        Standard deviation of the normal distribution
        
    Returns
    -------
    sample : float
        Random sample from normal distribution
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std)


def wavelet_coherence_worker(seed, signal1, signal2):
    """
    Example worker for wavelet coherence Monte Carlo significance testing.
    
    This is a simplified placeholder - actual implementation would compute
    WCT on surrogate data.
    
    Parameters
    ----------
    seed : int
        Random seed for this simulation
    signal1, signal2 : array-like
        Input signals (would be randomized for surrogates)
        
    Returns
    -------
    max_coherence : float
        Maximum coherence value in the surrogate
    """
    rng = np.random.default_rng(seed)
    
    # Placeholder: In real implementation, would:
    # 1. Generate surrogate data (AR process or phase randomization)
    # 2. Compute WCT on surrogates
    # 3. Return maximum coherence or other statistic
    
    # For now, just return random value
    return rng.uniform(0, 1)


def main():
    """Demonstrate backend usage."""
    
    print("=" * 60)
    print("PyCWT-mod Backend System Examples")
    print("=" * 60)
    print()
    
    # Example 1: List available backends
    print("Example 1: Discovering Available Backends")
    print("-" * 60)
    print(f"All registered backends: {list_backends()}")
    print(f"Available backends: {list_backends(available_only=True)}")
    print()
    
    # Example 2: Print detailed backend information
    print("Example 2: Backend Information")
    print("-" * 60)
    print_backend_info()
    print()
    
    # Example 3: Simple Monte Carlo with sequential backend
    print("Example 3: Sequential Backend (Simple)")
    print("-" * 60)
    backend = get_backend('sequential')
    results = backend.run_monte_carlo(
        simple_worker,
        n_simulations=100,
        worker_kwargs={'mean': 5.0, 'std': 2.0},
        seed=42,
        verbose=False
    )
    print(f"Ran 100 simulations")
    print(f"Mean: {np.mean(results):.4f} (expected: 5.0)")
    print(f"Std:  {np.std(results):.4f} (expected: 2.0)")
    print()
    
    # Example 4: Get recommended backend
    print("Example 4: Recommended Backend")
    print("-" * 60)
    recommended = get_recommended_backend(n_simulations=1000)
    print(f"Recommended backend for 1000 simulations: {recommended}")
    print()
    
    # Example 5: Use joblib backend if available
    print("Example 5: Joblib Backend (Parallel)")
    print("-" * 60)
    if 'joblib' in list_backends(available_only=True):
        backend = get_backend('joblib')
        print("Running 1000 simulations in parallel...")
        results = backend.run_monte_carlo(
            simple_worker,
            n_simulations=1000,
            worker_kwargs={'mean': 0.0, 'std': 1.0},
            seed=42,
            verbose=True,
            n_jobs=-1  # Use all CPUs
        )
        print(f"Mean: {np.mean(results):.4f} (expected: 0.0)")
        print(f"Std:  {np.std(results):.4f} (expected: 1.0)")
    else:
        print("Joblib backend not available (install with: pip install joblib)")
    print()
    
    # Example 6: Determinism test
    print("Example 6: Testing Determinism")
    print("-" * 60)
    backend = get_backend('sequential')
    
    results1 = backend.run_monte_carlo(
        simple_worker, 50, seed=42, verbose=False
    )
    results2 = backend.run_monte_carlo(
        simple_worker, 50, seed=42, verbose=False
    )
    
    if np.allclose(results1, results2):
        print("✓ Results are deterministic (identical with same seed)")
    else:
        print("✗ Results differ (unexpected!)")
    print()
    
    # Example 7: Wavelet coherence significance testing (conceptual)
    print("Example 7: Wavelet Coherence Significance (Conceptual)")
    print("-" * 60)
    print("This demonstrates how backends would be used for WCT significance:")
    print()
    print("# Pseudo-code for actual implementation:")
    print("backend = get_backend('joblib')")
    print("surrogate_max_coherence = backend.run_monte_carlo(")
    print("    wavelet_coherence_worker,")
    print("    n_simulations=1000,")
    print("    worker_args=(signal1, signal2),")
    print("    seed=42")
    print(")")
    print("# Then compute p-value from distribution of surrogate maxima")
    print()
    
    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

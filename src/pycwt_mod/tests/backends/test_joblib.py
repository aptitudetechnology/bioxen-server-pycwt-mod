"""
Tests for joblib parallel backend.
"""

import pytest
import numpy as np
from pycwt_mod.backends.joblib import JoblibBackend


# Check if joblib is available
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


@pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib not installed")
class TestJoblibBackend:
    """Tests for joblib backend (only run if joblib is installed)."""
    
    def test_joblib_initialization(self):
        """Test joblib backend initialization."""
        backend = JoblibBackend(n_jobs=2)
        
        assert backend.name == "Joblib"
        assert backend.n_jobs == 2
        assert backend.is_available() is True
    
    def test_joblib_basic_execution(self):
        """Test basic parallel execution."""
        backend = JoblibBackend(n_jobs=2)
        
        def worker(seed, x):
            rng = np.random.default_rng(seed)
            return rng.normal() + x
        
        results = backend.run_monte_carlo(
            worker,
            n_simulations=10,
            worker_args=(5.0,),
            seed=42,
            verbose=False
        )
        
        assert len(results) == 10
    
    def test_joblib_determinism(self):
        """Test that joblib backend is deterministic."""
        backend = JoblibBackend(n_jobs=2)
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            return rng.normal()
        
        results1 = backend.run_monte_carlo(worker, 100, seed=42, verbose=False)
        results2 = backend.run_monte_carlo(worker, 100, seed=42, verbose=False)
        
        assert np.allclose(results1, results2)
    
    def test_joblib_different_njobs(self):
        """Test with different n_jobs settings."""
        backend = JoblibBackend(n_jobs=1)
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            return rng.normal()
        
        results = backend.run_monte_carlo(worker, 10, seed=42, verbose=False)
        assert len(results) == 10
        
        # Test with all CPUs
        backend_all = JoblibBackend(n_jobs=-1)
        results_all = backend_all.run_monte_carlo(worker, 10, seed=42, verbose=False)
        
        # Should get same results (deterministic)
        assert np.allclose(results, results_all)
    
    def test_joblib_with_args_kwargs(self):
        """Test joblib with worker arguments."""
        backend = JoblibBackend(n_jobs=2)
        
        def worker(seed, a, b, c=0):
            rng = np.random.default_rng(seed)
            return rng.normal() + a + b + c
        
        results = backend.run_monte_carlo(
            worker,
            n_simulations=20,
            worker_args=(1.0, 2.0),
            worker_kwargs={'c': 3.0},
            seed=42,
            verbose=False
        )
        
        # Mean should be close to 6.0
        assert np.abs(np.mean(results) - 6.0) < 1.0
    
    def test_joblib_processes_vs_threads(self):
        """Test different joblib backends (processes vs threads)."""
        def worker(seed):
            rng = np.random.default_rng(seed)
            return rng.normal()
        
        backend_proc = JoblibBackend(n_jobs=2, prefer='processes')
        results_proc = backend_proc.run_monte_carlo(worker, 50, seed=42, verbose=False)
        
        backend_thread = JoblibBackend(n_jobs=2, prefer='threads')
        results_thread = backend_thread.run_monte_carlo(worker, 50, seed=42, verbose=False)
        
        # Both should give same results
        assert np.allclose(results_proc, results_thread)
    
    def test_joblib_override_njobs(self):
        """Test overriding n_jobs at runtime."""
        backend = JoblibBackend(n_jobs=2)
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            return rng.normal()
        
        # Override n_jobs at runtime
        results = backend.run_monte_carlo(
            worker, 10, seed=42, n_jobs=1, verbose=False
        )
        
        assert len(results) == 10
    
    def test_joblib_get_capabilities(self):
        """Test getting backend capabilities."""
        backend = JoblibBackend()
        capabilities = backend.get_capabilities()
        
        assert 'parallel' in capabilities
        assert 'deterministic' in capabilities
        assert any('joblib' in cap for cap in capabilities)
    
    def test_joblib_validate_config(self):
        """Test configuration validation."""
        backend = JoblibBackend()
        
        # Valid configs
        assert backend.validate_config(n_jobs=2) is True
        assert backend.validate_config(prefer='processes') is True
        assert backend.validate_config(prefer='threads') is True
        
        # Invalid configs
        with pytest.raises(ValueError):
            backend.validate_config(prefer='invalid')
        
        with pytest.raises(ValueError):
            backend.validate_config(n_jobs=0)
        
        with pytest.raises(ValueError):
            backend.validate_config(n_jobs='invalid')
    
    def test_joblib_large_workload(self):
        """Test with larger workload."""
        backend = JoblibBackend(n_jobs=-1)
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            return rng.normal()
        
        results = backend.run_monte_carlo(worker, 500, seed=42, verbose=False)
        
        assert len(results) == 500
        assert np.abs(np.mean(results)) < 0.1
        assert np.abs(np.std(results) - 1.0) < 0.1
    
    def test_joblib_error_handling(self):
        """Test error handling in parallel execution."""
        backend = JoblibBackend(n_jobs=2)
        
        def bad_worker(seed):
            if seed % 2 == 0:  # Some seeds will fail
                raise ValueError("Test error")
            return 1.0
        
        # Joblib should propagate the error
        with pytest.raises(ValueError):
            backend.run_monte_carlo(bad_worker, 5, seed=42, verbose=False)
    
    def test_joblib_complex_return_types(self):
        """Test with complex return types."""
        backend = JoblibBackend(n_jobs=2)
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            return {
                'scalar': rng.normal(),
                'array': rng.normal(size=3),
                'list': [1, 2, 3]
            }
        
        results = backend.run_monte_carlo(worker, 10, seed=42, verbose=False)
        
        assert len(results) == 10
        assert all(isinstance(r, dict) for r in results)


def test_joblib_not_available():
    """Test behavior when joblib is not available."""
    # This test should work whether joblib is installed or not
    backend = JoblibBackend()
    
    if not JOBLIB_AVAILABLE:
        assert backend.is_available() is False
    else:
        assert backend.is_available() is True


def test_joblib_import_error_handling():
    """Test error message when joblib not available."""
    if JOBLIB_AVAILABLE:
        pytest.skip("joblib is installed, can't test import error")
    
    backend = JoblibBackend()
    
    def worker(seed):
        return 1.0
    
    with pytest.raises(ImportError, match="joblib"):
        backend.run_monte_carlo(worker, 5, verbose=False)

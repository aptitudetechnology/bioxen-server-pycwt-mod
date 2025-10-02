"""
Tests for sequential backend.
"""

import pytest
import numpy as np
from pycwt_mod.backends.sequential import SequentialBackend


def test_sequential_backend_initialization():
    """Test sequential backend initialization."""
    backend = SequentialBackend()
    
    assert backend.name == "Sequential"
    assert backend.is_available() is True
    assert "deterministic" in backend.get_capabilities()


def test_sequential_backend_basic_execution():
    """Test basic sequential execution."""
    backend = SequentialBackend()
    
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
    assert all(isinstance(r, (int, float, np.number)) for r in results)


def test_sequential_determinism():
    """Test that sequential backend is deterministic."""
    backend = SequentialBackend()
    
    def worker(seed):
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    results1 = backend.run_monte_carlo(worker, 100, seed=42, verbose=False)
    results2 = backend.run_monte_carlo(worker, 100, seed=42, verbose=False)
    
    assert np.allclose(results1, results2)


def test_sequential_different_seeds():
    """Test that different seeds produce different results."""
    backend = SequentialBackend()
    
    def worker(seed):
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    results1 = backend.run_monte_carlo(worker, 100, seed=42, verbose=False)
    results2 = backend.run_monte_carlo(worker, 100, seed=123, verbose=False)
    
    # Results should be different
    assert not np.allclose(results1, results2)


def test_sequential_with_args():
    """Test sequential backend with worker arguments."""
    backend = SequentialBackend()
    
    def worker(seed, a, b, c):
        rng = np.random.default_rng(seed)
        return rng.normal() + a + b + c
    
    results = backend.run_monte_carlo(
        worker,
        n_simulations=10,
        worker_args=(1.0, 2.0, 3.0),
        seed=42,
        verbose=False
    )
    
    # Mean should be close to 6.0 (1+2+3)
    assert np.abs(np.mean(results) - 6.0) < 1.0


def test_sequential_with_kwargs():
    """Test sequential backend with keyword arguments."""
    backend = SequentialBackend()
    
    def worker(seed, x, y=0, z=0):
        rng = np.random.default_rng(seed)
        return rng.normal() + x + y + z
    
    results = backend.run_monte_carlo(
        worker,
        n_simulations=10,
        worker_args=(1.0,),
        worker_kwargs={'y': 2.0, 'z': 3.0},
        seed=42,
        verbose=False
    )
    
    # Mean should be close to 6.0 (1+2+3)
    assert np.abs(np.mean(results) - 6.0) < 1.0


def test_sequential_no_seed():
    """Test sequential backend without seed (random)."""
    backend = SequentialBackend()
    
    def worker(seed):
        # When no seed provided, we should still get results
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    results1 = backend.run_monte_carlo(worker, 10, verbose=False)
    results2 = backend.run_monte_carlo(worker, 10, verbose=False)
    
    # Without seed, results should be different
    assert len(results1) == 10
    assert len(results2) == 10


def test_sequential_single_simulation():
    """Test with single simulation."""
    backend = SequentialBackend()
    
    def worker(seed):
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    results = backend.run_monte_carlo(worker, 1, seed=42, verbose=False)
    
    assert len(results) == 1


def test_sequential_large_workload():
    """Test with larger number of simulations."""
    backend = SequentialBackend()
    
    def worker(seed):
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    results = backend.run_monte_carlo(worker, 1000, seed=42, verbose=False)
    
    assert len(results) == 1000
    # Mean should be close to 0, std close to 1
    assert np.abs(np.mean(results)) < 0.1
    assert np.abs(np.std(results) - 1.0) < 0.1


def test_sequential_verbose_output(capsys):
    """Test verbose output."""
    backend = SequentialBackend()
    
    def worker(seed):
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    backend.run_monte_carlo(worker, 10, seed=42, verbose=True)
    
    captured = capsys.readouterr()
    assert "Running" in captured.out
    assert "simulations" in captured.out


def test_sequential_error_handling():
    """Test error handling in worker function."""
    backend = SequentialBackend()
    
    def bad_worker(seed):
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        backend.run_monte_carlo(bad_worker, 5, verbose=False)


def test_sequential_complex_return_types():
    """Test with complex return types (arrays, dicts, etc)."""
    backend = SequentialBackend()
    
    def worker(seed):
        rng = np.random.default_rng(seed)
        return {
            'value': rng.normal(),
            'array': rng.normal(size=3)
        }
    
    results = backend.run_monte_carlo(worker, 5, seed=42, verbose=False)
    
    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)
    assert all('value' in r and 'array' in r for r in results)


def test_sequential_validate_config():
    """Test config validation."""
    backend = SequentialBackend()
    
    # Sequential backend should accept any config
    assert backend.validate_config() is True
    assert backend.validate_config(any_param=123) is True


def test_sequential_get_info():
    """Test getting backend info."""
    backend = SequentialBackend()
    info = backend.get_info()
    
    assert info['name'] == 'Sequential'
    assert info['available'] is True
    assert 'deterministic' in info['capabilities']
    assert 'low_memory' in info['capabilities']


def test_sequential_independent_seeds():
    """Test that each simulation gets independent seed."""
    backend = SequentialBackend()
    
    collected_seeds = []
    
    def worker(seed):
        collected_seeds.append(seed)
        rng = np.random.default_rng(seed)
        return rng.normal()
    
    backend.run_monte_carlo(worker, 10, seed=42, verbose=False)
    
    # All seeds should be different
    assert len(set(collected_seeds)) == 10

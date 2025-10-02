"""
Tests for backend base class and interface.
"""

import pytest
import numpy as np
from pycwt_mod.backends.base import MonteCarloBackend


class MockBackend(MonteCarloBackend):
    """Mock backend for testing."""
    
    def __init__(self, available=True):
        super().__init__()
        self.name = "Mock"
        self.description = "Mock backend for testing"
        self._available = available
        self.call_count = 0
    
    def run_monte_carlo(self, worker_func, n_simulations, worker_args=(),
                       worker_kwargs=None, seed=None, verbose=True, **backend_kwargs):
        self.call_count += 1
        if worker_kwargs is None:
            worker_kwargs = {}
        
        # Generate seeds
        if seed is not None:
            seed_sequence = np.random.SeedSequence(seed)
            child_seeds = seed_sequence.spawn(n_simulations)
            seeds = [s.generate_state(1)[0] for s in child_seeds]
        else:
            seeds = [None] * n_simulations
        
        # Execute
        results = []
        for sim_seed in seeds:
            result = worker_func(sim_seed, *worker_args, **worker_kwargs)
            results.append(result)
        
        return results
    
    def is_available(self):
        return self._available
    
    def get_capabilities(self):
        return ["mock", "testing"]


def test_backend_interface():
    """Test that backend interface is properly defined."""
    # Abstract methods should exist
    assert hasattr(MonteCarloBackend, 'run_monte_carlo')
    assert hasattr(MonteCarloBackend, 'is_available')
    
    # Utility methods should exist
    assert hasattr(MonteCarloBackend, 'get_info')
    assert hasattr(MonteCarloBackend, 'get_capabilities')
    assert hasattr(MonteCarloBackend, 'validate_config')


def test_mock_backend_basic():
    """Test basic mock backend functionality."""
    backend = MockBackend()
    
    assert backend.name == "Mock"
    assert backend.is_available() is True
    assert "mock" in backend.get_capabilities()


def test_backend_get_info():
    """Test backend info retrieval."""
    backend = MockBackend(available=True)
    info = backend.get_info()
    
    assert isinstance(info, dict)
    assert info['name'] == "Mock"
    assert info['available'] is True
    assert 'capabilities' in info


def test_backend_repr():
    """Test string representation."""
    backend = MockBackend(available=True)
    repr_str = repr(backend)
    
    assert "Mock" in repr_str
    assert "available" in repr_str


def test_backend_unavailable():
    """Test unavailable backend."""
    backend = MockBackend(available=False)
    
    assert backend.is_available() is False
    info = backend.get_info()
    assert info['available'] is False


def test_backend_validate_config():
    """Test config validation."""
    backend = MockBackend()
    
    # Default validation should pass
    assert backend.validate_config() is True
    assert backend.validate_config(some_param=42) is True


def test_mock_backend_execution():
    """Test mock backend can execute simulations."""
    backend = MockBackend()
    
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
    assert backend.call_count == 1


def test_mock_backend_determinism():
    """Test that mock backend produces deterministic results with seed."""
    backend = MockBackend()
    
    def worker(seed, x):
        rng = np.random.default_rng(seed)
        return rng.normal() + x
    
    # Run twice with same seed
    results1 = backend.run_monte_carlo(
        worker, 5, worker_args=(0.0,), seed=42, verbose=False
    )
    results2 = backend.run_monte_carlo(
        worker, 5, worker_args=(0.0,), seed=42, verbose=False
    )
    
    assert np.allclose(results1, results2)


def test_mock_backend_different_seeds():
    """Test that different seeds produce different results."""
    backend = MockBackend()
    
    def worker(seed, x):
        rng = np.random.default_rng(seed)
        return rng.normal() + x
    
    results1 = backend.run_monte_carlo(
        worker, 5, worker_args=(0.0,), seed=42, verbose=False
    )
    results2 = backend.run_monte_carlo(
        worker, 5, worker_args=(0.0,), seed=123, verbose=False
    )
    
    assert not np.allclose(results1, results2)


def test_backend_with_kwargs():
    """Test backend with worker kwargs."""
    backend = MockBackend()
    
    def worker(seed, x, y=0):
        rng = np.random.default_rng(seed)
        return rng.normal() + x + y
    
    results = backend.run_monte_carlo(
        worker,
        n_simulations=5,
        worker_args=(1.0,),
        worker_kwargs={'y': 2.0},
        seed=42,
        verbose=False
    )
    
    assert len(results) == 5
    # Results should be roughly around 3.0 (1.0 + 2.0 + noise)
    assert np.mean(results) > 2.0 and np.mean(results) < 4.0

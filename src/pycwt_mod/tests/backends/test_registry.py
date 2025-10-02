"""
Tests for backend registry system.
"""

import pytest
from pycwt_mod.backends.base import MonteCarloBackend
from pycwt_mod.backends.registry import BackendRegistry, get_backend, list_backends
from pycwt_mod.backends.sequential import SequentialBackend


class DummyBackend(MonteCarloBackend):
    """Dummy backend for testing registry."""
    
    def __init__(self):
        super().__init__()
        self.name = "Dummy"
    
    def run_monte_carlo(self, *args, **kwargs):
        return []
    
    def is_available(self):
        return True


class UnavailableBackend(MonteCarloBackend):
    """Backend that's never available."""
    
    def __init__(self):
        super().__init__()
        self.name = "Unavailable"
    
    def run_monte_carlo(self, *args, **kwargs):
        return []
    
    def is_available(self):
        return False


@pytest.fixture
def clean_registry():
    """Fixture to provide a clean registry for each test."""
    # Save original state
    original_backends = BackendRegistry._backends.copy()
    original_default = BackendRegistry._default_backend
    
    # Clear registry
    BackendRegistry.clear()
    
    yield BackendRegistry
    
    # Restore original state
    BackendRegistry._backends = original_backends
    BackendRegistry._default_backend = original_default


def test_register_backend(clean_registry):
    """Test backend registration."""
    clean_registry.register('dummy', DummyBackend)
    
    assert 'dummy' in clean_registry.list_all()


def test_register_duplicate_raises(clean_registry):
    """Test that registering duplicate backend raises error."""
    clean_registry.register('dummy', DummyBackend)
    
    with pytest.raises(ValueError, match="already registered"):
        clean_registry.register('dummy', DummyBackend)


def test_register_invalid_class_raises(clean_registry):
    """Test that registering invalid class raises error."""
    class NotABackend:
        pass
    
    with pytest.raises(TypeError, match="must inherit"):
        clean_registry.register('invalid', NotABackend)


def test_get_backend(clean_registry):
    """Test retrieving backend."""
    clean_registry.register('dummy', DummyBackend)
    
    backend = clean_registry.get('dummy')
    assert isinstance(backend, DummyBackend)


def test_get_nonexistent_backend_raises(clean_registry):
    """Test that getting nonexistent backend raises error."""
    with pytest.raises(ValueError, match="not found"):
        clean_registry.get('nonexistent')


def test_get_default_backend(clean_registry):
    """Test getting default backend."""
    clean_registry.register('dummy', DummyBackend)
    
    # First registered should be default
    backend = clean_registry.get()
    assert isinstance(backend, DummyBackend)


def test_set_default_backend(clean_registry):
    """Test setting default backend."""
    clean_registry.register('dummy1', DummyBackend)
    clean_registry.register('dummy2', DummyBackend)
    
    clean_registry.set_default('dummy2')
    assert clean_registry.get_default() == 'dummy2'


def test_set_invalid_default_raises(clean_registry):
    """Test that setting invalid default raises error."""
    with pytest.raises(ValueError, match="not registered"):
        clean_registry.set_default('nonexistent')


def test_list_all_backends(clean_registry):
    """Test listing all backends."""
    clean_registry.register('dummy1', DummyBackend)
    clean_registry.register('dummy2', DummyBackend)
    
    all_backends = clean_registry.list_all()
    assert 'dummy1' in all_backends
    assert 'dummy2' in all_backends


def test_list_available_backends(clean_registry):
    """Test listing only available backends."""
    clean_registry.register('available', DummyBackend)
    clean_registry.register('unavailable', UnavailableBackend)
    
    available = clean_registry.list_available()
    assert 'available' in available
    assert 'unavailable' not in available


def test_get_info(clean_registry):
    """Test getting backend info."""
    clean_registry.register('dummy', DummyBackend)
    
    info = clean_registry.get_info()
    assert 'dummy' in info
    assert info['dummy']['name'] == 'Dummy'
    assert info['dummy']['available'] is True


def test_clear_registry(clean_registry):
    """Test clearing registry."""
    clean_registry.register('dummy', DummyBackend)
    assert len(clean_registry.list_all()) > 0
    
    clean_registry.clear()
    assert len(clean_registry.list_all()) == 0
    assert clean_registry.get_default() is None


def test_convenience_functions():
    """Test convenience wrapper functions."""
    # These use the global registry, which should have built-in backends
    backends = list_backends()
    assert len(backends) > 0
    
    # Sequential should always be available
    assert 'sequential' in list_backends(available_only=True)
    
    backend = get_backend('sequential')
    assert isinstance(backend, SequentialBackend)


def test_get_backend_with_none():
    """Test getting default backend with None."""
    backend = get_backend(None)
    # Should return default backend (sequential for built-in registry)
    assert backend is not None


def test_registry_auto_registration():
    """Test that built-in backends are auto-registered."""
    # After module import, built-in backends should be registered
    backends = list_backends()
    
    assert 'sequential' in backends
    assert 'joblib' in backends
    assert 'dask' in backends
    assert 'gpu' in backends

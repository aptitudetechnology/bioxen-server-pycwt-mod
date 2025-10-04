"""
Pytest configuration and shared fixtures for PyCWT-mod API tests.

Provides:
- HTTPX client instances for remote server testing
- Mock data generators
- Backend mocks
- Database/Redis fixtures
"""

import pytest
import numpy as np
from typing import Dict, List, Generator
import json
import httpx


@pytest.fixture
def test_client() -> httpx.Client:
    """
    Create HTTP client for testing against remote server.
    
    Returns:
        httpx.Client instance configured for wavelet.local server
    """
    base_url = "http://wavelet.local:8000"
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        yield client


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API endpoints."""
    return "/api/v1"


@pytest.fixture
def sample_signal_short() -> np.ndarray:
    """
    Generate short test signal (100 points).
    
    Returns:
        NumPy array with synthetic sine wave + noise
    """
    t = np.linspace(0, 10, 100)
    signal = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100)
    return signal


@pytest.fixture
def sample_signal_medium() -> np.ndarray:
    """
    Generate medium test signal (1000 points).
    
    Returns:
        NumPy array with synthetic multi-frequency signal
    """
    t = np.linspace(0, 10, 1000)
    signal = (
        np.sin(2 * np.pi * 1 * t) +           # 1 Hz component
        0.5 * np.sin(2 * np.pi * 3 * t) +     # 3 Hz component
        0.2 * np.random.randn(1000)           # Noise
    )
    return signal


@pytest.fixture
def sample_signal_long() -> np.ndarray:
    """
    Generate long test signal (10000 points).
    
    Returns:
        NumPy array with complex multi-frequency signal
    """
    t = np.linspace(0, 100, 10000)
    signal = (
        np.sin(2 * np.pi * 0.5 * t) +         # 0.5 Hz slow oscillation
        0.7 * np.sin(2 * np.pi * 2 * t) +     # 2 Hz component
        0.3 * np.sin(2 * np.pi * 10 * t) +    # 10 Hz fast component
        0.1 * np.random.randn(10000)          # Noise
    )
    return signal


@pytest.fixture
def sample_signal_pair(sample_signal_medium) -> tuple:
    """
    Generate pair of correlated signals.
    
    Returns:
        Tuple of (signal1, signal2) with phase shift
    """
    t = np.linspace(0, 10, 1000)
    signal1 = sample_signal_medium
    signal2 = (
        np.sin(2 * np.pi * 1 * t + np.pi / 4) +  # Phase shifted
        0.5 * np.sin(2 * np.pi * 3 * t) +
        0.2 * np.random.randn(1000)
    )
    return signal1, signal2


@pytest.fixture
def cwt_request_basic(sample_signal_short) -> Dict:
    """
    Basic CWT request payload.
    
    Returns:
        Dictionary with minimal required parameters
    """
    return {
        "data": sample_signal_short.tolist(),
        "dt": 0.1
    }


@pytest.fixture
def cwt_request_full(sample_signal_medium) -> Dict:
    """
    Full CWT request payload with all parameters.
    
    Returns:
        Dictionary with all optional parameters specified
    """
    return {
        "data": sample_signal_medium.tolist(),
        "dt": 0.01,
        "dj": 0.125,
        "s0": -1,
        "J": -1,
        "mother": "morlet",
        "param": -1
    }


@pytest.fixture
def wct_request_basic(sample_signal_pair) -> Dict:
    """
    Basic WCT request payload.
    
    Returns:
        Dictionary with minimal required parameters
    """
    signal1, signal2 = sample_signal_pair
    return {
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.01
    }


@pytest.fixture
def wct_request_with_significance(sample_signal_pair) -> Dict:
    """
    WCT request with significance testing.
    
    Returns:
        Dictionary with Monte Carlo parameters
    """
    signal1, signal2 = sample_signal_pair
    return {
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.01,
        "significance_level": 0.95,
        "mc_count": 100,  # Reduced for testing speed
        "backend": "sequential"
    }


@pytest.fixture
def wct_request_fpga(sample_signal_short) -> Dict:
    """
    WCT request targeting FPGA backend.
    
    Returns:
        Dictionary configured for Tang Nano 9K
    """
    signal1 = sample_signal_short
    signal2 = np.sin(2 * np.pi * np.linspace(0, 10, 100) + np.pi/6)
    return {
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.1,
        "mc_count": 50,
        "backend": "elm11"
    }


@pytest.fixture
def xwt_request_basic(sample_signal_pair) -> Dict:
    """
    Basic XWT request payload.
    
    Returns:
        Dictionary with minimal required parameters
    """
    signal1, signal2 = sample_signal_pair
    return {
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.01
    }


@pytest.fixture
def benchmark_request_basic() -> Dict:
    """
    Basic benchmark request.
    
    Returns:
        Dictionary for small-scale benchmark
    """
    return {
        "signal_length": 500,
        "mc_count": 50,
        "backends": ["sequential"]
    }


@pytest.fixture
def benchmark_request_comprehensive() -> Dict:
    """
    Comprehensive benchmark request.
    
    Returns:
        Dictionary for multi-backend comparison
    """
    return {
        "signal_length": 1000,
        "mc_count": 100,
        "backends": ["sequential", "joblib"]
    }


@pytest.fixture
def batch_job_request(sample_signal_pair) -> Dict:
    """
    Batch job submission request.
    
    Returns:
        Dictionary with multiple wavelet tasks
    """
    signal1, signal2 = sample_signal_pair
    return {
        "tasks": [
            {
                "type": "wct",
                "signal1": signal1[:100].tolist(),
                "signal2": signal2[:100].tolist(),
                "dt": 0.01,
                "backend": "sequential"
            },
            {
                "type": "cwt",
                "data": signal1[:100].tolist(),
                "dt": 0.01
            },
            {
                "type": "xwt",
                "signal1": signal1[:100].tolist(),
                "signal2": signal2[:100].tolist(),
                "dt": 0.01
            }
        ]
    }


@pytest.fixture
def mock_backend_list() -> List[Dict]:
    """
    Mock backend list response.
    
    Returns:
        List of backend information dictionaries
    """
    return [
        {
            "name": "sequential",
            "available": True,
            "description": "Sequential Monte Carlo backend (single-core CPU)",
            "type": "SequentialBackend"
        },
        {
            "name": "joblib",
            "available": True,
            "description": "Parallel Monte Carlo backend using joblib (multi-core CPU)",
            "type": "JoblibBackend"
        },
        {
            "name": "elm11",
            "available": False,
            "description": "FPGA-accelerated backend using ELM11/Tang Nano 9K",
            "type": "ELM11Backend",
            "error": "Hardware not detected"
        }
    ]


@pytest.fixture
def mock_hardware_detection() -> Dict:
    """
    Mock hardware detection response.
    
    Returns:
        Dictionary with detected hardware information
    """
    return {
        "fpga": {
            "available": False,
            "device": "Tang Nano 9K",
            "status": "Not detected"
        },
        "embedded": {
            "available": False,
            "devices": []
        },
        "gpu": {
            "available": False,
            "type": "None"
        },
        "cpu": {
            "cores": 8,
            "available": True
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests requiring hardware (FPGA, GPU)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests for performance benchmarking"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on patterns."""
    for item in items:
        # Mark slow tests
        if "long" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark hardware tests
        if "fpga" in item.nodeid or "gpu" in item.nodeid or "elm11" in item.nodeid:
            item.add_marker(pytest.mark.hardware)
        
        # Mark integration tests
        if "test_integration" in item.nodeid or "test_workflow" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default for most)
        if "test_unit" in item.nodeid or "test_services" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# Helper functions for tests
def assert_valid_wavelet_response(response_data: Dict, expected_keys: List[str]):
    """
    Assert that wavelet response contains expected keys and valid data.
    
    Args:
        response_data: JSON response from API
        expected_keys: List of keys that must be present
    """
    for key in expected_keys:
        assert key in response_data, f"Missing key: {key}"
    
    # Check that arrays are lists (JSON serialized)
    if "wave" in response_data:
        assert isinstance(response_data["wave"], list)
    if "WCT" in response_data:
        assert isinstance(response_data["WCT"], list)
    if "freqs" in response_data:
        assert isinstance(response_data["freqs"], list)
        assert len(response_data["freqs"]) > 0
    
    # Check computation time is reasonable
    if "computation_time" in response_data:
        assert response_data["computation_time"] > 0
        assert response_data["computation_time"] < 300  # Max 5 minutes


def assert_backend_available(client: httpx.Client, backend_name: str):
    """
    Assert that a specific backend is available.
    
    Args:
        client: HTTPX client for API requests
        backend_name: Name of backend to check
    
    Raises:
        AssertionError if backend not available
    """
    response = client.get(f"/api/v1/backends/{backend_name}")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True, f"Backend {backend_name} not available"


def assert_valid_json(response) -> Dict:
    """
    Assert response is valid JSON and return parsed data.
    
    Args:
        response: FastAPI response object
    
    Returns:
        Parsed JSON data
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert data is not None
    return data


def generate_eeg_signal(duration: float, fs: float = 250.0) -> np.ndarray:
    """
    Generate realistic EEG-like signal.
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
    
    Returns:
        NumPy array with EEG-like signal
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Alpha band (8-13 Hz)
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    
    # Beta band (13-30 Hz)
    beta = 1.0 * np.sin(2 * np.pi * 20 * t)
    
    # Theta band (4-8 Hz)
    theta = 1.5 * np.sin(2 * np.pi * 6 * t)
    
    # Noise
    noise = 0.5 * np.random.randn(n_samples)
    
    return alpha + beta + theta + noise


def generate_circadian_signal(duration_hours: float, sampling_interval_hours: float = 1.0) -> np.ndarray:
    """
    Generate circadian rhythm signal.
    
    Args:
        duration_hours: Duration in hours
        sampling_interval_hours: Time between samples in hours
    
    Returns:
        NumPy array with circadian-like signal
    """
    n_samples = int(duration_hours / sampling_interval_hours)
    t = np.linspace(0, duration_hours, n_samples)
    
    # 24-hour rhythm
    circadian = np.sin(2 * np.pi * t / 24)
    
    # 12-hour harmonic
    harmonic = 0.3 * np.sin(2 * np.pi * t / 12)
    
    # Noise
    noise = 0.2 * np.random.randn(n_samples)
    
    return circadian + harmonic + noise

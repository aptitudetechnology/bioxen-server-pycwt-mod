"""
Pytest configuration for wishful BioXen computation API tests.

Provides fixtures for:
- HTTPX client instances for remote server testing
- Sample biological time-series data
- Circadian signals
- Metabolic data
"""

import pytest
import numpy as np
from typing import Dict, List
import httpx


@pytest.fixture
def test_client() -> httpx.Client:
    """
    Create HTTP client for testing against remote BioXen computation server.
    
    NOTE: This server doesn't exist yet! (Phase 6+)
    
    Returns:
        httpx.Client instance configured for bioxen.local server
    """
    base_url = "http://bioxen.local:8000"
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        yield client


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API endpoints."""
    return "/api/v1"


# ============================================================================
# Circadian Time Series Fixtures
# ============================================================================

@pytest.fixture
def circadian_time_series() -> Dict[str, List[float]]:
    """
    Generate synthetic circadian time series (48 hours, ~24h period).
    
    Returns:
        Dict with 'timestamps' (hours) and 'values' (normalized)
    """
    t = np.linspace(0, 48, 576)  # 5 min intervals for 48 hours
    period = 24.0
    values = np.sin(2 * np.pi * t / period) + 0.1 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


@pytest.fixture
def long_circadian_series() -> Dict[str, List[float]]:
    """
    Generate long circadian time series (96 hours, 4 cycles).
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 96, 1152)  # 5 min intervals for 96 hours
    period = 24.0
    values = (
        np.sin(2 * np.pi * t / period) + 
        0.05 * np.sin(2 * np.pi * t / 12.0) +  # Harmonic
        0.1 * np.random.randn(len(t))
    )
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


@pytest.fixture
def drifting_oscillation() -> Dict[str, List[float]]:
    """
    Generate oscillation with period drift.
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 96, 1152)
    # Period increases linearly from 24h to 26h over 96 hours
    instantaneous_period = 24.0 + (t / 96) * 2.0
    phase = np.cumsum(2 * np.pi / instantaneous_period) * (t[1] - t[0])
    values = np.sin(phase) + 0.1 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


@pytest.fixture
def decaying_oscillation() -> Dict[str, List[float]]:
    """
    Generate oscillation with exponential amplitude decay.
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 96, 1152)
    decay_rate = 0.01  # Decay constant
    amplitude = np.exp(-decay_rate * t)
    values = amplitude * np.sin(2 * np.pi * t / 24.0) + 0.05 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


# ============================================================================
# Stability Test Fixtures
# ============================================================================

@pytest.fixture
def stable_time_series() -> Dict[str, List[float]]:
    """
    Generate stable damped oscillation.
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 50, 500)
    damping = 0.1
    freq = 1.0
    values = np.exp(-damping * t) * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


@pytest.fixture
def unstable_time_series() -> Dict[str, List[float]]:
    """
    Generate unstable exponentially growing oscillation.
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 10, 100)
    growth_rate = 0.1
    freq = 1.0
    values = np.exp(growth_rate * t) * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


# ============================================================================
# Oscillation Test Fixtures
# ============================================================================

@pytest.fixture
def oscillating_signal() -> Dict[str, List[float]]:
    """
    Generate clean oscillating signal.
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 48, 576)
    values = np.sin(2 * np.pi * t / 24.0) + 0.05 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


@pytest.fixture
def noisy_oscillation() -> Dict[str, List[float]]:
    """
    Generate noisy oscillation (high noise).
    
    Returns:
        Dict with 'timestamps' and 'values'
    """
    t = np.linspace(0, 48, 576)
    values = np.sin(2 * np.pi * t / 24.0) + 0.5 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }


# ============================================================================
# Metabolic Data Fixtures
# ============================================================================

@pytest.fixture
def metabolic_time_series() -> Dict[str, List]:
    """
    Generate multi-metabolite time series.
    
    Returns:
        Dict with 'timestamps', 'atp', 'nadh', 'glucose'
    """
    t = np.linspace(0, 24, 288)  # 5 min intervals
    
    atp = 100 + 10 * np.sin(2 * np.pi * t / 2.0) + 2 * np.random.randn(len(t))
    nadh = 50 + 5 * np.sin(2 * np.pi * t / 3.0 + 0.5) + 1 * np.random.randn(len(t))
    glucose = 200 - 5 * t + 5 * np.random.randn(len(t))  # Declining
    
    return {
        "timestamps": t.tolist(),
        "atp": atp.tolist(),
        "nadh": nadh.tolist(),
        "glucose": glucose.tolist()
    }


@pytest.fixture
def gene_expression_series() -> Dict[str, List]:
    """
    Generate gene expression time series with circadian components.
    
    Returns:
        Dict with 'timestamps', 'values' dict containing gene_a and gene_r
    """
    t = np.linspace(0, 48, 576)  # 48 hours
    
    # Gene A (activator) - peaks at dawn
    gene_a = 100 + 50 * np.sin(2 * np.pi * t / 24.0) + 5 * np.random.randn(len(t))
    
    # Gene R (repressor) - peaks at dusk (phase shifted)
    gene_r = 80 + 40 * np.sin(2 * np.pi * t / 24.0 + np.pi) + 5 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": {
            "gene_a": gene_a.tolist(),
            "gene_r": gene_r.tolist()
        }
    }


@pytest.fixture
def noisy_oscillation() -> Dict[str, List[float]]:
    """
    Generate noisy oscillation for filter testing.
    
    Returns:
        Dict with 'timestamps' and 'values' (clean signal + noise)
    """
    t = np.linspace(0, 48, 576)
    clean_signal = np.sin(2 * np.pi * t / 24.0)
    noise = 0.3 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": (clean_signal + noise).tolist()
    }


# ============================================================================
# Analysis Request Fixtures
# ============================================================================

@pytest.fixture
def fourier_analysis_request(circadian_time_series) -> Dict:
    """
    Fourier analysis request payload.
    
    Returns:
        Dict with analysis parameters
    """
    return {
        "timestamps": circadian_time_series["timestamps"],
        "values": circadian_time_series["values"],
        "method": "fft"
    }


@pytest.fixture
def wavelet_analysis_request(circadian_time_series) -> Dict:
    """
    Wavelet analysis request payload.
    
    Returns:
        Dict with analysis parameters
    """
    return {
        "timestamps": circadian_time_series["timestamps"],
        "values": circadian_time_series["values"],
        "wavelet_type": "morlet",
        "scales": list(range(1, 100))
    }


@pytest.fixture
def parameter_sweep_request() -> Dict:
    """
    Parameter sweep request payload.
    
    Returns:
        Dict with sweep parameters
    """
    return {
        "parameter": "rate_constant_k1",
        "range": {
            "min": 0.1,
            "max": 1.0,
            "step": 0.1
        },
        "initial_conditions": {
            "atp": 100.0,
            "nadh": 50.0
        },
        "simulation_duration_hours": 24
    }


@pytest.fixture
def rate_tuning_request(circadian_time_series) -> Dict:
    """
    Rate constant tuning request.
    
    Returns:
        Dict with tuning parameters
    """
    return {
        "observed_data": {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"]
        },
        "target_period_hours": 24.0,
        "tunable_parameters": ["transcription_rate", "degradation_rate"],
        "optimization_method": "least_squares"
    }

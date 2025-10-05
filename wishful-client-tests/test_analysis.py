"""
Unit tests for BioXen signal analysis endpoints.

Tests:
- Fourier analysis (FFT, PSD, harmonics)
- Wavelet analysis (CWT, transients, coherence)
- Laplace analysis (poles, stability, frequency response)
- Z-Transform analysis (digital filters, discrete-time)
"""

import pytest
import numpy as np


class TestFourierAnalysis:
    """Test suite for POST /api/v1/analysis/fourier"""
    
    def test_fft_basic(self, test_client, api_base_url, circadian_time_series):
        """Test basic FFT computation."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "method": "fft"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/fourier",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "frequencies" in data
        assert "magnitudes" in data
        assert "phases" in data
        assert len(data["frequencies"]) == len(data["magnitudes"])
    
    def test_detect_circadian_period(self, test_client, api_base_url, long_circadian_series):
        """Detect circadian period using FFT."""
        request = {
            "timestamps": long_circadian_series["timestamps"],
            "values": long_circadian_series["values"],
            "method": "fft",
            "detect_peaks": True
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/fourier",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "dominant_period" in data
        # Should detect ~24h period
        assert 22 <= data["dominant_period"] <= 26
    
    def test_power_spectral_density(self, test_client, api_base_url, circadian_time_series):
        """Test PSD computation."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "method": "psd"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/fourier",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "frequencies" in data
        assert "power" in data
    
    def test_detect_harmonics(self, test_client, api_base_url, circadian_time_series):
        """Detect harmonic frequencies."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "detect_harmonics": True,
            "num_harmonics": 5
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/fourier",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "fundamental_frequency" in data
        assert "harmonics" in data
        assert len(data["harmonics"]) <= 5
    
    def test_fft_empty_data(self, test_client, api_base_url):
        """Test FFT with empty data."""
        invalid_request = {
            "timestamps": [],
            "values": [],
            "method": "fft"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/fourier",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422]
    
    @pytest.mark.slow
    def test_fft_large_dataset(self, test_client, api_base_url):
        """Test FFT with large dataset."""
        t = np.linspace(0, 200, 10000)
        values = np.sin(2 * np.pi * t / 24.0) + 0.1 * np.random.randn(len(t))
        
        request = {
            "timestamps": t.tolist(),
            "values": values.tolist(),
            "method": "fft"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/fourier",
            json=request
        )
        
        assert response.status_code == 200


class TestWaveletAnalysis:
    """Test suite for POST /api/v1/analysis/wavelet"""
    
    def test_cwt_basic(self, test_client, api_base_url, circadian_time_series):
        """Test basic continuous wavelet transform."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "wavelet_type": "morlet",
            "scales": list(range(1, 100))
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/wavelet",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "scales" in data
        assert "coefficients" in data
        assert "frequencies" in data
    
    def test_detect_transients(self, test_client, api_base_url):
        """Detect transient events using wavelets."""
        # Create signal with sudden change
        t = np.linspace(0, 48, 576)
        signal = np.sin(2 * np.pi * t / 24.0)
        signal[288:295] += 2.0  # Transient spike at 24h
        
        request = {
            "timestamps": t.tolist(),
            "values": signal.tolist(),
            "wavelet_type": "morlet",
            "detect_transients": True
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/wavelet",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "transients_detected" in data
        if data["transients_detected"]:
            assert "transient_times" in data
    
    def test_wavelet_coherence(self, test_client, api_base_url, circadian_time_series):
        """Test wavelet coherence between two signals."""
        # Create second signal with phase shift
        t = np.array(circadian_time_series["timestamps"])
        signal2 = np.sin(2 * np.pi * t / 24.0 + np.pi/4) + 0.1 * np.random.randn(len(t))
        
        request = {
            "signal1": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "signal2": {
                "timestamps": circadian_time_series["timestamps"],
                "values": signal2.tolist()
            },
            "wavelet_type": "morlet"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/wavelet-coherence",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "coherence" in data
        assert "phase_difference" in data
    
    def test_invalid_wavelet_type(self, test_client, api_base_url, circadian_time_series):
        """Test with invalid wavelet type."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "wavelet_type": "invalid_wavelet"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/wavelet",
            json=request
        )
        
        assert response.status_code in [400, 422]


class TestLaplaceAnalysis:
    """Test suite for POST /api/v1/analysis/laplace"""
    
    def test_laplace_poles_basic(self, test_client, api_base_url, stable_time_series):
        """Test Laplace pole-zero analysis."""
        request = {
            "timestamps": stable_time_series["timestamps"],
            "values": stable_time_series["values"]
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/laplace",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "poles" in data
        assert "zeros" in data
        
        # Check pole structure
        for pole in data["poles"]:
            assert "real" in pole
            assert "imaginary" in pole
    
    def test_stability_check(self, test_client, api_base_url, stable_time_series):
        """Test stability analysis via Laplace poles."""
        request = {
            "timestamps": stable_time_series["timestamps"],
            "values": stable_time_series["values"],
            "check_stability": True
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/laplace",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stable" in data
        assert data["stable"] is True
        
        # All poles should have negative real parts
        for pole in data["poles"]:
            assert pole["real"] < 0
    
    def test_unstable_system(self, test_client, api_base_url, unstable_time_series):
        """Test Laplace analysis on unstable system."""
        request = {
            "timestamps": unstable_time_series["timestamps"],
            "values": unstable_time_series["values"],
            "check_stability": True
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/laplace",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stable" in data
        assert data["stable"] is False
    
    def test_frequency_response(self, test_client, api_base_url, circadian_time_series):
        """Compute frequency response (Bode plot data)."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "frequency_range": {
                "min": 0.01,
                "max": 1.0,
                "num_points": 50
            }
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/laplace-frequency-response",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "frequencies" in data
        assert "magnitude_db" in data
        assert "phase_degrees" in data


class TestZTransformAnalysis:
    """Test suite for POST /api/v1/analysis/ztransform"""
    
    def test_ztransform_basic(self, test_client, api_base_url, circadian_time_series):
        """Test basic Z-transform computation."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"]
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/ztransform",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "z_values" in data
        assert "transform" in data
    
    def test_design_digital_filter(self, test_client, api_base_url):
        """Test digital filter design."""
        request = {
            "filter_type": "lowpass",
            "cutoff_frequency": 0.1,
            "order": 4,
            "sampling_rate": 1.0
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/design-filter",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "filter_coefficients" in data
        assert "b" in data["filter_coefficients"]
        assert "a" in data["filter_coefficients"]
    
    def test_apply_filter(self, test_client, api_base_url, noisy_oscillation):
        """Test applying digital filter to signal."""
        # First design filter
        filter_response = test_client.post(
            f"{api_base_url}/analysis/design-filter",
            json={
                "filter_type": "lowpass",
                "cutoff_frequency": 0.1,
                "order": 4
            }
        )
        filter_coeff = filter_response.json()["filter_coefficients"]
        
        # Apply filter
        request = {
            "timestamps": noisy_oscillation["timestamps"],
            "values": noisy_oscillation["values"],
            "filter_coefficients": filter_coeff
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/apply-filter",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "filtered_signal" in data
        assert "timestamps" in data
        assert len(data["filtered_signal"]) == len(data["timestamps"])
    
    def test_ztransform_stability(self, test_client, api_base_url, stable_time_series):
        """Test Z-transform stability check."""
        request = {
            "timestamps": stable_time_series["timestamps"],
            "values": stable_time_series["values"],
            "check_stability": True
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/ztransform",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stable" in data
        assert "poles" in data
        
        # Z-domain poles should be inside unit circle for stability
        if data["stable"]:
            for pole in data["poles"]:
                magnitude = (pole["real"]**2 + pole["imaginary"]**2)**0.5
                assert magnitude < 1.0


class TestMultiDomainAnalysis:
    """Test suite for POST /api/v1/analysis/multi-domain"""
    
    def test_four_lens_analysis(self, test_client, api_base_url, circadian_time_series):
        """Test comprehensive four-lens analysis."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "analyses": ["fourier", "wavelet", "laplace", "ztransform"]
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/multi-domain",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "fourier" in data
        assert "wavelet" in data
        assert "laplace" in data
        assert "ztransform" in data
        
        # Each lens should have key metrics
        assert "dominant_frequency" in data["fourier"]
        assert "scales" in data["wavelet"]
        assert "stable" in data["laplace"]
    
    def test_compare_time_frequency_domains(self, test_client, api_base_url, circadian_time_series):
        """Compare time-domain vs frequency-domain insights."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "comparison_mode": "time_vs_frequency"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/compare-domains",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "time_domain" in data
        assert "frequency_domain" in data
        assert "comparison_summary" in data
    
    def test_batch_analysis(self, test_client, api_base_url, metabolic_time_series):
        """Test batch analysis of multiple signals."""
        request = {
            "signals": [
                {
                    "id": "atp",
                    "timestamps": metabolic_time_series["timestamps"],
                    "values": metabolic_time_series["atp"]
                },
                {
                    "id": "nadh",
                    "timestamps": metabolic_time_series["timestamps"],
                    "values": metabolic_time_series["nadh"]
                }
            ],
            "analysis_type": "fourier"
        }
        
        response = test_client.post(
            f"{api_base_url}/analysis/batch",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 2
        for result in data["results"]:
            assert "signal_id" in result
            assert "frequencies" in result
            assert "magnitudes" in result

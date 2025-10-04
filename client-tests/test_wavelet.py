"""
Unit tests for wavelet analysis endpoints.

Tests:
- Continuous Wavelet Transform (CWT)
- Wavelet Coherence (WCT)
- Cross-Wavelet Transform (XWT)
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient


class TestContinuousWaveletTransform:
    """Test suite for POST /api/v1/wavelet/cwt"""
    
    def test_cwt_basic_request(self, test_client, api_base_url, cwt_request_basic):
        """Test basic CWT computation."""
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=cwt_request_basic
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required = ["wave", "scales", "freqs", "coi", "fft", "fftfreqs"]
        for field in required:
            assert field in data, f"Missing field: {field}"
        
        # Check data types
        assert isinstance(data["wave"], list)
        assert isinstance(data["scales"], list)
        assert len(data["scales"]) > 0
    
    def test_cwt_full_parameters(self, test_client, api_base_url, cwt_request_full):
        """Test CWT with all parameters specified."""
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=cwt_request_full
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "wave" in data
        assert "computation_time" in data
        assert data["computation_time"] > 0
    
    def test_cwt_invalid_data(self, test_client, api_base_url):
        """Test CWT with invalid input data."""
        invalid_request = {
            "data": [],  # Empty array
            "dt": 0.1
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=invalid_request
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_cwt_negative_dt(self, test_client, api_base_url, sample_signal_short):
        """Test CWT with negative time step."""
        invalid_request = {
            "data": sample_signal_short.tolist(),
            "dt": -0.1  # Invalid
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422]
    
    def test_cwt_missing_required_field(self, test_client, api_base_url):
        """Test CWT with missing required field."""
        incomplete_request = {
            "data": [1, 2, 3, 4, 5]
            # Missing 'dt'
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=incomplete_request
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.slow
    def test_cwt_large_signal(self, test_client, api_base_url, sample_signal_long):
        """Test CWT with large signal."""
        request = {
            "data": sample_signal_long.tolist(),
            "dt": 0.01
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "computation_time" in data
        assert data["computation_time"] < 60  # Should complete within 1 minute
    
    @pytest.mark.parametrize("mother", ["morlet", "paul", "dog"])
    def test_cwt_different_wavelets(self, test_client, api_base_url, sample_signal_short, mother):
        """Test CWT with different mother wavelets."""
        request = {
            "data": sample_signal_short.tolist(),
            "dt": 0.1,
            "mother": mother
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=request
        )
        
        assert response.status_code == 200


class TestWaveletCoherence:
    """Test suite for POST /api/v1/wavelet/wct"""
    
    def test_wct_basic_request(self, test_client, api_base_url, wct_request_basic):
        """Test basic WCT computation."""
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=wct_request_basic
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required = ["WCT", "aWCT", "coi", "freqs", "backend_used"]
        for field in required:
            assert field in data, f"Missing field: {field}"
        
        # Validate coherence values (should be 0-1)
        assert isinstance(data["WCT"], list)
        assert isinstance(data["backend_used"], str)
    
    def test_wct_with_significance(self, test_client, api_base_url, wct_request_with_significance):
        """Test WCT with significance testing."""
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=wct_request_with_significance
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "signif" in data
        assert isinstance(data["signif"], list)
        assert data["backend_used"] == "sequential"
    
    def test_wct_different_backends(self, test_client, api_base_url, sample_signal_pair):
        """Test WCT with different backends."""
        signal1, signal2 = sample_signal_pair
        
        backends = ["sequential", "joblib"]
        
        for backend in backends:
            request = {
                "signal1": signal1[:100].tolist(),
                "signal2": signal2[:100].tolist(),
                "dt": 0.01,
                "backend": backend,
                "mc_count": 50
            }
            
            response = test_client.post(
                f"{api_base_url}/wavelet/wct",
                json=request
            )
            
            # Sequential should always work, joblib depends on installation
            if backend == "sequential":
                assert response.status_code == 200
                data = response.json()
                assert data["backend_used"] == backend
    
    @pytest.mark.hardware
    def test_wct_fpga_backend(self, test_client, api_base_url, wct_request_fpga):
        """Test WCT with FPGA backend."""
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=wct_request_fpga
        )
        
        # Will work if FPGA available, otherwise should fallback or error
        if response.status_code == 200:
            data = response.json()
            # FPGA was used or fell back to another backend
            assert "backend_used" in data
    
    def test_wct_mismatched_signal_lengths(self, test_client, api_base_url):
        """Test WCT with different length signals."""
        invalid_request = {
            "signal1": [1, 2, 3, 4, 5],
            "signal2": [1, 2, 3],  # Different length
            "dt": 0.1
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=invalid_request
        )
        
        assert response.status_code == 400  # Bad request
        data = response.json()
        assert "detail" in data
    
    def test_wct_computation_time_reported(self, test_client, api_base_url, wct_request_basic):
        """Test WCT reports computation time."""
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=wct_request_basic
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "computation_time" in data
        assert data["computation_time"] > 0
        assert isinstance(data["computation_time"], (int, float))
    
    @pytest.mark.parametrize("mc_count", [10, 50, 100, 300])
    def test_wct_different_mc_counts(self, test_client, api_base_url, sample_signal_pair, mc_count):
        """Test WCT with different Monte Carlo iteration counts."""
        signal1, signal2 = sample_signal_pair
        
        request = {
            "signal1": signal1[:100].tolist(),
            "signal2": signal2[:100].tolist(),
            "dt": 0.01,
            "mc_count": mc_count,
            "backend": "sequential"
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=request
        )
        
        assert response.status_code == 200
    
    @pytest.mark.slow
    def test_wct_performance_scales(self, test_client, api_base_url):
        """Test WCT performance with different signal lengths."""
        times = []
        
        for n in [100, 500, 1000]:
            signal1 = np.sin(np.linspace(0, 10, n))
            signal2 = np.cos(np.linspace(0, 10, n))
            
            request = {
                "signal1": signal1.tolist(),
                "signal2": signal2.tolist(),
                "dt": 0.01,
                "mc_count": 50,
                "backend": "sequential"
            }
            
            response = test_client.post(
                f"{api_base_url}/wavelet/wct",
                json=request
            )
            
            assert response.status_code == 200
            data = response.json()
            times.append(data["computation_time"])
        
        # Computation time should increase with signal length
        assert times[1] > times[0]
        assert times[2] > times[1]


class TestCrossWaveletTransform:
    """Test suite for POST /api/v1/wavelet/xwt"""
    
    def test_xwt_basic_request(self, test_client, api_base_url, xwt_request_basic):
        """Test basic XWT computation."""
        response = test_client.post(
            f"{api_base_url}/wavelet/xwt",
            json=xwt_request_basic
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required = ["xwt", "phase", "coi", "freqs"]
        for field in required:
            assert field in data, f"Missing field: {field}"
    
    def test_xwt_mismatched_lengths(self, test_client, api_base_url):
        """Test XWT with mismatched signal lengths."""
        invalid_request = {
            "signal1": [1, 2, 3, 4, 5],
            "signal2": [1, 2, 3],  # Different length
            "dt": 0.1
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/xwt",
            json=invalid_request
        )
        
        assert response.status_code == 400
    
    def test_xwt_phase_angles(self, test_client, api_base_url, sample_signal_pair):
        """Test XWT returns valid phase angles."""
        signal1, signal2 = sample_signal_pair
        
        request = {
            "signal1": signal1[:100].tolist(),
            "signal2": signal2[:100].tolist(),
            "dt": 0.01
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/xwt",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Phase should be present
        assert "phase" in data
        assert isinstance(data["phase"], list)


class TestWaveletValidation:
    """Test suite for input validation."""
    
    def test_empty_signal_rejected(self, test_client, api_base_url):
        """Test empty signals are rejected."""
        invalid_requests = [
            {
                "data": [],
                "dt": 0.1
            },
            {
                "signal1": [],
                "signal2": [1, 2, 3],
                "dt": 0.1
            }
        ]
        
        for request in invalid_requests:
            if "data" in request:
                endpoint = "cwt"
            else:
                endpoint = "wct"
            
            response = test_client.post(
                f"{api_base_url}/wavelet/{endpoint}",
                json=request
            )
            
            assert response.status_code in [400, 422]
    
    def test_nan_values_handled(self, test_client, api_base_url):
        """Test NaN values in signals are handled."""
        signal = [1.0, 2.0, float('nan'), 4.0, 5.0]
        
        request = {
            "data": signal,
            "dt": 0.1
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=request
        )
        
        # Should either handle gracefully or return error
        assert response.status_code in [200, 400, 422, 500]
    
    def test_invalid_json(self, test_client, api_base_url):
        """Test invalid JSON is rejected."""
        response = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

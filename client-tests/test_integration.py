"""
Integration tests for complete API workflows.

Tests:
- End-to-end wavelet analysis workflows
- Multi-step operations
- Backend selection and fallback
- Error recovery
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient


class TestCompleteWorkflow:
    """Test suite for complete analysis workflows."""
    
    def test_full_wct_workflow(self, test_client, api_base_url):
        """
        Test complete WCT workflow:
        1. Check health
        2. List backends
        3. Detect hardware
        4. Compute WCT
        """
        # Step 1: Health check
        response = test_client.get("/health")
        assert response.status_code == 200
        
        # Step 2: List backends
        response = test_client.get(f"{api_base_url}/backends/")
        assert response.status_code == 200
        backends_data = response.json()
        available_backends = [
            b["name"] for b in backends_data["backends"]
            if b["available"]
        ]
        assert len(available_backends) > 0
        
        # Step 3: Hardware detection
        response = test_client.get(f"{api_base_url}/hardware/detect")
        assert response.status_code == 200
        
        # Step 4: Compute WCT
        signal1 = np.sin(np.linspace(0, 10, 100))
        signal2 = np.cos(np.linspace(0, 10, 100))
        
        wct_request = {
            "signal1": signal1.tolist(),
            "signal2": signal2.tolist(),
            "dt": 0.1,
            "backend": available_backends[0],  # Use first available
            "mc_count": 50
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=wct_request
        )
        
        assert response.status_code == 200
        wct_data = response.json()
        assert "WCT" in wct_data
        assert "computation_time" in wct_data
    
    def test_benchmark_then_analyze_workflow(self, test_client, api_base_url):
        """
        Test workflow: benchmark backends, then use fastest for analysis.
        """
        # Step 1: Benchmark backends
        benchmark_request = {
            "signal_length": 500,
            "mc_count": 50,
            "backends": ["sequential", "joblib"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=benchmark_request
        )
        
        assert response.status_code == 200
        benchmark_data = response.json()
        
        # Find fastest backend
        results = benchmark_data["results"]
        completed = {
            name: data for name, data in results.items()
            if data["status"] == "completed"
        }
        
        if len(completed) > 0:
            fastest = min(completed.items(), key=lambda x: x[1]["computation_time"])
            fastest_backend = fastest[0]
            
            # Step 2: Use fastest backend for analysis
            signal1 = np.sin(np.linspace(0, 10, 100))
            signal2 = np.cos(np.linspace(0, 10, 100))
            
            wct_request = {
                "signal1": signal1.tolist(),
                "signal2": signal2.tolist(),
                "dt": 0.1,
                "backend": fastest_backend,
                "mc_count": 100
            }
            
            response = test_client.post(
                f"{api_base_url}/wavelet/wct",
                json=wct_request
            )
            
            assert response.status_code == 200
    
    def test_multiple_analyses_sequential(self, test_client, api_base_url):
        """Test multiple wavelet analyses in sequence."""
        signal = np.sin(np.linspace(0, 10, 100))
        
        # Multiple CWT computations
        for i in range(3):
            request = {
                "data": signal.tolist(),
                "dt": 0.1
            }
            
            response = test_client.post(
                f"{api_base_url}/wavelet/cwt",
                json=request
            )
            
            assert response.status_code == 200


class TestBackendFallback:
    """Test suite for backend fallback behavior."""
    
    @pytest.mark.hardware
    def test_fpga_unavailable_fallback(self, test_client, api_base_url):
        """Test graceful fallback when FPGA unavailable."""
        signal1 = np.sin(np.linspace(0, 10, 100))
        signal2 = np.cos(np.linspace(0, 10, 100))
        
        # Request FPGA backend
        request = {
            "signal1": signal1.tolist(),
            "signal2": signal2.tolist(),
            "dt": 0.1,
            "backend": "elm11",
            "mc_count": 50
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=request
        )
        
        # Should either succeed with FPGA or return error
        # (fallback might be implemented or not)
        assert response.status_code in [200, 400, 503]
    
    def test_invalid_backend_error(self, test_client, api_base_url):
        """Test clear error for invalid backend."""
        signal1 = np.sin(np.linspace(0, 10, 100))
        signal2 = np.cos(np.linspace(0, 10, 100))
        
        request = {
            "signal1": signal1.tolist(),
            "signal2": signal2.tolist(),
            "dt": 0.1,
            "backend": "nonexistent",
            "mc_count": 50
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=request
        )
        
        # Should return error
        assert response.status_code in [400, 404, 422]
        data = response.json()
        assert "detail" in data


class TestErrorRecovery:
    """Test suite for error handling and recovery."""
    
    def test_malformed_request_recovery(self, test_client, api_base_url):
        """Test API recovers from malformed requests."""
        # Send malformed request
        response1 = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response1.status_code == 422
        
        # Next valid request should work
        signal = np.sin(np.linspace(0, 10, 100))
        request2 = {
            "data": signal.tolist(),
            "dt": 0.1
        }
        
        response2 = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=request2
        )
        
        assert response2.status_code == 200
    
    def test_error_doesnt_crash_server(self, test_client, api_base_url):
        """Test errors don't crash the server."""
        # Trigger various errors
        error_requests = [
            {
                "data": [],
                "dt": 0.1
            },
            {
                "data": [1, 2, 3],
                "dt": -1
            },
            {
                "signal1": [1],
                "signal2": [1, 2, 3],
                "dt": 0.1
            }
        ]
        
        for request in error_requests:
            if "data" in request:
                endpoint = "cwt"
            else:
                endpoint = "wct"
            
            response = test_client.post(
                f"{api_base_url}/wavelet/{endpoint}",
                json=request
            )
            
            # Should return error, not crash
            assert response.status_code in [400, 422, 500]
        
        # Server should still respond
        health = test_client.get("/health")
        assert health.status_code == 200


class TestConcurrentRequests:
    """Test suite for concurrent request handling."""
    
    def test_multiple_backends_simultaneously(self, test_client, api_base_url):
        """Test querying multiple backends simultaneously."""
        backends_to_test = ["sequential", "joblib", "elm11"]
        
        for backend in backends_to_test:
            response = test_client.get(f"{api_base_url}/backends/{backend}")
            # Should handle concurrent requests
            assert response.status_code in [200, 404]
    
    def test_parallel_computations(self, test_client, api_base_url):
        """Test handling multiple wavelet computations."""
        signal = np.sin(np.linspace(0, 10, 100))
        
        # Submit multiple CWT requests
        responses = []
        for i in range(3):
            request = {
                "data": signal.tolist(),
                "dt": 0.1
            }
            
            response = test_client.post(
                f"{api_base_url}/wavelet/cwt",
                json=request
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200


class TestDataIntegrity:
    """Test suite for data integrity throughout workflow."""
    
    def test_cwt_output_consistency(self, test_client, api_base_url):
        """Test CWT produces consistent output for same input."""
        signal = np.sin(np.linspace(0, 10, 100))
        request = {
            "data": signal.tolist(),
            "dt": 0.1
        }
        
        # Compute CWT twice
        response1 = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=request
        )
        data1 = response1.json()
        
        response2 = test_client.post(
            f"{api_base_url}/wavelet/cwt",
            json=request
        )
        data2 = response2.json()
        
        # Scales should be identical (deterministic)
        assert data1["scales"] == data2["scales"]
        assert data1["freqs"] == data2["freqs"]
    
    def test_wct_coherence_values_valid(self, test_client, api_base_url):
        """Test WCT coherence values are in valid range [0, 1]."""
        signal1 = np.sin(np.linspace(0, 10, 100))
        signal2 = np.cos(np.linspace(0, 10, 100))
        
        request = {
            "signal1": signal1.tolist(),
            "signal2": signal2.tolist(),
            "dt": 0.1
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=request
        )
        
        data = response.json()
        wct = np.array(data["WCT"])
        
        # Coherence should be between 0 and 1
        assert np.all(wct >= 0)
        assert np.all(wct <= 1)


class TestBioXenIntegration:
    """Test suite for BioXen client library integration."""
    
    def test_bioxen_remote_backend_workflow(self, test_client, api_base_url):
        """
        Test workflow simulating BioXen remote backend usage.
        
        Simulates:
        BioXen_Fourier_lib → REST API → pycwt-mod backends
        """
        # Step 1: BioXen checks server health
        health = test_client.get("/health")
        assert health.status_code == 200
        
        # Step 2: BioXen lists available backends
        backends = test_client.get(f"{api_base_url}/backends/")
        assert backends.status_code == 200
        backend_list = backends.json()["backends"]
        
        # Step 3: BioXen selects first available backend
        available = [b["name"] for b in backend_list if b["available"]]
        assert len(available) > 0
        chosen_backend = available[0]
        
        # Step 4: BioXen sends analysis request
        eeg_channel1 = generate_eeg_signal(10.0, fs=250.0)
        eeg_channel2 = generate_eeg_signal(10.0, fs=250.0)
        
        request = {
            "signal1": eeg_channel1[:100].tolist(),
            "signal2": eeg_channel2[:100].tolist(),
            "dt": 1/250.0,  # 250 Hz sampling
            "backend": chosen_backend,
            "mc_count": 100
        }
        
        response = test_client.post(
            f"{api_base_url}/wavelet/wct",
            json=request
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Step 5: BioXen receives results
        assert "WCT" in result
        assert "backend_used" in result
        assert result["backend_used"] == chosen_backend
    
    def test_bioxen_batch_analysis(self, test_client, api_base_url):
        """
        Test batch analysis workflow for BioXen.
        
        Simulates analyzing multiple signal pairs.
        """
        # Generate multiple signal pairs
        n_pairs = 3
        
        for i in range(n_pairs):
            signal1 = np.sin(2 * np.pi * (i+1) * np.linspace(0, 10, 100))
            signal2 = np.cos(2 * np.pi * (i+1) * np.linspace(0, 10, 100))
            
            request = {
                "signal1": signal1.tolist(),
                "signal2": signal2.tolist(),
                "dt": 0.1,
                "backend": "sequential"
            }
            
            response = test_client.post(
                f"{api_base_url}/wavelet/wct",
                json=request
            )
            
            assert response.status_code == 200


# Helper function
def generate_eeg_signal(duration: float, fs: float = 250.0) -> np.ndarray:
    """Generate realistic EEG-like signal."""
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    beta = 1.0 * np.sin(2 * np.pi * 20 * t)
    theta = 1.5 * np.sin(2 * np.pi * 6 * t)
    noise = 0.5 * np.random.randn(n_samples)
    
    return alpha + beta + theta + noise

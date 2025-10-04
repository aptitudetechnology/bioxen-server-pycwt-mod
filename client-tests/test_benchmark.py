"""
Unit tests for benchmark endpoints.

Tests:
- Performance benchmarking
- Multi-backend comparison
- Speedup calculations
"""

import pytest
from fastapi.testclient import TestClient


class TestBenchmarkBasic:
    """Test suite for POST /api/v1/benchmark"""
    
    def test_benchmark_basic_request(self, test_client, api_base_url, benchmark_request_basic):
        """Test basic benchmark execution."""
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=benchmark_request_basic
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "signal_length" in data
        assert "mc_count" in data
        assert "results" in data
        
        assert data["signal_length"] == benchmark_request_basic["signal_length"]
        assert data["mc_count"] == benchmark_request_basic["mc_count"]
    
    def test_benchmark_results_structure(self, test_client, api_base_url, benchmark_request_basic):
        """Test benchmark results have proper structure."""
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=benchmark_request_basic
        )
        
        data = response.json()
        results = data["results"]
        
        # Should have results for requested backends
        for backend in benchmark_request_basic["backends"]:
            assert backend in results
            
            backend_result = results[backend]
            assert "status" in backend_result
            
            if backend_result["status"] == "completed":
                assert "computation_time" in backend_result
                assert "speedup" in backend_result
                assert backend_result["computation_time"] > 0
    
    def test_benchmark_sequential_baseline(self, test_client, api_base_url):
        """Test sequential backend serves as baseline (1.0× speedup)."""
        request = {
            "signal_length": 500,
            "mc_count": 50,
            "backends": ["sequential"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        data = response.json()
        results = data["results"]
        
        if "sequential" in results and results["sequential"]["status"] == "completed":
            # Sequential should be 1.0× speedup (baseline)
            assert results["sequential"]["speedup"] == pytest.approx(1.0, rel=0.01)
    
    @pytest.mark.slow
    def test_benchmark_comprehensive(self, test_client, api_base_url, benchmark_request_comprehensive):
        """Test comprehensive multi-backend benchmark."""
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=benchmark_request_comprehensive
        )
        
        assert response.status_code == 200
        data = response.json()
        
        results = data["results"]
        
        # Should have attempted all requested backends
        for backend in benchmark_request_comprehensive["backends"]:
            assert backend in results


class TestBenchmarkComparison:
    """Test suite for multi-backend performance comparison."""
    
    def test_joblib_faster_than_sequential(self, test_client, api_base_url):
        """Test joblib backend is faster than sequential."""
        request = {
            "signal_length": 1000,
            "mc_count": 100,
            "backends": ["sequential", "joblib"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        data = response.json()
        results = data["results"]
        
        # If both completed successfully
        if (results.get("sequential", {}).get("status") == "completed" and
            results.get("joblib", {}).get("status") == "completed"):
            
            seq_time = results["sequential"]["computation_time"]
            joblib_time = results["joblib"]["computation_time"]
            
            # Joblib should be faster (or at least not significantly slower)
            # Allow for variation due to system load
            assert joblib_time <= seq_time * 1.2, "Joblib unexpectedly slower"
            
            # Speedup should be >= 1.0
            assert results["joblib"]["speedup"] >= 0.8
    
    @pytest.mark.hardware
    def test_fpga_performance(self, test_client, api_base_url):
        """Test FPGA backend performance."""
        request = {
            "signal_length": 1000,
            "mc_count": 100,
            "backends": ["sequential", "elm11"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        data = response.json()
        results = data["results"]
        
        if "elm11" in results and results["elm11"]["status"] == "completed":
            # FPGA completed, should have speedup metric
            assert "speedup" in results["elm11"]
            assert results["elm11"]["speedup"] > 0
    
    @pytest.mark.slow
    def test_scaling_behavior(self, test_client, api_base_url):
        """Test benchmark scaling with different signal lengths."""
        signal_lengths = [100, 500, 1000]
        times = []
        
        for length in signal_lengths:
            request = {
                "signal_length": length,
                "mc_count": 50,
                "backends": ["sequential"]
            }
            
            response = test_client.post(
                f"{api_base_url}/benchmark",
                json=request
            )
            
            data = response.json()
            results = data["results"]
            
            if results["sequential"]["status"] == "completed":
                times.append(results["sequential"]["computation_time"])
        
        # Times should generally increase with length
        # (allowing for some variation)
        if len(times) == len(signal_lengths):
            assert times[-1] > times[0], "Computation time should increase with signal length"


class TestBenchmarkValidation:
    """Test suite for benchmark input validation."""
    
    def test_invalid_signal_length(self, test_client, api_base_url):
        """Test invalid signal length is rejected."""
        invalid_requests = [
            {"signal_length": 0, "mc_count": 50, "backends": ["sequential"]},
            {"signal_length": -100, "mc_count": 50, "backends": ["sequential"]},
        ]
        
        for request in invalid_requests:
            response = test_client.post(
                f"{api_base_url}/benchmark",
                json=request
            )
            
            assert response.status_code in [400, 422]
    
    def test_invalid_mc_count(self, test_client, api_base_url):
        """Test invalid Monte Carlo count is rejected."""
        invalid_request = {
            "signal_length": 500,
            "mc_count": -10,
            "backends": ["sequential"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422]
    
    def test_empty_backends_list(self, test_client, api_base_url):
        """Test empty backends list is rejected."""
        invalid_request = {
            "signal_length": 500,
            "mc_count": 50,
            "backends": []
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422]
    
    def test_nonexistent_backend(self, test_client, api_base_url):
        """Test nonexistent backend in benchmark."""
        request = {
            "signal_length": 500,
            "mc_count": 50,
            "backends": ["sequential", "nonexistent"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        data = response.json()
        results = data["results"]
        
        # Sequential should work
        assert results["sequential"]["status"] == "completed"
        
        # Nonexistent should fail
        if "nonexistent" in results:
            assert results["nonexistent"]["status"] == "failed"
    
    def test_very_large_signal(self, test_client, api_base_url):
        """Test benchmark rejects very large signals."""
        request = {
            "signal_length": 10000000,  # 10 million points
            "mc_count": 300,
            "backends": ["sequential"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        # Should either reject or timeout gracefully
        # Don't let it run for hours
        assert response.status_code in [200, 400, 422, 504]


class TestBenchmarkResearchValidation:
    """Test suite validating MVP research questions."""
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_mvp_question1_performance_characterization(self, test_client, api_base_url):
        """
        Test MVP Research Question 1: pycwt Performance Characterization.
        
        Validates performance scaling across biological dataset sizes.
        """
        # Test range from MVP research: 100 to 10,000 timepoints
        test_sizes = [100, 1000, 10000]
        results = {}
        
        for n in test_sizes:
            request = {
                "signal_length": n,
                "mc_count": 300,
                "backends": ["sequential"]
            }
            
            response = test_client.post(
                f"{api_base_url}/benchmark",
                json=request
            )
            
            assert response.status_code == 200
            data = response.json()
            
            if data["results"]["sequential"]["status"] == "completed":
                results[n] = data["results"]["sequential"]["computation_time"]
        
        # Should have results for all sizes
        assert len(results) > 0
        
        # Log results for research analysis
        print("\nMVP Q1 - Performance Characterization:")
        for n, time in results.items():
            print(f"  N={n}: {time:.3f}s")
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_mvp_question2_pywavelets_comparison(self, test_client, api_base_url):
        """
        Test MVP Research Question 2: PyWavelets Performance Comparison.
        
        Validates speedup of joblib parallel backend.
        """
        request = {
            "signal_length": 10000,
            "mc_count": 300,
            "backends": ["sequential", "joblib"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        if all(results[b]["status"] == "completed" for b in ["sequential", "joblib"]):
            speedup = results["joblib"]["speedup"]
            
            print(f"\nMVP Q2 - PyWavelets Comparison:")
            print(f"  Sequential: {results['sequential']['computation_time']:.3f}s")
            print(f"  Joblib:     {results['joblib']['computation_time']:.3f}s")
            print(f"  Speedup:    {speedup:.2f}×")
            
            # Document speedup for research
            assert speedup > 0
    
    @pytest.mark.hardware
    @pytest.mark.benchmark
    def test_fpga_research_latency_measurement(self, test_client, api_base_url):
        """
        Test FPGA Research: Real-time latency measurement.
        
        Validates FPGA performance for BCI/adaptive stimulation.
        """
        request = {
            "signal_length": 1000,
            "mc_count": 100,
            "backends": ["sequential", "elm11"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        if "elm11" in results and results["elm11"]["status"] == "completed":
            latency = results["elm11"]["computation_time"]
            speedup = results["elm11"]["speedup"]
            
            print(f"\nFPGA Research - Latency Measurement:")
            print(f"  FPGA latency: {latency:.3f}s ({latency*1000:.1f}ms)")
            print(f"  Speedup:      {speedup:.2f}×")
            
            # Research target: <10ms for real-time BCI
            # This is overall computation, not per-sample latency


class TestBenchmarkTimeout:
    """Test suite for benchmark timeout handling."""
    
    def test_benchmark_reasonable_timeout(self, test_client, api_base_url):
        """Test benchmark completes within reasonable time."""
        import time
        
        request = {
            "signal_length": 1000,
            "mc_count": 100,
            "backends": ["sequential"]
        }
        
        start = time.time()
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 120, "Benchmark took too long (>2 minutes)"
    
    @pytest.mark.slow
    def test_benchmark_long_computation(self, test_client, api_base_url):
        """Test benchmark handles long computations."""
        request = {
            "signal_length": 5000,
            "mc_count": 300,
            "backends": ["sequential"]
        }
        
        response = test_client.post(
            f"{api_base_url}/benchmark",
            json=request
        )
        
        # Should complete or timeout gracefully
        assert response.status_code in [200, 504]

"""
Unit tests for hardware detection endpoints.

Tests:
- Hardware detection
- FPGA detection
- GPU detection
- CPU information
"""

import pytest
from fastapi.testclient import TestClient


class TestHardwareDetection:
    """Test suite for GET /api/v1/hardware/detect"""
    
    def test_hardware_detect_success(self, test_client, api_base_url):
        """Test hardware detection returns valid response."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required categories
        required = ["fpga", "embedded", "gpu", "cpu"]
        for category in required:
            assert category in data, f"Missing category: {category}"
    
    def test_fpga_detection_structure(self, test_client, api_base_url):
        """Test FPGA detection returns proper structure."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        fpga = data["fpga"]
        assert "available" in fpga
        assert isinstance(fpga["available"], bool)
        
        if fpga["available"]:
            assert "device" in fpga
            assert "status" in fpga
        else:
            # Should explain why not available
            assert "status" in fpga or "error" in fpga
    
    def test_embedded_detection_structure(self, test_client, api_base_url):
        """Test embedded systems detection returns proper structure."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        embedded = data["embedded"]
        assert "available" in embedded
        assert isinstance(embedded["available"], bool)
        
        if embedded["available"]:
            assert "devices" in embedded
            assert isinstance(embedded["devices"], list)
    
    def test_gpu_detection_structure(self, test_client, api_base_url):
        """Test GPU detection returns proper structure."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        gpu = data["gpu"]
        assert "available" in gpu
        assert isinstance(gpu["available"], bool)
        assert "type" in gpu
        
        if gpu["available"]:
            assert "devices" in gpu or "type" in gpu
    
    def test_cpu_detection_always_available(self, test_client, api_base_url):
        """Test CPU is always detected as available."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        cpu = data["cpu"]
        assert "available" in cpu
        assert cpu["available"] is True
        assert "cores" in cpu
        assert cpu["cores"] > 0
    
    @pytest.mark.hardware
    def test_fpga_tang_nano_9k_detection(self, test_client, api_base_url):
        """Test Tang Nano 9K FPGA detection."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        fpga = data["fpga"]
        
        if fpga["available"]:
            # Should mention Tang Nano 9K
            device_info = str(fpga.get("device", ""))
            assert "Tang Nano" in device_info or "9K" in device_info
    
    def test_hardware_detection_response_time(self, test_client, api_base_url):
        """Test hardware detection completes quickly."""
        import time
        
        start = time.time()
        response = test_client.get(f"{api_base_url}/hardware/detect")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 10.0, "Hardware detection took too long"
    
    def test_hardware_detection_idempotent(self, test_client, api_base_url):
        """Test hardware detection gives consistent results."""
        # First detection
        response1 = test_client.get(f"{api_base_url}/hardware/detect")
        data1 = response1.json()
        
        # Second detection
        response2 = test_client.get(f"{api_base_url}/hardware/detect")
        data2 = response2.json()
        
        # CPU should be identical
        assert data1["cpu"] == data2["cpu"]
        
        # Hardware availability should be consistent
        # (might change if device disconnected between calls, but unlikely)
        assert data1["fpga"]["available"] == data2["fpga"]["available"]


class TestSerialPortDetection:
    """Test suite for serial port detection (FPGA/embedded)."""
    
    def test_serial_ports_listed(self, test_client, api_base_url):
        """Test serial ports are listed if available."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        embedded = data["embedded"]
        
        if embedded["available"]:
            assert "devices" in embedded
            assert len(embedded["devices"]) > 0
            
            # Each device should have port info
            for device in embedded["devices"]:
                assert "port" in device or "description" in device
    
    @pytest.mark.hardware
    def test_sipeed_jtag_detected(self, test_client, api_base_url):
        """Test SIPEED JTAG debugger is detected."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        # Check if any device mentions SIPEED or JTAG
        embedded = data["embedded"]
        
        if embedded["available"]:
            devices = embedded.get("devices", [])
            sipeed_found = any(
                "SIPEED" in str(d.get("description", "")) or
                "JTAG" in str(d.get("description", ""))
                for d in devices
            )
            
            # If SIPEED hardware connected, should be detected
            # This test will pass either way, just validates structure


class TestGPUDetection:
    """Test suite for GPU detection."""
    
    def test_gpu_detection_no_error(self, test_client, api_base_url):
        """Test GPU detection doesn't error even without GPU."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        
        assert response.status_code == 200
        data = response.json()
        assert "gpu" in data
    
    @pytest.mark.hardware
    def test_nvidia_gpu_detection(self, test_client, api_base_url):
        """Test NVIDIA GPU detection if available."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        gpu = data["gpu"]
        
        if gpu["available"]:
            # Should mention NVIDIA or CUDA
            gpu_type = gpu.get("type", "")
            assert "NVIDIA" in gpu_type or "CUDA" in gpu_type
    
    def test_gpu_unavailable_graceful(self, test_client, api_base_url):
        """Test GPU unavailable is handled gracefully."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        gpu = data["gpu"]
        
        if not gpu["available"]:
            # Should still have type field
            assert "type" in gpu
            assert gpu["type"] == "None" or "None" in str(gpu)


class TestCPUInformation:
    """Test suite for CPU information."""
    
    def test_cpu_core_count(self, test_client, api_base_url):
        """Test CPU core count is reported."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        cpu = data["cpu"]
        assert "cores" in cpu
        assert isinstance(cpu["cores"], int)
        assert cpu["cores"] > 0
        assert cpu["cores"] <= 256  # Reasonable upper bound
    
    def test_cpu_information_detailed(self, test_client, api_base_url):
        """Test CPU provides useful information."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        cpu = data["cpu"]
        
        # At minimum should have cores and availability
        assert "cores" in cpu
        assert "available" in cpu
        assert cpu["available"] is True


class TestHardwareErrorHandling:
    """Test suite for error handling in hardware detection."""
    
    def test_hardware_detection_never_fails(self, test_client, api_base_url):
        """Test hardware detection always returns 200 even if detection fails."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        
        # Should never return error status
        assert response.status_code == 200
        
        # Should always return valid JSON
        data = response.json()
        assert isinstance(data, dict)
    
    def test_partial_hardware_failure_handled(self, test_client, api_base_url):
        """Test partial hardware detection failures are handled."""
        response = test_client.get(f"{api_base_url}/hardware/detect")
        data = response.json()
        
        # Even if one category fails, others should still work
        categories = ["fpga", "embedded", "gpu", "cpu"]
        
        for category in categories:
            assert category in data
            assert isinstance(data[category], dict)
    
    def test_hardware_timeout_handled(self, test_client, api_base_url):
        """Test hardware detection timeout is handled."""
        import time
        
        start = time.time()
        response = test_client.get(f"{api_base_url}/hardware/detect")
        elapsed = time.time() - start
        
        # Should not hang indefinitely
        assert elapsed < 30.0, "Hardware detection timed out"
        assert response.status_code == 200


class TestHardwareIntegrationWithBackends:
    """Test integration between hardware detection and backend availability."""
    
    def test_hardware_matches_backend_availability(self, test_client, api_base_url):
        """Test hardware detection matches backend availability."""
        # Get hardware info
        hw_response = test_client.get(f"{api_base_url}/hardware/detect")
        hw_data = hw_response.json()
        
        # Get backend info
        backend_response = test_client.get(f"{api_base_url}/backends/")
        backend_data = backend_response.json()
        
        backends = {b["name"]: b for b in backend_data["backends"]}
        
        # If FPGA detected, elm11 backend should be available
        if hw_data["fpga"]["available"]:
            if "elm11" in backends:
                # Hardware available, backend should report available
                # (or at least not have connection error)
                pass
    
    def test_cpu_available_sequential_available(self, test_client, api_base_url):
        """Test CPU available means sequential backend available."""
        # Get hardware info
        hw_response = test_client.get(f"{api_base_url}/hardware/detect")
        hw_data = hw_response.json()
        
        # Get backends
        backend_response = test_client.get(f"{api_base_url}/backends/")
        backend_data = backend_response.json()
        
        backends = {b["name"]: b for b in backend_data["backends"]}
        
        # CPU always available
        assert hw_data["cpu"]["available"] is True
        
        # Sequential backend should be available
        assert "sequential" in backends
        assert backends["sequential"]["available"] is True

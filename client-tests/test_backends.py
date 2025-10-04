"""
Unit tests for backend management endpoints.

Tests:
- Backend listing
- Backend information retrieval
- Backend availability detection
"""

import pytest
from fastapi.testclient import TestClient


class TestBackendListing:
    """Test suite for GET /api/v1/backends/"""
    
    def test_list_backends_success(self, test_client, api_base_url):
        """Test listing all backends returns valid response."""
        response = test_client.get(f"{api_base_url}/backends/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "backends" in data
        assert isinstance(data["backends"], list)
        assert len(data["backends"]) > 0
    
    def test_backends_have_required_fields(self, test_client, api_base_url):
        """Test each backend has required fields."""
        response = test_client.get(f"{api_base_url}/backends/")
        data = response.json()
        
        required_fields = ["name", "available", "description", "type"]
        
        for backend in data["backends"]:
            for field in required_fields:
                assert field in backend, f"Missing field: {field}"
            
            # Validate types
            assert isinstance(backend["name"], str)
            assert isinstance(backend["available"], bool)
            assert isinstance(backend["description"], str)
            assert isinstance(backend["type"], str)
    
    def test_sequential_backend_always_available(self, test_client, api_base_url):
        """Test sequential backend is always available."""
        response = test_client.get(f"{api_base_url}/backends/")
        data = response.json()
        
        backends = {b["name"]: b for b in data["backends"]}
        
        assert "sequential" in backends
        assert backends["sequential"]["available"] is True
    
    def test_backend_names_unique(self, test_client, api_base_url):
        """Test backend names are unique."""
        response = test_client.get(f"{api_base_url}/backends/")
        data = response.json()
        
        names = [b["name"] for b in data["backends"]]
        assert len(names) == len(set(names)), "Duplicate backend names found"
    
    def test_list_backends_response_time(self, test_client, api_base_url):
        """Test backend listing responds quickly."""
        import time
        
        start = time.time()
        response = test_client.get(f"{api_base_url}/backends/")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0, "Backend listing took too long"


class TestBackendInfo:
    """Test suite for GET /api/v1/backends/{backend_name}"""
    
    def test_get_sequential_backend(self, test_client, api_base_url):
        """Test getting sequential backend information."""
        response = test_client.get(f"{api_base_url}/backends/sequential")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "sequential"
        assert data["available"] is True
        assert "description" in data
        assert "type" in data
    
    def test_get_joblib_backend(self, test_client, api_base_url):
        """Test getting joblib backend information."""
        response = test_client.get(f"{api_base_url}/backends/joblib")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "joblib"
        # Available depends on joblib installation
        assert isinstance(data["available"], bool)
    
    @pytest.mark.hardware
    def test_get_elm11_backend(self, test_client, api_base_url):
        """Test getting ELM11/FPGA backend information."""
        response = test_client.get(f"{api_base_url}/backends/elm11")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "elm11"
        # Available depends on hardware connection
        assert isinstance(data["available"], bool)
        
        if not data["available"]:
            # Should have error message explaining why
            assert "error" in data or "status" in data
    
    def test_get_nonexistent_backend(self, test_client, api_base_url):
        """Test getting nonexistent backend returns 404."""
        response = test_client.get(f"{api_base_url}/backends/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.parametrize("backend_name", [
        "sequential",
        "joblib"
    ])
    def test_get_known_backends(self, test_client, api_base_url, backend_name):
        """Test getting information for known backends."""
        response = test_client.get(f"{api_base_url}/backends/{backend_name}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == backend_name


class TestBackendAvailability:
    """Test suite for backend availability detection."""
    
    def test_cpu_backends_available(self, test_client, api_base_url):
        """Test CPU-based backends are available."""
        response = test_client.get(f"{api_base_url}/backends/")
        data = response.json()
        
        backends = {b["name"]: b for b in data["backends"]}
        
        # Sequential should always work
        assert backends["sequential"]["available"] is True
        
        # Joblib might be available
        if "joblib" in backends:
            # If listed, should report correct availability
            assert isinstance(backends["joblib"]["available"], bool)
    
    @pytest.mark.hardware
    def test_hardware_backend_detection(self, test_client, api_base_url):
        """Test hardware backends report availability correctly."""
        response = test_client.get(f"{api_base_url}/backends/")
        data = response.json()
        
        backends = {b["name"]: b for b in data["backends"]}
        
        if "elm11" in backends:
            # Should have availability status
            assert isinstance(backends["elm11"]["available"], bool)
            
            # If unavailable, might have error
            if not backends["elm11"]["available"]:
                # Error message optional but helpful
                pass
    
    def test_backend_cache_consistency(self, test_client, api_base_url):
        """Test backend availability is consistent across requests."""
        # First request
        response1 = test_client.get(f"{api_base_url}/backends/")
        data1 = response1.json()
        
        # Second request (should use cache)
        response2 = test_client.get(f"{api_base_url}/backends/")
        data2 = response2.json()
        
        # Results should be identical
        assert len(data1["backends"]) == len(data2["backends"])
        
        # Availability should be consistent
        backends1 = {b["name"]: b["available"] for b in data1["backends"]}
        backends2 = {b["name"]: b["available"] for b in data2["backends"]}
        assert backends1 == backends2


class TestBackendValidation:
    """Test suite for backend parameter validation."""
    
    def test_backend_name_case_sensitive(self, test_client, api_base_url):
        """Test backend names are case-sensitive."""
        # Correct case
        response1 = test_client.get(f"{api_base_url}/backends/sequential")
        assert response1.status_code == 200
        
        # Wrong case
        response2 = test_client.get(f"{api_base_url}/backends/Sequential")
        assert response2.status_code == 404
    
    def test_backend_name_special_characters(self, test_client, api_base_url):
        """Test backend names with special characters."""
        # Valid name with underscore
        response = test_client.get(f"{api_base_url}/backends/sequential")
        assert response.status_code in [200, 404]  # Exists or doesn't
        
        # Invalid characters
        response = test_client.get(f"{api_base_url}/backends/invalid@backend")
        assert response.status_code == 404

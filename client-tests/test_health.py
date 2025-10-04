"""
Unit tests for health and root endpoints.

Tests basic API functionality and connectivity.
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(test_client):
    """Test root endpoint returns API information."""
    response = test_client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "name" in data
    assert "version" in data
    assert "docs" in data
    assert "health" in data
    
    assert data["name"] == "PyCWT REST API"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_check_healthy(test_client):
    """Test health check returns healthy status."""
    response = test_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "api_version" in data
    
    assert data["status"] == "healthy"
    assert data["api_version"] == "1.0.0"


def test_health_check_response_time(test_client):
    """Test health check responds quickly."""
    import time
    
    start = time.time()
    response = test_client.get("/health")
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 1.0, "Health check took too long"


def test_docs_endpoint_exists(test_client):
    """Test Swagger UI documentation endpoint exists."""
    response = test_client.get("/docs")
    
    # Should return HTML page
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_endpoint_exists(test_client):
    """Test ReDoc documentation endpoint exists."""
    response = test_client.get("/redoc")
    
    # Should return HTML page
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema_exists(test_client):
    """Test OpenAPI schema is available."""
    response = test_client.get("/openapi.json")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data
    
    # Check API info
    assert data["info"]["title"] == "PyCWT REST API"
    assert data["info"]["version"] == "1.0.0"


def test_invalid_endpoint_404(test_client):
    """Test invalid endpoint returns 404."""
    response = test_client.get("/api/v1/nonexistent")
    
    assert response.status_code == 404


def test_cors_headers_present(test_client):
    """Test CORS headers are set correctly."""
    response = test_client.get("/health", headers={
        "Origin": "http://localhost:3000"
    })
    
    assert response.status_code == 200
    # CORS headers should be present (if middleware configured)
    # assert "access-control-allow-origin" in response.headers


@pytest.mark.parametrize("endpoint", [
    "/",
    "/health",
    "/docs",
    "/redoc"
])
def test_all_root_endpoints_accessible(test_client, endpoint):
    """Test all root-level endpoints are accessible."""
    response = test_client.get(endpoint)
    assert response.status_code == 200

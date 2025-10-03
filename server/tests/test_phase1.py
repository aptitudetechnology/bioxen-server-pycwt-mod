"""Tests for Phase 1: Backend discovery endpoints."""

import pytest
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PyCWT REST API"
    assert data["version"] == "1.0.0"
    assert "docs" in data
    assert "health" in data


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api_version"] == "1.0.0"


def test_list_backends():
    """Test listing all available backends."""
    response = client.get("/api/v1/backends/")
    assert response.status_code == 200
    data = response.json()
    assert "backends" in data
    assert isinstance(data["backends"], list)
    
    # Verify each backend has expected fields
    for backend in data["backends"]:
        assert "name" in backend
        assert "available" in backend
        assert isinstance(backend["available"], bool)


def test_get_backend_info():
    """Test getting information about a specific backend."""
    # First get list of backends
    response = client.get("/api/v1/backends/")
    backends = response.json()["backends"]
    
    if backends:
        # Test first available backend
        backend_name = backends[0]["name"]
        response = client.get(f"/api/v1/backends/{backend_name}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == backend_name
        assert "available" in data
        assert "description" in data
        assert "type" in data


def test_get_invalid_backend():
    """Test getting information about a non-existent backend."""
    response = client.get("/api/v1/backends/nonexistent_backend")
    assert response.status_code == 404


def test_openapi_docs():
    """Test that OpenAPI documentation is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "PyCWT REST API"
    assert data["info"]["version"] == "1.0.0"

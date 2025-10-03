#!/usr/bin/env python3
"""
Example client for testing PyCWT REST API.
Run this after starting the server to verify functionality.
"""

import sys
import json
import time
import requests
from typing import Dict, Any

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

BASE_URL = "http://localhost:8000"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{text}{NC}")
    print(f"{BLUE}{'='*60}{NC}")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{NC} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{NC} {text}")


def print_info(text: str):
    """Print info message."""
    print(f"{BLUE}ℹ{NC} {text}")


def pretty_json(data: Dict[Any, Any]) -> str:
    """Format JSON data for display."""
    return json.dumps(data, indent=2)


def test_connection() -> bool:
    """Test if server is reachable."""
    print_header("Connection Test")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print_success(f"Server is reachable at {BASE_URL}")
            return True
        else:
            print_error(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Cannot connect to server: {e}")
        print_info("Make sure the server is running: python -m server.main")
        return False


def test_health_endpoint():
    """Test health check endpoint."""
    print_header("Health Check Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print_success("Health check passed")
        print(f"  Status: {data.get('status')}")
        print(f"  API Version: {data.get('api_version')}")
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def test_root_endpoint():
    """Test root endpoint."""
    print_header("Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        
        data = response.json()
        print_success("Root endpoint accessible")
        print(f"  API Name: {data.get('name')}")
        print(f"  Version: {data.get('version')}")
        print(f"  Docs URL: {data.get('docs')}")
        print(f"  Health URL: {data.get('health')}")
        return True
    except Exception as e:
        print_error(f"Root endpoint failed: {e}")
        return False


def test_list_backends():
    """Test listing all backends."""
    print_header("List Backends Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/backends/")
        response.raise_for_status()
        
        data = response.json()
        backends = data.get('backends', [])
        
        print_success(f"Found {len(backends)} registered backend(s)")
        
        available_count = sum(1 for b in backends if b.get('available'))
        print(f"  Available: {available_count}/{len(backends)}")
        
        print("\n  Backend Details:")
        for backend in backends:
            status = "✓" if backend.get('available') else "✗"
            name = backend.get('name', 'Unknown')
            desc = backend.get('description', 'No description')
            print(f"    {status} {name}")
            print(f"      {desc}")
        
        return True
    except Exception as e:
        print_error(f"List backends failed: {e}")
        return False


def test_backend_details(backend_name: str = "sequential"):
    """Test getting details for a specific backend."""
    print_header(f"Backend Details: {backend_name}")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/backends/{backend_name}")
        response.raise_for_status()
        
        data = response.json()
        print_success(f"Retrieved details for '{backend_name}'")
        print(f"  Name: {data.get('name')}")
        print(f"  Available: {data.get('available')}")
        print(f"  Type: {data.get('type')}")
        print(f"  Description: {data.get('description', 'N/A')[:80]}...")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print_error(f"Backend '{backend_name}' not found (404)")
        else:
            print_error(f"HTTP error: {e}")
        return False
    except Exception as e:
        print_error(f"Backend details failed: {e}")
        return False


def test_invalid_backend():
    """Test that invalid backend returns 404."""
    print_header("Invalid Backend Test (Should Fail)")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/backends/nonexistent_backend")
        
        if response.status_code == 404:
            print_success("Correctly returned 404 for invalid backend")
            error_data = response.json()
            print(f"  Error detail: {error_data.get('detail', 'N/A')}")
            return True
        else:
            print_error(f"Expected 404, got {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Invalid backend test failed: {e}")
        return False


def test_api_docs():
    """Test that API documentation is accessible."""
    print_header("API Documentation")
    try:
        # Test OpenAPI schema
        response = requests.get(f"{BASE_URL}/openapi.json")
        response.raise_for_status()
        schema = response.json()
        
        print_success("OpenAPI schema accessible")
        print(f"  Title: {schema.get('info', {}).get('title')}")
        print(f"  Version: {schema.get('info', {}).get('version')}")
        print(f"  Endpoints: {len(schema.get('paths', {}))}")
        
        # Test Swagger UI (just check if it returns HTML)
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200 and 'html' in response.headers.get('content-type', '').lower():
            print_success(f"Swagger UI accessible at {BASE_URL}/docs")
        
        return True
    except Exception as e:
        print_error(f"API docs test failed: {e}")
        return False


def run_performance_test(endpoint: str = "/health", requests_count: int = 100):
    """Run simple performance test."""
    print_header(f"Performance Test: {endpoint}")
    print(f"  Running {requests_count} requests...")
    
    url = f"{BASE_URL}{endpoint}"
    times = []
    errors = 0
    
    for i in range(requests_count):
        try:
            start = time.time()
            response = requests.get(url, timeout=5)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if response.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = len(times) / sum(times)
        
        print_success("Performance test complete")
        print(f"  Average response time: {avg_time*1000:.2f} ms")
        print(f"  Min response time: {min_time*1000:.2f} ms")
        print(f"  Max response time: {max_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} req/sec")
        print(f"  Errors: {errors}/{requests_count}")
        
        return errors == 0
    else:
        print_error("No successful requests")
        return False


def main():
    """Run all tests."""
    # Add src directory to Python path
    from pathlib import Path
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         PyCWT REST API - Test Client                      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{NC}")
    
    # Check connection first
    if not test_connection():
        sys.exit(1)
    
    # Run tests
    results = {
        'Health Check': test_health_endpoint(),
        'Root Endpoint': test_root_endpoint(),
        'List Backends': test_list_backends(),
        'Backend Details': test_backend_details('sequential'),
        'Invalid Backend': test_invalid_backend(),
        'API Documentation': test_api_docs(),
        'Performance (Health)': run_performance_test('/health', 50),
        'Performance (Backends)': run_performance_test('/api/v1/backends/', 20),
    }
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}✓ PASS{NC}" if result else f"{RED}✗ FAIL{NC}"
        print(f"{status}  {test_name}")
    
    print(f"\n{BLUE}Results: {passed}/{total} tests passed{NC}")
    
    if passed == total:
        print(f"\n{GREEN}✓ All tests passed! Phase 1 is working correctly.{NC}")
        print(f"\n{BLUE}Next steps:{NC}")
        print(f"  1. Review results above")
        print(f"  2. Test interactively at {BASE_URL}/docs")
        print(f"  3. Run automated tests: pytest server/tests/")
        print(f"  4. Review server/TESTING.md for more tests")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some tests failed. Review errors above.{NC}")
        print(f"\n{BLUE}Troubleshooting:{NC}")
        print(f"  1. Check server logs")
        print(f"  2. Run: python diagnose-server.py")
        print(f"  3. See: server/TESTING.md")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Tests interrupted by user{NC}")
        sys.exit(130)

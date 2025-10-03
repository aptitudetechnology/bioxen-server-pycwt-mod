# Testing Guide for PyCWT REST API Server

This guide covers all aspects of testing the PyCWT REST API server, from basic functionality to performance testing.

## Prerequisites

```bash
# Install server dependencies
pip install -r server/requirements.txt

# Install additional testing tools (optional)
pip install pytest-cov pytest-benchmark
```

## Quick Start Testing

### 1. Start the Server

```bash
# Option A: Using the startup script
./start-server.sh

# Option B: Direct Python
python -m server.main

# Option C: Using uvicorn
uvicorn server.main:app --reload
```

The server will start at `http://localhost:8000`

### 2. Verify Server is Running

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","api_version":"1.0.0"}
```

## Phase 1 Testing: Backend Discovery

### Manual Testing with curl

#### Test 1: Root Endpoint
```bash
curl http://localhost:8000/

# Expected: API information with name, version, docs, health
```

#### Test 2: Health Check
```bash
curl http://localhost:8000/health

# Expected: {"status":"healthy","api_version":"1.0.0"}
```

#### Test 3: List All Backends
```bash
curl http://localhost:8000/api/v1/backends/

# Expected: JSON with list of available backends
# Example response:
# {
#   "backends": [
#     {
#       "name": "sequential",
#       "available": true,
#       "description": "Sequential backend for wavelet analysis"
#     },
#     {
#       "name": "joblib",
#       "available": true,
#       "description": "Joblib parallel backend"
#     }
#   ]
# }
```

#### Test 4: Get Specific Backend Info
```bash
# Test with sequential backend
curl http://localhost:8000/api/v1/backends/sequential

# Expected: Detailed backend information
```

#### Test 5: Test Invalid Backend
```bash
curl http://localhost:8000/api/v1/backends/nonexistent

# Expected: 404 error with detail message
```

### Interactive Testing with API Docs

1. **Open Swagger UI**: http://localhost:8000/docs
2. **Navigate through endpoints**:
   - Try the "GET /" endpoint
   - Try the "GET /health" endpoint
   - Try "GET /api/v1/backends/" 
   - Try "GET /api/v1/backends/{backend_name}" with different backend names

3. **Open ReDoc**: http://localhost:8000/redoc
   - Browse the comprehensive API documentation

### Automated Testing with pytest

```bash
# Run all Phase 1 tests
cd /home/chris/pycwt-mod
pytest server/tests/test_phase1.py -v

# Run with coverage
pytest server/tests/test_phase1.py --cov=server --cov-report=html

# Run specific test
pytest server/tests/test_phase1.py::test_health_check -v
```

### Testing with Python Requests

Create a test script (`test_manual.py`):

```python
import requests

BASE_URL = "http://localhost:8000"

# Test health
response = requests.get(f"{BASE_URL}/health")
print(f"Health: {response.json()}")

# Test backends list
response = requests.get(f"{BASE_URL}/api/v1/backends/")
backends = response.json()["backends"]
print(f"\nAvailable backends: {len(backends)}")
for backend in backends:
    print(f"  - {backend['name']}: {'✓' if backend['available'] else '✗'}")

# Test specific backend
if backends:
    backend_name = backends[0]["name"]
    response = requests.get(f"{BASE_URL}/api/v1/backends/{backend_name}")
    print(f"\n{backend_name} details:")
    print(f"  Available: {response.json()['available']}")
    print(f"  Type: {response.json()['type']}")
```

Run it:
```bash
python test_manual.py
```

## Performance Testing

### Load Testing with Apache Bench

```bash
# Install ab (Apache Bench)
sudo apt-get install apache2-utils  # Ubuntu/Debian

# Test health endpoint (100 requests, 10 concurrent)
ab -n 100 -c 10 http://localhost:8000/health

# Test backends list endpoint
ab -n 100 -c 10 http://localhost:8000/api/v1/backends/
```

### Load Testing with wrk (Advanced)

```bash
# Install wrk
sudo apt-get install wrk

# Run 30-second test with 10 threads and 100 connections
wrk -t10 -c100 -d30s http://localhost:8000/health

# Test backends endpoint
wrk -t10 -c100 -d30s http://localhost:8000/api/v1/backends/
```

## Integration Testing

### Test Backend Integration

```bash
# Create integration test script
cat > test_backend_integration.py << 'EOF'
"""Test that server correctly integrates with pycwt_mod backends."""
import requests

BASE_URL = "http://localhost:8000"

# Get available backends from API
response = requests.get(f"{BASE_URL}/api/v1/backends/")
api_backends = {b["name"] for b in response.json()["backends"]}

# Get backends directly from pycwt_mod
from pycwt_mod.backends import list_backends
direct_backends = set(list_backends())

print("Backend Integration Test")
print("=" * 50)
print(f"Backends from API: {api_backends}")
print(f"Backends from pycwt_mod: {direct_backends}")
print(f"Match: {api_backends == direct_backends}")

# Test each backend availability
for backend_name in api_backends:
    response = requests.get(f"{BASE_URL}/api/v1/backends/{backend_name}")
    api_available = response.json()["available"]
    
    from pycwt_mod.backends import get_backend
    backend = get_backend(backend_name)
    direct_available = backend.is_available()
    
    status = "✓" if api_available == direct_available else "✗"
    print(f"{status} {backend_name}: API={api_available}, Direct={direct_available}")
EOF

python test_backend_integration.py
```

## Common Issues and Troubleshooting

### Issue: Server won't start

**Problem**: `Address already in use`
```bash
# Solution: Check what's using port 8000
lsof -i :8000

# Kill the process or change port in .env
echo "PORT=8001" >> server/.env
```

### Issue: Import errors

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`
```bash
# Solution: Install dependencies
pip install -r server/requirements.txt
```

### Issue: Backend not available

**Problem**: Backend shows `"available": false`
```bash
# Check backend dependencies
python -c "from pycwt_mod.backends import get_backend; print(get_backend('joblib').is_available())"

# Install missing dependencies
pip install joblib  # for joblib backend
pip install dask distributed  # for dask backend
pip install cupy  # for gpu backend
```

### Issue: Tests failing

**Problem**: `ImportError: cannot import name 'app' from 'server.main'`
```bash
# Solution: Make sure PYTHONPATH includes project root
export PYTHONPATH=/home/chris/pycwt-mod:$PYTHONPATH
pytest server/tests/test_phase1.py
```

## Test Checklist

Use this checklist to verify Phase 1 is complete:

- [ ] Server starts without errors
- [ ] Health endpoint returns 200 OK
- [ ] Root endpoint returns API information
- [ ] Backends list endpoint returns at least one backend
- [ ] Each backend endpoint returns correct information
- [ ] Invalid backend returns 404
- [ ] OpenAPI docs accessible at /docs
- [ ] ReDoc accessible at /redoc
- [ ] All automated tests pass
- [ ] Backend availability matches direct pycwt_mod checks
- [ ] Server handles concurrent requests

## Performance Benchmarks (Expected)

For Phase 1 endpoints on typical hardware:

| Endpoint | Expected Response Time | Concurrent Capacity |
|----------|----------------------|---------------------|
| /health | < 5ms | > 1000 req/sec |
| /api/v1/backends/ | < 50ms | > 100 req/sec |
| /api/v1/backends/{name} | < 50ms | > 100 req/sec |

## Next Steps

After Phase 1 testing is complete:
1. Document any issues found
2. Verify all backends are properly detected
3. Check performance meets expectations
4. Move to Phase 2: Analysis Endpoints testing

## Reporting Issues

When reporting issues, include:
1. Server logs
2. Request details (method, URL, body)
3. Response (status code, body, headers)
4. Environment info (Python version, OS, dependencies)
5. Steps to reproduce

Example:
```bash
# Capture server logs
python -m server.main > server.log 2>&1

# Make request with verbose output
curl -v http://localhost:8000/api/v1/backends/
```

## Additional Resources

- FastAPI Testing: https://fastapi.tiangolo.com/tutorial/testing/
- pytest Documentation: https://docs.pytest.org/
- HTTP Status Codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status

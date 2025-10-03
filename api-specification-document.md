# PyCWT REST API Specification

**Version:** 0.1.0-alpha  
**Base URL:** `http://localhost:8000`  
**Protocol:** HTTP/HTTPS  
**Content-Type:** `application/json`

## Overview

The PyCWT REST API provides programmatic access to continuous wavelet transform analysis with hardware-accelerated Monte Carlo backends. This API enables integration with external programs, web applications, and data analysis pipelines.

### Key Features

- 🔌 RESTful architecture for easy integration
- ⚡ Hardware-accelerated backend selection (CPU, FPGA, embedded)
- 🔄 Asynchronous processing for long-running computations
- 📊 Comprehensive wavelet analysis endpoints
- 🛡️ CORS-enabled for web application integration
- 📖 Interactive API documentation (Swagger UI & ReDoc)

### Base Endpoints

- **API Documentation:** `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs:** `http://localhost:8000/redoc` (ReDoc)
- **Health Check:** `http://localhost:8000/health`

---

## Authentication

**Current Version:** No authentication required (development mode)

**Production Recommendations:**
- Implement API key authentication
- Use JWT tokens for user sessions
- Enable HTTPS/TLS encryption
- Configure CORS for specific origins

---

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Get API information and available endpoints.

#### Response

```json
{
  "name": "PyCWT REST API",
  "version": "0.1.0-alpha",
  "docs": "/docs",
  "health": "/health"
}
```

#### cURL Example

```bash
curl http://localhost:8000/
```

---

### 2. Health Check

**GET** `/health`

Check API server health status.

#### Response

```json
{
  "status": "healthy",
  "api_version": "0.1.0-alpha"
}
```

#### cURL Example

```bash
curl http://localhost:8000/health
```

#### Status Codes

- `200 OK`: Server is healthy
- `503 Service Unavailable`: Server is experiencing issues

---

## Backend Management

### 3. List Available Backends

**GET** `/api/v1/backends/`

List all registered computation backends with availability status.

#### Response

```json
{
  "backends": [
    {
      "name": "sequential",
      "available": true,
      "description": "Sequential Monte Carlo backend (single-core CPU)"
    },
    {
      "name": "joblib",
      "available": true,
      "description": "Parallel Monte Carlo backend using joblib (multi-core CPU)"
    },
    {
      "name": "elm11",
      "available": true,
      "description": "FPGA-accelerated backend using ELM11/Tang Nano 9K"
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Backend identifier |
| `available` | boolean | Hardware/software availability |
| `description` | string | Backend description |
| `error` | string | Error message (if unavailable) |

#### cURL Example

```bash
curl http://localhost:8000/api/v1/backends/
```

#### Python Example

```python
import requests

response = requests.get("http://localhost:8000/api/v1/backends/")
backends = response.json()

for backend in backends["backends"]:
    print(f"{backend['name']}: {'✓' if backend['available'] else '✗'}")
```

---

### 4. Get Backend Information

**GET** `/api/v1/backends/{backend_name}`

Get detailed information about a specific backend.

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `backend_name` | string | Yes | Backend identifier (e.g., `sequential`, `joblib`, `elm11`) |

#### Response

```json
{
  "name": "elm11",
  "available": true,
  "description": "FPGA-accelerated backend using ELM11 microcontroller...",
  "type": "ELM11Backend"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Backend identifier |
| `available` | boolean | Hardware/software availability |
| `description` | string | Full backend documentation |
| `type` | string | Backend class name |

#### cURL Example

```bash
curl http://localhost:8000/api/v1/backends/elm11
```

#### Python Example

```python
import requests

response = requests.get("http://localhost:8000/api/v1/backends/elm11")
backend_info = response.json()

if backend_info["available"]:
    print(f"✓ {backend_info['name']} is available")
    print(f"Type: {backend_info['type']}")
else:
    print(f"✗ {backend_info['name']} is not available")
```

#### Status Codes

- `200 OK`: Backend information retrieved
- `404 Not Found`: Backend does not exist

---

## Wavelet Analysis Endpoints

### 5. Continuous Wavelet Transform (CWT)

**POST** `/api/v1/wavelet/cwt`

Perform continuous wavelet transform on time series data.

#### Request Body

```json
{
  "data": [1.2, 3.4, 2.1, 4.5, 3.2],
  "dt": 0.1,
  "dj": 0.125,
  "s0": -1,
  "J": -1,
  "mother": "morlet",
  "param": -1
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data` | array[float] | Yes | - | Time series data |
| `dt` | float | Yes | - | Time step |
| `dj` | float | No | 0.25 | Scale resolution |
| `s0` | float | No | -1 | Smallest scale (auto if -1) |
| `J` | int | No | -1 | Number of scales (auto if -1) |
| `mother` | string | No | "morlet" | Mother wavelet type |
| `param` | float | No | -1 | Wavelet parameter |

#### Response

```json
{
  "wave": [[complex_values]],
  "scales": [0.1, 0.2, 0.4],
  "freqs": [10.0, 5.0, 2.5],
  "coi": [0.1, 0.15, 0.2],
  "fft": [complex_fft_values],
  "fftfreqs": [0.0, 0.1, 0.2]
}
```

#### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/wavelet/cwt \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.0, 2.0, 1.5, 3.0, 2.5],
    "dt": 0.1,
    "dj": 0.125
  }'
```

---

### 6. Wavelet Coherence (WCT)

**POST** `/api/v1/wavelet/wct`

Calculate wavelet coherence between two time series.

#### Request Body

```json
{
  "signal1": [1.2, 3.4, 2.1, 4.5],
  "signal2": [2.1, 3.2, 2.8, 4.1],
  "dt": 0.1,
  "dj": 0.125,
  "s0": -1,
  "J": -1,
  "significance_level": 0.95,
  "mc_count": 300,
  "backend": "elm11"
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `signal1` | array[float] | Yes | - | First time series |
| `signal2` | array[float] | Yes | - | Second time series |
| `dt` | float | Yes | - | Time step |
| `dj` | float | No | 0.25 | Scale resolution |
| `s0` | float | No | -1 | Smallest scale |
| `J` | int | No | -1 | Number of scales |
| `significance_level` | float | No | 0.95 | Significance level (0-1) |
| `mc_count` | int | No | 300 | Monte Carlo simulations |
| `backend` | string | No | "sequential" | Backend: `sequential`, `joblib`, `elm11` |

#### Response

```json
{
  "WCT": [[coherence_matrix]],
  "aWCT": [[phase_angle_matrix]],
  "coi": [cone_of_influence],
  "freqs": [frequencies],
  "signif": [[significance_matrix]],
  "backend_used": "elm11",
  "computation_time": 2.34
}
```

#### Python Example

```python
import requests
import numpy as np

# Generate sample data
t = np.linspace(0, 10, 100)
signal1 = np.sin(2 * np.pi * t)
signal2 = np.sin(2 * np.pi * t + np.pi/4)

# Request wavelet coherence with FPGA acceleration
response = requests.post(
    "http://localhost:8000/api/v1/wavelet/wct",
    json={
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.1,
        "mc_count": 500,
        "backend": "elm11"  # Use Tang Nano 9K FPGA
    }
)

result = response.json()
print(f"Backend used: {result['backend_used']}")
print(f"Computation time: {result['computation_time']:.2f}s")
```

---

### 7. Cross-Wavelet Transform (XWT)

**POST** `/api/v1/wavelet/xwt`

Calculate cross-wavelet transform between two time series.

#### Request Body

```json
{
  "signal1": [1.2, 3.4, 2.1],
  "signal2": [2.1, 3.2, 2.8],
  "dt": 0.1,
  "dj": 0.125
}
```

#### Response

```json
{
  "xwt": [[cross_wavelet_values]],
  "phase": [[phase_angles]],
  "coi": [cone_of_influence],
  "freqs": [frequencies]
}
```

---

## Batch Processing

### 8. Submit Batch Job

**POST** `/api/v1/jobs/batch`

Submit multiple wavelet analysis tasks for batch processing.

#### Request Body

```json
{
  "tasks": [
    {
      "type": "wct",
      "signal1": [1.0, 2.0, 1.5],
      "signal2": [1.5, 2.5, 2.0],
      "dt": 0.1,
      "backend": "elm11"
    },
    {
      "type": "cwt",
      "data": [1.0, 2.0, 1.5],
      "dt": 0.1
    }
  ]
}
```

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "tasks_count": 2,
  "estimated_time": 15.5
}
```

---

### 9. Check Job Status

**GET** `/api/v1/jobs/{job_id}`

Check the status of a batch job.

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "results": [
    {"task_id": 0, "status": "completed"},
    {"task_id": 1, "status": "completed"}
  ]
}
```

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Server temporarily unavailable |

---

## Rate Limiting

**Current Version:** No rate limiting

**Production Recommendations:**
- Implement per-IP rate limiting
- Use API key-based quotas
- Queue long-running computations

---

## Client Libraries

### Python

```python
import requests

class PyCWTClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def list_backends(self):
        """List available backends."""
        response = requests.get(f"{self.base_url}/api/v1/backends/")
        return response.json()
    
    def wavelet_coherence(self, signal1, signal2, dt, backend="sequential", **kwargs):
        """Calculate wavelet coherence."""
        data = {
            "signal1": signal1,
            "signal2": signal2,
            "dt": dt,
            "backend": backend,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/api/v1/wavelet/wct", json=data)
        return response.json()
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()


# Usage
client = PyCWTClient()
backends = client.list_backends()
print(backends)
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class PyCWTClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({ baseURL });
    }
    
    async listBackends() {
        const response = await this.client.get('/api/v1/backends/');
        return response.data;
    }
    
    async waveletCoherence(signal1, signal2, dt, options = {}) {
        const data = { signal1, signal2, dt, ...options };
        const response = await this.client.post('/api/v1/wavelet/wct', data);
        return response.data;
    }
    
    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }
}

// Usage
const client = new PyCWTClient();
client.listBackends().then(backends => console.log(backends));
```

### cURL

```bash
# List backends
curl http://localhost:8000/api/v1/backends/

# Check backend status
curl http://localhost:8000/api/v1/backends/elm11

# Health check
curl http://localhost:8000/health
```

---

## Configuration

### Environment Variables

Create a `.env` file in the `server/` directory:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Backend Configuration
DEFAULT_BACKEND=sequential
AVAILABLE_BACKENDS=sequential,joblib,elm11

# Job Management
MAX_CONCURRENT_JOBS=10
JOB_TIMEOUT_SECONDS=3600
```

### Starting the Server

```bash
# Development mode
cd pycwt-mod
python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Integration Examples

### MATLAB

```matlab
% List available backends
url = 'http://localhost:8000/api/v1/backends/';
response = webread(url);
disp(response.backends);

% Calculate wavelet coherence
url = 'http://localhost:8000/api/v1/wavelet/wct';
signal1 = sin(2*pi*(0:0.1:10));
signal2 = cos(2*pi*(0:0.1:10));
data = struct('signal1', signal1, 'signal2', signal2, 'dt', 0.1);
options = weboptions('MediaType', 'application/json');
response = webwrite(url, data, options);
```

### R

```r
library(httr)
library(jsonlite)

# List backends
response <- GET("http://localhost:8000/api/v1/backends/")
backends <- content(response, "parsed")
print(backends)

# Wavelet coherence
signal1 <- sin(2*pi*seq(0, 10, 0.1))
signal2 <- cos(2*pi*seq(0, 10, 0.1))
data <- list(signal1 = signal1, signal2 = signal2, dt = 0.1, backend = "elm11")
response <- POST("http://localhost:8000/api/v1/wavelet/wct",
                 body = toJSON(data, auto_unbox = TRUE),
                 content_type_json())
result <- content(response, "parsed")
```

---

## Performance Optimization

### Backend Selection

Choose the appropriate backend based on your requirements:

| Backend | Best For | Performance |
|---------|----------|-------------|
| `sequential` | Small datasets, debugging | Baseline |
| `joblib` | Multi-core systems, moderate datasets | 3-4× faster |
| `elm11` | FPGA-accelerated, large Monte Carlo runs | Variable speedup |

### Batch Processing

For multiple analyses, use batch endpoints to reduce overhead:

```python
# Instead of multiple individual requests
for signal_pair in signal_pairs:
    result = requests.post("/api/v1/wavelet/wct", json=signal_pair)

# Use batch processing
tasks = [{"type": "wct", **pair} for pair in signal_pairs]
job = requests.post("/api/v1/jobs/batch", json={"tasks": tasks})
```

---

## Troubleshooting

### Connection Refused

**Problem:** `Connection refused` error

**Solution:**
1. Verify server is running: `curl http://localhost:8000/health`
2. Check port availability: `lsof -i :8000`
3. Review server logs for errors

### Backend Not Available

**Problem:** Backend shows `"available": false`

**Solution:**
1. Check hardware connection (for `elm11`)
2. Verify dependencies: `pip install pyserial joblib`
3. Run hardware detection: `python3 test-tang-nano-9k.py`
4. Check permissions: `groups` (should include `dialout`)

### Slow Response Times

**Problem:** API requests timing out

**Solution:**
1. Use appropriate backend for workload size
2. Reduce `mc_count` for faster (less accurate) results
3. Enable batch processing for multiple analyses
4. Check server resource usage: `htop`

---

## Security Considerations

### Production Deployment

**⚠️ Important:** The current API has no authentication for development purposes.

For production deployment:

1. **Enable Authentication:**
   ```python
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   
   security = HTTPBearer()
   
   @app.get("/api/v1/backends/")
   async def list_backends(credentials: HTTPAuthorizationCredentials = Depends(security)):
       # Verify credentials
       pass
   ```

2. **Configure CORS:**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Specific origins
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

3. **Enable HTTPS:**
   ```bash
   uvicorn server.main:app --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
   ```

4. **Input Validation:**
   - Validate array sizes
   - Limit `mc_count` to reasonable values
   - Sanitize file uploads

---

## Support & Documentation

- **GitHub:** https://github.com/aptitudetechnology/pycwt-mod
- **Issues:** https://github.com/aptitudetechnology/pycwt-mod/issues
- **Documentation:** https://pycwt-mod.readthedocs.io
- **Hardware Setup:** See `TANG-NANO-9K-TESTING.md`

---

## Changelog

### Version 0.1.0-alpha (Current)

- ✅ Backend management endpoints
- ✅ Health check and status endpoints
- ✅ Interactive API documentation
- ✅ CORS support for web applications
- 📋 Wavelet analysis endpoints (planned)
- 📋 Batch processing (planned)
- 📋 Job queue management (planned)

---

## License

PyCWT-mod is released under a BSD-style open source licence.

Copyright (c) 2024-2025 Sebastian Krieger, Nabil Freij, and contributors.

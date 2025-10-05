# PyCWT REST API Specification

**Version:** 1.0.0  
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
  "version": "1.0.0",
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
  "api_version": "1.0.0"
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

## Hardware Detection

### 3. Detect Hardware

**GET** `/api/v1/hardware/detect`

Detect available hardware resources for wavelet computation.

#### Response

```json
{
  "cpu": {
    "available": true,
    "cores": 8,
    "model": "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz"
  },
  "gpu": {
    "available": true,
    "count": 1,
    "devices": ["NVIDIA GeForce RTX 3080"]
  },
  "fpga": {
    "available": true,
    "devices": [
      {
        "port": "/dev/ttyUSB0",
        "vid": "0x1A86",
        "pid": "0x55D4",
        "description": "Tang Nano 9K FPGA"
      }
    ]
  },
  "embedded": {
    "available": true,
    "devices": [
      {
        "port": "/dev/ttyUSB1",
        "vid": "0x1A86",
        "pid": "0x7523",
        "description": "CH340 serial converter"
      }
    ]
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `cpu.available` | boolean | CPU availability |
| `cpu.cores` | integer | Number of CPU cores |
| `cpu.model` | string | CPU model name |
| `gpu.available` | boolean | GPU availability (requires CUDA/cupy) |
| `gpu.count` | integer | Number of GPU devices |
| `gpu.devices` | array[string] | GPU device names |
| `fpga.available` | boolean | FPGA device availability |
| `fpga.devices` | array[object] | FPGA device details |
| `embedded.available` | boolean | Embedded device availability |
| `embedded.devices` | array[object] | Embedded device details |

#### cURL Example

```bash
curl http://localhost:8000/api/v1/hardware/detect
```

#### Python Example

```python
import requests

response = requests.get("http://localhost:8000/api/v1/hardware/detect")
hardware = response.json()

if hardware["fpga"]["available"]:
    print(f"✓ FPGA available: {len(hardware['fpga']['devices'])} device(s)")
    for device in hardware["fpga"]["devices"]:
        print(f"  - {device['description']} on {device['port']}")

if hardware["gpu"]["available"]:
    print(f"✓ GPU available: {hardware['gpu']['count']} device(s)")
    for gpu in hardware["gpu"]["devices"]:
        print(f"  - {gpu}")
```

---

## Backend Management

### 4. List Available Backends

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

### 5. Get Backend Information

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

## Performance Benchmarking

### 6. Backend Benchmark

**POST** `/api/v1/benchmark`

Benchmark wavelet computation performance across different backends.

#### Request Body

```json
{
  "signal_length": 1000,
  "mc_count": 100,
  "backends": ["sequential", "joblib", "elm11"]
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `signal_length` | integer | No | 1000 | Test signal length (100-10000) |
| `mc_count` | integer | No | 100 | Monte Carlo iterations (10-1000) |
| `backends` | array[string] | No | ["sequential"] | Backends to benchmark |

#### Response

```json
{
  "signal_length": 1000,
  "mc_count": 100,
  "results": [
    {
      "backend": "sequential",
      "status": "completed",
      "computation_time": 5.234,
      "speedup": 1.0
    },
    {
      "backend": "joblib",
      "status": "completed",
      "computation_time": 1.456,
      "speedup": 3.59
    },
    {
      "backend": "elm11",
      "status": "completed",
      "computation_time": 2.123,
      "speedup": 2.47
    },
    {
      "backend": "nonexistent",
      "status": "failed",
      "error": "Backend not available"
    }
  ],
  "fastest_backend": "joblib",
  "test_parameters": {
    "signal_length": 1000,
    "dt": 0.25,
    "mother": "morlet"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `signal_length` | integer | Test signal length used |
| `mc_count` | integer | Monte Carlo iterations used |
| `results` | array[object] | Benchmark results per backend |
| `results[].backend` | string | Backend name |
| `results[].status` | string | "completed", "failed", or "unavailable" |
| `results[].computation_time` | float | Execution time in seconds |
| `results[].speedup` | float | Speedup vs sequential backend |
| `results[].error` | string | Error message (if failed) |
| `fastest_backend` | string | Name of fastest backend |
| `test_parameters` | object | Test configuration details |

#### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "signal_length": 1000,
    "mc_count": 100,
    "backends": ["sequential", "joblib", "elm11"]
  }'
```

#### Python Example

```python
import requests

# Benchmark all backends
response = requests.post(
    "http://localhost:8000/api/v1/benchmark",
    json={
        "signal_length": 2000,
        "mc_count": 200,
        "backends": ["sequential", "joblib", "elm11"]
    }
)

result = response.json()
print(f"Fastest backend: {result['fastest_backend']}")

for backend_result in result["results"]:
    if backend_result["status"] == "completed":
        print(f"{backend_result['backend']}: {backend_result['computation_time']:.2f}s "
              f"(speedup: {backend_result['speedup']:.2f}x)")
```

---

## Wavelet Analysis Endpoints

### 7. Continuous Wavelet Transform (CWT)

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

### 8. Wavelet Coherence (WCT)

**POST** `/api/v1/wavelet/wct`

Calculate wavelet coherence between two time series with optional significance testing.

#### Request Body

```json
{
  "signal1": [1.2, 3.4, 2.1, 4.5],
  "signal2": [2.1, 3.2, 2.8, 4.1],
  "dt": 0.1,
  "dj": 0.125,
  "s0": -1,
  "J": -1,
  "sig": true,
  "significance_level": 0.95,
  "mc_count": 30,
  "backend": "elm11"
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `signal1` | array[float] | Yes | - | First time series (minimum 200 points recommended) |
| `signal2` | array[float] | Yes | - | Second time series (minimum 200 points recommended) |
| `dt` | float | Yes | - | Time step |
| `dj` | float | No | 0.25 | Scale resolution |
| `s0` | float | No | -1 | Smallest scale (auto if -1) |
| `J` | int | No | -1 | Number of scales (auto if -1) |
| `sig` | boolean | No | false | Compute significance testing |
| `significance_level` | float | No | null | Significance level (0-1), enables significance if set |
| `mc_count` | int | No | 30 | Monte Carlo simulations (higher = more accurate but slower) |
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

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `WCT` | array[array[float]] | Wavelet coherence matrix |
| `aWCT` | array[array[float]] | Phase angle matrix (radians) |
| `coi` | array[float] | Cone of influence boundary |
| `freqs` | array[float] | Frequency values |
| `signif` | array[array[float]] | Significance matrix (only if sig=true or significance_level set) |
| `backend_used` | string | Backend used for computation |
| `computation_time` | float | Execution time in seconds |

#### Performance Notes

⚠️ **Significance Testing:** Computing significance with Monte Carlo simulations can be slow (30+ seconds for `mc_count >= 100`). For faster results:
- Set `sig=false` (default) to skip significance testing
- Use `mc_count=30` or lower for quick approximations
- Use FPGA backend (`elm11`) for hardware acceleration
- Consider async/background processing for large datasets

⚠️ **Signal Length Requirements:** Signals shorter than 200 points may fail with "Series too short" errors due to autocorrelation requirements. Use longer signals or adjust wavelet parameters.

#### Python Example

```python
import requests
import numpy as np

# Generate sample data (use 1000+ points for best results)
t = np.linspace(0, 10, 1000)
signal1 = np.sin(2 * np.pi * t)
signal2 = np.sin(2 * np.pi * t + np.pi/4)

# Request wavelet coherence WITHOUT significance testing (fast)
response = requests.post(
    "http://localhost:8000/api/v1/wavelet/wct",
    json={
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.01,
        "sig": False,  # Skip significance testing for speed
        "backend": "elm11"  # Use Tang Nano 9K FPGA
    }
)

result = response.json()
print(f"Backend used: {result['backend_used']}")
print(f"Computation time: {result['computation_time']:.2f}s")

# Request WITH significance testing (slower)
response_sig = requests.post(
    "http://localhost:8000/api/v1/wavelet/wct",
    json={
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.01,
        "sig": True,
        "mc_count": 30,  # Low count for speed
        "significance_level": 0.95,
        "backend": "elm11"
    }
)

result_sig = response_sig.json()
print(f"With significance - Time: {result_sig['computation_time']:.2f}s")
```

---

### 9. Cross-Wavelet Transform (XWT)

**POST** `/api/v1/wavelet/xwt`

Calculate cross-wavelet transform between two time series.

#### Request Body

```json
{
  "signal1": [1.2, 3.4, 2.1, 4.5],
  "signal2": [2.1, 3.2, 2.8, 4.1],
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
| `signal1` | array[float] | Yes | - | First time series (minimum 200 points recommended) |
| `signal2` | array[float] | Yes | - | Second time series (minimum 200 points recommended) |
| `dt` | float | Yes | - | Time step |
| `dj` | float | No | 0.25 | Scale resolution |
| `s0` | float | No | -1 | Smallest scale (auto if -1) |
| `J` | int | No | -1 | Number of scales (auto if -1) |
| `mother` | string | No | "morlet" | Mother wavelet type |
| `param` | float | No | -1 | Wavelet parameter (auto if -1) |

#### Response

```json
{
  "amplitude": [[amplitude_matrix]],
  "phase": [[phase_matrix]],
  "coi": [cone_of_influence],
  "freqs": [frequencies]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `amplitude` | array[array[float]] | Cross-wavelet amplitude matrix |
| `phase` | array[array[float]] | Phase angle matrix (radians) |
| `coi` | array[float] | Cone of influence boundary |
| `freqs` | array[float] | Frequency values |

#### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/wavelet/xwt \
  -H "Content-Type: application/json" \
  -d '{
    "signal1": [1.0, 2.0, 1.5, 3.0, 2.5],
    "signal2": [1.5, 2.5, 2.0, 3.5, 3.0],
    "dt": 0.1
  }'
```

#### Python Example

```python
import requests
import numpy as np

# Generate test signals (use 1000+ points for best results)
t = np.linspace(0, 10, 1000)
signal1 = np.sin(2 * np.pi * t)
signal2 = np.cos(2 * np.pi * t)

response = requests.post(
    "http://localhost:8000/api/v1/wavelet/xwt",
    json={
        "signal1": signal1.tolist(),
        "signal2": signal2.tolist(),
        "dt": 0.01,
        "mother": "morlet"
    }
)

result = response.json()
amplitude = np.array(result["amplitude"])
phase = np.array(result["phase"])
print(f"Amplitude shape: {amplitude.shape}")
print(f"Phase shape: {phase.shape}")
```

#### Performance Notes

⚠️ **Signal Length Requirements:** Signals shorter than 200 points may fail with "Series too short" errors. Use longer signals or adjust wavelet parameters (`s0`, `J`, `dj`).

---

## Batch Processing

### 10. Submit Batch Job

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

### 11. Check Job Status

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

### Version 1.0.0 (Current)

**Released:** 2025-01-XX

#### New Features
- ✅ **Hardware Detection Endpoint** (`/api/v1/hardware/detect`)
  - Automatic detection of CPU, GPU, FPGA, and embedded devices
  - Detailed hardware specifications and port information
  - Support for Tang Nano 9K FPGA detection

- ✅ **Performance Benchmarking** (`/api/v1/benchmark`)
  - Compare backend performance with configurable test parameters
  - Automatic speedup calculations
  - Support for multiple backend comparisons in single request

- ✅ **Enhanced Wavelet Analysis**
  - CWT: Continuous Wavelet Transform with full parameter control
  - WCT: Wavelet Coherence with optional significance testing
  - XWT: Cross-Wavelet Transform with amplitude/phase outputs
  - All endpoints support multiple mother wavelet types

- ✅ **Backend Management**
  - List available backends with status
  - Get detailed backend information
  - Backend type field shows implementation class

- ✅ **Interactive Documentation**
  - Swagger UI at `/docs`
  - ReDoc alternative at `/redoc`
  - Comprehensive API examples

#### API Changes
- **WCT Endpoint Updates:**
  - Added `sig` parameter (default: `false`) to control significance testing
  - Changed `mc_count` default from 300 to 30 for faster computation
  - Changed `significance_level` default from 0.95 to `null`
  - Significance only computed if `sig=true` OR `significance_level` is set

- **XWT Endpoint Updates:**
  - Renamed response fields: `WXamp` → `amplitude`, `WXangle` → `phase`
  - Returns proper amplitude (absolute value) and phase (angle) of cross-wavelet

- **Backend Information:**
  - Added `type` field showing backend class name
  - Improved error messages for unavailable backends

#### Performance Improvements
- Optimized WCT endpoint for faster computation without significance testing
- Reduced default Monte Carlo count for better response times
- FPGA backend support for hardware-accelerated computations

#### Known Limitations
- ⚠️ WCT significance testing with `mc_count > 100` may timeout (30+ seconds)
- ⚠️ Signals shorter than 200 points may fail with "Series too short" error
- ⚠️ Monte Carlo simulations are synchronous and may block on large datasets
- 📋 Batch processing endpoints (planned for v1.1)
- 📋 Async/background job queue (planned for v1.1)

#### Bug Fixes
- Fixed backend parameter handling in benchmark endpoint
- Fixed XWT unpacking error (was expecting 5 values, now correctly handles 4)
- Fixed field name mismatches in XWT response
- Improved validation for nonexistent backends

#### Test Coverage
- 88/104 tests passing (84.6%)
- Full coverage for: CWT, backends, hardware, benchmark, health endpoints
- Partial coverage for: WCT (with/without significance), XWT

---

### Version 0.1.0-alpha

**Released:** 2024-12-XX (Initial Development)

- ✅ Basic backend management endpoints
- ✅ Health check and status endpoints
- ✅ CORS support for web applications
- ✅ Initial wavelet analysis endpoints

---

## License

PyCWT-mod is released under a BSD-style open source licence.

Copyright (c) 2024-2025 Sebastian Krieger, Nabil Freij, and contributors.

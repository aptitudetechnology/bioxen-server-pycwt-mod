# KaiABC Circadian Oscillator - Open Source Software Stack 📚

**Purpose:** Comprehensive list of open source libraries for implementing the KaiABC Circadian Oscillator API server

**Date:** October 6, 2025  
**Target Hardware:** Raspberry Pi Pico (MicroPython) and ELM11 (Lua) sensor nodes  
**Server Architecture:** FastAPI-based REST/WebSocket server with ODE integration

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Server-Side Libraries](#server-side-libraries)
3. [Client-Side Libraries](#client-side-libraries)
4. [Database & Storage](#database--storage)
5. [Monitoring & Operations](#monitoring--operations)
6. [Complete Requirements Files](#complete-requirements-files)

---

## System Architecture

```
┌──────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│ RPi Pico / ELM11 │────────▶│  KaiABC Server   │◀────────│  Web Dashboard  │
│  (Sensor Node)   │  MQTT/  │  (Computation)   │  HTTP/  │   (Monitor)     │
│  - BME280        │  HTTP   │  - ODE Solver    │  WS     │  - Visualization│
│  - MicroPython/  │         │  - Kalman Filter │         │  - Control      │
│    Lua Runtime   │         │  - ETF Engine    │         └─────────────────┘
└──────────────────┘         └──────────────────┘
                                     │
                            ┌────────▼────────┐
                            │  Time Series DB │
                            │  (InfluxDB/     │
                            │   TimescaleDB)  │
                            └─────────────────┘
```

---

## Server-Side Libraries

### 🌐 Web Framework & API

#### **FastAPI** - Modern Async Web Framework
- **Version:** 0.104.1+
- **Purpose:** REST API server with async support
- **Features:**
  - Automatic OpenAPI/Swagger documentation
  - Pydantic data validation
  - Async/await support for WebSockets
  - High performance (on par with NodeJS, Go)
  - Dependency injection
- **Use Cases:**
  - All REST endpoints
  - WebSocket connections for real-time streaming
  - Request validation and error handling
- **License:** MIT
- **Installation:** `pip install fastapi==0.104.1`
- **Documentation:** https://fastapi.tiangolo.com/
- **Why FastAPI:** Native WebSocket support, perfect for real-time oscillator state streaming

#### **Uvicorn** - ASGI Server
- **Version:** 0.24.0+
- **Purpose:** Lightning-fast ASGI server
- **Features:**
  - HTTP/1.1 and HTTP/2
  - WebSocket support
  - Hot reload in development
  - Production-ready performance
- **Use Cases:** Production server deployment
- **License:** BSD-3-Clause
- **Installation:** `pip install uvicorn[standard]==0.24.0`
- **Documentation:** https://www.uvicorn.org/

#### **Pydantic** - Data Validation
- **Version:** 2.4.2+
- **Purpose:** Data validation using Python type hints
- **Features:**
  - Automatic validation
  - JSON Schema generation
  - Custom validators
  - Nested model support
- **Use Cases:**
  - OscillatorState model
  - SensorReading validation
  - ETFParameters validation
  - API request/response schemas
- **License:** MIT
- **Installation:** `pip install pydantic==2.4.2`
- **Documentation:** https://docs.pydantic.dev/

#### **python-socketio** - WebSocket Management
- **Version:** Latest
- **Purpose:** Socket.IO integration for advanced real-time features
- **Features:**
  - Room/namespace support
  - Event-based communication
  - Automatic reconnection
- **Use Cases:** Alternative to raw WebSockets for complex real-time features
- **License:** MIT
- **Installation:** `pip install python-socketio`
- **Documentation:** https://python-socketio.readthedocs.io/

---

### 🔬 Scientific Computing & ODE Solvers

#### **NumPy** - Array Operations
- **Version:** 1.24.3+
- **Purpose:** Foundation for numerical computing
- **Features:**
  - N-dimensional arrays
  - Linear algebra
  - FFT operations
  - Broadcasting
- **Use Cases:**
  - State vector operations for 6-state KaiABC model
  - Matrix operations for Kalman filter
  - Numerical computations
- **License:** BSD-3-Clause
- **Installation:** `pip install numpy==1.24.3`
- **Documentation:** https://numpy.org/doc/

#### **SciPy** - Scientific Algorithms
- **Version:** 1.11.3+
- **Purpose:** Advanced scientific computing
- **Key Modules:**
  - `scipy.integrate` - **ODE solvers** (RKF45, Radau, LSODA)
  - `scipy.signal` - Signal processing, filtering
  - `scipy.optimize` - Parameter optimization
  - `scipy.linalg` - Linear algebra operations
  - `scipy.interpolate` - Interpolation for PRC lookup
- **Use Cases:**
  - **Critical:** ODE integration for KaiABC dynamics
  - Kalman filter implementation
  - PRC (Phase Response Curve) interpolation
  - Period detection and analysis
- **License:** BSD-3-Clause
- **Installation:** `pip install scipy==1.11.3`
- **Documentation:** https://scipy.org/

**Example ODE Integration:**
```python
from scipy.integrate import solve_ivp
import numpy as np

def kaiabc_ode(t, y, temp_k, params):
    """
    KaiABC 6-state ODE system
    y = [C_U, C_S, C_T, C_ST, A_free, CABC_complex]
    """
    C_U, C_S, C_T, C_ST, A_free, CABC = y
    
    # Temperature-dependent rate constants (Arrhenius)
    k = arrhenius_scale(params, temp_k)
    
    # ODEs (simplified example)
    dC_U_dt = -k['phos'] * C_U + k['dephos'] * C_S
    dC_S_dt = k['phos'] * C_U - k['trans'] * C_S
    # ... (complete 6-state system)
    
    return [dC_U_dt, dC_S_dt, dC_T_dt, dC_ST_dt, dA_free_dt, dCABC_dt]

# Integrate
solution = solve_ivp(
    kaiabc_ode, 
    t_span=(0, 86400),  # 24 hours
    y0=[0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
    method='RK45',
    args=(298.15, params)
)
```

#### **PyDSTool** - Dynamical Systems Toolkit (Optional)
- **Version:** Latest
- **Purpose:** Advanced dynamical systems analysis
- **Features:**
  - Phase plane analysis
  - Bifurcation analysis
  - Parameter continuation
  - Event detection
- **Use Cases:**
  - Advanced oscillator analysis
  - Bifurcation studies
  - Research and development
- **License:** BSD-3-Clause
- **Installation:** `pip install pydstool`
- **Documentation:** https://pydstool.github.io/PyDSTool/

---

### 🎯 Kalman Filtering & State Estimation

#### **FilterPy** - Kalman Filters
- **Version:** Latest
- **Purpose:** Kalman filter and state estimation library
- **Features:**
  - Kalman Filter (linear)
  - Extended Kalman Filter (EKF)
  - Unscented Kalman Filter (UKF)
  - Particle filters
- **Use Cases:**
  - Temperature sensor filtering
  - Noise reduction for BME280 readings
  - State estimation with noisy measurements
- **License:** MIT
- **Installation:** `pip install filterpy`
- **Documentation:** https://filterpy.readthedocs.io/

**Example Kalman Filter for Temperature:**
```python
from filterpy.kalman import KalmanFilter
import numpy as np

# Initialize filter
kf = KalmanFilter(dim_x=1, dim_z=1)  # 1D temperature

# State transition matrix (constant temperature)
kf.F = np.array([[1.]])

# Measurement matrix
kf.H = np.array([[1.]])

# Process noise
kf.Q = np.array([[0.01]])

# Measurement noise (BME280 noise)
kf.R = np.array([[0.05]])

# Initial state
kf.x = np.array([[298.15]])  # 25°C in Kelvin

# Predict and update cycle
kf.predict()
kf.update(temp_measurement)
filtered_temp = kf.x[0, 0]
```

#### **statsmodels** - Statistical Models
- **Version:** Latest
- **Purpose:** Statistical analysis and time series
- **Features:**
  - Time series analysis
  - ARIMA models
  - Seasonal decomposition
  - Statistical tests
- **Use Cases:**
  - Period stability analysis
  - Trend detection
  - Statistical validation
- **License:** BSD-3-Clause
- **Installation:** `pip install statsmodels`
- **Documentation:** https://www.statsmodels.org/

---

### 📊 Time Series Analysis & Circadian Metrics

#### **Astropy** - Astronomy/Time Series Tools
- **Version:** 5.3.4+
- **Purpose:** Lomb-Scargle periodogram for period detection
- **Features:**
  - `astropy.timeseries.LombScargle` - Frequency analysis
  - Handles irregular sampling
  - Statistical significance testing
- **Use Cases:**
  - Circadian period detection
  - Period stability analysis
  - Detecting deviations from 24h period
- **License:** BSD-3-Clause
- **Installation:** `pip install astropy==5.3.4`
- **Documentation:** https://docs.astropy.org/

#### **CosinorPy** - Cosinor Analysis
- **Version:** Latest
- **Purpose:** Cosinor analysis for biological rhythms
- **Features:**
  - Single/multi-component cosinor
  - Population-mean cosinor
  - Acrophase calculation
  - Amplitude and MESOR estimation
- **Use Cases:**
  - Circadian rhythm characterization
  - Phase (acrophase) calculation
  - Period validation
- **License:** GPL
- **Installation:** `pip install cosinorpy`
- **Documentation:** https://github.com/mmoskon/CosinorPy

#### **PyWavelets** - Wavelet Analysis
- **Version:** 1.4.1+
- **Purpose:** Wavelet transforms for time-frequency analysis
- **Features:**
  - Continuous Wavelet Transform (CWT)
  - Discrete Wavelet Transform (DWT)
  - Multiple wavelet families
- **Use Cases:**
  - Transient detection in oscillator
  - Time-frequency analysis
  - Non-stationary dynamics
- **License:** MIT
- **Installation:** `pip install PyWavelets==1.4.1`
- **Documentation:** https://pywavelets.readthedocs.io/

---

### 🔄 MQTT & Message Queuing

#### **paho-mqtt** - MQTT Client
- **Version:** 1.6.1+
- **Purpose:** MQTT protocol for lightweight sensor communication
- **Features:**
  - MQTT 3.1.1 and 5.0 support
  - TLS/SSL support
  - QoS levels 0, 1, 2
  - Will messages
- **Use Cases:**
  - Sensor data ingestion from Pico/ELM11 nodes
  - Low-bandwidth communication
  - Publish/subscribe pattern
- **License:** EPL/EDL
- **Installation:** `pip install paho-mqtt==1.6.1`
- **Documentation:** https://www.eclipse.org/paho/

**Example MQTT Subscription:**
```python
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    """Handle incoming sensor data"""
    if msg.topic.startswith("kaiabc/sensors/"):
        node_id = msg.topic.split('/')[-1]
        data = json.loads(msg.payload)
        process_sensor_reading(node_id, data)

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt.kaiabc.local", 1883)
client.subscribe("kaiabc/sensors/#")
client.loop_forever()
```

#### **celery** - Distributed Task Queue (Optional)
- **Version:** Latest
- **Purpose:** Asynchronous task processing
- **Features:**
  - Task scheduling
  - Periodic tasks
  - Result backends
- **Use Cases:**
  - Batch processing
  - Scheduled analysis
  - Long-running computations
- **License:** BSD-3-Clause
- **Installation:** `pip install celery`
- **Documentation:** https://docs.celeryq.dev/

---

### 🗄️ Database & Storage

#### **SQLAlchemy** - SQL ORM
- **Version:** 2.0+
- **Purpose:** Database abstraction and ORM
- **Features:**
  - Multiple database backends
  - Async support
  - Migration tools (Alembic)
  - Query builder
- **Use Cases:**
  - Node registration
  - Configuration storage
  - User management
- **License:** MIT
- **Installation:** `pip install sqlalchemy[asyncio]`
- **Documentation:** https://www.sqlalchemy.org/

#### **asyncpg** - Async PostgreSQL Driver
- **Version:** Latest
- **Purpose:** High-performance PostgreSQL driver
- **Features:**
  - Native async/await
  - Connection pooling
  - Prepared statements
- **Use Cases:** PostgreSQL database for metadata
- **License:** Apache 2.0
- **Installation:** `pip install asyncpg`
- **Documentation:** https://magicstack.github.io/asyncpg/

#### **aioinflux** - InfluxDB Client
- **Version:** Latest
- **Purpose:** Async InfluxDB client for time series
- **Features:**
  - Async API
  - Batch writes
  - Downsampling
- **Use Cases:**
  - Time series storage for oscillator state
  - Sensor data logging
  - Historical analysis
- **License:** MIT
- **Installation:** `pip install aioinflux`
- **Documentation:** https://github.com/gusutabopb/aioinflux

**Alternative: TimescaleDB with asyncpg**
- PostgreSQL extension for time series
- Better integration if already using PostgreSQL
- SQL queries on time series data

---

### 🔐 Authentication & Security

#### **python-jose** - JWT Tokens
- **Version:** Latest
- **Purpose:** JSON Web Tokens for authentication
- **Features:**
  - JWT creation and validation
  - Multiple algorithms
  - Claims management
- **Use Cases:**
  - Web client authentication
  - Token-based API access
- **License:** MIT
- **Installation:** `pip install python-jose[cryptography]`
- **Documentation:** https://python-jose.readthedocs.io/

#### **passlib** - Password Hashing
- **Version:** Latest
- **Purpose:** Password hashing and verification
- **Features:**
  - bcrypt, argon2, scrypt support
  - Context-based password policies
- **Use Cases:** User password management
- **License:** BSD
- **Installation:** `pip install passlib[bcrypt]`
- **Documentation:** https://passlib.readthedocs.io/

#### **cryptography** - Cryptographic Primitives
- **Version:** Latest
- **Purpose:** Low-level cryptographic operations
- **Features:**
  - Symmetric encryption
  - Key derivation
  - X.509 certificates
- **Use Cases:**
  - API key generation
  - Secure storage
- **License:** Apache 2.0 / BSD
- **Installation:** `pip install cryptography`
- **Documentation:** https://cryptography.io/

---

### 📈 Monitoring & Operations

#### **prometheus-client** - Metrics Export
- **Version:** 0.18.0+
- **Purpose:** Prometheus instrumentation
- **Features:**
  - Counter, Gauge, Histogram, Summary metrics
  - HTTP exposition
  - Multi-process support
- **Use Cases:**
  - ODE solver performance metrics
  - Request latency tracking
  - Node connectivity monitoring
- **License:** Apache 2.0
- **Installation:** `pip install prometheus-client==0.18.0`
- **Documentation:** https://github.com/prometheus/client_python

**Example Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Track sensor readings
sensor_readings = Counter(
    'kaiabc_sensor_readings_total',
    'Total sensor readings received',
    ['node_id', 'sensor_type']
)

# Track ODE solver time
ode_solver_time = Histogram(
    'kaiabc_ode_solver_seconds',
    'ODE solver execution time',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

# Track connected nodes
connected_nodes = Gauge(
    'kaiabc_connected_nodes',
    'Number of connected sensor nodes'
)
```

#### **psutil** - System Monitoring
- **Version:** 5.9.6+
- **Purpose:** System resource monitoring
- **Features:**
  - CPU, memory, disk, network stats
  - Process management
  - Cross-platform
- **Use Cases:**
  - Health checks
  - Resource alerting
  - Performance diagnostics
- **License:** BSD-3-Clause
- **Installation:** `pip install psutil==5.9.6`
- **Documentation:** https://psutil.readthedocs.io/

#### **structlog** - Structured Logging
- **Version:** Latest
- **Purpose:** Structured logging for better analysis
- **Features:**
  - JSON output
  - Context binding
  - Log processors
- **Use Cases:**
  - Detailed event logging
  - Debugging
  - Audit trails
- **License:** MIT/Apache 2.0
- **Installation:** `pip install structlog`
- **Documentation:** https://www.structlog.org/

---

### 🧪 Testing

#### **pytest** - Testing Framework
- **Version:** 7.4.3+
- **Purpose:** Comprehensive testing
- **Features:**
  - Simple assertions
  - Fixtures
  - Parameterized tests
  - Plugin ecosystem
- **Use Cases:**
  - Unit tests for ODE solver
  - API endpoint tests
  - Integration tests
- **License:** MIT
- **Installation:** `pip install pytest==7.4.3`
- **Documentation:** https://docs.pytest.org/

#### **pytest-asyncio** - Async Test Support
- **Version:** 0.21.1+
- **Purpose:** Test async functions
- **Features:**
  - Async fixtures
  - Event loop management
- **Use Cases:** Testing async API endpoints, WebSocket handlers
- **License:** Apache 2.0
- **Installation:** `pip install pytest-asyncio==0.21.1`
- **Documentation:** https://pytest-asyncio.readthedocs.io/

#### **httpx** - HTTP Client for Testing
- **Version:** Latest
- **Purpose:** Modern HTTP client with async support
- **Features:**
  - Sync and async APIs
  - HTTP/2 support
  - Test client for FastAPI
- **Use Cases:**
  - API endpoint testing
  - Integration tests
- **License:** BSD-3-Clause
- **Installation:** `pip install httpx`
- **Documentation:** https://www.python-httpx.org/

---

## Client-Side Libraries

### 📱 Raspberry Pi Pico (MicroPython)

#### **MicroPython Core**
- **Version:** v1.20+
- **Purpose:** Python 3 for microcontrollers
- **Built-in Modules:**
  - `machine` - Hardware access (I2C, PWM, Pin)
  - `network` - WiFi connectivity
  - `urequests` - HTTP client
  - `ujson` - JSON encoding/decoding
  - `utime` - Time functions
- **Installation:** Flash Pico with MicroPython firmware
- **Documentation:** https://docs.micropython.org/

#### **micropython-bme280** - BME280 Sensor Driver
- **Purpose:** Temperature, humidity, pressure sensor
- **Features:**
  - I2C communication
  - Calibration support
  - Multiple oversampling modes
- **Installation:** Copy to Pico filesystem
- **Repository:** https://github.com/robert-hh/BME280

#### **umqtt.simple** - MQTT Client
- **Purpose:** Lightweight MQTT for MicroPython
- **Features:**
  - Publish/subscribe
  - QoS support
  - Minimal memory footprint
- **Use Cases:** Send sensor data to server via MQTT
- **Installation:** Built-in or from micropython-lib
- **Documentation:** https://github.com/micropython/micropython-lib

**Example Pico Client:**
```python
import urequests
import ujson
from machine import Pin, I2C, PWM
import bme280

# Setup I2C for BME280
i2c = I2C(0, scl=Pin(1), sda=Pin(0))
bme = bme280.BME280(i2c=i2c)

# Setup PWM for output
pwm = PWM(Pin(15))
pwm.freq(1000)

def send_sensor_data():
    temp, pressure, humidity = bme.values
    
    data = {
        "node_id": "pico_001",
        "timestamp": get_iso_timestamp(),
        "temperature_kelvin": float(temp) + 273.15,
        "humidity_percent": float(humidity),
        "pressure_pa": float(pressure) * 100
    }
    
    response = urequests.post(
        "http://192.168.1.100:8000/v1/sensors/temperature",
        headers={"X-API-Key": "kaiabc_node_pico_001"},
        json=data
    )
    return response.json()
```

---

### 🔧 ELM11 (Lua)

#### **Lua Core**
- **Version:** Lua 5.1-5.3 (depending on ELM11 model)
- **Purpose:** Lightweight scripting language
- **Built-in Libraries:**
  - `string`, `table`, `math`
  - `os`, `io`

#### **LuaSocket** - Networking
- **Purpose:** TCP/UDP sockets, HTTP client
- **Features:**
  - HTTP requests
  - TCP/UDP communication
  - URL parsing
- **Use Cases:** HTTP requests to server
- **Documentation:** http://w3.impa.br/~diego/software/luasocket/

#### **lua-cjson** - JSON Encoding
- **Purpose:** Fast JSON encoder/decoder
- **Features:**
  - Parse JSON responses
  - Encode sensor data
- **Installation:** Often included with NodeMCU-like firmware
- **Repository:** https://github.com/mpx/lua-cjson

#### **ELM11 I2C Library**
- **Purpose:** I2C communication for sensors
- **Features:** Read/write to I2C devices like BME280
- **Documentation:** Check ELM11-specific documentation

**Example ELM11 Client:**
```lua
local http = require("socket.http")
local json = require("cjson")

function send_sensor_data(temp_k, humidity)
    local payload = json.encode({
        node_id = "elm11_001",
        timestamp = os.date("!%Y-%m-%dT%H:%M:%SZ"),
        temperature_kelvin = temp_k,
        humidity_percent = humidity
    })
    
    local response = http.request({
        url = "http://192.168.1.100:8000/v1/sensors/temperature",
        method = "POST",
        headers = {
            ["X-API-Key"] = "kaiabc_node_elm11_001",
            ["Content-Type"] = "application/json"
        },
        source = ltn12.source.string(payload)
    })
    
    return json.decode(response)
end
```

---

## Database & Storage

### Time Series Databases

#### **InfluxDB** - Optimized for Time Series
- **Version:** 2.x
- **Purpose:** High-performance time series database
- **Features:**
  - Efficient compression
  - Downsampling
  - Retention policies
  - Flux query language
- **Use Cases:**
  - Store oscillator state history
  - Sensor data logging
  - Long-term storage
- **License:** MIT
- **Installation:** Docker or native
- **Documentation:** https://docs.influxdata.com/

**Flux Query Example:**
```flux
from(bucket: "kaiabc")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "oscillator_state")
  |> filter(fn: (r) => r.node_id == "pico_001")
  |> filter(fn: (r) => r._field == "circadian_time")
```

#### **TimescaleDB** - PostgreSQL Extension
- **Version:** 2.x
- **Purpose:** Time series on PostgreSQL
- **Features:**
  - SQL interface
  - Automatic partitioning
  - Continuous aggregates
  - Compression
- **Use Cases:**
  - Time series with relational data
  - Complex queries
  - Existing PostgreSQL infrastructure
- **License:** Apache 2.0 (Community) / Commercial
- **Installation:** PostgreSQL extension
- **Documentation:** https://docs.timescale.com/

**SQL Query Example:**
```sql
SELECT time_bucket('1 hour', timestamp) AS hour,
       avg(circadian_time) AS avg_ct,
       avg(koa_metric) AS avg_koa
FROM oscillator_states
WHERE node_id = 'pico_001'
  AND timestamp > NOW() - INTERVAL '7 days'
GROUP BY hour
ORDER BY hour;
```

### Caching

#### **Redis** - In-Memory Data Store
- **Version:** 7.0+
- **Purpose:** High-speed caching and pub/sub
- **Features:**
  - Key-value store
  - Pub/sub messaging
  - Sorted sets for time series
  - Persistence options
- **Use Cases:**
  - Cache latest oscillator state
  - Real-time pub/sub for WebSocket
  - Session storage
- **License:** BSD-3-Clause
- **Installation:** Docker or native
- **Documentation:** https://redis.io/docs/

#### **redis-py** - Redis Python Client
- **Version:** Latest
- **Purpose:** Async Redis client for Python
- **Installation:** `pip install redis[hiredis]`
- **Documentation:** https://redis-py.readthedocs.io/

---

## Monitoring & Operations

### Observability Stack

#### **Prometheus** - Metrics Collection
- **Version:** 2.x
- **Purpose:** Time series metrics database
- **Features:**
  - Pull-based metrics collection
  - PromQL query language
  - Alerting with Alertmanager
- **Use Cases:**
  - System metrics
  - Application metrics
  - Performance monitoring
- **License:** Apache 2.0
- **Documentation:** https://prometheus.io/docs/

#### **Grafana** - Visualization
- **Version:** 10.x
- **Purpose:** Metrics and logs visualization
- **Features:**
  - Dashboards
  - Alerts
  - Multiple data sources
  - Variables and templating
- **Use Cases:**
  - Real-time oscillator monitoring
  - System health dashboards
  - Historical analysis
- **License:** AGPL (open source edition)
- **Documentation:** https://grafana.com/docs/

**Example Dashboard Panels:**
- Circadian time vs clock time
- KOA metric over 24h cycle
- Node connectivity status
- Temperature entrainment events
- ODE solver performance

#### **Loki** - Log Aggregation
- **Version:** 2.x
- **Purpose:** Like Prometheus, but for logs
- **Features:**
  - Label-based indexing
  - LogQL query language
  - Integration with Grafana
- **Use Cases:**
  - Centralized logging
  - Error tracking
  - Debugging
- **License:** AGPL
- **Documentation:** https://grafana.com/docs/loki/

---

## Complete Requirements Files

### Server Core Requirements

```txt
# requirements-server.txt
# KaiABC Server Core Dependencies

# ===== Web Framework =====
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6
python-socketio==5.10.0

# ===== Scientific Computing =====
numpy==1.24.3
scipy==1.11.3

# ===== Time Series & Circadian Analysis =====
astropy==5.3.4
PyWavelets==1.4.1
cosinorpy

# ===== Kalman Filtering =====
filterpy
statsmodels

# ===== MQTT Communication =====
paho-mqtt==1.6.1

# ===== Database =====
sqlalchemy[asyncio]==2.0.23
asyncpg==0.29.0
aioinflux

# ===== Authentication =====
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
cryptography==41.0.5

# ===== Monitoring =====
prometheus-client==0.18.0
psutil==5.9.6
structlog==23.2.0

# ===== HTTP Client =====
httpx==0.25.1

# ===== Testing =====
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
```

### Optional Advanced Features

```txt
# requirements-advanced.txt
# Optional Advanced Features

# ===== Dynamical Systems Analysis =====
pydstool

# ===== Task Queue =====
celery[redis]==5.3.4

# ===== Redis =====
redis[hiredis]==5.0.1

# ===== Additional Time Series =====
pandas==2.1.3
xarray==2023.11.0

# ===== Machine Learning (Future) =====
scikit-learn==1.3.2
pytorch  # For adaptive entrainment

# ===== Visualization (Development) =====
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
```

### Development Tools

```txt
# requirements-dev.txt
# Development and Testing Tools

# ===== Code Quality =====
black==23.11.0
flake8==6.1.0
mypy==1.7.1
isort==5.12.0

# ===== Testing =====
pytest-watch==4.2.0
pytest-benchmark==4.0.0
hypothesis==6.92.0

# ===== Documentation =====
mkdocs==1.5.3
mkdocs-material==9.4.14
mkdocstrings[python]==0.24.0

# ===== Profiling =====
py-spy==0.3.14
memory-profiler==0.61.0
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  kaiabc-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/kaiabc
      - REDIS_URL=redis://redis:6379
      - INFLUX_URL=http://influxdb:8086
    depends_on:
      - postgres
      - redis
      - influxdb

  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_PASSWORD=secretpass
      - POSTGRES_DB=kaiabc
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  influxdb:
    image: influxdb:2.7
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=secretpass
      - DOCKER_INFLUXDB_INIT_ORG=kaiabc
      - DOCKER_INFLUXDB_INIT_BUCKET=oscillator
    volumes:
      - influxdb_data:/var/lib/influxdb2

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  influxdb_data:
  prometheus_data:
  grafana_data:
```

---

## Installation Guide

### Step 1: Server Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements-server.txt

# Install development tools (optional)
pip install -r requirements-dev.txt
```

### Step 2: Database Setup

```bash
# Start database services with Docker
docker-compose up -d postgres redis influxdb

# Initialize TimescaleDB extension
psql -h localhost -U user -d kaiabc -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Create InfluxDB bucket
influx bucket create --name oscillator --org kaiabc
```

### Step 3: Run Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode (with workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 4: Raspberry Pi Pico Setup

```bash
# Flash MicroPython firmware
# Download from: https://micropython.org/download/rp2-pico/

# Copy sensor libraries to Pico
ampy --port /dev/ttyACM0 put bme280.py
ampy --port /dev/ttyACM0 put main.py

# Or use Thonny IDE for easier development
```

### Step 5: Monitoring Setup

```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access Grafana
# http://localhost:3000 (admin/admin)

# Import dashboard
# Use dashboard JSON from ./grafana/dashboards/
```

---

## Hardware-Specific Considerations

### Raspberry Pi Pico

**Strengths:**
- ✅ MicroPython: Excellent documentation and community
- ✅ Dual-core: Can dedicate one core to sensor reading
- ✅ PIO: Programmable I/O for custom protocols
- ✅ Low cost (~$4)

**Limitations:**
- ❌ No FPU: Software floating-point (slower)
- ❌ Limited RAM: 264KB SRAM
- ❌ Requires external WiFi: ESP8266/ESP32 module or Pico W

**Recommendation:** Perfect for sensor nodes, server-side ODE computation essential

### ELM11

**Strengths:**
- ✅ Lua scripting: Easy to modify in field
- ✅ Built-in WiFi: Some models include networking
- ✅ Industrial-grade: Ruggedized options available

**Limitations:**
- ❌ Lua overhead: Slower than compiled code
- ❌ Variable specs: Different ELM11 models have different capabilities
- ❌ Less documentation: Compared to Raspberry Pi ecosystem

**Recommendation:** Good for research deployments where scripting flexibility is valuable

---

## Performance Benchmarks

### Server-Side ODE Integration

**SciPy RK45 Solver:**
- 1 hour simulation: ~50ms (Intel i7, single core)
- 24 hour simulation: ~500ms
- Memory: ~10MB per node

**Optimization:**
- Use compiled Numba JIT: 5-10x speedup
- Batch multiple nodes: Amortize overhead
- Cache intermediate results

### Database Performance

**InfluxDB:**
- Write: 100,000+ points/second
- Query: <10ms for recent data
- Compression: ~10:1 ratio

**TimescaleDB:**
- Write: 50,000+ rows/second
- Query: <50ms with proper indexes
- Compression: ~5:1 ratio

### WebSocket Latency

- Server → Client: <5ms (local network)
- Update rate: 1-10 Hz (configurable)
- Concurrent clients: 1000+ (with proper scaling)

---

## License Summary

### Permissive Licenses (Commercial-Friendly)
- **MIT:** FastAPI, NumPy, Pydantic, FilterPy, Redis
- **BSD:** SciPy, Astropy, SQLAlchemy, PyWavelets
- **Apache 2.0:** asyncpg, Cryptography, Prometheus

### Copyleft Licenses (Check Requirements)
- **GPL:** CosinorPy (use for research, check for commercial)
- **AGPL:** Grafana, Loki (open source edition)

**Note:** All core dependencies use permissive licenses suitable for commercial use.

---

## Related Documents

- `Kai-ABC-api-specification-planning.md` - Full API specification
- `wishful-server/wishful-software.md` - BioXen Wishful server software stack
- `api-specification-document.md` - PyCWT-mod API specification

---

**Last Updated:** October 6, 2025  
**Maintained By:** BioXen Development Team  
**Version:** 1.0.0

# KaiABC Circadian Oscillator API Server

**Version:** 1.0.0  
**Status:** 🏗️ Design Complete, Implementation In Progress  
**Date:** October 9, 2025

---

## Overview

The KaiABC Circadian Oscillator API Server is a REST/WebSocket service for simulating and monitoring the cyanobacterial KaiABC protein circadian clock system. Unlike traditional circadian models that require transcription-translation feedback loops, the KaiABC system operates purely through protein phosphorylation cycles and can oscillate **in vitro** without DNA.

### Key Features

- ⏰ **24-hour oscillation** without transcription
- 🌡️ **Temperature compensation** (Q10 ≈ 1)
- 🔄 **External Temperature Forcing (ETF)** for day/night entrainment
- 📡 **Real-time WebSocket streaming** of oscillator state
- 🔌 **IoT integration** with Raspberry Pi Pico and ELM11 sensor nodes
- 🧮 **ODE-based simulation** using SciPy
- 📊 **Kalman filtering** for sensor noise reduction
- 💾 **Time-series database** storage (InfluxDB/TimescaleDB)

---

## Project Structure

```
kaiabc-server/
├── README.md                                      # This file
├── Kai-ABC-api-specification-planning.md          # Complete API specification (1976 lines)
├── Kai-ABC-api-specification-planning-software.md # Software stack & implementation guide
├── start-KaiABC-server.sh                         # Automated startup script
│
├── api/                                           # FastAPI application (to be created)
│   ├── __init__.py
│   ├── main.py                                    # FastAPI app entry point
│   ├── models/                                    # Pydantic data models
│   │   ├── __init__.py
│   │   ├── oscillator.py                          # OscillatorState, ETFParameters
│   │   └── sensors.py                             # SensorReading, KalmanFilterConfig
│   └── routes/                                    # API endpoints
│       ├── __init__.py
│       ├── oscillator.py                          # Oscillator state endpoints
│       ├── sensors.py                             # Sensor ingestion endpoints
│       ├── entrainment.py                         # ETF and PRC endpoints
│       ├── simulation.py                          # ODE simulation endpoints
│       └── analytics.py                           # Period detection, analysis
│
├── core/                                          # Core simulation logic
│   ├── __init__.py
│   ├── kaiabc_ode.py                              # 6-state KaiABC ODE system
│   ├── etf.py                                     # External Temperature Forcing
│   ├── prc.py                                     # Phase Response Curve
│   ├── kalman.py                                  # Kalman filter for sensors
│   └── period_detection.py                        # Lomb-Scargle period detection
│
├── hardware/                                      # Hardware client code
│   ├── pico/                                      # Raspberry Pi Pico (MicroPython)
│   │   ├── main.py                                # Pico client main loop
│   │   ├── bme280_reader.py                       # Temperature sensor
│   │   └── mqtt_client.py                         # MQTT communication
│   └── elm11/                                     # ELM11 (Lua)
│       ├── main.lua                               # ELM11 client main loop
│       └── http_client.lua                        # HTTP requests
│
├── tests/                                         # Unit and integration tests
│   ├── __init__.py
│   ├── test_ode.py                                # ODE solver tests
│   ├── test_etf.py                                # ETF tests
│   ├── test_api.py                                # API endpoint tests
│   └── test_kalman.py                             # Kalman filter tests
│
├── requirements-server.txt                        # Python dependencies
├── requirements-dev.txt                           # Development dependencies
├── docker-compose.yml                             # Docker services stack
├── Dockerfile                                     # Server container
└── .env.example                                   # Environment variables template
```

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (optional, for databases)
- **Raspberry Pi Pico** with MicroPython (optional, for hardware)
- **ELM11 board** with Lua (optional, for hardware)

### Installation

```bash
# Clone or navigate to this directory
cd kaiabc-server/

# Make startup script executable
chmod +x start-KaiABC-server.sh

# Run setup check
./start-KaiABC-server.sh --check

# Start in development mode
./start-KaiABC-server.sh --dev

# Or start with Docker services
./start-KaiABC-server.sh --docker --dev
```

### Access the API

- **Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **KaiABC API:** http://localhost:8000/api/v1/kaiabc/
- **WebSocket:** ws://localhost:8000/api/v1/kaiabc/ws

### Quick Test

```bash
# Get current oscillator state
curl http://localhost:8000/api/v1/kaiabc/oscillator/state

# Submit temperature reading (from Pico/ELM11)
curl -X POST http://localhost:8000/api/v1/kaiabc/sensor/reading \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "pico_001",
    "temperature_kelvin": 298.15,
    "timestamp": "2025-10-09T12:00:00Z"
  }'

# Run 24-hour simulation
curl -X POST http://localhost:8000/api/v1/kaiabc/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 24.0,
    "dt": 0.1
  }'
```

---

## Documentation

### 📚 API Specification

See **`Kai-ABC-api-specification-planning.md`** (1976 lines) for:
- Complete REST endpoint documentation
- WebSocket event specifications
- Data model schemas
- Client-server architecture
- Authentication & security
- Error handling & rate limiting
- Hardware integration guides (Pico/ELM11)

### 🛠️ Software Stack

See **`Kai-ABC-api-specification-planning-software.md`** for:
- Complete list of open source libraries
- Server-side dependencies (FastAPI, SciPy, FilterPy)
- Client-side dependencies (MicroPython, Lua)
- Database setup (InfluxDB, TimescaleDB, Redis)
- Monitoring stack (Prometheus, Grafana)
- Docker Compose configuration
- Code examples and tutorials
- Performance benchmarks

---

## Core Concepts

### The KaiABC System

The KaiABC circadian oscillator from *Synechococcus elongatus* consists of three proteins:

- **KaiA:** Enhances KaiC phosphorylation
- **KaiB:** Sequesters KaiA when bound to phosphorylated KaiC
- **KaiC:** Central hexamer with autokinase/phosphatase activity

**Key Properties:**
- **Period:** ~24 hours (can oscillate in a test tube!)
- **Temperature Compensation:** Period remains stable across 15-35°C
- **Entrainment:** Can synchronize to temperature cycles (day/night)

### 6-State ODE Model

The API simulates the following state variables:

```python
# Concentrations in μM (micromolar)
KaiA     # Free KaiA
KaiB     # Free KaiB
KaiC     # Free KaiC (unphosphorylated/hypophosphorylated)
KaiAB    # KaiA-KaiB complex
KaiAC    # KaiA-KaiC complex (phosphorylation active)
KaiBC    # KaiB-KaiC complex (sequestration)
```

### External Temperature Forcing (ETF)

Temperature acts as a zeitgeber (time cue) through:

1. **Arrhenius Scaling:** Reaction rates increase with temperature
2. **Structural Coupling:** Conformational changes affect protein interactions
3. **ATPase Constraint:** Temperature affects ATP hydrolysis rates

**Day/Night Simulation:**
- **Day (warm):** 28-30°C → faster kinetics
- **Night (cool):** 22-25°C → slower kinetics
- **Cycle:** 12h warm : 12h cool → entrains oscillator to 24h period

---

## API Endpoints Overview

### Oscillator State

```
GET  /api/v1/kaiabc/oscillator/state         # Get current state
POST /api/v1/kaiabc/oscillator/state         # Set initial state
GET  /api/v1/kaiabc/oscillator/history       # Historical data
```

### Sensor Data

```
POST /api/v1/kaiabc/sensors/temperature      # Single reading
POST /api/v1/kaiabc/sensors/batch            # Batch readings
GET  /api/v1/kaiabc/sensors/nodes            # List all nodes
```

### Entrainment Control

```
GET  /api/v1/kaiabc/entrainment/parameters   # Get ETF config
PUT  /api/v1/kaiabc/entrainment/parameters   # Set ETF config
POST /api/v1/kaiabc/entrainment/phase-shift  # Manual phase shift
GET  /api/v1/kaiabc/entrainment/prc          # Phase Response Curve
```

### Simulation

```
POST /api/v1/kaiabc/simulate                 # Run ODE simulation
GET  /api/v1/kaiabc/simulation/config        # Get solver config
PUT  /api/v1/kaiabc/simulation/config        # Set solver config
```

### Analytics

```
GET  /api/v1/kaiabc/analytics/period         # Detect circadian period
GET  /api/v1/kaiabc/analytics/phase          # Calculate current phase
GET  /api/v1/kaiabc/analytics/entrainment    # Entrainment efficiency
```

### WebSocket Streaming

```
WS   /api/v1/kaiabc/ws                       # Real-time state updates
```

---

## Hardware Integration

### Raspberry Pi Pico (MicroPython)

**Capabilities:**
- BME280 temperature/humidity/pressure sensor
- MQTT or HTTP client for data transmission
- Local PWM control based on circadian phase
- Offline buffering during network outages

**Example Client:**
```python
# hardware/pico/main.py
import bme280
import urequests
import ujson
from machine import Pin, I2C

i2c = I2C(0, scl=Pin(1), sda=Pin(0))
bme = bme280.BME280(i2c=i2c)

while True:
    temp = bme.temperature  # Read sensor
    
    # Send to server
    data = {
        "node_id": "pico_001",
        "temperature_kelvin": temp + 273.15,
        "timestamp": get_timestamp()
    }
    
    response = urequests.post(
        "http://server:8000/api/v1/kaiabc/sensors/temperature",
        json=data
    )
    
    time.sleep(60)  # Sample every minute
```

### ELM11 (Lua)

**Capabilities:**
- HTTP client for sensor submission
- Lower power consumption
- Lua scripting for field deployment

**Example Client:**
```lua
-- hardware/elm11/main.lua
local http = require("socket.http")
local json = require("cjson")

function send_temperature(temp)
    local data = json.encode({
        node_id = "elm11_001",
        temperature_kelvin = temp + 273.15,
        timestamp = os.date("!%Y-%m-%dT%H:%M:%SZ")
    })
    
    http.request({
        url = "http://server:8000/api/v1/kaiabc/sensors/temperature",
        method = "POST",
        headers = {["Content-Type"] = "application/json"},
        source = ltn12.source.string(data)
    })
end

-- Main loop
while true do
    local temp = read_sensor()  -- Hardware-specific
    send_temperature(temp)
    socket.sleep(60)
end
```

---

## Database Schema

### InfluxDB (Time Series)

```python
# Measurement: oscillator_state
fields:
  - KaiA (float)
  - KaiB (float)
  - KaiC (float)
  - KaiAB (float)
  - KaiAC (float)
  - KaiBC (float)
  - phase (float, 0-2π)
  - period (float, hours)
tags:
  - node_id (string)
  - hardware_type (string)

# Measurement: sensor_readings
fields:
  - temperature_kelvin (float)
  - humidity_percent (float)
  - pressure_pa (float)
tags:
  - node_id (string)
  - sensor_type (string)
```

### TimescaleDB (PostgreSQL Extension)

```sql
-- Hypertable for high-frequency data
CREATE TABLE oscillator_states (
    timestamp TIMESTAMPTZ NOT NULL,
    node_id TEXT NOT NULL,
    kaia DOUBLE PRECISION,
    kaib DOUBLE PRECISION,
    kaic DOUBLE PRECISION,
    kaiab DOUBLE PRECISION,
    kaiac DOUBLE PRECISION,
    kaibc DOUBLE PRECISION,
    phase DOUBLE PRECISION,
    PRIMARY KEY (timestamp, node_id)
);

SELECT create_hypertable('oscillator_states', 'timestamp');
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=api --cov=core

# Run specific test file
pytest tests/test_ode.py -v
```

### Development Mode

```bash
# Auto-reload on code changes
./start-KaiABC-server.sh --dev

# Run on different port
./start-KaiABC-server.sh --dev --port 8080

# Enable debug logging
export LOG_LEVEL=DEBUG
./start-KaiABC-server.sh --dev
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f kaiabc-server

# Rebuild after code changes
docker-compose up -d --build

# Stop all services
docker-compose down
```

---

## Performance

### Benchmarks (Target)

- **24h simulation:** ~500ms (Python + SciPy)
- **Period detection:** ~200ms (Lomb-Scargle on 72h data)
- **WebSocket latency:** <50ms for state updates
- **Sensor ingestion:** >1000 readings/sec
- **Database writes:** >10,000 points/sec (InfluxDB)

### Optimization Tips

1. **Use adaptive ODE solver:** RK45 or Radau for efficiency
2. **Batch sensor writes:** Buffer 10-100 readings before DB insert
3. **Cache period calculations:** Recalculate only when needed
4. **Use Redis for hot data:** Recent state, active nodes
5. **Downsample historical data:** Keep 1s resolution for 24h, then aggregate

---

## Deployment

### Production Checklist

- [ ] Set strong JWT secret keys
- [ ] Configure HTTPS/TLS certificates
- [ ] Set up rate limiting (e.g., 100 req/min per IP)
- [ ] Enable CORS for specific origins only
- [ ] Configure database backups (InfluxDB, TimescaleDB)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (Loki)
- [ ] Set resource limits (CPU, memory)
- [ ] Enable authentication for all endpoints
- [ ] Set up health checks and alerting

### Environment Variables

```bash
# Server
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
JWT_SECRET=your-secret-key-here

# Database
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=kaiabc
INFLUXDB_BUCKET=oscillator

TIMESCALEDB_HOST=timescaledb
TIMESCALEDB_PORT=5432
TIMESCALEDB_USER=kaiabc
TIMESCALEDB_PASSWORD=your-password
TIMESCALEDB_DATABASE=kaiabc

REDIS_HOST=redis
REDIS_PORT=6379

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

---

## Research Applications

### Use Cases

1. **Circadian Period Validation**
   - Test if model maintains ~24h period across temperatures
   - Validate Q10 ≈ 1 (temperature compensation)

2. **Entrainment Studies**
   - Simulate day/night temperature cycles
   - Measure phase-locking to external zeitgebers
   - Test conflicting zeitgeber hierarchy

3. **Phase Response Experiments**
   - Apply temperature pulses at different circadian times
   - Measure phase shifts (advances/delays)
   - Build experimental PRC curves

4. **Multi-Node Coordination**
   - Synchronize multiple oscillators
   - Test distributed entrainment strategies
   - Model population-level phenomena

5. **Real-Time Environmental Sensing**
   - Monitor temperature in multiple locations
   - Track oscillator response to natural cycles
   - Validate model predictions against hardware

---

## References

### Key Papers

1. **Rust, M. J., Markson, J. S., Lane, W. S., Fisher, D. S., & O'Shea, E. K. (2007)**  
   *Ordered phosphorylation governs oscillation of a three-protein circadian clock.*  
   Science, 318(5851), 809-812.

2. **Nakajima, M., Imai, K., Ito, H., Nishiwaki, T., Murayama, Y., Iwasaki, H., ... & Kondo, T. (2005)**  
   *Reconstitution of circadian oscillation of cyanobacterial KaiC phosphorylation in vitro.*  
   Science, 308(5720), 414-415.

3. **Qin, X., Byrne, M., Xu, Y., Mori, T., & Johnson, C. H. (2010)**  
   *Coupling of a core post-translational pacemaker to a slave transcription/translation feedback loop in a circadian system.*  
   PLoS Biology, 8(6), e1000394.

4. **Terauchi, K., Kitayama, Y., Nishiwaki, T., Miwa, K., Murayama, Y., Oyama, T., & Kondo, T. (2007)**  
   *ATPase activity of KaiC determines the basic timing for circadian clock of cyanobacteria.*  
   Proceedings of the National Academy of Sciences, 104(41), 16377-16381.

### Related Projects

- **BioXen Wishful Server** - General biological signal analysis API
- **PyCWT-mod** - Wavelet analysis with FPGA acceleration
- **Kakeya IoT** - Circadian-driven home automation

---

## Contributing

### Development Guidelines

1. **Code Style:** Follow PEP 8 for Python, use `black` formatter
2. **Type Hints:** Use type annotations for all functions
3. **Documentation:** Docstrings for all public APIs (Google style)
4. **Testing:** Maintain >80% code coverage
5. **Git:** Descriptive commit messages, feature branches

### Adding New Features

1. Update API specification first (`Kai-ABC-api-specification-planning.md`)
2. Add data models to `api/models/`
3. Implement routes in `api/routes/`
4. Add core logic to `core/`
5. Write tests in `tests/`
6. Update this README

---

## License

See `LICENSE` file in project root.

---

## Support

- **Issues:** Open a GitHub issue
- **Discussions:** Use GitHub Discussions
- **Email:** Contact maintainers

---

## Roadmap

### Phase 1: Core Implementation (Current)
- [x] API specification complete
- [x] Software stack documentation
- [x] Startup script automation
- [ ] FastAPI routes scaffold
- [ ] Pydantic models
- [ ] Basic ODE solver

### Phase 2: Simulation Engine
- [ ] 6-state KaiABC ODE system
- [ ] External Temperature Forcing (ETF)
- [ ] Phase Response Curve (PRC)
- [ ] Kalman filter for sensors
- [ ] Period detection (Lomb-Scargle)

### Phase 3: Hardware Integration
- [ ] Raspberry Pi Pico client (MicroPython)
- [ ] ELM11 client (Lua)
- [ ] MQTT broker setup
- [ ] Sensor data pipeline
- [ ] Real-time WebSocket streaming

### Phase 4: Database & Monitoring
- [ ] InfluxDB time-series storage
- [ ] TimescaleDB PostgreSQL integration
- [ ] Redis caching layer
- [ ] Prometheus metrics
- [ ] Grafana dashboards

### Phase 5: Advanced Features
- [ ] Machine learning for parameter optimization
- [ ] Multi-node synchronization
- [ ] Adaptive entrainment algorithms
- [ ] Web dashboard UI
- [ ] Mobile app integration

---

**Last Updated:** October 9, 2025  
**Version:** 1.0.0  
**Status:** 🏗️ In Development

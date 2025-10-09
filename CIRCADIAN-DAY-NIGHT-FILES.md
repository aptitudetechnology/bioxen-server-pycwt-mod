# Day/Night Cycle & Circadian Rhythm Related Files

**Project:** BioXen Server - PyCWT-mod with KaiABC Circadian Oscillator  
**Date:** October 9, 2025  
**Purpose:** Catalog of all files related to day/night cycles, circadian rhythms, temperature entrainment, and light-dark cycles for use in other projects

---

## 📋 Core Files Overview

This repository contains both **implemented** (PyCWT wavelet analysis) and **aspirational** (circadian simulation) components. Files marked 🔮 are "wishful" - they define ideal APIs but are not yet implemented.

---

## 🧬 KaiABC Circadian Oscillator Project

**Note:** As of October 9, 2025, KaiABC files have been moved to a dedicated `kaiabc-server/` directory to separate this specific implementation from the general BioXen Wishful API.

### Project Location

All KaiABC files are now in: **`kaiabc-server/`**

### Primary API Specification Files

**1. `kaiabc-server/Kai-ABC-api-specification-planning.md`** (1976 lines)
- **Type:** Complete API specification document
- **Status:** Planning/Design phase
- **Contents:**
  - Client-server architecture for KaiABC oscillator
  - REST endpoints for oscillator state, sensor ingestion, entrainment control
  - WebSocket streaming for real-time state updates
  - Data models: OscillatorState, SensorReading, ETFParameters
  - Support for Raspberry Pi Pico (MicroPython) and ELM11 (Lua) sensor nodes
  - External Temperature Forcing (ETF) for day/night entrainment
  - Phase Response Curve (PRC) implementation
  - Kalman filtering for sensor noise reduction
- **Key Sections:**
  - `/oscillator/state` - Get/set 6-protein concentrations (KaiA, KaiB, KaiC, complexes)
  - `/sensors/temperature` - Ingest temperature readings from hardware
  - `/entrainment/parameters` - Configure temperature forcing for day/night cycles
  - `/entrainment/phase-shift` - Apply phase shifts using PRC
  - `/simulation/run` - ODE integration for multi-day simulations
  - `/analytics/period` - Detect circadian period using Lomb-Scargle
- **Related Biology:**
  - Cyanobacteria KaiABC protein system
  - Temperature compensation (Q10 ≈ 1)
  - Circadian period ~24 hours
  - Temperature entrainment without transcription

**2. `kaiabc-server/Kai-ABC-api-specification-planning-software.md`** (1200+ lines)
- **Type:** Software stack documentation
- **Status:** Documentation complete, implementation pending
- **Contents:**
  - Complete list of open source libraries for KaiABC server
  - Server-side: FastAPI, SciPy (ODE solvers), FilterPy (Kalman), CosinorPy, Astropy
  - Client-side: MicroPython for Pico, Lua for ELM11
  - Databases: InfluxDB (time series), TimescaleDB, Redis (caching)
  - Monitoring: Prometheus, Grafana, Loki
  - Docker Compose configuration with all services
  - Code examples: ODE integration, Kalman filtering, MQTT, Pico/ELM11 clients
  - Hardware considerations: Pico (no FPU, 264KB RAM), ELM11 (Lua overhead)
  - Performance benchmarks: 24h simulation ~500ms
- **Key Libraries:**
  - `scipy.integrate.solve_ivp` - ODE solver for 6-state KaiABC model
  - `filterpy` - Kalman filter for temperature sensor noise
  - `astropy.timeseries.LombScargle` - Period detection
  - `micropython-bme280` - Temperature sensor driver for Pico
  - `paho-mqtt` - MQTT for sensor data ingestion

---

## 🔮 Wishful Test Suite (Aspirational APIs)

### Test Files - Circadian & Environmental

**3. `wishful-client-tests/test_circadian_entrainment.py`** (403 lines)
- **Type:** 🔮 Wishful test suite
- **Status:** Defines ideal APIs, not yet implemented
- **Contents:**
  - Light-dark cycle tests (12L:12D, 16L:8D, 8L:16D)
  - Natural dawn/dusk transitions (30min gradual changes)
  - Skeleton photoperiods (brief light pulses)
  - Circadian entrainment validation
  - Phase coherence measurements
  - Kai protein oscillation tests
  - Multi-VM coordination
- **Test Classes:**
  - `TestLightDarkCycles` - Configure photoperiods
  - `TestCircadianEntrainment` - Validate entrainment to L:D cycles
  - `TestPhaseResponseCurve` - PRC for light pulses
  - `TestFreeRunning` - Test in constant darkness (DD)
- **Example Tests:**
  - `test_configure_12L_12D_cycle()` - Standard 12:12 light:dark
  - `test_simulate_natural_dawn_dusk()` - Gradual transitions
  - `test_yeast_entrainment_to_12L_12D()` - Entrainment validation
  - `test_cyanobacteria_kai_oscillation()` - KaiABC specific
- **API Endpoints Defined:**
  - `POST /api/v1/environment/light-cycle/configure`
  - `POST /api/v1/vms/{vm_id}/environment/light-cycle`
  - `GET /api/v1/environment/light-cycle/current`
  - `POST /api/v1/vms/{vm_id}/validate/circadian-entrainment`

**4. `wishful-client-tests/test_temperature_compensation.py`** (372 lines)
- **Type:** 🔮 Wishful test suite
- **Status:** Defines ideal APIs, not yet implemented
- **Contents:**
  - Q10 temperature compensation studies (Q10 ≈ 1 for circadian)
  - Temperature cycle entrainment (12W:12C - 12h warm:12h cool)
  - Heat shock responses (37°C → 42°C)
  - Cold shock responses (37°C → 15°C)
  - Arrhenius kinetics validation
  - Conflicting light/temperature zeitgebers
- **Test Classes:**
  - `TestTemperatureCompensation` - Q10 studies at 15°C, 25°C, 35°C
  - `TestHeatShockResponse` - HSP upregulation, cell death at 50°C
  - `TestTemperatureCycles` - Day/night temperature entrainment
  - `TestTemperatureGradients` - Spatial gradients, optimal temperature
  - `TestArrheniusKinetics` - Arrhenius equation validation
- **Example Tests:**
  - `test_measure_period_at_15_celsius()` - Period at low temp
  - `test_calculate_q10_coefficient()` - Q10 should be ~1.0
  - `test_temperature_cycle_entrainment()` - 12h warm:cool cycles
  - `test_conflicting_light_and_temperature_cycles()` - Zeitgeber hierarchy
  - `test_heat_shock_37_to_42_celsius()` - HSP expression
- **API Endpoints Defined:**
  - `POST /api/v1/vms/{vm_id}/environment/temperature`
  - `POST /api/v1/vms/{vm_id}/environment/temperature-cycle`
  - `POST /api/v1/vms/{vm_id}/environment/temperature-shock`
  - `POST /api/v1/vms/{vm_id}/environment/conflicting-cycles`
  - `GET /api/v1/vms/{vm_id}/analysis/circadian-period`
  - `POST /api/v1/experiments/temperature-compensation`

**5. `wishful-client-tests/conftest.py`** (500+ lines)
- **Type:** Pytest configuration and fixtures
- **Status:** Partially implemented
- **Contents:**
  - `generate_circadian_signal()` - Create 24h sine wave + 12h harmonic + noise
  - Test fixtures for VM configurations
  - Helper functions for EEG-like signals
  - Data generation utilities
- **Key Functions:**
  ```python
  def generate_circadian_signal(duration_hours, sampling_interval_hours=1.0):
      # 24-hour rhythm with 12-hour harmonic + noise
      circadian = np.sin(2 * np.pi * t / 24)
      harmonic = 0.3 * np.sin(2 * np.pi * t / 12)
      noise = 0.2 * np.random.randn(n_samples)
      return circadian + harmonic + noise
  ```

**6. `wishful-client-tests/test_sensor_hardware.py`**
- **Type:** 🔮 Wishful test suite
- **Status:** Hardware integration tests (aspirational)
- **Contents:**
  - BME280 environmental sensor tests
  - LTR-559 light sensor tests
  - Real-time sensor data streaming
  - Kalman filtering for noise reduction

**7. `wishful-client-tests/test_validation.py`**
- **Type:** 🔮 Wishful test suite
- **Status:** Validation framework (aspirational)
- **Contents:**
  - Model validation against experimental data
  - Period detection using Lomb-Scargle
  - Phase coherence measurements

---

## 📚 Documentation Files

**8. `wishful-client-tests/wishful-client-tests-circadian-clock.md`** (821 lines)
- **Type:** Documentation
- **Contents:**
  - Rationale for environmental sensor integration
  - BME280 specifications (temperature, humidity, pressure)
  - LTR-559 specifications (light intensity, spectral response)
  - Why sensors matter for circadian biology
  - Response to critique about emergent vs imposed clocks
  - Example use cases: entrainment, temperature compensation, stress responses
- **Key Points:**
  - Circadian clocks are EMERGENT from feedback loops
  - Light is environmental INPUT that synchronizes oscillators
  - Temperature compensation (Q10 ≈ 1) is validation metric
  - Real biology uses photoreceptors (cryptochromes) for light input

**9. `wishful-client-tests/circadian-clock-claude-thoughts.md`**
- **Type:** Discussion/analysis document
- **Contents:**
  - Conceptual discussion about circadian systems
  - Emergent properties vs external forcing
  - Biological accuracy considerations

**10. `kakeya-prompt.md`**
- **Type:** Project vision document
- **Contents:**
  - Environmental sensors → KaiABC oscillator → Circadian-driven actions
  - IoT integration concepts
  - Distributed oscillator networks
  - Entrainment strategies (autonomous vs environmental consensus)
  - Example use cases: circadian lighting, HVAC, garden irrigation
  - ESP32 + BME280 proof-of-concept ideas
- **Key Concepts:**
  ```python
  class CircadianIoT:
      if 0.7 < phase < 0.9:  # "Night" phase
          # Trigger actions based on circadian state
  ```

---

## 🛠️ Implementation Files

**11. `kaiabc-server/start-KaiABC-server.sh`** (700+ lines)
- **Type:** Bash startup script
- **Status:** Complete, ready to use
- **Contents:**
  - Automated server startup with dependency checks
  - Python 3.9+ version verification
  - Virtual environment creation/activation
  - Dependency installation from requirements
  - Server structure creation if missing
  - Development mode (auto-reload) and production mode (multi-worker)
  - Docker stack integration
  - Health checks for databases (PostgreSQL, Redis, InfluxDB)
- **Features:**
  - Extends existing `server/` directory
  - Adds KaiABC routes to current FastAPI server
  - Creates `server/api/models/kaiabc.py` with data models
  - Creates `server/api/routes/kaiabc.py` with REST endpoints
  - Updates `server/main.py` to include KaiABC router
- **Usage:**
  ```bash
  ./start-KaiABC-server.sh --dev         # Development mode
  ./start-KaiABC-server.sh --prod        # Production mode
  ./start-KaiABC-server.sh --docker      # With Docker services
  ./start-KaiABC-server.sh --check       # Check dependencies only
  ```

---

## 🌐 Wishful Server (Aspirational API)

**12. `wishful-server/wishful-api-specification-document.md`** (1673 lines)
- **Type:** Complete API specification (broader than just KaiABC)
- **Status:** 🔮 Aspirational
- **Contents:**
  - General biological signal processing API
  - Fourier analysis endpoints
  - Wavelet analysis endpoints
  - Laplace transform endpoints
  - Z-transform endpoints
  - Lomb-Scargle periodogram for circadian detection
  - Parameter optimization (SALib, scikit-optimize)
- **Circadian-Relevant Endpoints:**
  - `POST /api/v1/analysis/fourier/periodogram` - Detect 24h periods
  - `POST /api/v1/analysis/lomb-scargle` - Irregular sampling period detection
  - `POST /api/v1/validation/period-stability` - Validate circadian period consistency
- **Example Response:**
  ```json
  {
    "dominant_frequency": 0.0417,  // Hz (~24 hour period)
    "dominant_period": 24.0,       // Hours
    "circadian": true
  }
  ```

**13. `wishful-server/wishful-software.md`** (800+ lines)
- **Type:** Software stack documentation (general BioXen)
- **Contents:**
  - Scientific computing libraries (NumPy, SciPy, Astropy)
  - Wavelet libraries (PyWavelets, pycwt-mod)
  - Optimization libraries (scikit-optimize, SALib)
  - Web framework (FastAPI, Uvicorn)
  - Hardware acceleration (Numba, CuPy, Tang Nano 9K FPGA)
  - Sensor drivers (BME280, LTR-559)
- **Circadian-Relevant:**
  - `astropy.timeseries.LombScargle` - Period detection
  - `scipy.fft` - Frequency analysis
  - `pywavelets` - Wavelet analysis for rhythm detection
  - `ltr559` - Light sensor for photoperiod studies
  - `bme280` - Temperature sensor for entrainment

---

## 📦 Related Library Files

**14. `client-tests/conftest.py`** (500+ lines)
- **Type:** Test configuration (for implemented PyCWT tests)
- **Status:** Implemented
- **Contents:**
  - `generate_circadian_signal()` - 24h rhythm generator
  - Test fixtures for wavelet analysis
  - BME280 mock data generators
  - EEG signal generators

**15. `server/api/routes/kaiabc.py`** (Generated by start script)
- **Type:** FastAPI route handlers
- **Status:** Generated scaffold, needs implementation
- **Contents:**
  - `GET /api/v1/kaiabc/oscillator/state` - Current oscillator state
  - `POST /api/v1/kaiabc/sensor/reading` - Ingest temperature from Pico/ELM11
  - `POST /api/v1/kaiabc/simulate` - Run ODE simulation
  - `POST /api/v1/kaiabc/etf/configure` - Configure temperature forcing
  - `GET /api/v1/kaiabc/analytics/period` - Period detection
  - `WS /api/v1/kaiabc/ws` - WebSocket streaming

**16. `server/api/models/kaiabc.py`** (Generated by start script)
- **Type:** Pydantic data models
- **Status:** Generated scaffold, needs implementation
- **Contents:**
  - `OscillatorState` - 6 protein concentrations (KaiA, KaiB, KaiC, complexes)
  - `SensorReading` - Temperature/humidity/light from sensor nodes
  - `ETFParameters` - External Temperature Forcing config
  - `SimulationRequest` - ODE simulation parameters
  - `KalmanFilterConfig` - Sensor noise reduction config

---

## 🔬 Implementation Roadmaps

**17. `wishful-client-tests/IMPLEMENTATION_ROADMAP.md`**
- **Type:** Implementation guide
- **Contents:**
  - Phase 1: Core VM simulation engine
  - Phase 2: Environmental sensors (light, temperature)
  - Phase 3: Circadian entrainment validation
  - Phase 4: Hardware integration (Pico, ELM11)

**18. `wishful-client-tests/COMPLETE_IMPLEMENTATION_SUMMARY.md`**
- **Type:** Summary document
- **Contents:**
  - Overview of wishful test suite
  - Implementation status
  - Dependencies and requirements

---

## 📝 Additional Context Files

**19. `wishful-server/IMPLEMENTATION_CHECKLIST.md`**
- **Type:** Implementation checklist (4 phases)
- **Contents:**
  - Phase 1: Core API framework
  - Phase 2: Analysis endpoints
  - Phase 3: Optimization and tuning
  - Phase 4: Validation and hardware

**20. `wishful-server/QUICK_START.md`**
- **Type:** Tutorial (30-minute quick start)
- **Contents:**
  - Step-by-step API usage examples
  - Circadian period detection example
  - Fourier analysis examples

---

## 🗂️ File Organization Summary

```
bioxen-server-pycwt-mod/
├── kakeya-prompt.md                               # IoT vision document
├── CIRCADIAN-DAY-NIGHT-FILES.md                   # This file - Complete catalog
│
├── kaiabc-server/                                  # 🆕 Dedicated KaiABC project
│   ├── README.md                                  # KaiABC project overview
│   ├── Kai-ABC-api-specification-planning.md      # Complete KaiABC API (1976 lines)
│   ├── Kai-ABC-api-specification-planning-software.md # Software stack
│   └── start-KaiABC-server.sh                     # Server startup script
│
├── wishful-server/
│   ├── wishful-api-specification-document.md      # General signal processing API
│   ├── wishful-software.md                        # Software stack (general)
│   ├── IMPLEMENTATION_CHECKLIST.md                # 4-phase implementation
│   └── QUICK_START.md                             # 30-minute tutorial
│
├── wishful-client-tests/
│   ├── test_circadian_entrainment.py              # 🔮 Light-dark cycle tests
│   ├── test_temperature_compensation.py           # 🔮 Temperature cycle tests
│   ├── test_sensor_hardware.py                    # 🔮 Hardware sensor tests
│   ├── test_validation.py                         # 🔮 Validation tests
│   ├── conftest.py                                # Test fixtures + circadian signal generator
│   ├── wishful-client-tests-circadian-clock.md    # Rationale document
│   ├── circadian-clock-claude-thoughts.md         # Discussion document
│   └── IMPLEMENTATION_ROADMAP.md                  # Implementation phases
│
├── server/                                         # Existing FastAPI server
│   ├── main.py                                    # Main app (can add KaiABC routes)
│   ├── api/
│   │   ├── models/                                # Add kaiabc.py here
│   │   └── routes/                                # Add kaiabc.py here
│   └── requirements.txt
│
└── client-tests/
    └── conftest.py                                # generate_circadian_signal() helper
```

---

## 🎯 Key Concepts & Terminology

### Circadian Biology
- **Circadian Period:** ~24-hour rhythm (free-running period)
- **Entrainment:** Synchronization to external cycles (light, temperature)
- **Zeitgeber:** Time cue (German: "time giver") - light, temperature
- **Phase:** Position in circadian cycle (CT 0-24)
- **PRC (Phase Response Curve):** How phase shifts depend on timing of stimulus
- **Q10:** Temperature coefficient (Q10 ≈ 1 for temperature-compensated clocks)
- **Free-running:** Oscillation in constant conditions (no zeitgebers)
- **DD:** Constant darkness
- **LL:** Constant light
- **LD:** Light-dark cycles (e.g., 12L:12D)

### KaiABC System
- **KaiA, KaiB, KaiC:** Three proteins in cyanobacteria circadian clock
- **ETF:** External Temperature Forcing (day/night temperature cycles)
- **Temperature Compensation:** Period stays ~24h despite temperature changes
- **In Vitro Oscillation:** KaiABC oscillates without DNA/transcription

### Day/Night Cycles
- **12L:12D:** 12 hours light, 12 hours dark (equinox)
- **16L:8D:** Long day (summer)
- **8L:16D:** Short day (winter)
- **12W:12C:** 12 hours warm, 12 hours cool (temperature cycle)
- **Skeleton Photoperiod:** Brief light pulses instead of continuous light
- **Dawn/Dusk:** Gradual transitions (twilight simulation)

---

## 🚀 Usage in Other Projects

### To Reuse These Files:

1. **For API Design:**
   - Copy `Kai-ABC-api-specification-planning.md` as template
   - Adapt endpoints for your specific oscillator model
   - Use data models (OscillatorState, ETFParameters) as starting point

2. **For Testing:**
   - Copy test files from `wishful-client-tests/`
   - Adapt `test_circadian_entrainment.py` for your light-dark cycles
   - Adapt `test_temperature_compensation.py` for temperature studies
   - Use `generate_circadian_signal()` for synthetic data

3. **For Implementation:**
   - Use `start-KaiABC-server.sh` as startup script template
   - Adapt software stack from `Kai-ABC-api-specification-planning-software.md`
   - Use Docker Compose configuration for services

4. **For Hardware Integration:**
   - Follow sensor specifications in `wishful-client-tests-circadian-clock.md`
   - BME280 for temperature (day/night simulation)
   - LTR-559 for light intensity (photoperiod)
   - Use MicroPython (Pico) or Lua (ELM11) client examples

5. **For Documentation:**
   - Use `kakeya-prompt.md` for project vision
   - Adapt circadian biology concepts for your domain
   - Reference Q10 studies for validation methods

---

## 📊 Implementation Status

| Component | Status | Lines | Completeness |
|-----------|--------|-------|--------------|
| KaiABC API Spec | 📋 Design Complete | 1976 | 100% (design) |
| KaiABC Software Stack | 📋 Design Complete | 1200+ | 100% (design) |
| Light-Dark Tests | 🔮 Wishful | 403 | 0% (impl) |
| Temperature Tests | 🔮 Wishful | 372 | 0% (impl) |
| Startup Script | ✅ Complete | 700+ | 100% |
| Signal Generator | ✅ Complete | ~50 | 100% |
| Routes Scaffold | 🏗️ Generated | ~200 | 20% (impl) |
| Models Scaffold | 🏗️ Generated | ~100 | 50% (impl) |

**Legend:**
- ✅ Complete - Fully implemented
- 🏗️ Generated - Scaffold exists, needs implementation
- 📋 Design Complete - Specification done, implementation pending
- 🔮 Wishful - Aspirational, defines ideal API

---

## 🔗 Dependencies

### Python Libraries (Server)
```
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.4.2
numpy>=1.24.3
scipy>=1.11.3
filterpy>=1.4.5
astropy>=5.3.4
pywavelets>=1.4.1
paho-mqtt>=1.6.1
influxdb-client>=1.38.0
psycopg2-binary>=2.9.9
redis>=5.0.1
prometheus-client>=0.19.0
```

### Python Libraries (Client - Pico)
```
micropython-bme280
umqtt.simple
urequests
ujson
```

### Lua Libraries (Client - ELM11)
```
luasocket
lua-cjson
```

### Infrastructure
```
Docker >= 20.10
Docker Compose >= 2.0
PostgreSQL 15 + TimescaleDB 2.x
InfluxDB 2.x
Redis 7.x
Prometheus + Grafana
```

---

## 📞 Contact & Next Steps

**For Use in Other Projects:**
1. Copy this file list
2. Identify which components you need
3. Adapt API specifications for your model
4. Reuse test patterns for validation
5. Follow software stack recommendations

**Key Questions to Answer:**
- Which oscillator model? (KaiABC, mammalian clock, plant clock, etc.)
- Which zeitgebers? (light, temperature, nutrients, etc.)
- Hardware platform? (server-only, Pico, ESP32, etc.)
- Time scale? (hours, days, weeks)
- Validation metrics? (period, phase, amplitude, Q10)

---

**Document Version:** 1.0  
**Last Updated:** October 9, 2025  
**Maintainer:** BioXen Development Team

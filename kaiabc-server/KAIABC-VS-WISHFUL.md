# KaiABC vs Wishful API - Key Differences

**Date:** October 9, 2025

## Why Separate Projects?

The **KaiABC API** is now in its own `kaiabc-server/` directory because it's **specific enough** to warrant standalone development, while the **Wishful API** remains aspirational and general-purpose.

---

## Quick Comparison

| Aspect | KaiABC API | Wishful API |
|--------|------------|-------------|
| **Location** | `kaiabc-server/` | `wishful-server/` |
| **Status** | 🏗️ In Progress | 🔮 Aspirational |
| **Focus** | Circadian oscillator (KaiABC proteins) | General biological signals |
| **Scope** | Narrow & deep | Broad & shallow |
| **Biology** | Cyanobacteria protein clock | Any biological system |
| **Hardware** | Pico/ELM11 + BME280 sensors | Generic sensors |
| **Analysis** | ODE simulation, entrainment | FFT, wavelets, Laplace, Z-transform |
| **Real-time** | Yes (WebSocket streaming) | Batch processing |
| **Implementation** | ~30% complete | 0% (specs only) |

---

## Detailed Comparison

### 🧬 KaiABC Circadian Oscillator API

**Purpose:** Simulate and monitor the KaiABC protein circadian clock

**Key Features:**
- 6-state ODE system (KaiA, KaiB, KaiC + complexes)
- External Temperature Forcing (ETF) for day/night entrainment
- Phase Response Curve (PRC) for phase shifts
- Real-time WebSocket streaming of oscillator state
- IoT integration with Pico/ELM11 sensor nodes
- Kalman filtering for temperature sensor noise
- 24-hour period with temperature compensation (Q10 ≈ 1)

**Endpoints:**
```
/api/v1/kaiabc/oscillator/state      # Get/set protein concentrations
/api/v1/kaiabc/sensors/temperature   # Ingest temperature from sensors
/api/v1/kaiabc/entrainment/*         # Configure day/night forcing
/api/v1/kaiabc/simulate              # Run ODE integration
/api/v1/kaiabc/analytics/period      # Detect circadian period
/api/v1/kaiabc/ws                    # Real-time streaming
```

**Data Models:**
- `OscillatorState` - 6 protein concentrations + phase
- `ETFParameters` - Temperature forcing config (amplitude, period, phase)
- `SensorReading` - Temperature/humidity/pressure from hardware
- `SimulationRequest` - ODE solver parameters

**Core Libraries:**
- `scipy.integrate.solve_ivp` - ODE solver
- `filterpy` - Kalman filter
- `astropy.timeseries.LombScargle` - Period detection
- `micropython-bme280` - Pico sensor driver

**Use Cases:**
1. Validate 24-hour period across temperatures (Q10 studies)
2. Test entrainment to 12h warm : 12h cool cycles
3. Measure phase response to temperature pulses
4. Monitor real-time oscillator state from distributed sensors
5. Coordinate multiple sensor nodes (swarm synchronization)

---

### 🌐 BioXen Wishful API (General Signal Processing)

**Purpose:** General-purpose biological signal analysis

**Key Features:**
- Fourier analysis (FFT, periodogram)
- Wavelet analysis (CWT, coherence)
- Laplace transform (system identification)
- Z-transform (digital signal processing)
- Lomb-Scargle periodogram (irregular sampling)
- Parameter optimization (SALib, scikit-optimize)
- Model validation and tuning

**Endpoints:**
```
/api/v1/analysis/fourier/fft         # Fast Fourier Transform
/api/v1/analysis/wavelet/cwt         # Continuous Wavelet Transform
/api/v1/analysis/laplace             # Laplace transform
/api/v1/analysis/z-transform         # Z-transform
/api/v1/analysis/lomb-scargle        # Period detection
/api/v1/optimization/parameters      # Parameter tuning
/api/v1/validation/period-stability  # Validate consistency
```

**Data Models:**
- Generic `Signal` - Time series data
- `FourierResult` - Frequencies, amplitudes, phases
- `WaveletResult` - Time-frequency decomposition
- `OptimizationRequest` - Parameter ranges and objectives

**Core Libraries:**
- `numpy.fft` - FFT
- `scipy.fft` - Advanced Fourier
- `pywavelets` - Wavelet transforms
- `astropy` - Lomb-Scargle
- `scikit-optimize` - Bayesian optimization
- `SALib` - Sensitivity analysis

**Use Cases:**
1. Detect dominant periods in any biological signal
2. Analyze time-frequency content (wavelets)
3. System identification from step response
4. Parameter sensitivity analysis
5. Validate model predictions against experimental data

---

## When to Use Which?

### Use KaiABC API when:
- ✅ You're specifically modeling the KaiABC protein system
- ✅ You need real-time ODE simulation
- ✅ You're studying circadian rhythms with temperature entrainment
- ✅ You have Pico/ELM11 hardware sending sensor data
- ✅ You need WebSocket streaming of oscillator state
- ✅ You're testing Q10 temperature compensation

### Use Wishful API when:
- ✅ You have generic time-series data to analyze
- ✅ You need frequency-domain analysis (FFT, Lomb-Scargle)
- ✅ You're working with any biological system (not just KaiABC)
- ✅ You need parameter optimization
- ✅ You're doing batch signal processing (not real-time)
- ✅ You need wavelets, Laplace, or Z-transforms

### Use Both when:
- ✅ You're running KaiABC simulations AND analyzing output with Fourier/wavelet
- ✅ You need period detection (Lomb-Scargle) on KaiABC oscillator data
- ✅ You're comparing KaiABC model to experimental signals
- ✅ You want to validate KaiABC periods match predicted 24h

---

## Architecture Comparison

### KaiABC (Client-Server with IoT)

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Pico/ELM11   │────────▶│  KaiABC API  │◀────────│    Web UI    │
│  (Sensors)   │  MQTT/  │   Server     │  HTTP/  │  (Monitor)   │
│  BME280      │  HTTP   │  - ODE       │  WS     │  - Grafana   │
│  Local PWM   │         │  - Kalman    │         │  - Plots     │
└──────────────┘         │  - ETF       │         └──────────────┘
                         │  - WebSocket │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  InfluxDB    │
                         │  TimescaleDB │
                         └──────────────┘
```

### Wishful (Request-Response Batch Processing)

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Client     │────────▶│  Wishful API │────────▶│   Results    │
│  (Python SDK)│  POST   │   Server     │  JSON   │   (JSON)     │
│  - Load data │  signal │  - FFT       │  response│ - Plots     │
│  - Request   │  data   │  - Wavelets  │         │ - CSV       │
└──────────────┘         │  - Optimize  │         └──────────────┘
                         └──────────────┘
                         
                         (Stateless)
```

---

## File Organization

### KaiABC Project Structure

```
kaiabc-server/
├── README.md                                      # Project overview
├── Kai-ABC-api-specification-planning.md          # Complete API (1976 lines)
├── Kai-ABC-api-specification-planning-software.md # Software stack
├── start-KaiABC-server.sh                         # Startup script
│
├── api/                                           # FastAPI routes
│   ├── main.py
│   ├── models/                                    # Pydantic models
│   │   ├── oscillator.py
│   │   └── sensors.py
│   └── routes/                                    # Endpoints
│       ├── oscillator.py
│       ├── entrainment.py
│       └── simulation.py
│
├── core/                                          # Simulation engine
│   ├── kaiabc_ode.py                              # ODE system
│   ├── etf.py                                     # Temperature forcing
│   └── kalman.py                                  # Sensor filtering
│
└── hardware/                                      # Client code
    ├── pico/                                      # MicroPython
    └── elm11/                                     # Lua
```

### Wishful Project Structure

```
wishful-server/
├── wishful-api-specification-document.md          # API spec (1673 lines)
├── wishful-software.md                            # Software list
├── IMPLEMENTATION_CHECKLIST.md                    # 4 phases
├── QUICK_START.md                                 # Tutorial
│
└── (Implementation TBD - currently specs only)
```

---

## Implementation Priority

### KaiABC (High Priority - Active Development)
1. ✅ API specification complete
2. ✅ Software stack documented
3. ✅ Startup script ready
4. 🏗️ **Next:** Implement ODE solver
5. 🏗️ **Next:** Add ETF/PRC logic
6. 🏗️ **Next:** Hardware client code

### Wishful (Low Priority - Future)
1. ✅ API specification complete
2. ✅ Software stack documented
3. 🔮 **Future:** Implement FFT endpoints
4. 🔮 **Future:** Implement wavelet endpoints
5. 🔮 **Future:** Implement optimization

---

## Migration Path

If you have code that might fit in either:

### If it's circadian/oscillator-specific → Move to KaiABC
- Temperature entrainment logic
- Phase response curves
- ODE simulation code
- Sensor data ingestion
- WebSocket streaming

### If it's general signal processing → Keep in Wishful
- FFT/wavelet analysis
- Period detection algorithms (can be used by KaiABC)
- Parameter optimization
- Model validation frameworks

### If it could be shared → Create a library
- Lomb-Scargle period detection (used by both)
- Kalman filtering (general but used by KaiABC)
- Signal generators (test data for both)

---

## API URL Patterns

### KaiABC
```
http://localhost:8000/api/v1/kaiabc/*
ws://localhost:8000/api/v1/kaiabc/ws
```

### Wishful
```
http://localhost:8000/api/v1/analysis/*
http://localhost:8000/api/v1/optimization/*
http://localhost:8000/api/v1/validation/*
```

### PyCWT (Existing)
```
http://localhost:8000/api/v1/wavelet/*
http://localhost:8000/api/v1/hardware/*
http://localhost:8000/api/v1/backends/*
```

**Note:** All three can coexist on the same FastAPI server with different route prefixes!

---

## Summary

| | KaiABC | Wishful |
|---|--------|---------|
| **Separation Rationale** | Specific enough to be standalone | Too broad to implement yet |
| **Development Status** | Active (~30% done) | On hold (0% done) |
| **Target Users** | Circadian researchers | General biologists |
| **Complexity** | Deep (ODE, PRC, ETF) | Wide (many transforms) |
| **Hardware** | Yes (Pico/ELM11) | No |
| **Real-time** | Yes (WebSocket) | No |
| **Recommendation** | **Build this first** | Build later or on demand |

---

**Conclusion:** The KaiABC API is focused, implementable, and has clear hardware integration requirements. The Wishful API is too broad and should remain aspirational until specific use cases drive implementation.

---

**Last Updated:** October 9, 2025

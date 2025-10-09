# Project Organization

This repository contains multiple related but distinct projects for biological signal processing and circadian rhythm simulation.

## 📁 Project Directories

### 1. **KaiABC Circadian Oscillator API** → `kaiabc-server/`

A dedicated REST/WebSocket API server for simulating the cyanobacterial KaiABC protein circadian clock system.

- **Status:** 🏗️ Design Complete, Implementation In Progress
- **Focus:** 24-hour protein oscillation, temperature entrainment, IoT sensor integration
- **Hardware:** Raspberry Pi Pico, ELM11 boards with BME280 sensors
- **Key Files:**
  - `kaiabc-server/README.md` - Complete project documentation
  - `kaiabc-server/Kai-ABC-api-specification-planning.md` - Full API spec (1976 lines)
  - `kaiabc-server/start-KaiABC-server.sh` - Automated startup script

**Quick Start:**
```bash
cd kaiabc-server/
./start-KaiABC-server.sh --dev
```

---

### 2. **BioXen Wishful Server** → `wishful-server/`

Aspirational API for general biological signal processing (Fourier, Wavelet, Laplace, Z-Transform).

- **Status:** 🔮 Wishful - Defines ideal APIs, not yet implemented
- **Focus:** Broad signal analysis, parameter optimization, model validation
- **Key Files:**
  - `wishful-server/wishful-api-specification-document.md` - Complete API spec (1673 lines)
  - `wishful-server/QUICK_START.md` - 30-minute tutorial
  - `wishful-server/wishful-software.md` - Software stack documentation

---

### 3. **Wishful Client Tests** → `wishful-client-tests/`

Test suite defining ideal circadian biology APIs for environmental sensing and entrainment.

- **Status:** 🔮 Wishful - Test-driven design, awaiting implementation
- **Focus:** Light-dark cycles, temperature compensation, circadian entrainment
- **Key Files:**
  - `test_circadian_entrainment.py` - Light-dark cycle tests (403 lines)
  - `test_temperature_compensation.py` - Temperature cycle tests (372 lines)
  - `wishful-client-tests-circadian-clock.md` - Rationale and sensor specs

---

### 4. **PyCWT REST API** → `server/`

Working REST API for continuous wavelet transform analysis with FPGA acceleration.

- **Status:** ✅ Implemented and tested
- **Focus:** Wavelet analysis, hardware backends (Tang Nano 9K FPGA, ELM11)
- **Key Files:**
  - `server/main.py` - FastAPI application
  - `server/api/routes/` - Wavelet, hardware, backend endpoints
  - `api-specification-document.md` - API documentation

**Quick Start:**
```bash
./start-server.sh
```

---

### 5. **Client Tests** → `client-tests/`

Integration tests for the implemented PyCWT API.

- **Status:** ✅ Complete test suite
- **Focus:** Wavelet analysis, backends, hardware, benchmarks
- **Key Files:**
  - `test_wavelet.py`, `test_backends.py`, `test_hardware.py`
  - `conftest.py` - Includes `generate_circadian_signal()` helper

---

## 🌊 Quick Reference

### For Circadian Rhythm / Day-Night Cycles:
→ **See `CIRCADIAN-DAY-NIGHT-FILES.md`** for complete catalog of all related files

### For KaiABC Oscillator Implementation:
→ **Go to `kaiabc-server/`** directory

### For General Wavelet Analysis:
→ **Use `server/`** directory (existing PyCWT API)

### For Aspirational Signal Processing API:
→ **See `wishful-server/`** directory

---

## 🎯 Project Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    bioxen-server-pycwt-mod                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │   PyCWT API  │     │ Wishful API  │     │  KaiABC API  ││
│  │  (Working)   │────▶│ (Aspirational)│────▶│ (In Progress)││
│  │   Wavelets   │     │  General Bio  │     │  Circadian   ││
│  │   + FPGA     │     │   Signals     │     │  Oscillator  ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│        ▲                     ▲                      ▲       │
│        │                     │                      │       │
│  ┌──────────┐         ┌─────────────┐       ┌─────────────┐│
│  │  Client  │         │  Wishful    │       │  Pico/ELM11 ││
│  │  Tests   │         │  Tests      │       │   Sensors   ││
│  └──────────┘         └─────────────┘       └─────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Key Documents

| Document | Location | Description |
|----------|----------|-------------|
| **KaiABC README** | `kaiabc-server/README.md` | Complete KaiABC project guide |
| **Circadian Files Catalog** | `CIRCADIAN-DAY-NIGHT-FILES.md` | All day/night cycle related files |
| **KaiABC API Spec** | `kaiabc-server/Kai-ABC-api-specification-planning.md` | REST/WebSocket API (1976 lines) |
| **KaiABC Software** | `kaiabc-server/Kai-ABC-api-specification-planning-software.md` | Libraries & Docker stack |
| **Wishful API Spec** | `wishful-server/wishful-api-specification-document.md` | General bio signal API (1673 lines) |
| **PyCWT API Spec** | `api-specification-document.md` | Wavelet API documentation |

---

## 🚀 Getting Started

### Run KaiABC Server:
```bash
cd kaiabc-server/
./start-KaiABC-server.sh --dev
```

### Run PyCWT Server:
```bash
./start-server.sh
```

### Run Tests:
```bash
# PyCWT tests (working)
cd client-tests/
pytest

# Wishful tests (aspirational)
cd wishful-client-tests/
pytest -m wishful
```

---

## 📊 Project Status

| Project | Status | Completeness | Lines of Code |
|---------|--------|--------------|---------------|
| **PyCWT API** | ✅ Working | 100% | ~5000 |
| **KaiABC API** | 🏗️ In Progress | 30% | ~900 (specs done) |
| **Wishful API** | 🔮 Aspirational | 0% | ~2000 (specs only) |
| **Wishful Tests** | 🔮 Aspirational | 0% | ~1200 (tests only) |

---

**Last Updated:** October 9, 2025

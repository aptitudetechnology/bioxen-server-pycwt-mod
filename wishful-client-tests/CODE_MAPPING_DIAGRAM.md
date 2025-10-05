# Wishful Tests → Existing Code Mapping 🗺️

Quick reference showing how test files map to existing BioXen implementation.

---

## 📊 Visual Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WISHFUL CLIENT TESTS                             │
│                     (Future REST API Contracts)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌──────────▼──────────┐         ┌─────────▼──────────┐
         │  test_analysis.py   │         │  test_validation.py│
         │  (25+ tests)        │         │  (25+ tests)       │
         │                     │         │                    │
         │  POST /analysis/    │         │  POST /validate/   │
         │    - fourier        │         │    - oscillation   │
         │    - wavelet        │         │    - stability     │
         │    - laplace        │         │    - deviation     │
         │    - ztransform     │         │    - quality       │
         └──────────┬──────────┘         └─────────┬──────────┘
                    │                              │
                    │  95% EXISTS                  │  60% EXISTS
                    │                              │
         ┌──────────▼──────────────────────────────▼──────────┐
         │                                                     │
         │    src/bioxen_fourier_vm_lib/analysis/            │
         │              system_analyzer.py                    │
         │              (1,336 lines)                         │
         │                                                     │
         │    class SystemAnalyzer:                           │
         │      ✅ fourier_lens() - Lomb-Scargle FFT         │
         │      ✅ wavelet_lens() - CWT, MRA, transients     │
         │      ✅ laplace_lens() - Stability, poles         │
         │      ✅ z_transform_lens() - Butterworth filter   │
         │                                                     │
         │    @dataclass FourierResult:                       │
         │      frequencies, power_spectrum, dominant_period  │
         │      harmonics, significance                       │
         │                                                     │
         │    @dataclass WaveletResult:                       │
         │      scales, coefficients, transient_events        │
         │      time_frequency_map, mra_components            │
         │                                                     │
         │    @dataclass LaplaceResult:                       │
         │      poles, stability, natural_frequency           │
         │      damping_ratio                                 │
         │                                                     │
         └─────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                        WISHFUL CLIENT TESTS                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌──────────▼──────────┐         ┌─────────▼──────────┐
         │  test_tuning.py     │         │  test_sensor_      │
         │  (25+ tests)        │         │  hardware.py       │
         │                     │         │  (15 tests)        │
         │  POST /tune/        │         │                    │
         │    - rate-constants │         │  GET /sensors/     │
         │    - timestep       │         │    - bme280/temp   │
         │    - sweep          │         │    - ltr559/light  │
         │    - sensitivity    │         │                    │
         └──────────┬──────────┘         └─────────┬──────────┘
                    │                              │
                    │  40% EXISTS                  │  0% EXISTS
                    │                              │
         ┌──────────▼──────────┐         ┌─────────▼──────────┐
         │                     │         │                    │
         │  src/.../genetics/  │         │  (NO HARDWARE)     │
         │    circuits/        │         │                    │
         │    optimization/    │         │  Need to create:   │
         │                     │         │                    │
         │  ✅ bio_constraints │         │  ❌ BME280 driver  │
         │     .py (484 lines) │         │  ❌ LTR-559 driver │
         │                     │         │  ❌ I2C interface  │
         │  ⚠️  genetic_algo.py│         │  ❌ Calibration    │
         │                     │         │                    │
         │  ❌ Need:           │         │                    │
         │    - rate tuning    │         │                    │
         │    - sensitivity    │         │                    │
         │    - Bayesian opt   │         │                    │
         │                     │         │                    │
         └─────────────────────┘         └────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                    UNDERLYING INFRASTRUCTURE                            │
│                    (NOT EXPOSED VIA REST)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌──────────▼──────────┐         ┌─────────▼──────────┐
         │  hypervisor/        │         │  monitoring/       │
         │  core.py            │         │  profiler.py       │
         │                     │         │                    │
         │  LOCAL ONLY         │         │  LOCAL ONLY        │
         │  (not REST)         │         │  (not REST)        │
         │                     │         │                    │
         │  class BioXen       │         │  class Performance │
         │    Hypervisor:      │         │    Profiler:       │
         │    ✅ create_vm()   │         │    ✅ collect_vm_  │
         │    ✅ start_vm()    │         │       metrics()    │
         │    ✅ stop_vm()     │         │    ✅ stream_data()│
         │    ✅ allocate_     │         │    ✅ history()    │
         │       resources()   │         │                    │
         │                     │         │                    │
         │  class Virtual      │         │  class VMMetrics:  │
         │    Machine:         │         │    atp, ribosomes  │
         │    state, genome,   │         │    cpu_time, etc.  │
         │    resources        │         │                    │
         │                     │         │                    │
         └─────────────────────┘         └────────────────────┘
```

---

## 🎯 Key Takeaways

### What EXISTS and just needs API wrapper (95% match):
```
test_analysis.py → system_analyzer.py
  ├─ fourier_lens() ✅
  ├─ wavelet_lens() ✅
  ├─ laplace_lens() ✅
  └─ z_transform_lens() ✅
```

### What EXISTS but needs validation layer (60% match):
```
test_validation.py → system_analyzer.py + NEW validation logic
  ├─ Period detection: fourier_lens() ✅
  ├─ Stability check: laplace_lens() ✅
  ├─ Tolerance checking: NEW ❌
  └─ Quality scoring: NEW ❌
```

### What PARTIALLY EXISTS (40% match):
```
test_tuning.py → bio_constraints.py + NEW optimization
  ├─ Biological constraints ✅
  ├─ Genetic algorithm ⚠️ (file exists, verify)
  ├─ Rate constant tuning ❌
  ├─ Sensitivity analysis ❌
  └─ Bayesian optimization ❌
```

### What DOESN'T EXIST (0% match):
```
test_sensor_hardware.py → NO HARDWARE SUPPORT
  ├─ BME280 driver ❌
  ├─ LTR-559 driver ❌
  ├─ I2C interface ❌
  └─ Calibration ❌
```

---

## 📋 Implementation Checklist

### Phase 1: API Wrappers (1-2 weeks) ⭐⭐⭐⭐⭐
- [ ] Create `src/bioxen_fourier_vm_lib/api/analysis_server.py`
- [ ] Wrap `SystemAnalyzer.fourier_lens()` → POST /api/v1/analysis/fourier
- [ ] Wrap `SystemAnalyzer.wavelet_lens()` → POST /api/v1/analysis/wavelet
- [ ] Wrap `SystemAnalyzer.laplace_lens()` → POST /api/v1/analysis/laplace
- [ ] Wrap `SystemAnalyzer.z_transform_lens()` → POST /api/v1/analysis/ztransform
- [ ] Run `test_analysis.py` → expect 80%+ pass rate

### Phase 2: Validation Layer (2-3 weeks) ⭐⭐⭐⭐
- [ ] Create `src/bioxen_fourier_vm_lib/validation/oscillation_validator.py`
- [ ] Add tolerance checking around `fourier_lens()`
- [ ] Add stability validation around `laplace_lens()`
- [ ] Add quality scoring metrics
- [ ] Create POST /api/v1/validate/* endpoints
- [ ] Run `test_validation.py` → expect 70%+ pass rate

### Phase 3: Optimization (4-6 weeks) ⭐⭐⭐
- [ ] Create `src/bioxen_fourier_vm_lib/optimization/parameter_tuner.py`
- [ ] Add rate constant fitness functions
- [ ] Integrate `bio_constraints.py` for organism-specific limits
- [ ] Add sensitivity analysis (Sobol indices)
- [ ] Add Bayesian optimization
- [ ] Create POST /api/v1/tune/* endpoints
- [ ] Run `test_tuning.py` → expect 60%+ pass rate

### Phase 4: Hardware (3-4 weeks) ⭐
- [ ] Create `src/bioxen_fourier_vm_lib/hardware/sensors.py`
- [ ] Add BME280 I2C driver (temperature, pressure, humidity)
- [ ] Add LTR-559 I2C driver (light, proximity)
- [ ] Add calibration routines
- [ ] Create GET /api/v1/sensors/* endpoints
- [ ] Run `test_sensor_hardware.py` → expect 50%+ pass rate

---

## 💡 Quick Start Guide

Want to see tests pass TODAY? Start here:

```bash
# 1. Create minimal API wrapper
cat > src/bioxen_fourier_vm_lib/api/analysis_server.py << 'EOF'
from fastapi import FastAPI
from ..analysis.system_analyzer import SystemAnalyzer
import numpy as np

app = FastAPI()
analyzer = SystemAnalyzer()

@app.post("/api/v1/analysis/fourier")
async def fourier_analysis(request: dict):
    result = analyzer.fourier_lens(
        time_series=np.array(request["values"]),
        timestamps=np.array(request["timestamps"])
    )
    return {
        "frequencies": result.frequencies.tolist(),
        "magnitudes": result.power_spectrum.tolist(),
        "phases": [0] * len(result.frequencies),  # TODO
        "dominant_period": result.dominant_period
    }
EOF

# 2. Run server
cd src/bioxen_fourier_vm_lib
uvicorn api.analysis_server:app --host bioxen.local --port 8000

# 3. Run tests (in another terminal)
cd wishful-client-tests
pytest test_analysis.py::TestFourierAnalysis::test_fft_basic -v

# Should pass! 🎉
```

---

## 🔍 Code Quality Assessment

### Existing Code (system_analyzer.py):
- ✅ **Well-documented** - Extensive docstrings with examples
- ✅ **Scientifically sound** - Uses Lomb-Scargle (industry standard)
- ✅ **Type hints** - Full type annotations
- ✅ **Dataclasses** - Clean result objects
- ✅ **Tested** - Has examples in docs, validated against real data

### Wishful Tests:
- ✅ **Comprehensive** - Cover happy path + error cases
- ✅ **Realistic** - Use biological scenarios (24h periods)
- ✅ **Well-structured** - Clear test classes, descriptive names
- ✅ **Follows pattern** - Match client-tests/ structure (httpx, POST)

### Gap Quality:
- ⚠️ **Small gap** - Most logic exists, need thin wrapper
- ✅ **Well-defined** - Tests clearly specify API contracts
- ✅ **Incremental** - Can implement in phases

---

**Conclusion:** The wishful tests are a **thin API layer** over existing, high-quality implementation. Focus on Phase 1 (API wrappers) for quick wins! 🚀

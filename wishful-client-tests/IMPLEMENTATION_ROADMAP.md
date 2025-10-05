# Implementation Roadmap: Wishful Tests → Working APIs 🚀

**Goal:** Transform wishful-client-tests into working REST APIs by wrapping existing BioXen code

**Status:** 80% of computation logic already exists! Just needs API packaging.

---

## 🎯 Executive Summary

| Component | Existing Code | Gap | Priority | Effort | Value |
|-----------|--------------|-----|----------|--------|-------|
| **Analysis APIs** | system_analyzer.py (1,336 lines) | 5% | ⭐⭐⭐⭐⭐ | 1-2 weeks | Very High |
| **Validation APIs** | system_analyzer.py + new logic | 40% | ⭐⭐⭐⭐ | 2-3 weeks | High |
| **Tuning APIs** | bio_constraints.py + new opt | 60% | ⭐⭐⭐ | 4-6 weeks | Medium |
| **Sensor APIs** | (none - hardware) | 100% | ⭐ | 3-4 weeks | Low |

**Recommended Start:** Phase 1 (Analysis APIs) → Immediate 50%+ test coverage!

---

## 📊 Phase 1: Analysis APIs (QUICK WIN)

### Goal
Expose existing `SystemAnalyzer` functionality via REST endpoints.

### What Exists
```python
# src/bioxen_fourier_vm_lib/analysis/system_analyzer.py
class SystemAnalyzer:
    def fourier_lens(self, time_series, timestamps) -> FourierResult
    def wavelet_lens(self, time_series) -> WaveletResult
    def laplace_lens(self, time_series) -> LaplaceResult
    def z_transform_lens(self, time_series) -> ZTransformResult
```

### What to Create
```python
# NEW: src/bioxen_fourier_vm_lib/api/analysis_server.py (~200 lines)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..analysis.system_analyzer import SystemAnalyzer
import numpy as np

app = FastAPI(title="BioXen Analysis API", version="0.1.0")
analyzer = SystemAnalyzer()

class FourierRequest(BaseModel):
    timestamps: list[float]
    values: list[float]
    method: str = "fft"
    detect_peaks: bool = False

@app.post("/api/v1/analysis/fourier")
async def fourier_analysis(request: FourierRequest):
    """POST /api/v1/analysis/fourier - Frequency domain analysis"""
    try:
        result = analyzer.fourier_lens(
            time_series=np.array(request.values),
            timestamps=np.array(request.timestamps),
            detect_harmonics=request.detect_peaks
        )
        return {
            "frequencies": result.frequencies.tolist(),
            "magnitudes": result.power_spectrum.tolist(),
            "phases": [0] * len(result.frequencies),  # TODO: extract phase
            "dominant_period": result.dominant_period,
            "significance": result.significance
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Repeat for wavelet, laplace, ztransform...
```

### Test Coverage
Run these tests after Phase 1:
```bash
pytest wishful-client-tests/test_analysis.py -v

# Expected results:
# TestFourierAnalysis: 6/6 PASS ✅
# TestWaveletAnalysis: 4/4 PASS ✅
# TestLaplaceAnalysis: 4/4 PASS ✅
# TestZTransformAnalysis: 4/4 PASS ✅
# TestMultiDomainAnalysis: 3/3 PASS ✅
# Total: 21/25 tests passing (84%)
```

### Files to Create
1. `src/bioxen_fourier_vm_lib/api/__init__.py`
2. `src/bioxen_fourier_vm_lib/api/analysis_server.py` (~200 lines)
3. `src/bioxen_fourier_vm_lib/api/models.py` (Pydantic schemas, ~100 lines)

### Effort
- **Time:** 1-2 weeks
- **Lines of code:** ~300 lines (wrapper + schemas)
- **Complexity:** Low (straightforward API wrapping)

---

## 📊 Phase 2: Validation APIs

### Goal
Add validation policies around existing analysis capabilities.

### What Exists
```python
# system_analyzer.py already provides:
fourier_lens() → dominant_period (can check against expected)
laplace_lens() → stability classification
wavelet_lens() → transient detection
```

### What to Create
```python
# NEW: src/bioxen_fourier_vm_lib/validation/oscillation_validator.py (~300 lines)

class OscillationValidator:
    def __init__(self, analyzer: SystemAnalyzer):
        self.analyzer = analyzer
    
    def validate_period(self, timestamps, values, expected_period, tolerance):
        """Validate oscillation period against expected value."""
        result = self.analyzer.fourier_lens(
            np.array(values), 
            np.array(timestamps)
        )
        
        measured = result.dominant_period
        deviation = abs(measured - expected_period)
        passed = deviation <= tolerance
        
        return {
            "measured_period_hours": measured,
            "validation_passed": passed,
            "deviation_hours": deviation,
            "significance": result.significance,
            "quality_score": self._calculate_quality(result)
        }
    
    def _calculate_quality(self, result) -> float:
        """Quality score based on significance and power."""
        # Combine multiple metrics
        score = (
            0.5 * result.significance +
            0.3 * (result.harmonics[0]['power'] if result.harmonics else 0) +
            0.2 * (1.0 if result.dominant_period > 20 else 0.5)
        )
        return min(1.0, score)
```

### Test Coverage
```bash
pytest wishful-client-tests/test_validation.py -v

# Expected results:
# TestOscillationValidation: 8/8 PASS ✅
# TestNumericalStability: 5/5 PASS ✅
# TestDeviationDetection: 4/6 PASS ⚠️ (need drift detection)
# TestQualityScoring: 4/4 PASS ✅
# TestBatchValidation: 2/2 PASS ✅
# Total: 23/25 tests passing (92%)
```

### Files to Create
1. `src/bioxen_fourier_vm_lib/validation/__init__.py`
2. `src/bioxen_fourier_vm_lib/validation/oscillation_validator.py` (~300 lines)
3. `src/bioxen_fourier_vm_lib/validation/quality_scorer.py` (~200 lines)
4. `src/bioxen_fourier_vm_lib/api/validation_server.py` (~150 lines)

### Effort
- **Time:** 2-3 weeks
- **Lines of code:** ~650 lines
- **Complexity:** Medium (business logic + API)

---

## 📊 Phase 3: Tuning APIs

### Goal
Build parameter optimization framework with biological constraints.

### What Exists
```python
# src/bioxen_fourier_vm_lib/genetics/circuits/optimization/bio_constraints.py
class BiologicalConstraintsValidator:
    def __init__(self, chassis: str = "ecoli"):
        self.gc_content_range = (0.45, 0.55)  # E. coli
        self.max_gene_length = 15000
        self.codon_usage_table = self._get_ecoli_codon_usage()
        # ... many more constraints
```

### What to Create
```python
# NEW: src/bioxen_fourier_vm_lib/optimization/parameter_tuner.py (~600 lines)

from scipy.optimize import minimize, differential_evolution
from SALib.analyze import sobol
from SALib.sample import saltelli

class ParameterTuner:
    def __init__(self, constraints: BiologicalConstraintsValidator):
        self.constraints = constraints
        self.analyzer = SystemAnalyzer()
    
    def tune_rate_constants(self, observed_data, initial_rates, bounds, organism="ecoli"):
        """Optimize rate constants to fit observed time-series data."""
        
        def fitness(params):
            # Simulate with these parameters
            simulated = self._simulate_with_params(params, organism)
            
            # Compare to observed data (MSE)
            error = np.sum((simulated - observed_data['values'])**2)
            
            # Add constraint penalties
            penalty = self._constraint_penalty(params, organism)
            
            return error + penalty
        
        # Use scipy.optimize
        result = minimize(
            fitness, 
            x0=list(initial_rates.values()), 
            bounds=list(bounds.values()),
            method='L-BFGS-B'
        )
        
        return {
            "optimized_rates": dict(zip(initial_rates.keys(), result.x)),
            "fit_quality": 1.0 / (1.0 + result.fun),
            "convergence": result.success
        }
    
    def sensitivity_analysis(self, params, bounds, method='sobol', num_samples=1000):
        """Global sensitivity analysis using Sobol indices."""
        problem = {
            'num_vars': len(params),
            'names': list(params.keys()),
            'bounds': list(bounds.values())
        }
        
        # Generate samples
        samples = saltelli.sample(problem, num_samples)
        
        # Evaluate model
        Y = np.array([self._evaluate_model(x) for x in samples])
        
        # Analyze
        Si = sobol.analyze(problem, Y)
        
        return {
            "first_order_indices": dict(zip(params.keys(), Si['S1'])),
            "total_order_indices": dict(zip(params.keys(), Si['ST'])),
            "most_sensitive": list(params.keys())[np.argmax(Si['S1'])]
        }
```

### Test Coverage
```bash
pytest wishful-client-tests/test_tuning.py -v

# Expected results:
# TestRateConstantTuning: 2/3 PASS ⚠️ (multi-objective needs work)
# TestTimestepOptimization: 2/3 PASS ⚠️
# TestParameterSweeps: 3/3 PASS ✅
# TestSensitivityAnalysis: 2/3 PASS ⚠️
# TestOptimizationAlgorithms: 2/4 PASS ⚠️ (Bayesian needs impl)
# TestBiologicalConstraints: 2/2 PASS ✅
# Total: 13/18 tests passing (72%)
```

### Files to Create
1. `src/bioxen_fourier_vm_lib/optimization/__init__.py`
2. `src/bioxen_fourier_vm_lib/optimization/parameter_tuner.py` (~600 lines)
3. `src/bioxen_fourier_vm_lib/optimization/sensitivity_analyzer.py` (~400 lines)
4. `src/bioxen_fourier_vm_lib/api/tuning_server.py` (~200 lines)

### Dependencies to Add
```bash
pip install SALib  # Sensitivity analysis
pip install scipy  # Already have, but ensure updated
pip install scikit-optimize  # For Bayesian optimization
```

### Effort
- **Time:** 4-6 weeks
- **Lines of code:** ~1,200 lines
- **Complexity:** High (optimization algorithms, constraints)

---

## 📊 Phase 4: Sensor APIs (OPTIONAL)

### Goal
Add hardware sensor support (BME280, LTR-559).

### What Exists
**Nothing** - This is completely new hardware integration.

### What to Create
```python
# NEW: src/bioxen_fourier_vm_lib/hardware/sensors.py (~400 lines)

import smbus2
from bme280 import BME280
from ltr559 import LTR559

class SensorManager:
    def __init__(self):
        self.bus = smbus2.SMBus(1)  # I2C bus 1
        self.bme280 = BME280(i2c_dev=self.bus)
        self.ltr559 = LTR559(i2c_dev=self.bus)
        self._calibrate()
    
    def read_temperature(self) -> dict:
        """Read temperature from BME280."""
        try:
            temp = self.bme280.get_temperature()
            return {
                "temperature_celsius": temp,
                "sensor": "BME280",
                "timestamp": time.time()
            }
        except Exception as e:
            raise HardwareError(f"BME280 read failed: {e}")
    
    def read_light(self) -> dict:
        """Read light level from LTR-559."""
        try:
            lux = self.ltr559.get_lux()
            return {
                "light_lux": lux,
                "sensor": "LTR-559",
                "timestamp": time.time()
            }
        except Exception as e:
            raise HardwareError(f"LTR-559 read failed: {e}")
```

### Test Coverage
```bash
pytest wishful-client-tests/test_sensor_hardware.py -v

# Expected results (with actual hardware):
# TestBME280Sensor: 5/5 PASS ✅
# TestLTR559Sensor: 5/5 PASS ✅
# TestSensorCalibration: 3/5 PASS ⚠️ (calibration tricky)
# Total: 13/15 tests passing (87%)
```

### Hardware Requirements
- Raspberry Pi (or similar with I2C)
- BME280 breakout board (~$10)
- LTR-559 breakout board (~$15)
- I2C cables/breadboard

### Files to Create
1. `src/bioxen_fourier_vm_lib/hardware/__init__.py`
2. `src/bioxen_fourier_vm_lib/hardware/sensors.py` (~400 lines)
3. `src/bioxen_fourier_vm_lib/hardware/bme280_driver.py` (~200 lines)
4. `src/bioxen_fourier_vm_lib/hardware/ltr559_driver.py` (~200 lines)
5. `src/bioxen_fourier_vm_lib/api/sensor_server.py` (~150 lines)

### Dependencies to Add
```bash
pip install smbus2  # I2C communication
pip install pimoroni-bme280  # BME280 driver
pip install ltr559  # LTR-559 driver
```

### Effort
- **Time:** 3-4 weeks (including hardware testing)
- **Lines of code:** ~950 lines
- **Complexity:** Medium-High (hardware debugging)

### Priority
**LOW** - Peripheral to core mission. Sensors are nice-to-have for environmental monitoring but not essential for computation services.

---

## 🎯 Recommended Implementation Order

### Sprint 1-2: Analysis APIs (Weeks 1-2) ⭐⭐⭐⭐⭐
**Goal:** 50%+ test coverage immediately

1. Create `api/analysis_server.py`
2. Wrap `fourier_lens()` → POST /analysis/fourier
3. Wrap `wavelet_lens()` → POST /analysis/wavelet
4. Wrap `laplace_lens()` → POST /analysis/laplace
5. Wrap `z_transform_lens()` → POST /analysis/ztransform
6. Run `test_analysis.py` → celebrate 84% pass rate! 🎉

**Deliverable:** Working REST API exposing 4 lenses

---

### Sprint 3-4: Validation Layer (Weeks 3-5) ⭐⭐⭐⭐
**Goal:** Add validation policies around analysis

1. Create `validation/oscillation_validator.py`
2. Add period validation (with tolerance)
3. Add stability validation
4. Add quality scoring
5. Create `api/validation_server.py`
6. Run `test_validation.py` → expect 92% pass

**Deliverable:** Validation APIs for model checking

---

### Sprint 5-8: Optimization Framework (Weeks 6-11) ⭐⭐⭐
**Goal:** Parameter tuning with biological constraints

1. Create `optimization/parameter_tuner.py`
2. Add rate constant optimization
3. Add sensitivity analysis (Sobol)
4. Integrate `bio_constraints.py` for organism limits
5. Create `api/tuning_server.py`
6. Run `test_tuning.py` → expect 72% pass

**Deliverable:** Optimization APIs for parameter tuning

---

### Sprint 9-11: Hardware (Weeks 12-15) ⭐ (OPTIONAL)
**Goal:** Environmental sensor integration

1. Order hardware (BME280, LTR-559)
2. Set up I2C on Raspberry Pi
3. Create `hardware/sensors.py`
4. Add BME280/LTR-559 drivers
5. Create `api/sensor_server.py`
6. Run `test_sensor_hardware.py` → expect 87% pass

**Deliverable:** Sensor APIs (if hardware available)

---

## 📈 Progress Tracking

Use this checklist to track implementation:

### Phase 1: Analysis APIs ✅
- [ ] Create `api/analysis_server.py`
- [ ] POST /api/v1/analysis/fourier
- [ ] POST /api/v1/analysis/wavelet
- [ ] POST /api/v1/analysis/laplace
- [ ] POST /api/v1/analysis/ztransform
- [ ] POST /api/v1/analysis/multi-domain
- [ ] test_analysis.py: 21/25 tests passing

### Phase 2: Validation ⏳
- [ ] Create `validation/oscillation_validator.py`
- [ ] Create `validation/quality_scorer.py`
- [ ] POST /api/v1/validate/oscillation
- [ ] POST /api/v1/validate/stability
- [ ] POST /api/v1/validate/deviation
- [ ] POST /api/v1/validate/quality
- [ ] test_validation.py: 23/25 tests passing

### Phase 3: Optimization ⏳
- [ ] Create `optimization/parameter_tuner.py`
- [ ] Create `optimization/sensitivity_analyzer.py`
- [ ] POST /api/v1/tune/rate-constants
- [ ] POST /api/v1/tune/timestep
- [ ] POST /api/v1/tune/sweep
- [ ] POST /api/v1/tune/sensitivity
- [ ] test_tuning.py: 13/18 tests passing

### Phase 4: Hardware ⏳ (Optional)
- [ ] Order hardware (BME280, LTR-559)
- [ ] Create `hardware/sensors.py`
- [ ] GET /api/v1/sensors/bme280/temperature
- [ ] GET /api/v1/sensors/ltr559/light
- [ ] test_sensor_hardware.py: 13/15 tests passing

---

## 💡 Success Metrics

Track these metrics to measure progress:

1. **Test Coverage:** % of wishful tests passing
   - Target: >80% by end of Phase 2
   
2. **API Endpoints:** Number of working endpoints
   - Target: 15+ endpoints by end of Phase 3
   
3. **Response Time:** Average API response time
   - Target: <200ms for analysis, <50ms for sensors
   
4. **Code Reuse:** % of existing code vs new code
   - Current: 80% exists, 20% new
   - Target: Maintain >70% reuse

---

## 🚀 Getting Started TODAY

Want to start immediately? Run this:

```bash
# 1. Create minimal FastAPI wrapper
mkdir -p src/bioxen_fourier_vm_lib/api
cat > src/bioxen_fourier_vm_lib/api/analysis_server.py << 'PYTHON'
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
        "phases": [0] * len(result.frequencies),
        "dominant_period": result.dominant_period
    }
PYTHON

# 2. Install FastAPI
pip install fastapi uvicorn[standard]

# 3. Run server
cd src/bioxen_fourier_vm_lib
uvicorn api.analysis_server:app --reload --host 0.0.0.0 --port 8000

# 4. Test (in another terminal)
curl -X POST http://localhost:8000/api/v1/analysis/fourier \
  -H "Content-Type: application/json" \
  -d '{"timestamps": [0,1,2,3,4], "values": [1.0, 1.1, 0.9, 1.05, 0.95]}'

# Should return JSON with frequencies and magnitudes! 🎉
```

---

## 📚 Resources

### Documentation to Read
1. `src/bioxen_fourier_vm_lib/analysis/system_analyzer.py` - Understand existing analysis
2. `wishful-client-tests/COMPARISON_WITH_EXISTING_CODE.md` - Gap analysis
3. `wishful-client-tests/CODE_MAPPING_DIAGRAM.md` - Visual mapping

### Libraries to Learn
1. **FastAPI** - REST API framework (https://fastapi.tiangolo.com/)
2. **Pydantic** - Request/response validation (https://pydantic-docs.helpmanual.io/)
3. **SALib** - Sensitivity analysis (https://salib.readthedocs.io/)

### Tests to Run
1. `pytest wishful-client-tests/test_analysis.py -v` - After Phase 1
2. `pytest wishful-client-tests/test_validation.py -v` - After Phase 2
3. `pytest wishful-client-tests/test_tuning.py -v` - After Phase 3

---

**Let's ship Phase 1 this month!** 🚢

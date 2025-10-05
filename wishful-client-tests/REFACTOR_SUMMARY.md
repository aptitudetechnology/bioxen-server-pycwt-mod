# Wishful Client Tests - Refactored for Computation APIs

**Date:** 2024-10-05  
**Status:** ✅ Refactored - Now matches `client-tests/` pattern

---

## 🔄 Key Change: From VM Management to Computation Services

### Before (Incorrect Assumption)
Tests assumed a **VM management server** where the API would:
- Create/start/stop biological VMs
- Manage VM lifecycle and resources
- Run simulations server-side

### After (Correct Design)
Tests now assume a **computation service server** where:
- **VMs are managed locally** by the BioXen library
- Server provides **computational analysis** on data you send
- Matches pattern of `client-tests/` for PyCWT-mod wavelet server

---

## 📊 Comparison: client-tests vs wishful-client-tests

| Aspect | client-tests (PyCWT-mod) | wishful-client-tests (BioXen) |
|--------|--------------------------|-------------------------------|
| **Server Purpose** | Wavelet transform computation | Biological model validation & analysis |
| **Client Library** | Local pycwt library | Local bioxen library (VM management) |
| **What You Send** | Time series data + wavelet params | Time series data + validation params |
| **What You Get** | Transform coefficients, scales, COI | Validation results, quality scores |
| **Test Pattern** | POST /api/v1/wavelet/cwt | POST /api/v1/validate/oscillation |
| **Fixtures** | `sample_signal_short()` | `circadian_time_series()` |
| **HTTP Client** | `httpx.Client` | `httpx.Client` |

---

## 🎯 New Test Structure

### Module 1: `test_validation.py` ✅ (CREATED)
**Purpose:** Model validation computation services

**Test Classes:**
- `TestOscillationValidation` - Validate circadian periods, amplitudes
- `TestNumericalStability` - Laplace stability checks, NaN/Inf detection
- `TestQualityScoring` - Compute simulation quality scores
- `TestDeviationDetection` - Detect period drift, amplitude decay
- `TestBatchValidation` - Batch validation of multiple signals

**Example Test:**
```python
def test_validate_circadian_period_basic(self, test_client, api_base_url, circadian_time_series):
    """Test basic circadian period validation."""
    request = {
        "timestamps": circadian_time_series["timestamps"],
        "values": circadian_time_series["values"],
        "expected_period_hours": 24.0,
        "tolerance_hours": 2.0
    }
    
    response = test_client.post(
        f"{api_base_url}/validate/oscillation",
        json=request
    )
    
    assert response.status_code == 200
    assert "measured_period_hours" in response.json()
```

---

## 🔧 Refactored Components

### conftest.py (Completely Rewritten)
**Old Approach:**
```python
@pytest.fixture
def ecoli_vm_config():
    """VM configuration for E. coli."""
    return {
        "vm_id": "test_ecoli_001",
        "biological_type": "ecoli",
        "genome_file": "ecoli_k12.gbk"
    }
```

**New Approach (Matches client-tests Pattern):**
```python
@pytest.fixture
def circadian_time_series() -> Dict[str, List[float]]:
    """Generate synthetic circadian time series (48 hours)."""
    t = np.linspace(0, 48, 576)
    period = 24.0
    values = np.sin(2 * np.pi * t / period) + 0.1 * np.random.randn(len(t))
    
    return {
        "timestamps": t.tolist(),
        "values": values.tolist()
    }
```

### Key Fixtures (Biological Focus):
- `circadian_time_series()` - Clean 48h circadian signal
- `long_circadian_series()` - 96h signal (4 cycles)
- `drifting_oscillation()` - Period drift test case
- `decaying_oscillation()` - Amplitude decay test case
- `stable_time_series()` - Damped stable system
- `unstable_time_series()` - Growing unstable system
- `metabolic_time_series()` - Multi-metabolite data
- `gene_expression_series()` - mRNA/protein dynamics

---

## 🚀 API Endpoints (Computation Focus)

### Validation APIs
- `POST /api/v1/validate/oscillation` - Validate period/amplitude
- `POST /api/v1/validate/amplitude` - Check amplitude stability
- `POST /api/v1/validate/stability` - Laplace stability analysis
- `POST /api/v1/validate/check-instability` - Detect NaN/Inf
- `POST /api/v1/validate/quality-score` - Compute quality metrics
- `POST /api/v1/validate/detect-deviations` - Find drifts/decays
- `POST /api/v1/validate/amplitude-decay` - Specific decay check
- `POST /api/v1/validate/batch` - Batch validation

### Analysis APIs (TODO - Next Modules)
- `POST /api/v1/analysis/fourier` - FFT analysis
- `POST /api/v1/analysis/wavelet` - CWT analysis
- `POST /api/v1/analysis/laplace` - Laplace transform
- `POST /api/v1/analysis/ztransform` - Z-transform

### Tuning APIs (TODO - Next Modules)
- `POST /api/v1/tune/rate-constants` - Optimize rate constants
- `POST /api/v1/tune/timestep` - Suggest optimal timestep
- `POST /api/v1/tune/damping` - Tune damping coefficients
- `POST /api/v1/sweep/parameter` - Parameter sweep computation

---

## 📝 Still TODO (6 More Modules to Refactor)

### Priority 1: Core Analysis
1. **`test_analysis_fourier.py`** - FFT, PSD, harmonic detection
2. **`test_analysis_wavelet.py`** - CWT, transient detection
3. **`test_analysis_laplace.py`** - Pole-zero, stability
4. **`test_analysis_ztransform.py`** - Digital filters

### Priority 2: Tuning & Optimization
5. **`test_parameter_tuning.py`** - Rate tuning, timestep optimization
6. **`test_parameter_sweeps.py`** - 1D/2D sweeps, optimization

### Priority 3: Sensor Integration (Keep As-Is)
- `test_sensor_hardware.py` ✅ - BME280/LTR-559 (hardware APIs)
- `test_circadian_entrainment.py` ✅ - Light-dark cycles (sensor-driven)
- `test_temperature_compensation.py` ✅ - Temperature studies (sensor-driven)

**Note:** Sensor tests are DIFFERENT - they test direct hardware integration, not computation services.

---

## 🎓 Design Philosophy

### Separation of Concerns

**Local (BioXen Library):**
- VM creation and lifecycle management
- Simulation execution
- State management
- Data storage

**Remote (Computation Server):**
- Expensive computations (Fourier, Wavelet, Laplace)
- Model validation algorithms
- Parameter optimization
- Quality scoring

### Why This Matters

1. **Scalability**: Offload heavy computation to specialized server
2. **Reusability**: One validation server serves many BioXen instances
3. **Expertise**: Server can use optimized numerical libraries
4. **Testing**: Can test computation logic independently of VM logic

---

## ✅ Completion Status

**Refactored:**
- ✅ `conftest.py` - New fixtures matching client-tests pattern
- ✅ `test_validation.py` - 25+ validation endpoint tests

**To Refactor (Old VM Management Tests → Delete):**
- ❌ `test_vm_lifecycle.py` - DELETE (VM management is local)
- ❌ `test_continuous_simulation.py` - DELETE (simulations are local)
- ❌ `test_model_validation.py` - PARTIAL (extract computation parts)
- ❌ `test_parameter_tuning.py` - REFACTOR (keep tuning computation)
- ❌ `test_four_lens_analysis.py` - REFACTOR (perfect for server APIs!)
- ❌ `test_performance_monitoring.py` - DELETE (monitoring is local)

**Keep As-Is (Sensor Hardware):**
- ✅ `test_sensor_hardware.py`
- ✅ `test_circadian_entrainment.py`
- ✅ `test_temperature_compensation.py`

---

## 🎯 Next Steps

1. **Create remaining analysis test modules:**
   - `test_analysis_fourier.py`
   - `test_analysis_wavelet.py`
   - `test_analysis_laplace.py`
   - `test_analysis_ztransform.py`

2. **Refactor tuning tests:**
   - `test_parameter_tuning.py` (focus on computation)
   - `test_parameter_sweeps.py` (server-side sweeps)

3. **Delete VM management tests:**
   - Remove `test_vm_lifecycle.py`
   - Remove `test_continuous_simulation.py`
   - Remove `test_performance_monitoring.py`

4. **Update README:**
   - Reflect new computation-focused design
   - Clarify local vs remote responsibilities
   - Update test count and coverage

---

## 🔍 Example: What Changed?

### OLD (VM Management Focus):
```python
def test_create_ecoli_vm(self, test_client):
    """POST /api/v1/vms - Create E. coli VM"""
    response = test_client.post("/api/v1/vms", json={
        "vm_id": "test_ecoli_001",
        "biological_type": "ecoli",
        "genome_file": "ecoli_k12.gbk"
    })
    assert response.status_code == 201
```

**Problem:** VMs are managed by local BioXen library, not server!

### NEW (Computation Focus):
```python
def test_validate_circadian_period(self, test_client, circadian_time_series):
    """POST /api/v1/validate/oscillation"""
    response = test_client.post("/api/v1/validate/oscillation", json={
        "timestamps": circadian_time_series["timestamps"],
        "values": circadian_time_series["values"],
        "expected_period_hours": 24.0
    })
    assert response.status_code == 200
```

**Correct:** Send data to server, get validation results back!

---

This refactor aligns wishful-client-tests with the proven pattern from client-tests/ and correctly separates local VM management from remote computation services. 🎉

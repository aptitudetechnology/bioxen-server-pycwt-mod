# Wishful Client Tests - Refactor Status Report

**Date:** 2024-10-05  
**Status:** 🚧 Partially Complete - Only 2 of 10 files refactored

---

## ✅ Refactored Files (2 of 10)

### 1. `conftest.py` ✅ COMPLETE
- **Status:** Fully refactored to match `client-tests/` pattern
- **Changes:** Uses `httpx.Client`, biological time-series fixtures
- **Pattern:** Sends data to computation server, not VM management

### 2. `test_validation.py` ✅ COMPLETE  
- **Status:** New file, properly designed
- **Tests:** 25+ validation endpoint tests
- **Pattern:** `POST /api/v1/validate/*` endpoints
- **Example:**
  ```python
  # Correct: Send data, get validation back
  response = test_client.post(
      f"{api_base_url}/validate/oscillation",
      json={
          "timestamps": [...],
          "values": [...],
          "expected_period_hours": 24.0
      }
  )
  ```

---

## ❌ NOT Refactored (8 of 10)

### Files That SHOULD Be Deleted (VM Management)

#### 1. `test_vm_lifecycle.py` ❌ DELETE
- **Problem:** Tests VM creation/start/stop APIs
- **Reality:** VMs are managed by local BioXen library, not server
- **Example Issue:**
  ```python
  # WRONG: VMs are not managed by server!
  response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
  test_client.post(f"/api/v1/vms/{vm_id}/start")
  ```

#### 2. `test_continuous_simulation.py` ❌ DELETE
- **Problem:** Tests server-side simulation management
- **Reality:** Simulations run locally via BioXen library
- **Example Issue:**
  ```python
  # WRONG: Simulations run locally!
  test_client.post("/api/v1/vms/ecoli_001/simulate/continuous")
  ```

#### 3. `test_performance_monitoring.py` ❌ DELETE
- **Problem:** Tests profiler streaming, resource monitoring
- **Reality:** Monitoring happens locally, not on computation server
- **Lines:** 550+ lines (25+ tests)

#### 4. `test_model_validation.py` ❌ DELETE (Replaced)
- **Problem:** Duplicate of `test_validation.py` but with VM management
- **Status:** Already replaced by new `test_validation.py`
- **Lines:** 550+ lines (25+ tests)

### Files That NEED Refactoring (Computation Logic)

#### 5. `test_parameter_tuning.py` 🔄 NEEDS REFACTOR
- **Current:** VM-focused (`POST /api/v1/vms/{vm_id}/tuning/*`)
- **Should Be:** Data-focused (`POST /api/v1/tune/*`)
- **Lines:** 600+ lines (25+ tests)
- **What to Keep:** Rate tuning algorithms, optimization methods
- **Example Refactor:**
  ```python
  # BEFORE (Wrong - VM management)
  create_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
  vm_id = create_response.json()["vm_id"]
  response = test_client.post(f"/api/v1/vms/{vm_id}/tuning/rate-constants")
  
  # AFTER (Correct - Send data, get tuning suggestions)
  response = test_client.post(
      f"{api_base_url}/tune/rate-constants",
      json={
          "observed_data": {"timestamps": [...], "values": [...]},
          "target_period_hours": 24.0,
          "tunable_parameters": ["k1", "k2"]
      }
  )
  ```

#### 6. `test_four_lens_analysis.py` 🔄 NEEDS REFACTOR
- **Current:** VM-focused analysis
- **Should Be:** Perfect for computation APIs!
- **Lines:** 700+ lines (30+ tests)
- **What to Keep:** All analysis logic (Fourier, Wavelet, Laplace, Z-Transform)
- **Example Refactor:**
  ```python
  # BEFORE (Wrong)
  response = test_client.post(f"/api/v1/vms/{vm_id}/analysis/fourier")
  
  # AFTER (Correct - matches client-tests wavelet pattern)
  response = test_client.post(
      f"{api_base_url}/analysis/fourier",
      json={
          "timestamps": circadian_time_series["timestamps"],
          "values": circadian_time_series["values"],
          "method": "fft"
      }
  )
  ```

### Files That Are Correct (Sensor Hardware)

#### 7. `test_sensor_hardware.py` ✅ KEEP AS-IS
- **Pattern:** Hardware integration (different from computation APIs)
- **Example:** `GET /api/v1/sensors/bme280/read`
- **Lines:** 250 lines (15 tests)
- **Status:** Correct - sensors are direct hardware access, not computation

#### 8. `test_circadian_entrainment.py` ✅ KEEP AS-IS
- **Pattern:** Sensor-driven experiments with light cycles
- **Lines:** 420 lines (20 tests)
- **Status:** Correct - uses sensors for light control

#### 9. `test_temperature_compensation.py` ✅ KEEP AS-IS
- **Pattern:** Temperature sensor experiments
- **Lines:** 300 lines (12 tests)
- **Status:** Correct - uses BME280 for temperature control

---

## 📊 Refactor Progress

| File | Lines | Tests | Status | Action Needed |
|------|-------|-------|--------|---------------|
| `conftest.py` | 275 | - | ✅ DONE | None |
| `test_validation.py` | 280 | 25+ | ✅ DONE | None |
| `test_sensor_hardware.py` | 250 | 15 | ✅ CORRECT | None |
| `test_circadian_entrainment.py` | 420 | 20 | ✅ CORRECT | None |
| `test_temperature_compensation.py` | 300 | 12 | ✅ CORRECT | None |
| `test_vm_lifecycle.py` | 400 | 30+ | ❌ WRONG | **DELETE** |
| `test_continuous_simulation.py` | 500 | 25+ | ❌ WRONG | **DELETE** |
| `test_model_validation.py` | 550 | 25+ | ❌ WRONG | **DELETE** (replaced) |
| `test_performance_monitoring.py` | 550 | 25+ | ❌ WRONG | **DELETE** |
| `test_parameter_tuning.py` | 600 | 25+ | 🔄 NEEDS REFACTOR | **REFACTOR** |
| `test_four_lens_analysis.py` | 700 | 30+ | 🔄 NEEDS REFACTOR | **REFACTOR** |

### Summary
- ✅ **Done:** 2 files (conftest, test_validation)
- ✅ **Correct:** 3 files (sensor tests - different pattern)
- ❌ **Delete:** 4 files (VM management - 2,000+ lines)
- 🔄 **Refactor:** 2 files (computation logic - 1,300+ lines)

**Total Work Remaining:** 6 files to handle (4 delete, 2 refactor)

---

## 🎯 Recommended Action Plan

### Step 1: Delete VM Management Files (Quick)
```bash
rm wishful-client-tests/test_vm_lifecycle.py
rm wishful-client-tests/test_continuous_simulation.py
rm wishful-client-tests/test_performance_monitoring.py
rm wishful-client-tests/test_model_validation.py  # Replaced by test_validation.py
```

### Step 2: Refactor Analysis File (High Value)
Create **`test_analysis.py`** from `test_four_lens_analysis.py`:
- Extract Fourier/Wavelet/Laplace/Z-Transform tests
- Change from `POST /api/v1/vms/{vm_id}/analysis/*` 
- To `POST /api/v1/analysis/*`
- Send time-series data directly

### Step 3: Refactor Tuning File (Medium Value)
Create **`test_tuning.py`** from `test_parameter_tuning.py`:
- Extract rate constant tuning tests
- Change from `POST /api/v1/vms/{vm_id}/tuning/*`
- To `POST /api/v1/tune/*`
- Send observed data + tuning parameters

### Step 4: Update Documentation
- Update README.md with new file structure
- Update test count (remove ~100 VM tests, keep ~50 computation tests)
- Clarify local vs remote responsibilities

---

## 🔍 Key Distinction

### ❌ Wrong Pattern (VM Management on Server)
```python
# Server manages VMs
create_response = test_client.post("/api/v1/vms", json=config)
vm_id = create_response.json()["vm_id"]
test_client.post(f"/api/v1/vms/{vm_id}/start")
test_client.post(f"/api/v1/vms/{vm_id}/simulations", json={...})
```

### ✅ Correct Pattern (Computation Services)
```python
# Send data, get results (like PyCWT-mod wavelet server)
response = test_client.post(
    f"{api_base_url}/validate/oscillation",
    json={
        "timestamps": [...],
        "values": [...],
        "expected_period_hours": 24.0
    }
)
assert response.status_code == 200
assert "measured_period_hours" in response.json()
```

---

## 📝 Estimated Work

- **Delete 4 files:** 10 minutes
- **Refactor `test_four_lens_analysis.py`:** 2 hours (complex analysis logic)
- **Refactor `test_parameter_tuning.py`:** 1.5 hours (optimization logic)
- **Update docs:** 30 minutes
- **Total:** ~4 hours

---

**Bottom Line:** No, only 2 of 10 files have been refactored. We need to delete 4 VM management files and refactor 2 computation files to complete the transition. 🚧

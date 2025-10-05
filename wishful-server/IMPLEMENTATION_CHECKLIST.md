# BioXen Wishful Server - Implementation Checklist 📋

**Status:** Ready to Build  
**Target:** Implement REST API wrapping existing BioXen functionality  
**Timeline:** 4-15 weeks (depending on phases completed)

---

## 🎯 Phase 1: Analysis APIs (Week 1-2) ⭐⭐⭐⭐⭐

**Goal:** Expose SystemAnalyzer via REST (QUICK WIN - 84% test coverage!)

### Setup
- [ ] Create project structure
  ```bash
  cd wishful-server
  mkdir -p api core tests
  touch main.py
  touch requirements.txt
  ```

- [ ] Install dependencies
  ```bash
  pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.4.2
  pip install numpy scipy astropy pywavelets
  ```

### Core Implementation
- [ ] Create `api/models.py` (Pydantic schemas)
  - [ ] FourierRequest
  - [ ] FourierResponse
  - [ ] WaveletRequest
  - [ ] WaveletResponse
  - [ ] LaplaceRequest
  - [ ] LaplaceResponse
  - [ ] ZTransformRequest
  - [ ] ZTransformResponse

- [ ] Create `core/analyzer_wrapper.py`
  - [ ] Import SystemAnalyzer from BioXen
  - [ ] Wrap fourier_lens()
  - [ ] Wrap wavelet_lens()
  - [ ] Wrap laplace_lens()
  - [ ] Wrap z_transform_lens()
  - [ ] Add error handling
  - [ ] Add metadata (computation time, version)

- [ ] Create `api/analysis.py`
  - [ ] POST /api/v1/analysis/fourier
  - [ ] POST /api/v1/analysis/wavelet
  - [ ] POST /api/v1/analysis/laplace
  - [ ] POST /api/v1/analysis/ztransform
  - [ ] POST /api/v1/analysis/multi-domain

- [ ] Create `main.py`
  - [ ] Initialize FastAPI app
  - [ ] Add CORS middleware
  - [ ] Include analysis router
  - [ ] Add error handlers

### Testing
- [ ] Run server: `uvicorn main:app --reload`
- [ ] Test with curl:
  ```bash
  curl -X POST http://localhost:8000/api/v1/analysis/fourier \
    -H "Content-Type: application/json" \
    -d '{"timestamps": [0,1,2,3,4], "values": [1.0, 1.1, 0.9, 1.05, 0.95]}'
  ```
- [ ] Run wishful tests:
  ```bash
  pytest ../wishful-client-tests/test_analysis.py -v
  ```
- [ ] Target: 21/25 tests passing (84%)

### Documentation
- [ ] Add OpenAPI docs (automatic via FastAPI)
- [ ] Test docs at `http://localhost:8000/docs`
- [ ] Create README.md with setup instructions

**Estimated Time:** 1-2 weeks  
**Lines of Code:** ~800 lines  
**Complexity:** Low (wrapping existing code)

---

## 🎯 Phase 2: Validation APIs (Week 3-5) ⭐⭐⭐⭐

**Goal:** Add validation policies around analysis

### Core Implementation
- [ ] Create `core/oscillation_validator.py`
  - [ ] validate_period() - Check period against expected
  - [ ] validate_amplitude() - Check for decay
  - [ ] _calculate_quality() - Quality scoring

- [ ] Create `core/stability_validator.py`
  - [ ] validate_stability() - Laplace pole check
  - [ ] check_oscillatory() - Oscillation classification

- [ ] Create `core/quality_scorer.py`
  - [ ] calculate_quality_score() - Multi-metric scoring
  - [ ] calculate_snr() - Signal-to-noise ratio
  - [ ] calculate_regularity() - Oscillation regularity

- [ ] Create `api/validation.py`
  - [ ] POST /api/v1/validate/oscillation
  - [ ] POST /api/v1/validate/amplitude
  - [ ] POST /api/v1/validate/stability
  - [ ] POST /api/v1/validate/deviation
  - [ ] POST /api/v1/validate/quality
  - [ ] POST /api/v1/validate/batch

### Testing
- [ ] Run wishful tests:
  ```bash
  pytest ../wishful-client-tests/test_validation.py -v
  ```
- [ ] Target: 23/25 tests passing (92%)

**Estimated Time:** 2-3 weeks  
**Lines of Code:** ~1,200 lines  
**Complexity:** Medium (business logic + API)

---

## 🎯 Phase 3: Optimization APIs (Week 6-11) ⭐⭐⭐

**Goal:** Build parameter tuning framework

### Setup
- [ ] Install optimization dependencies
  ```bash
  pip install SALib==1.4.7 scikit-optimize==0.9.0
  ```

### Core Implementation
- [ ] Create `core/parameter_tuner.py`
  - [ ] tune_rate_constants() - scipy.optimize wrapper
  - [ ] _create_fitness_function() - MSE + constraints
  - [ ] _apply_constraints() - Use bio_constraints.py
  - [ ] optimize_timestep() - Adaptive timestep finding

- [ ] Create `core/sensitivity_analyzer.py`
  - [ ] local_sensitivity() - Finite differences
  - [ ] global_sensitivity() - Sobol indices (SALib)
  - [ ] _evaluate_model() - Model evaluation

- [ ] Create `core/sweep_engine.py`
  - [ ] parameter_sweep_1d() - Single parameter
  - [ ] parameter_sweep_2d() - Grid search
  - [ ] _parallel_evaluate() - Parallel execution

- [ ] Create `api/tuning.py`
  - [ ] POST /api/v1/tune/rate-constants
  - [ ] POST /api/v1/tune/timestep
  - [ ] POST /api/v1/tune/sweep
  - [ ] POST /api/v1/tune/sweep-2d
  - [ ] POST /api/v1/tune/sensitivity
  - [ ] POST /api/v1/tune/sensitivity-global
  - [ ] POST /api/v1/tune/multi-objective

### Testing
- [ ] Run wishful tests:
  ```bash
  pytest ../wishful-client-tests/test_tuning.py -v
  ```
- [ ] Target: 13/18 tests passing (72%)

**Estimated Time:** 4-6 weeks  
**Lines of Code:** ~2,000 lines  
**Complexity:** High (optimization algorithms)

---

## 🎯 Phase 4: Hardware APIs (Week 12-15) ⭐ (OPTIONAL)

**Goal:** Add sensor integration

### Hardware Setup
- [ ] Order hardware
  - [ ] Raspberry Pi (or similar with I2C)
  - [ ] BME280 breakout board (~$10)
  - [ ] LTR-559 breakout board (~$15)
  - [ ] Breadboard + jumper wires

- [ ] Enable I2C on Raspberry Pi
  ```bash
  sudo raspi-config  # Enable I2C
  sudo i2cdetect -y 1  # Verify sensors detected
  ```

- [ ] Install sensor libraries
  ```bash
  pip install smbus2 pimoroni-bme280 ltr559
  ```

### Core Implementation
- [ ] Create `hardware/sensor_manager.py`
  - [ ] __init__() - Initialize I2C bus
  - [ ] detect_sensors() - Auto-detect connected sensors
  - [ ] _calibrate_sensor() - Calibration routine

- [ ] Create `hardware/bme280_driver.py`
  - [ ] read_temperature()
  - [ ] read_pressure()
  - [ ] read_humidity()
  - [ ] read_all()

- [ ] Create `hardware/ltr559_driver.py`
  - [ ] read_light()
  - [ ] read_proximity()
  - [ ] set_gain()
  - [ ] set_integration_time()

- [ ] Create `api/sensors.py`
  - [ ] GET /api/v1/sensors/bme280/temperature
  - [ ] GET /api/v1/sensors/bme280/all
  - [ ] GET /api/v1/sensors/ltr559/light
  - [ ] POST /api/v1/sensors/{type}/calibrate
  - [ ] GET /api/v1/sensors/status

### Testing
- [ ] Test with actual hardware
- [ ] Run wishful tests:
  ```bash
  pytest ../wishful-client-tests/test_sensor_hardware.py -v
  ```
- [ ] Target: 13/15 tests passing (87%)

**Estimated Time:** 3-4 weeks  
**Lines of Code:** ~1,500 lines  
**Complexity:** Medium-High (hardware debugging)

---

## 🔧 Common Tasks

### Adding a New Endpoint

1. **Define Pydantic models** (`api/models.py`):
   ```python
   class MyRequest(BaseModel):
       data: List[float]
       param: float = 1.0
   
   class MyResponse(BaseModel):
       result: float
       metadata: dict
   ```

2. **Create business logic** (`core/my_feature.py`):
   ```python
   def my_computation(data: np.ndarray, param: float) -> dict:
       # Your logic here
       return {"result": 42.0}
   ```

3. **Add API endpoint** (`api/my_router.py`):
   ```python
   @router.post("/my-endpoint")
   async def my_endpoint(request: MyRequest):
       result = my_computation(np.array(request.data), request.param)
       return MyResponse(**result)
   ```

4. **Register router** (`main.py`):
   ```python
   app.include_router(my_router, prefix="/api/v1/my", tags=["my"])
   ```

### Error Handling Pattern

```python
from fastapi import HTTPException

@router.post("/endpoint")
async def endpoint(request: MyRequest):
    try:
        # Validation
        if len(request.data) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Data array cannot be empty"
                    }
                }
            )
        
        # Computation
        result = my_computation(request.data)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "COMPUTATION_ERROR",
                    "message": str(e)
                }
            }
        )
```

### Testing Pattern

```python
# tests/test_analysis.py
import httpx
import pytest

@pytest.fixture
def client():
    with httpx.Client(base_url="http://localhost:8000") as client:
        yield client

def test_fourier_basic(client):
    response = client.post(
        "/api/v1/analysis/fourier",
        json={
            "timestamps": [0, 1, 2, 3, 4],
            "values": [1.0, 1.1, 0.9, 1.05, 0.95]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "frequencies" in data
    assert "magnitudes" in data
```

---

## 📊 Progress Tracking

Use this to track overall progress:

### Phase 1: Analysis APIs
- [ ] Setup complete (10%)
- [ ] Models defined (20%)
- [ ] Core wrappers implemented (40%)
- [ ] API endpoints created (30%)
- [ ] Tests passing >80% (100%)

### Phase 2: Validation APIs
- [ ] Oscillation validator (30%)
- [ ] Stability validator (20%)
- [ ] Quality scorer (20%)
- [ ] API endpoints (20%)
- [ ] Tests passing >90% (10%)

### Phase 3: Optimization APIs
- [ ] Parameter tuner (40%)
- [ ] Sensitivity analyzer (30%)
- [ ] Sweep engine (20%)
- [ ] API endpoints (10%)

### Phase 4: Hardware APIs
- [ ] Hardware acquired (20%)
- [ ] Sensor drivers (40%)
- [ ] Calibration (20%)
- [ ] API endpoints (20%)

---

## 🚀 Getting Started TODAY

```bash
# 1. Navigate to wishful-server
cd /home/chris/BioXen_Fourier_lib/wishful-server

# 2. Create directory structure
mkdir -p api core hardware tests
touch main.py requirements.txt
touch api/__init__.py core/__init__.py hardware/__init__.py

# 3. Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
numpy==1.24.3
scipy==1.11.3
astropy==5.3.4
PyWavelets==1.4.1
python-multipart==0.0.6
EOF

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create minimal main.py
cat > main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BioXen Computation API",
    version="0.1.0",
    description="Remote computation services for biological signal analysis"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "BioXen Computation API",
        "version": "0.1.0",
        "status": "online",
        "endpoints": {
            "analysis": "/api/v1/analysis",
            "validation": "/api/v1/validate",
            "tuning": "/api/v1/tune",
            "sensors": "/api/v1/sensors"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF

# 6. Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 7. Test (in another terminal)
curl http://localhost:8000/
curl http://localhost:8000/health

# 8. Check OpenAPI docs
# Open browser: http://localhost:8000/docs
```

---

## 📚 Resources

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- OpenAPI: https://swagger.io/specification/

### BioXen Code
- SystemAnalyzer: `../src/bioxen_fourier_vm_lib/analysis/system_analyzer.py`
- BioConstraints: `../src/bioxen_fourier_vm_lib/genetics/circuits/optimization/bio_constraints.py`

### Tests
- Analysis: `../wishful-client-tests/test_analysis.py`
- Validation: `../wishful-client-tests/test_validation.py`
- Tuning: `../wishful-client-tests/test_tuning.py`
- Sensors: `../wishful-client-tests/test_sensor_hardware.py`

### Specifications
- Full API Spec: `wishful-api-specification-document.md`
- Implementation Guide: `../wishful-client-tests/IMPLEMENTATION_ROADMAP.md`

---

**Ready to build!** Start with Phase 1 and get 84% test coverage in 1-2 weeks! 🚀

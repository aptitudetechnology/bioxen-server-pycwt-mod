# PyCWT-mod REST API Test Suite

Comprehensive test suite for the PyCWT-mod REST API server, implementing test-driven development (TDD) for hardware-accelerated wavelet analysis.

## 📋 Test Overview

| Test Module | Purpose | Test Classes | Coverage |
|------------|---------|--------------|----------|
| `test_health.py` | Root endpoints, health checks | 1 | `/`, `/health`, `/docs` |
| `test_backends.py` | Backend management | 4 | List, info, availability |
| `test_wavelet.py` | Wavelet analysis | 4 | CWT, WCT, XWT |
| `test_hardware.py` | Hardware detection | 5 | FPGA, GPU, CPU, serial |
| `test_benchmark.py` | Performance benchmarking | 5 | Speedup, research validation |
| `test_integration.py` | End-to-end workflows | 6 | Complete workflows, BioXen |

**Total:** 100+ test functions across 6 test modules

## 🚀 Quick Start

### Install Test Dependencies

```bash
pip install -r server/tests/requirements-test.txt
```

### Run All Tests

```bash
# Using pytest directly
pytest server/tests/

# Using test runner script
python run_tests.py
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Hardware tests (requires FPGA/GPU)
pytest -m hardware

# Benchmark tests
pytest -m benchmark
```

## 📊 Test Organization

### Test Markers

Tests are organized with pytest markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests (< 1s each)
- `@pytest.mark.integration` - Integration tests (1-5s each)
- `@pytest.mark.slow` - Long-running tests (> 5s)
- `@pytest.mark.hardware` - Requires physical hardware (FPGA/GPU)
- `@pytest.mark.benchmark` - Performance benchmarking

### Test Fixtures (conftest.py)

**Client Fixtures:**
- `test_client` - FastAPI TestClient for API requests

**Signal Fixtures:**
- `sample_signal_short` - 100 samples (fast tests)
- `sample_signal_medium` - 1000 samples
- `sample_signal_long` - 10000 samples (slow tests)
- `signal_pair` - Two correlated signals for WCT/XWT

**Request Fixtures:**
- `cwt_request_basic` - Basic CWT request payload
- `wct_request_with_significance` - WCT with Monte Carlo
- `benchmark_request` - Benchmark configuration

**Mock Fixtures:**
- `mock_backend_list` - Available backends
- `mock_hardware_detection` - Hardware capabilities

**Helper Functions:**
- `assert_valid_wavelet_response()` - Validate wavelet output
- `generate_eeg_signal()` - Generate realistic EEG data
- `generate_circadian_signal()` - Generate circadian rhythm data

## 🧪 Test Examples

### Run Health Checks

```bash
pytest server/tests/test_health.py -v
```

Expected output:
```
test_health.py::test_root_endpoint PASSED
test_health.py::test_health_check_healthy PASSED
test_health.py::test_docs_endpoint_exists PASSED
```

### Run Backend Tests

```bash
pytest server/tests/test_backends.py -v
```

Tests:
- ✅ Sequential backend always available
- ✅ Joblib backend on multi-core systems
- ✅ ELM11 backend when Tang Nano 9K detected
- ✅ Backend info consistency

### Run Wavelet Analysis Tests

```bash
pytest server/tests/test_wavelet.py -v
```

Tests:
- ✅ CWT with basic parameters
- ✅ CWT with all parameters (wavelet, scales, backend)
- ✅ WCT with significance testing
- ✅ XWT with phase angles
- ✅ Input validation (empty signals, NaN values)

### Run Hardware Detection Tests

```bash
pytest server/tests/test_hardware.py -v
```

Tests:
- ✅ FPGA detection (Tang Nano 9K)
- ✅ Serial port enumeration (SIPEED JTAG)
- ✅ GPU detection (NVIDIA CUDA)
- ✅ CPU information (cores, architecture)

### Run Research Validation Tests

```bash
pytest server/tests/test_benchmark.py::TestBenchmarkResearchValidation -v
```

**MVP Research Q1:** Performance characterization
```python
# Tests speedup across signal lengths
# Expected: Sequential 1.0×, Joblib 2-4×, FPGA 10-100×
pytest -k "test_mvp_question1"
```

**MVP Research Q2:** PyWavelets comparison
```python
# Tests joblib speedup vs PyWavelets baseline
# Expected: 2-8× speedup for multi-core
pytest -k "test_mvp_question2"
```

**FPGA Research:** BCI latency measurement
```python
# Tests FPGA latency for real-time BCI
# Expected: < 10ms for 256-sample window
pytest -k "test_fpga_research_latency" -m hardware
```

### Run Integration Tests

```bash
pytest server/tests/test_integration.py -v
```

**Complete Workflow:**
```python
# Tests full WCT workflow:
# 1. Health check
# 2. List backends
# 3. Detect hardware
# 4. Compute WCT with significance
pytest -k "test_full_wct_workflow"
```

**BioXen Integration:**
```python
# Simulates BioXen remote backend
pytest -k "TestBioXenIntegration"
```

## 📈 Coverage Reports

### Generate Coverage Report

```bash
pytest --cov=server --cov-report=html --cov-report=term
```

### View HTML Report

```bash
# Open in browser
firefox server/tests/htmlcov/index.html
```

**Coverage Goals:**
- Overall: > 80%
- Critical paths (wavelet, backends): > 95%
- Error handling: > 90%

## 🔧 Test Configuration

### pytest.ini

Global configuration in `pytest.ini`:

```ini
[pytest]
testpaths = server/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Fast unit tests
    integration: Integration tests
    slow: Tests that take > 5s
    hardware: Tests requiring FPGA/GPU
    benchmark: Performance benchmarking

addopts =
    --strict-markers
    --tb=short
    --cov-report=html
    --timeout=300
```

## 🎯 Test-Driven Development Workflow

### Phase 1: Run Tests (Expect Failures)

```bash
pytest server/tests/ -v
```

All tests will fail initially since server implementation doesn't exist yet.

### Phase 2: Implement Features

Implement features to make tests pass:

1. **Health endpoints** → `test_health.py` passes
2. **Backend service** → `test_backends.py` passes
3. **Wavelet endpoints** → `test_wavelet.py` passes
4. **Hardware detection** → `test_hardware.py` passes
5. **Benchmarking** → `test_benchmark.py` passes
6. **Integration** → `test_integration.py` passes

### Phase 3: Validate Implementation

```bash
# Run all tests
pytest server/tests/

# Check coverage
pytest --cov=server --cov-report=term-missing
```

## 🐛 Debugging Failed Tests

### Verbose Output

```bash
pytest -vv server/tests/test_wavelet.py::TestContinuousWaveletTransform::test_cwt_basic_request
```

### Show Print Statements

```bash
pytest -s server/tests/test_wavelet.py
```

### Stop at First Failure

```bash
pytest -x server/tests/
```

### Drop into Debugger on Failure

```bash
pytest --pdb server/tests/
```

### Run Last Failed Tests

```bash
pytest --lf server/tests/
```

## 🔬 Research Validation

### MVP Research Questions

**Q1: Performance Characterization**
```bash
pytest server/tests/test_benchmark.py::TestBenchmarkResearchValidation::test_mvp_question1_performance_characterization -v
```

Validates:
- Sequential baseline (1.0× speedup)
- Joblib scaling with CPU cores
- FPGA speedup across signal lengths

**Q2: PyWavelets Comparison**
```bash
pytest server/tests/test_benchmark.py::TestBenchmarkResearchValidation::test_mvp_question2_pywavelets_comparison -v
```

Validates:
- Joblib speedup vs PyWavelets sequential
- Expected: 2-8× speedup for multi-core systems

### FPGA Research

**BCI Latency Measurement**
```bash
pytest server/tests/test_benchmark.py::TestBenchmarkResearchValidation::test_fpga_research_latency_measurement -m hardware -v
```

Validates:
- FPGA latency < 10ms for 256-sample window
- Suitable for real-time BCI applications

## 🌐 BioXen Integration Testing

### Remote Backend Workflow

```bash
pytest server/tests/test_integration.py::TestBioXenIntegration -v
```

**Test Scenarios:**
1. Client connects to remote API
2. Lists available backends
3. Computes WCT with Monte Carlo
4. Validates results format
5. Batch analysis workflow

**Example Client Code (from tests):**
```python
# Simulates BioXen remote backend
client = TestClient(app)

# 1. Check health
response = client.get("/health")
assert response.status_code == 200

# 2. List backends
response = client.get("/api/v1/backends/")
backends = response.json()

# 3. Compute WCT
payload = {
    "signal1": signal1.tolist(),
    "signal2": signal2.tolist(),
    "dt": 1.0,
    "backend": "joblib",
    "significance_test": True,
    "mc_count": 100
}
response = client.post("/api/v1/wavelet/wct", json=payload)
result = response.json()

# 4. Validate
assert "wcoh" in result
assert "significance" in result
```

## 📝 Adding New Tests

### Template for New Test

```python
import pytest

class TestNewFeature:
    """Test suite for new feature."""
    
    @pytest.mark.unit
    def test_basic_functionality(self, test_client):
        """Test basic feature functionality."""
        response = test_client.get("/api/v1/new-endpoint")
        
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data
    
    @pytest.mark.integration
    def test_feature_integration(self, test_client, sample_signal_medium):
        """Test feature integration with other components."""
        payload = {
            "signal": sample_signal_medium.tolist(),
            "parameter": "value"
        }
        response = test_client.post("/api/v1/new-endpoint", json=payload)
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
```

## 🚨 CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r server/tests/requirements-test.txt
    
    - name: Run tests
      run: |
        pytest server/tests/ --cov=server --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## 📚 References

- **API Specification:** `PyCWT-mod-api-specification.md`
- **Implementation Plan:** `PyCWT-mod-api-specification-claudes-implementation-plan.md`
- **Research Documents:** 
  - `wavelets-deep-research-prompt2-mvp.md`
  - `FFT-based-performance-bottleneck-massive-datasets-prompt.FPGA.md`

## 🤝 Contributing

When adding new features:

1. Write tests first (TDD)
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Add fixtures to `conftest.py` if reusable
4. Document test purpose in docstrings
5. Ensure tests pass before committing

## 📞 Support

For issues or questions:
- Check test output for error messages
- Review API specification for expected behavior
- Consult implementation plan for architecture details
- Run with `-vv` for verbose debugging output

# Laptop Testing Instructions for pycwt-mod

**Project:** pycwt-mod - Modular Python Continuous Wavelet Transform Library  
**Branch:** development  
**Date:** October 2, 2025  
**Phases Complete:** Phase 1 (Backend Architecture) + Phase 2 (Integration)

---

## 📋 What We've Built

We've successfully integrated a modular backend system into pycwt-mod that enables parallel Monte Carlo simulations for wavelet coherence significance testing. The system includes:

✅ **Plugin-style backend architecture**
- Abstract base class for extensible backends
- Registry system for backend discovery
- Auto-selection of optimal backend

✅ **Two working backends**
- Sequential: Single-core execution (always available)
- Joblib: Multi-core parallel execution (if joblib installed)

✅ **Core integration**
- `wct_significance()` now supports parallel execution
- Full backward compatibility maintained
- New optional parameters: `backend` and `n_jobs`

✅ **Comprehensive test suite**
- Backend system tests
- Integration tests
- Performance validation tests

---

## 🚀 Quick Start Testing

### Step 1: Install the Package

```bash
cd /home/chris/pycwt-mod

# Install in development mode with dependencies
pip install -e .

# Or if you prefer pip3
pip3 install -e .
```

This will install:
- `pycwt-mod` package in editable mode
- All required dependencies (numpy, scipy, matplotlib, tqdm)
- Optional dependency (joblib) for parallel execution

### Step 2: Verify Installation

```bash
# Check that the package is installed
python3 -c "import pycwt_mod; print('✓ pycwt_mod installed:', pycwt_mod.__version__)"

# Check backend system
python3 -c "from pycwt_mod.backends import list_backends; print('✓ Available backends:', list(list_backends().keys()))"

# Check backend availability
python3 -c "
from pycwt_mod.backends import get_backend
for name in ['sequential', 'joblib']:
    backend = get_backend(name)
    status = '✓' if backend.is_available() else '✗'
    print(f'{status} {name}: {backend.is_available()}')
"
```

Expected output:
```
✓ pycwt_mod installed: 0.0.001
✓ Available backends: ['sequential', 'joblib', 'dask', 'gpu']
✓ sequential: True
✓ joblib: True  (or False if joblib not installed)
```

### Step 3: Run Quick Integration Test

```bash
# Run the simple integration test
python3 test_phase2_integration.py
```

This will test:
- Default backend selection
- Explicit sequential backend
- Joblib backend (if available)
- Backward compatibility

Expected output:
```
======================================================================
Phase 2 Integration Test: wct_significance with backend system
======================================================================

Available backends:
  ✓ sequential: SequentialBackend
  ✓ joblib: JoblibBackend
  ✗ dask: DaskBackend
  ✗ gpu: GPUBackend

Test parameters:
  Red noise coefficients: al1=0.72, al2=0.72
  Scales: dj=0.25, s0=2*dt, J=7/dj
  Monte Carlo count: 10 (quick test)

----------------------------------------------------------------------
Test 1: Default backend (auto-select)
----------------------------------------------------------------------
Calculating wavelet coherence significance
✓ Success! Result shape: (29,)
  Sample values: [0.xxx 0.xxx 0.xxx]

----------------------------------------------------------------------
Test 2: Explicit sequential backend
----------------------------------------------------------------------
Calculating wavelet coherence significance
✓ Success! Result shape: (29,)
  Sample values: [0.xxx 0.xxx 0.xxx]

----------------------------------------------------------------------
Test 3: Joblib backend with n_jobs=2
----------------------------------------------------------------------
Calculating wavelet coherence significance
✓ Success! Result shape: (29,)
  Sample values: [0.xxx 0.xxx 0.xxx]
  Comparing with sequential results:
    Max difference: 0.xxxxxx
    Mean difference: 0.xxxxxx

----------------------------------------------------------------------
Test 4: Backward compatibility (original function signature)
----------------------------------------------------------------------
Calculating wavelet coherence significance
✓ Success! Result shape: (29,)
  Sample values: [0.xxx 0.xxx 0.xxx]
  Backward compatibility maintained!

======================================================================
Phase 2 Integration Test Complete!
======================================================================
```

---

## 🧪 Comprehensive Testing

### Option A: Interactive Test Runner (Recommended)

```bash
python3 run_phase2_tests.py
```

This will:
1. Run all backend system tests
2. Run integration tests
3. Optionally run performance tests (asks for confirmation)
4. Provide detailed pass/fail summary

### Option B: Run with pytest

```bash
# Install pytest if needed
pip install pytest pytest-cov

# Run all tests
pytest src/pycwt_mod/tests/ -v

# Run with coverage report
pytest src/pycwt_mod/tests/ --cov=pycwt_mod --cov-report=term-missing

# Run only backend tests
pytest src/pycwt_mod/tests/backends/ -v

# Run only integration tests
pytest src/pycwt_mod/tests/test_wct_significance_integration.py -v

# Run performance tests (slow, marked)
pytest src/pycwt_mod/tests/test_performance.py -v -m slow -s
```

### Option C: Individual Test Files

```bash
# Integration tests (fast)
python3 -m pytest src/pycwt_mod/tests/test_wct_significance_integration.py -v -s

# Performance tests (slow, optional)
python3 -m pytest src/pycwt_mod/tests/test_performance.py -v -s -m slow
```

---

## 💡 Quick Usage Examples

### Example 1: Basic Usage (Auto Backend)

```python
from pycwt_mod import wct_significance

# Compute significance levels (backend auto-selected)
sig95 = wct_significance(
    al1=0.72,      # AR(1) coefficient for signal 1
    al2=0.72,      # AR(1) coefficient for signal 2
    dt=0.25,       # Time step
    dj=0.25,       # Scale resolution
    s0=0.5,        # Smallest scale
    J=28,          # Number of scales
    mc_count=300   # Monte Carlo simulations
)

print(f"Significance levels shape: {sig95.shape}")
print(f"95% threshold at scale 0: {sig95[0]:.4f}")
```

### Example 2: Explicit Sequential Backend

```python
from pycwt_mod import wct_significance

# Force sequential execution
sig95 = wct_significance(
    al1=0.72,
    al2=0.72,
    dt=0.25,
    dj=0.25,
    s0=0.5,
    J=28,
    mc_count=300,
    backend='sequential',  # Explicit backend
    progress=True          # Show progress bar
)
```

### Example 3: Parallel Execution

```python
from pycwt_mod import wct_significance

# Use parallel backend with 4 cores
sig95 = wct_significance(
    al1=0.72,
    al2=0.72,
    dt=0.25,
    dj=0.25,
    s0=0.5,
    J=28,
    mc_count=300,
    backend='joblib',  # Parallel backend
    n_jobs=4,          # Use 4 cores
    progress=True
)
```

### Example 4: Backward Compatible (Old Code)

```python
from pycwt_mod import wct_significance

# Old-style call (still works!)
sig95 = wct_significance(
    0.72,  # al1 (positional)
    0.72,  # al2 (positional)
    dt=0.25,
    dj=0.25,
    s0=0.5,
    J=28,
    mc_count=300
)
# Backend is auto-selected, backward compatible
```

### Example 5: Check Available Backends

```python
from pycwt_mod.backends import list_backends, get_backend

# List all registered backends
print("Registered backends:")
for name in list_backends():
    backend = get_backend(name)
    available = "✓" if backend.is_available() else "✗"
    print(f"  {available} {name}")

# Get recommended backend for a workload
from pycwt_mod.backends import get_recommended_backend

backend_name = get_recommended_backend(n_simulations=500)
print(f"\nRecommended for 500 simulations: {backend_name}")
```

---

## 🔍 Verification Tests

### Test 1: Import Check

```python
python3 << 'EOF'
# Test all imports work
from pycwt_mod import wct_significance, cwt, xwt
from pycwt_mod.backends import (
    get_backend, 
    list_backends,
    get_recommended_backend,
    MonteCarloBackend,
    SequentialBackend,
    JoblibBackend
)
from pycwt_mod.mothers import Morlet, Paul, DOG, MexicanHat

print("✓ All imports successful!")
EOF
```

### Test 2: Backend Functionality

```python
python3 << 'EOF'
import numpy as np
from pycwt_mod.backends import get_backend

# Test sequential backend
backend = get_backend('sequential')
print(f"Sequential backend: {backend.is_available()}")
print(f"Backend name: {backend.name}")
print(f"Backend class: {backend.__class__.__name__}")

# Test worker function can be called
def dummy_worker(seed):
    np.random.seed(seed)
    return np.random.rand(10)

results = backend.run_monte_carlo(
    dummy_worker,
    n_simulations=5,
    worker_args=(),
    verbose=False
)

print(f"✓ Worker function executed: {len(results)} results")
EOF
```

### Test 3: Integration Smoke Test

```python
python3 << 'EOF'
from pycwt_mod import wct_significance
import numpy as np

# Quick smoke test
sig95 = wct_significance(
    al1=0.5,
    al2=0.5,
    dt=0.25,
    dj=0.25,
    s0=0.5,
    J=5,
    mc_count=10,
    progress=False,
    cache=False
)

assert sig95 is not None
assert isinstance(sig95, np.ndarray)
assert sig95.shape[0] == 6  # J+1
assert np.all((sig95[~np.isnan(sig95)] >= 0) & (sig95[~np.isnan(sig95)] <= 1))

print("✓ Integration smoke test passed!")
EOF
```

### Test 4: Performance Comparison

```python
python3 << 'EOF'
import time
import numpy as np
from pycwt_mod import wct_significance
from pycwt_mod.backends import get_backend

# Only run if joblib available
if get_backend('joblib').is_available():
    print("Comparing sequential vs parallel performance...")
    
    # Sequential
    start = time.time()
    sig_seq = wct_significance(
        al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
        mc_count=50, backend='sequential', progress=False, cache=False
    )
    seq_time = time.time() - start
    
    # Parallel
    start = time.time()
    sig_par = wct_significance(
        al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
        mc_count=50, backend='joblib', n_jobs=2, progress=False, cache=False
    )
    par_time = time.time() - start
    
    speedup = seq_time / par_time
    print(f"\nSequential: {seq_time:.2f}s")
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {speedup:.2f}×")
    
    # Check results are similar
    correlation = np.corrcoef(
        sig_seq[~np.isnan(sig_seq)],
        sig_par[~np.isnan(sig_par)]
    )[0,1]
    print(f"Correlation: {correlation:.3f}")
    
    print("\n✓ Performance comparison complete!")
else:
    print("⚠ Joblib not available, skipping performance test")
EOF
```

---

## 📊 Expected Test Results

### All Tests Passing
```
✓ Import check: All modules importable
✓ Backend functionality: Sequential and joblib work
✓ Integration: wct_significance produces valid results
✓ Backward compatibility: Old code still works
✓ Performance: Parallel faster than sequential (if joblib available)
```

### Pytest Summary
```
src/pycwt_mod/tests/backends/test_base.py ............ PASSED
src/pycwt_mod/tests/backends/test_registry.py ........ PASSED
src/pycwt_mod/tests/backends/test_sequential.py ...... PASSED
src/pycwt_mod/tests/backends/test_joblib.py .......... PASSED
src/pycwt_mod/tests/test_wct_significance_integration.py .... PASSED

======================== X passed in X.XXs =========================
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError: No module named 'pycwt_mod'

**Solution:**
```bash
# Make sure you installed in development mode
cd /home/chris/pycwt-mod
pip install -e .

# Or check if it's in your Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Issue: ImportError: No module named 'scipy' (or numpy, matplotlib)

**Solution:**
```bash
# Install dependencies
pip install numpy scipy matplotlib tqdm

# Or install with the package
pip install -e .
```

### Issue: Joblib backend not available

**Solution:**
```bash
# Install joblib for parallel execution
pip install joblib

# Verify installation
python3 -c "import joblib; print('✓ joblib installed:', joblib.__version__)"
```

### Issue: Tests fail with "basestring is not defined"

**Solution:**
This was a Python 2/3 compatibility issue that should be fixed. If you see this:
```bash
# Make sure you pulled the latest changes
git status
git log --oneline -5
```

### Issue: Permission denied on test runner

**Solution:**
```bash
chmod +x run_phase2_tests.py
# Or just run with python3
python3 run_phase2_tests.py
```

### Issue: Cache directory errors

**Solution:**
```bash
# Tests run with cache=False, but if you see cache errors:
mkdir -p ~/.pycwt/cache
```

---

## 📈 Performance Expectations

Based on Phase 1 benchmarks and Phase 2 validation:

### Small Problems (mc_count < 100)
- Sequential: ~1-5 seconds
- Parallel: Similar or slightly slower (overhead dominates)
- **Recommendation:** Use sequential backend

### Medium Problems (mc_count = 100-300)
- Sequential: ~10-30 seconds
- Parallel (2 cores): ~5-15 seconds
- **Speedup:** 1.5-2.5×
- **Recommendation:** Use joblib backend

### Large Problems (mc_count > 300)
- Sequential: ~1-5 minutes
- Parallel (4 cores): ~15-75 seconds
- **Speedup:** 2-4×
- **Recommendation:** Use joblib backend with n_jobs=4 or more

### Very Large Problems (mc_count > 1000)
- Sequential: ~10-30 minutes
- Parallel (8 cores): ~2-5 minutes
- **Speedup:** 4-8×
- **Recommendation:** Use joblib backend with all available cores

---

## ✅ Success Checklist

After testing on your laptop, verify:

- [ ] Package installs without errors
- [ ] All imports work
- [ ] Backend system lists sequential and joblib
- [ ] Sequential backend is available
- [ ] Joblib backend is available (if joblib installed)
- [ ] `test_phase2_integration.py` passes all tests
- [ ] `run_phase2_tests.py` shows all tests passing
- [ ] `wct_significance()` works with no backend parameter
- [ ] `wct_significance()` works with `backend='sequential'`
- [ ] `wct_significance()` works with `backend='joblib'` (if available)
- [ ] Parallel execution is faster than sequential for mc_count > 100
- [ ] Results are similar between sequential and parallel backends
- [ ] Old code (without backend parameter) still works

---

## 📝 What to Report Back

After testing, please share:

1. **Installation Status**
   - Did `pip install -e .` work?
   - Any dependency issues?

2. **Backend Availability**
   - Is joblib backend available?
   - Number of CPU cores detected?

3. **Test Results**
   - Did `test_phase2_integration.py` pass?
   - Did `run_phase2_tests.py` pass?
   - Any failed tests?

4. **Performance**
   - Speedup observed (if you ran performance tests)?
   - Time for sequential vs parallel execution?

5. **Issues Encountered**
   - Any errors or warnings?
   - Any unexpected behavior?

---

## 🎯 Next Steps After Testing

Once testing is complete on your laptop:

1. **If all tests pass:** Ready for Phase 3 (Documentation)
2. **If some tests fail:** Debug and fix issues
3. **If performance is good:** Consider benchmarking with real datasets
4. **If ready to commit:** We can commit Phase 1 + Phase 2 changes

---

**Document Version:** 1.0  
**Last Updated:** October 2, 2025  
**Status:** Ready for laptop testing  
**Contact:** Report issues back for debugging

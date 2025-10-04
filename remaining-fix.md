# Remaining Fixes to Reach 100% Test Pass Rate

## Current Status: 71/104 Tests Passing (68.3%)

### ✅ Completed Endpoints (71 tests passing):
- **Health Endpoints**: 12/12 (100%)
- **Backend Endpoints**: 16/16 (100%)
- **Hardware Detection**: 20/20 (100%) ✨ NEW!
- **CWT (Continuous Wavelet Transform)**: 9/9 (100%)
- **Integration Tests**: 9/13 (69%) - partial, depends on other endpoints

### ❌ Remaining Issues (33 tests failing):

---

## Issue 1: Benchmark Endpoint Not Implemented (17 tests)
**Impact**: 0/17 tests passing  
**Priority**: HIGH (easiest win, +17 tests)  
**Estimated Effort**: 2-3 hours

### What's Needed:
Create `/api/v1/benchmark` POST endpoint that:
1. Accepts a benchmark request with:
   - `signal_length`: Length of test signal to generate
   - `backends`: List of backend names to benchmark
   - `mc_count`: Monte Carlo iterations (optional)
   - `wavelet`: Wavelet type (optional, default: 'morlet')

2. Returns benchmark results with:
   - Per-backend timing results
   - Mean, median, std dev for each backend
   - Comparison metrics (speedup ratios)
   - System information

### Implementation Steps:
1. Create `server/api/models/benchmark.py` with Pydantic models:
   - `BenchmarkRequest`
   - `BenchmarkResult` (per backend)
   - `BenchmarkResponse` (overall results)

2. Create `server/api/routes/benchmark.py` with:
   - POST `/benchmark` endpoint
   - Generate test signal with numpy
   - Run CWT on each requested backend
   - Measure execution time (use `time.perf_counter()`)
   - Collect results and statistics
   - Handle errors gracefully (skip unavailable backends)

3. Register router in `server/main.py`:
   ```python
   from server.api.routes import benchmark
   app.include_router(benchmark.router, prefix="/api/v1", tags=["benchmark"])
   ```

4. Test requirements (from `client-tests/test_benchmark.py`):
   - Must complete within reasonable time (<30s for basic tests)
   - Handle invalid inputs (signal_length <= 0, empty backends list)
   - Return structured results with timing data
   - Support comparison between backends
   - Validate against research questions (MVP requirements)

**Expected Outcome**: +17 tests → 88/104 (85%)

---

## Issue 2: WCT (Wavelet Coherence) Timeout Issues (11 tests)
**Impact**: 0/11 tests passing  
**Priority**: MEDIUM (functional but slow)  
**Estimated Effort**: 1-2 hours

### Problem:
- WCT endpoint exists and works
- Monte Carlo significance testing exceeds 30s client timeout
- Some tests get 500 errors (likely timeout-related)

### Root Cause:
```python
# In server/api/routes/wavelet.py
significance = wct(..., sig=True, mc_count=request.mc_count or 300)
```
Default `mc_count=300` is too high for API context (takes >30s).

### Solution Options:

**Option A: Reduce Default mc_count** (Quick fix)
```python
# Change default from 300 to 30
mc_count = request.mc_count or 30  # Much faster, still statistically valid
```

**Option B: Make Significance Optional** (Better)
```python
# Add sig parameter to request model
class WCTRequest(BaseModel):
    ...
    sig: bool = Field(False, description="Calculate significance (slower)")
    mc_count: Optional[int] = Field(30, description="Monte Carlo iterations")

# In endpoint:
if request.sig:
    significance = wct(..., sig=True, mc_count=request.mc_count)
else:
    # Skip significance calculation
    wct_result = wct(..., sig=False)
```

**Option C: Async/Background Processing** (Most robust)
- Return immediately with task ID
- Client polls for results
- Requires more infrastructure

**Recommendation**: Start with Option A (change default to 30), then implement Option B.

**Expected Outcome**: +11 tests → 82/104 (79%) or combined with benchmark → 99/104 (95%)

---

## Issue 3: XWT (Cross Wavelet Transform) 500 Errors (6 tests)
**Impact**: 0/6 tests passing  
**Priority**: MEDIUM  
**Estimated Effort**: 1-2 hours

### Problem:
- XWT endpoint returns 500 Internal Server Error
- Endpoint exists but has implementation issues

### Likely Causes:
1. **Return Value Format**: XWT returns complex arrays that may not serialize properly
2. **Missing Parameters**: Function signature mismatch
3. **Error in Response Model**: Pydantic validation failing

### Debugging Steps:
1. Check server logs for actual error message
2. Test XWT function directly:
   ```python
   from pycwt_mod import xwt
   import numpy as np
   
   signal1 = np.random.randn(100)
   signal2 = np.random.randn(100)
   t = np.linspace(0, 1, 100)
   
   result = xwt(signal1, signal2, dt=0.01, dj=0.25, s0=2, J=7, wavelet='morlet')
   print(type(result), len(result))
   ```

3. Check return value structure:
   ```python
   # XWT returns: (Wxy, coi, freqs, signif)
   # Wxy is complex - needs [real, imag] conversion
   ```

4. Fix in `server/api/routes/wavelet.py`:
   ```python
   @router.post("/xwt", response_model=XWTResponse)
   async def cross_wavelet_transform(request: XWTRequest):
       try:
           # ... existing code ...
           
           # Convert complex Wxy to [real, imag] lists
           xwt_real = np.real(xwt_result).tolist()
           xwt_imag = np.imag(xwt_result).tolist()
           
           return XWTResponse(
               xwt_real=xwt_real,
               xwt_imag=xwt_imag,
               coi=coi.tolist(),
               freqs=freqs.tolist(),
               signif=signif.tolist() if signif is not None else None
           )
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   ```

**Expected Outcome**: +6 tests → 77/104 (74%) or combined → 94/104 (90%)

---

## Issue 4: Integration Test Failures (4 tests)
**Impact**: 4/13 failing  
**Priority**: LOW (depend on other fixes)  
**Estimated Effort**: Minimal (will pass after above fixes)

### Failing Tests:
1. `test_full_wct_workflow` - Depends on WCT fix
2. `test_benchmark_then_analyze_workflow` - Depends on benchmark endpoint
3. `test_fpga_unavailable_fallback` - Backend error handling
4. `test_invalid_backend_error` - Backend error handling
5. `test_wct_coherence_values_valid` - Depends on WCT fix

### Solution:
These should mostly pass automatically after fixing WCT, XWT, and implementing benchmark.
May need minor error handling improvements.

---

## Recommended Implementation Order:

### Phase 1: Quick Wins (Target: 85% - 88/104 tests)
**Time: 2-3 hours**
1. ✅ Implement benchmark endpoint (+17 tests)
2. Test and validate

### Phase 2: WCT Optimization (Target: 95% - 99/104 tests)
**Time: 1-2 hours**
1. Fix WCT timeout by reducing default mc_count to 30
2. Make significance calculation optional
3. Test all WCT tests (+11 tests)

### Phase 3: XWT Debugging (Target: 98% - 102/104 tests)
**Time: 1-2 hours**
1. Debug 500 errors (check server logs)
2. Fix complex number serialization
3. Test all XWT tests (+6 tests)

### Phase 4: Integration Cleanup (Target: 100% - 104/104 tests)
**Time: 30 minutes**
1. Verify integration tests pass
2. Fix any remaining edge cases
3. Celebrate! 🎉

---

## Total Estimated Time: 4-8 hours
## Expected Final Pass Rate: 100% (104/104 tests)

---

## Testing Strategy:

After each phase:
```bash
# Test specific endpoint
pytest client-tests/test_benchmark.py -v
pytest client-tests/test_wavelet.py::TestWaveletCoherence -v
pytest client-tests/test_wavelet.py::TestCrossWaveletTransform -v

# Full test suite
pytest client-tests/ -v --tb=short

# Quick summary
pytest client-tests/ -v --tb=no -q
```

---

## Notes:
- All fixes should be made on the server (wavelet.local:8000)
- Tests run from laptop client
- Remember to git push and restart server after changes
- Server logs available at: (check systemd journal or uvicorn logs)

---

## Success Metrics:
- ✅ 71/104 tests passing (68.3%) - CURRENT
- 🎯 88/104 tests passing (85%) - After Phase 1
- 🎯 99/104 tests passing (95%) - After Phase 2
- 🎯 102/104 tests passing (98%) - After Phase 3
- 🎯 104/104 tests passing (100%) - After Phase 4 🏆

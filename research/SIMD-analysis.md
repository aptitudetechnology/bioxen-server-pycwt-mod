# SIMD Optimization Analysis for pycwt Wavelet Coherence

**Date:** October 2, 2025  
**Context:** Performance optimization opportunities beyond parallelization and GPU

---

## What is SIMD?

**SIMD (Single Instruction, Multiple Data)** are CPU instructions that perform the same operation on multiple data points simultaneously. Modern CPUs support:

- **SSE (Streaming SIMD Extensions)**: 128-bit (4× float32 or 2× float64)
- **AVX (Advanced Vector Extensions)**: 256-bit (8× float32 or 4× float64)
- **AVX-512**: 512-bit (16× float32 or 8× float64)
- **ARM NEON**: 128-bit (ARM processors)

---

## SIMD Opportunities in pycwt

### 1. **FFT Operations (Already Optimized)**

**Current State:**
```python
# helpers.py - FFT backend selection
import pyfftw.interfaces.scipy_fftpack as fft  # FFTW
# or
import scipy.fftpack as fft  # FFTPACK/MKL
```

**SIMD Status:** ✅ **Already Optimized**
- **FFTW**: Heavily optimized with SIMD (SSE2, AVX, AVX-512)
- **MKL**: Intel's Math Kernel Library uses SIMD extensively
- **Benefit**: FFT operations already achieve near-peak FLOPS

**No additional work needed** - FFTW and MKL are state-of-the-art.

---

### 2. **Element-Wise Array Operations (NumPy Optimized)**

**Current Operations in wavelet.py:**

```python
# Power spectrum calculation
power = np.abs(W) ** 2

# Cross-spectrum
W12 = W1 * W2.conj()

# Coherence calculation
WCT = np.abs(S12) ** 2 / (S1 * S2)

# Scale normalization
W_normalized = W / scales
```

**SIMD Status:** ✅ **Already Optimized via NumPy**
- NumPy uses vectorized operations
- Automatically leverages BLAS/LAPACK libraries
- Modern NumPy builds use:
  - OpenBLAS (SIMD-optimized)
  - Intel MKL (AVX-512 optimized)
  - Apple Accelerate (ARM NEON)

**Minimal gains** - NumPy already vectorized.

---

### 3. **Smoothing Operations (Potential Target)**

**Current Implementation (mothers.py, lines 72-95):**

```python
def smooth(self, W, dt, dj, scales):
    # Time smoothing via FFT (already SIMD-optimized)
    F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)
    smooth = fft.ifft(F * fft.fft(W, axis=1, ...), axis=1, ...)
    
    # Scale smoothing via 2D convolution
    wsize = self.deltaj0 / dj * 2
    win = rect(int(np.round(wsize)), normalize=True)
    T = convolve2d(T, win[:, np.newaxis], 'same')  # ← Potential target
    
    return T
```

**SIMD Opportunity:** 🟡 **Marginal Gains**
- `convolve2d` from SciPy already uses optimized routines
- Could be replaced with hand-tuned SIMD convolution
- **Expected speedup:** 1.2-1.5× (not worth the complexity)

---

### 4. **Monte Carlo Surrogate Generation (Best Target)**

**Current Implementation (helpers.py, lines 148-180):**

```python
def rednoise(N, g, a=1.):
    """Red noise generator using filter."""
    if g == 0:
        yr = np.randn(N, 1) * a
    else:
        tau = int(np.ceil(-2 / np.log(np.abs(g))))
        yr = lfilter([1, 0], [1, -g], np.random.randn(N + tau, 1) * a)
        yr = yr[tau:]
    return yr.flatten()
```

**SIMD Opportunity:** 🟢 **Good Target**
- `lfilter` (SciPy) applies IIR filter sequentially
- AR(1) process: `x[t] = g * x[t-1] + noise[t]`
- **Issue**: Sequential dependency makes SIMD difficult

**However:** Could vectorize across multiple Monte Carlo iterations:
```python
# Generate 300 surrogates simultaneously (vectorized)
def rednoise_batch(N, g, a, batch_size=300):
    """Generate multiple red noise series in parallel."""
    # Generate all random numbers at once
    noise = np.random.randn(batch_size, N + tau) * a
    
    # Apply filter to all series (vectorized along batch dimension)
    yr_batch = lfilter_vectorized([1, 0], [1, -g], noise, axis=1)
    
    return yr_batch[:, tau:]
```

**Expected speedup:** 2-4× for surrogate generation (small part of total time)

---

### 5. **Wavelet Transform Convolution (Limited Opportunity)**

**Current Implementation (wavelet.py, lines 98-107):**

```python
# Outer product for all scales
sj_col = sj[:, numpy.newaxis]
psi_ft_bar = (sj_col * ftfreqs[1] * N) ** 0.5 * numpy.conjugate(
    wavelet.psi_ft(sj_col * ftfreqs)
)
W = fft.ifft(signal_ft * psi_ft_bar, axis=1, ...)
```

**SIMD Status:** ✅ **Already Optimized**
- Outer product uses NumPy broadcasting (vectorized)
- Complex multiplication is SIMD-optimized in modern NumPy
- FFT is SIMD-optimized via FFTW/MKL

---

## SIMD Implementation Approaches

### Approach 1: **Trust NumPy/SciPy (Current - Recommended)**

**Pros:**
- ✅ Already highly optimized
- ✅ Portable across architectures (x86, ARM, PowerPC)
- ✅ Automatically uses best SIMD available
- ✅ No maintenance burden

**Cons:**
- ❌ Can't squeeze out last 10-20% performance

**Recommendation:** ✅ **Stick with this for MVP**

---

### Approach 2: **Use Numba with SIMD Decorators**

```python
from numba import jit, vectorize
import numpy as np

@jit(nopython=True, fastmath=True, parallel=True)
def fast_power_spectrum(W_real, W_imag):
    """SIMD-optimized power spectrum calculation."""
    return W_real * W_real + W_imag * W_imag

@vectorize(['float64(complex128)'], target='parallel')
def fast_abs_squared(c):
    """Vectorized absolute value squared."""
    return c.real * c.real + c.imag * c.imag
```

**Pros:**
- ✅ JIT compilation with SIMD
- ✅ Minimal code changes
- ✅ Automatic loop vectorization

**Cons:**
- ❌ Numba doesn't always generate optimal SIMD
- ❌ Limited benefit for FFT-dominated code
- ❌ Additional dependency

**Expected speedup:** 1.1-1.3× for non-FFT portions

---

### Approach 3: **Cython with SIMD Intrinsics**

```cython
# cython: boundscheck=False, wraparound=False
from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_coherence(double[:, ::1] S12_real, double[:, ::1] S12_imag,
                   double[:, ::1] S1, double[:, ::1] S2):
    """SIMD-optimized coherence calculation."""
    cdef int i, j
    cdef int n_scales = S12_real.shape[0]
    cdef int n_times = S12_real.shape[1]
    cdef double[:, ::1] WCT = np.empty((n_scales, n_times))
    
    for i in range(n_scales):
        for j in range(n_times):
            WCT[i, j] = (S12_real[i, j]**2 + S12_imag[i, j]**2) / (S1[i, j] * S2[i, j])
    
    return np.asarray(WCT)
```

**Pros:**
- ✅ Fine-grained control
- ✅ C-level performance
- ✅ Can use explicit SIMD intrinsics

**Cons:**
- ❌ High development effort
- ❌ Platform-specific optimization needed
- ❌ NumPy already does this internally

**Expected speedup:** 1.2-1.5× for non-FFT portions

---

### Approach 4: **Intel oneAPI/DPCPP (Aggressive Optimization)**

```cpp
// Using Intel SYCL/DPC++ for explicit SIMD
#include <sycl/sycl.hpp>

void wavelet_coherence_simd(
    sycl::queue& q,
    const std::complex<double>* S12,
    const double* S1,
    const double* S2,
    double* WCT,
    size_t n_scales,
    size_t n_times
) {
    q.parallel_for(sycl::range<2>(n_scales, n_times), [=](sycl::id<2> idx) {
        size_t i = idx[0];
        size_t j = idx[1];
        size_t offset = i * n_times + j;
        
        double abs_sq = S12[offset].real() * S12[offset].real() +
                        S12[offset].imag() * S12[offset].imag();
        WCT[offset] = abs_sq / (S1[offset] * S2[offset]);
    }).wait();
}
```

**Pros:**
- ✅ Maximum performance potential
- ✅ Unified CPU/GPU programming model
- ✅ Explicit SIMD + GPU offload

**Cons:**
- ❌ Major rewrite required
- ❌ Intel-specific (limited portability)
- ❌ High maintenance burden
- ❌ Overkill for current bottlenecks

---

## Benchmark: SIMD Impact Analysis

### Hypothetical SIMD Speedups (Optimistic Estimates)

| Operation | % of Total Time | Current Impl | SIMD Speedup | Net Gain |
|-----------|----------------|--------------|--------------|----------|
| FFT (forward/inverse) | 60% | FFTW+SIMD | 1.0× | 0% |
| Element-wise ops | 15% | NumPy+BLAS | 1.2× | 3% |
| Smoothing (convolve2d) | 10% | SciPy | 1.3× | 3% |
| Array allocations | 10% | NumPy | 1.0× | 0% |
| Control flow | 5% | Python | 1.0× | 0% |
| **Total** | **100%** | | | **~6%** |

**Amdahl's Law Analysis:**
- Parallelizable: 85% (non-FFT operations)
- SIMD speedup on 85%: 1.2×
- Overall speedup: 1 / (0.15 + 0.85/1.2) = **1.06×**

**Conclusion:** ⚠️ **SIMD offers <10% speedup - not worth the effort**

---

## Better Alternatives to SIMD

### 1. **Parallel Monte Carlo** (300× the data to process)

```python
from multiprocessing import Pool

def compute_mc_iteration(args):
    noise1, noise2, kwargs = args
    nW1, nW2 = cwt(noise1, **kwargs), cwt(noise2, **kwargs)
    # ... compute coherence ...
    return R2

# Parallelize across iterations
with Pool(cpu_count()) as pool:
    results = pool.map(compute_mc_iteration, iteration_args)
```

**Expected speedup:** 4-8× (on 8-core CPU)  
**Development effort:** 1 day  
**ROI:** ✅ Excellent

---

### 2. **GPU Acceleration** (Thousands of parallel threads)

```python
import cupy as cp

def wct_gpu(y1, y2, dt, dj, s0, J):
    # Transfer to GPU
    y1_gpu = cp.asarray(y1)
    y2_gpu = cp.asarray(y2)
    
    # CWT on GPU (using CuPy's FFT)
    W1_gpu = cwt_cupy(y1_gpu, dt, dj, s0, J)
    W2_gpu = cwt_cupy(y2_gpu, dt, dj, s0, J)
    
    # Coherence calculation (element-wise ops on GPU)
    WCT_gpu = compute_coherence_gpu(W1_gpu, W2_gpu)
    
    # Transfer back to CPU
    return cp.asnumpy(WCT_gpu)
```

**Expected speedup:** 10-50× (on modern GPU)  
**Development effort:** 1-2 weeks  
**ROI:** ✅ Excellent for large datasets

---

### 3. **Reduced Monte Carlo Iterations** (300 → 100)

```python
# Current: 300 iterations (slow)
sig = wct_significance(..., mc_count=300)

# Adaptive: Stop early if clear significance
sig = wct_significance_adaptive(..., mc_count_max=300, confidence_threshold=0.95)
```

**Expected speedup:** 3× for significance testing  
**Development effort:** 3 days  
**ROI:** ✅ Good, statistically valid

---

## Recommendations

### For MVP Phase (Current Priority):

**DO:**
1. ✅ **Ensure FFTW is installed** - Already has SIMD
2. ✅ **Use optimized NumPy build** (OpenBLAS or MKL)
3. ✅ **Parallelize Monte Carlo** - 4-8× speedup
4. ✅ **Optimize algorithm** before optimizing code

**DON'T:**
1. ❌ Hand-code SIMD intrinsics - diminishing returns
2. ❌ Use Cython for SIMD - NumPy already optimized
3. ❌ Invest in AVX-512 tuning - marginal gains

### For Phase 2 (After MVP):

**DO:**
4. ✅ **GPU acceleration** - Real game-changer
5. ✅ **Hybrid PyWavelets/pycwt** - Architectural improvement
6. ✅ **Batch processing API** - Multiple signals at once

### For Phase 3 (If Needed):

**DO:**
7. ✅ **FPGA** - For real-time, low-latency applications
8. ✅ **Distributed computing** - For massive datasets
9. 🟡 **Numba JIT** - Easy wins for custom functions

---

## SIMD Status in Current Libraries

### NumPy SIMD Support

Check your NumPy build:
```python
import numpy as np
print(np.__config__.show())
```

Look for:
- `lapack_opt_info` → OpenBLAS or MKL
- `blas_opt_info` → SIMD-optimized BLAS

**Ensure you have:**
```bash
# For Intel CPUs
pip install numpy[mkl]

# For general use (AMD, ARM)
pip install numpy  # Uses OpenBLAS
```

### FFTW SIMD Support

```python
import pyfftw
print(pyfftw.simd_alignment)  # Should be 16, 32, or 64
print(pyfftw.export_wisdom_string())  # Check for SIMD wisdom
```

**Ensure you have:**
```bash
pip install pyfftw
```

---

## Conclusion

### SIMD Verdict: 🟡 **Not a Priority**

**Why?**
1. **Already optimized**: FFTW and NumPy use SIMD extensively
2. **Marginal gains**: <10% speedup potential
3. **High complexity**: Manual SIMD coding is error-prone
4. **Better alternatives**: Parallelization (4-8×), GPU (10-50×), algorithmic (2-3×)

### Priority Order for Optimization:

| Optimization | Expected Speedup | Development Effort | ROI | Priority |
|--------------|------------------|-------------------|-----|----------|
| **Parallel Monte Carlo** | 4-8× | 1-2 days | ⭐⭐⭐⭐⭐ | 🔴 High |
| **Reduce MC iterations** | 3× | 3 days | ⭐⭐⭐⭐ | 🔴 High |
| **Ensure FFTW installed** | 2× | 0 days | ⭐⭐⭐⭐⭐ | 🔴 High |
| **Hybrid PyWavelets** | 2-5× | 2-3 weeks | ⭐⭐⭐⭐ | 🟡 Medium |
| **GPU acceleration** | 10-50× | 1-2 weeks | ⭐⭐⭐⭐⭐ | 🟡 Medium |
| **Batch API** | 2-4× | 1 week | ⭐⭐⭐ | 🟡 Medium |
| **Manual SIMD** | 1.06× | 4+ weeks | ⭐ | 🟢 Low |
| **FPGA** | 5-100× | 8-12 weeks | ⭐⭐ | 🟢 Low |

### The Bottom Line:

**Trust the libraries.** NumPy, SciPy, and FFTW are maintained by world-class numerical computing experts who have already squeezed every last drop of SIMD performance. Your time is better spent on:

1. **Parallelizing embarrassingly parallel workloads** (Monte Carlo)
2. **Using GPU for massive parallelism** (thousands of threads)
3. **Algorithmic improvements** (hybrid architecture, reduced iterations)

**SIMD is a solved problem** in modern numerical Python. Focus on the 10× gains, not the 1.06× gains.

---

**Last Updated:** October 2, 2025  
**Verdict:** SIMD optimization is not recommended for pycwt - the infrastructure already handles it optimally.

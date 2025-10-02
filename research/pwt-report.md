# PyCWT Library Code Analysis Report

**Date:** October 2, 2025  
**Analyst:** GitHub Copilot  
**Repository:** aptitudetechnology/pycwt  
**Purpose:** Technical analysis to inform FFT performance bottleneck research

---

## Executive Summary

This report provides a detailed code analysis of the `pycwt` library to validate and contextualize the performance concerns raised in the FFT bottleneck research prompts. The analysis confirms that:

1. ✅ **pycwt is indeed FFT-based** - All continuous wavelet transforms use FFT via the convolution theorem
2. ✅ **Multiple FFT operations per WCT** - Each wavelet coherence computation involves 6-8 FFT calls
3. ✅ **Monte Carlo testing is the primary bottleneck** - Significance testing performs 600+ CWT operations (1,200+ FFT calls)
4. ✅ **No API for pre-computed coefficients** - Hybrid architecture would require custom implementation
5. ✅ **Parallelization opportunities exist** - Monte Carlo simulations are embarrassingly parallel

---

## 1. Library Architecture Overview

### File Structure

```
pycwt/
├── __init__.py          # Package initialization (100 lines)
├── wavelet.py           # Core transform functions (664 lines)
├── mothers.py           # Wavelet classes (234 lines)
├── helpers.py           # Utility functions (237 lines)
└── sample/              # Example datasets and scripts
```

### Core Functions Exported

From `__init__.py`:
- `cwt()` - Continuous Wavelet Transform
- `icwt()` - Inverse CWT
- `xwt()` - Cross-Wavelet Transform
- `wct()` - Wavelet Coherence Transform ⭐ (main focus)
- `wct_significance()` - Monte Carlo significance testing
- `significance()` - Statistical significance for CWT

### Mother Wavelet Classes

- **Morlet** (default, f0=6) - Most common for biological signals
- **Paul** (m=4)
- **DOG** (Derivative of Gaussian, m=2)
- **MexicanHat** (DOG with m=2)

---

## 2. FFT-Based Implementation Details

### 2.1 CWT Algorithm (wavelet.py, lines 12-126)

**Function signature:**
```python
def cwt(signal, dt, dj=1/12, s0=-1, J=-1, wavelet='morlet', freqs=None):
```

**FFT-based convolution implementation** (lines 94-107):

```python
# Signal Fourier transform
signal_ft = fft.fft(signal, **fft_kwargs(signal))
N = len(signal_ft)

# Fourier angular frequencies
ftfreqs = 2 * np.pi * fft.fftfreq(N, dt)

# Creates wavelet transform matrix as outer product of scaled transformed
# wavelets and transformed signal according to the convolution theorem.
sj_col = sj[:, np.newaxis]
psi_ft_bar = ((sj_col * ftfreqs[1] * N) ** .5 *
              np.conjugate(wavelet.psi_ft(sj_col * ftfreqs)))
W = fft.ifft(signal_ft * psi_ft_bar, axis=1,
             **fft_kwargs(signal_ft, overwrite_x=True))
```

**Key observations:**
- Uses **convolution theorem**: convolution in time domain = multiplication in frequency domain
- **One forward FFT** per signal
- **One inverse FFT** per scale (but vectorized across all scales simultaneously)
- Complexity: O(N log N) for FFT + O(S × N) for scale operations, where S = number of scales

**Mathematical approach:**
Follows Torrence & Compo (1998) methodology:
1. Transform signal to frequency domain: `F[signal]`
2. Multiply by conjugate of scaled wavelet: `F[signal] × conj(F[wavelet(scale)])`
3. Inverse transform for each scale: `F^-1[product]`

### 2.2 FFT Backend Selection (helpers.py, lines 6-27)

The library supports **two FFT backends** with automatic fallback:

**Primary backend: FFTW (via pyfftw)**
```python
import pyfftw.interfaces.scipy_fftpack as fft
from multiprocessing import cpu_count

_FFTW_KWARGS_DEFAULT = {
    'planner_effort': 'FFTW_ESTIMATE',
    'threads': cpu_count()
}
```

**Fallback backend: scipy.fftpack**
```python
import scipy.fftpack as fft

_FFT_NEXT_POW2 = True  # Pads to next power of 2 for speed
```

**Performance implications:**
- **With FFTW**: Multi-threaded FFT, faster for large signals
- **Without FFTW**: Single-threaded, power-of-2 padding adds memory overhead
- **User transparency**: No indication in API which backend is used

**Recommendation for benchmarking:**
Always check which backend is active during performance tests:
```python
import pycwt
print(pycwt.wavelet.fft.__name__)  # Check backend
```

---

## 3. Wavelet Coherence Transform (WCT) Implementation

### 3.1 WCT Algorithm (wavelet.py, lines 421-540)

**Function signature:**
```python
def wct(y1, y2, dt, dj=1/12, s0=-1, J=-1, sig=True,
        significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
```

**Complete workflow:**

```python
# Step 1: Normalize signals
y1_normal = (y1 - y1.mean()) / y1.std()
y2_normal = (y2 - y2.mean()) / y2.std()

# Step 2: Compute CWT for both signals (2 × FFT operations)
W1, sj, freq, coi, _, _ = cwt(y1_normal, dt, **_kwargs)
W2, sj, freq, coi, _, _ = cwt(y2_normal, dt, **_kwargs)

# Step 3: Smooth individual power spectra (2 × smoothing operations)
S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

# Step 4: Cross-spectrum and smoothing (1 × smoothing operation)
W12 = W1 * W2.conj()
S12 = wavelet.smooth(W12 / scales, dt, dj, sj)

# Step 5: Coherence calculation
WCT = np.abs(S12) ** 2 / (S1 * S2)
aWCT = np.angle(W12)  # Phase angle

# Step 6: Significance testing (optional, see section 4)
if sig:
    sig = wct_significance(a1, a2, dt=dt, dj=dj, s0=s0, J=J, ...)
```

**FFT operation count per WCT:**
- 2 × CWT computations (2 FFT forward + 2 FFT inverse)
- 3 × smoothing operations (each with FFT forward + FFT inverse)
- **Total: 8 FFT operations per signal pair**

### 3.2 Smoothing Algorithm (mothers.py, lines 64-95)

**Critical component for coherence calculation:**

```python
def smooth(self, W, dt, dj, scales):
    """Smoothing function used in coherence analysis."""
    
    # Filter in time using FFT
    k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])['n'])
    k2 = k ** 2
    snorm = scales / dt
    
    # Gaussian window in Fourier domain (absolute value of wavelet)
    F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)
    smooth = fft.ifft(
        F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
        axis=1,
        **fft_kwargs(W[0, :], overwrite_x=True)
    )
    T = smooth[:, :n]
    
    # Filter in scale using boxcar (for Morlet wavelet)
    wsize = self.deltaj0 / dj * 2  # 0.6 width for Morlet
    win = rect(int(np.round(wsize)), normalize=True)
    T = convolve2d(T, win[:, np.newaxis], 'same')
    
    return T
```

**Smoothing methodology:**
- **Time smoothing**: Gaussian convolution via FFT (multiplication in frequency domain)
- **Scale smoothing**: Boxcar filter via 2D convolution
- **Mathematical basis**: Torrence & Webster (1999), Grinsted et al. (2004)

**Performance note:**
Each `smooth()` call involves:
- 1 × FFT forward transform
- 1 × FFT inverse transform
- 1 × 2D convolution (via `scipy.signal.convolve2d`)

---

## 4. Monte Carlo Significance Testing (The Primary Bottleneck)

### 4.1 Algorithm (wavelet.py, lines 543-650)

**Function signature:**
```python
def wct_significance(al1, al2, dt, dj, s0, J, 
                     significance_level=0.95,
                     wavelet='morlet', 
                     mc_count=300,      # ⚠️ Key parameter
                     progress=True,
                     cache=True):
```

**Complete Monte Carlo workflow:**

```python
# Setup
N = int(np.ceil(ms * 6))  # Signal length for simulations
nbins = 1000
wlc = np.ma.zeros([J + 1, nbins])  # Coherence histogram

# Monte Carlo iterations (DEFAULT: 300)
for _ in tqdm(range(mc_count), disable=not progress):
    
    # 1. Generate red-noise surrogates with AR(1) coefficients
    noise1 = rednoise(N, al1, 1)
    noise2 = rednoise(N, al2, 1)
    
    # 2. Compute CWT for BOTH surrogate signals
    kwargs = dict(dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    nW1, sj, freq, coi, _, _ = cwt(noise1, **kwargs)
    nW2, sj, freq, coi, _, _ = cwt(noise2, **kwargs)
    
    # 3. Compute cross-spectrum
    nW12 = nW1 * nW2.conj()
    
    # 4. Smooth spectra (3 × smoothing operations)
    S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, dt, dj, sj)
    S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)
    
    # 5. Compute coherence
    R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
    
    # 6. Build histogram of coherence values
    for s in range(maxscale):
        cd = np.floor(R2[s, :] * nbins)
        for j, t in enumerate(cd[~cd.mask]):
            wlc[s, int(t)] += 1

# Determine significance threshold from histogram percentiles
for s in range(maxscale):
    P = wlc[s, sel].data.cumsum()
    P = (P - 0.5) / P[-1]
    sig95[s] = np.interp(significance_level, P, R2y[sel])
```

**Computational cost analysis:**

| Operation | Count per Iteration | Total (300 iterations) |
|-----------|---------------------|------------------------|
| Red noise generation | 2 | 600 |
| CWT computations | 2 | **600** |
| FFT operations | 8 (from CWT + smoothing) | **2,400** |
| Smoothing operations | 3 | 900 |
| Coherence calculations | 1 | 300 |

**Critical bottleneck identified:**
- Each Monte Carlo iteration performs **full WCT workflow** on surrogate data
- 300 iterations × 8 FFT operations = **2,400 FFT calls**
- This is the **dominant computational cost** for significance testing
- **Embarrassingly parallel** - each iteration is independent

### 4.2 Caching Mechanism (lines 569-578)

**Intelligent caching implemented:**

```python
if cache:
    aa = np.round(np.arctanh(np.array([al1, al2]) * 4))
    aa = np.abs(aa) + 0.5 * (aa < 0)
    cache_file = 'wct_sig_{:0.5f}_{:0.5f}_{:0.5f}_{:0.5f}_{:d}_{}' \
        .format(aa[0], aa[1], dj, s0 / dt, J, wavelet.name)
    cache_dir = get_cache_dir()  # ~/.cache/pycwt/
    
    try:
        dat = np.loadtxt('{}/{}.gz'.format(cache_dir, cache_file))
        print('NOTE: WCT significance loaded from cache.\n')
        return dat
    except IOError:
        pass  # Compute if not cached
```

**Cache key components:**
- AR(1) coefficients (al1, al2) - transformed via arctanh
- Scale parameters (dj, s0/dt, J)
- Wavelet type

**Implications:**
- Same signals with same AR(1) characteristics = cache hit
- Different signals with similar autocorrelation = cache hit
- **Significant performance improvement** for repeated analyses
- Cache persists across Python sessions

---

## 5. Performance Bottleneck Analysis

### 5.1 Computational Complexity Breakdown

**Single CWT operation:**
- FFT: O(N log N) where N = signal length
- Scale processing: O(S × N) where S = number of scales
- Typical S ≈ 50-100 for biological signals

**Single WCT operation (without significance):**
- 2 × CWT: 2 × O(N log N)
- 3 × smoothing: 3 × O(N log N) for FFT portion
- Cross-spectrum: O(S × N)
- **Total: O(N log N)** (FFT dominates)

**WCT with Monte Carlo significance (mc_count=300):**
- 300 iterations × [2 CWT + 3 smoothing]
- **Total: 300 × O(N log N)**
- This is **300× slower** than single WCT

### 5.2 Scaling Analysis

**Expected performance scaling:**

| Signal Length | Scales | CWT Time | WCT Time | WCT + Sig (300 MC) |
|---------------|--------|----------|----------|--------------------|
| 100 | 50 | 0.001s | 0.003s | 0.9s |
| 1,000 | 60 | 0.01s | 0.03s | 9s |
| 10,000 | 70 | 0.1s | 0.3s | 90s (1.5 min) |
| 100,000 | 80 | 1s | 3s | 900s (15 min) |
| 1,000,000 | 90 | 10s | 30s | 9000s (2.5 hours) |
| 10,000,000 | 100 | 100s | 300s | 90,000s (25 hours) |

**Note:** These are theoretical estimates. Actual times depend on:
- FFT backend (FFTW vs scipy)
- CPU specifications
- Memory bandwidth
- Number of scales

**Memory scaling:**
- CWT output: O(S × N) complex128 = 16 bytes × S × N
- For N=1M, S=90: ~1.4 GB per signal
- WCT requires 2 signals + cross-spectrum: ~4 GB
- Monte Carlo needs temporary storage: additional 4 GB

### 5.3 Identified Bottlenecks

**Ranked by severity:**

1. **🔴 Monte Carlo significance testing** (300× slowdown)
   - Sequential iteration (no parallelization)
   - Each iteration performs full CWT + smoothing
   - Dominates total computation time for N > 10,000

2. **🟡 Multiple smoothing operations** (3× overhead per WCT)
   - Each smoothing = 1 FFT forward + 1 FFT inverse
   - Could potentially be optimized or fused

3. **🟡 FFT padding** (memory overhead, minor speed impact)
   - scipy backend pads to next power of 2
   - Wastes memory but speeds up FFT

4. **🟢 Individual CWT computation** (baseline, acceptable)
   - O(N log N) is theoretically optimal for convolution
   - FFT is well-optimized algorithm

---

## 6. Implications for Research Prompts

### 6.1 Validation of Prompt Concerns

**From FFT-based-performance-bottleneck-massive-datasets-prompt.md:**

> "The computational method employed by pycwt relies on the Fast Fourier Transform (FFT) algorithm... it presents a functional trade-off. For the analysis of extremely large or high-sampling-rate biological signals... the FFT-based approach may introduce performance bottlenecks..."

**✅ CONFIRMED:** Code analysis validates this concern:
- All CWT operations use FFT
- Multiple FFT calls per WCT
- Monte Carlo testing compounds the issue

### 6.2 Answers to Research Questions

**Question 1: Quantifying the bottleneck**

**Code evidence:**
- For N=1M timepoints with significance testing:
  - ~2,400 FFT operations (300 iterations × 8 FFTs)
  - Estimated 2.5 hours on typical hardware
  - **Bottleneck threshold: N > 100,000 with significance testing**

**Question 2: Alternative architectures**

**Code evidence:**
```python
# Current architecture in wct():
W1, sj, freq, coi, _, _ = cwt(y1_normal, dt, **_kwargs)
W2, sj, freq, coi, _, _ = cwt(y2_normal, dt, **_kwargs)

# No API to inject pre-computed coefficients!
# Function signature doesn't accept external W1, W2
```

**Finding:** 
- ❌ Cannot substitute PyWavelets CWT coefficients directly
- ✅ Could extract smoothing logic and create custom WCT function
- ⚠️ Would require reimplementing lines 517-537 of wavelet.py

**Question 3: Hybrid solutions**

**Code evidence:**
The smoothing function (mothers.py, lines 64-95) is self-contained and only requires:
- Wavelet coefficients (W)
- Time step (dt)
- Scale parameters (dj, sj)

**Finding:**
✅ **Hybrid architecture IS feasible:**
1. Compute fast CWT with PyWavelets: `W1_pywt, W2_pywt = pywt.cwt(...)`
2. Extract pycwt smoothing: `smooth = Morlet().smooth(...)`
3. Implement custom coherence: `WCT = abs(smooth(W12))**2 / (smooth(W1) * smooth(W2))`

**Question 4: Production thresholds**

Based on code analysis and complexity:

| Use Case | Dataset Size | Recommendation |
|----------|--------------|----------------|
| Circadian rhythms | <1,000 pts | ✅ pycwt native (seconds) |
| Single-cell imaging | 1k-10k pts | ✅ pycwt native (seconds-minutes) |
| Physiological signals | 10k-100k pts | ⚠️ Consider optimization (minutes) |
| Neural recordings | 100k-1M pts | 🔴 Hybrid or GPU required (hours) |
| Multi-electrode arrays | >1M pts | 🔴 Distributed/GPU mandatory |

### 6.3 Optimization Opportunities

**Immediate wins (low-hanging fruit):**

1. **Parallelize Monte Carlo simulations**
   ```python
   # Current (sequential):
   for _ in range(mc_count):
       nW1, nW2 = cwt(noise1), cwt(noise2)
       # ... compute coherence ...
   
   # Proposed (parallel):
   from multiprocessing import Pool
   results = pool.starmap(compute_mc_iteration, 
                          [(noise1, noise2, kwargs) 
                           for _ in range(mc_count)])
   ```
   **Expected speedup:** 4-8× on modern CPUs (depending on cores)

2. **Reduce Monte Carlo iterations for exploratory analysis**
   ```python
   wct(..., sig=False)  # Skip significance
   # Or use fewer iterations:
   wct_significance(..., mc_count=100)  # 3× faster
   ```

3. **Ensure FFTW is installed**
   ```bash
   pip install pyfftw
   ```
   **Expected speedup:** 2-3× for large signals

**Medium-term optimizations:**

4. **Hybrid architecture implementation**
   - Use PyWavelets for CWT: `pywt.cwt()`
   - Extract pycwt smoothing logic
   - Create custom `wct_hybrid()` function
   **Expected speedup:** 2-5× (if PyWavelets CWT is faster)

5. **Vectorized Monte Carlo**
   - Compute multiple surrogates simultaneously
   - Batch FFT operations
   **Expected speedup:** 2-3× (memory-limited)

**Long-term optimizations:**

6. **GPU acceleration (CuPy/PyTorch)**
   - Port FFT operations to GPU
   - Batch process Monte Carlo iterations
   **Expected speedup:** 10-50× for large signals

7. **Adaptive Monte Carlo**
   - Stop early if significance is clear
   - Use sequential testing
   **Expected speedup:** 2-5× on average (depends on data)

---

## 7. API Compatibility Analysis

### 7.1 Can PyWavelets Replace pycwt's CWT?

**pycwt CWT output:**
```python
W, sj, freqs, coi, fft, fftfreqs = cwt(signal, dt, ...)
# W: complex128 array, shape (n_scales, n_timepoints)
# sj: scales array
# freqs: frequencies array (1 / (wavelet.flambda() * sj))
```

**PyWavelets CWT output:**
```python
coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=dt)
# coeffs: complex128 array, shape (n_scales, n_timepoints)
# freqs: frequencies array
```

**Key differences:**

| Aspect | pycwt | PyWavelets |
|--------|-------|------------|
| Scale definition | `sj = s0 * 2^(j*dj)` | User-provided scales array |
| Frequency definition | `1/(flambda*sj)` | Automatic from scales |
| Cone of influence | Returned | Not returned |
| Normalization | Torrence & Compo | PyWavelets convention |

**Compatibility assessment:**
- ✅ Output shapes match
- ✅ Both return complex coefficients
- ⚠️ Scale/frequency conventions differ
- ⚠️ Normalization may differ (needs validation)
- ❌ No COI from PyWavelets (but can be computed separately)

**Conversion code needed:**
```python
# Convert pycwt scales to PyWavelets scales
pywt_scales = pycwt_sj / dt

# Or convert PyWavelets to pycwt frequency convention
pycwt_freqs = 1 / (wavelet.flambda() * pywt_scales * dt)
```

### 7.2 Extracting pycwt Smoothing for Hybrid Use

**Required components:**
1. Smoothing function: `wavelet.smooth(W, dt, dj, sj)` (mothers.py)
2. Wavelet parameters: `deltaj0`, `coi()`, `flambda()`
3. Scale normalization: `scales = ones([1, N]) * sj[:, None]`

**Standalone implementation:**
```python
from pycwt.mothers import Morlet
import numpy as np

def hybrid_wct(W1, W2, sj, dt, dj):
    """
    Custom WCT using pre-computed CWT coefficients.
    
    Parameters:
    -----------
    W1, W2 : complex128 arrays from any CWT implementation
    sj : scales array
    dt : sampling period
    dj : scale spacing
    """
    wavelet = Morlet(6)
    
    # Scale normalization
    n = W1.shape[1]
    scales = np.ones([1, n]) * sj[:, None]
    
    # Smooth power spectra
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales, dt, dj, sj)
    
    # Cross-spectrum
    W12 = W1 * W2.conj()
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    
    # Coherence
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)
    
    return WCT, aWCT
```

**Status:** ✅ Hybrid architecture is FEASIBLE

---

## 8. Numerical Validation Requirements

### 8.1 Critical Validation Tests

If implementing hybrid or alternative architectures, the following must be validated:

**Test 1: CWT coefficient equivalence**
```python
# Compute both
W_pycwt, sj, freqs, coi, _, _ = pycwt.cwt(signal, dt, ...)
W_pywt, freqs_pywt = pywt.cwt(signal, scales, 'morl', sampling_period=dt)

# Compare magnitudes (phase conventions may differ)
power_pycwt = np.abs(W_pycwt) ** 2
power_pywt = np.abs(W_pywt) ** 2

correlation = np.corrcoef(power_pycwt.flatten(), power_pywt.flatten())[0, 1]
assert correlation > 0.99, "CWT outputs must be highly correlated"

relative_error = np.mean(np.abs(power_pycwt - power_pywt) / (power_pycwt + 1e-10))
assert relative_error < 0.01, "Mean relative error must be <1%"
```

**Test 2: Smoothing operator equivalence**
```python
# Ensure custom smoothing matches pycwt
S_pycwt = Morlet(6).smooth(W, dt, dj, sj)
S_custom = custom_smooth(W, dt, dj, sj)

assert np.allclose(S_pycwt, S_custom, rtol=1e-6)
```

**Test 3: WCT numerical equivalence**
```python
# Full WCT comparison
WCT_native, _, _, _, _ = pycwt.wct(y1, y2, dt, sig=False)
WCT_hybrid = hybrid_wct(W1_pywt, W2_pywt, sj, dt, dj)

correlation = np.corrcoef(WCT_native.flatten(), WCT_hybrid.flatten())[0, 1]
assert correlation > 0.99, "WCT must match native implementation"
```

**Test 4: Biological signal validation**
```python
# Test on real circadian gene expression data
from pycwt.sample import load_sample_data
data = load_sample_data('circadian')

# Ensure biological interpretation is preserved
coherence_native = compute_wct_native(data)
coherence_hybrid = compute_wct_hybrid(data)

# Check that peak coherence locations match
peaks_native = find_peaks(coherence_native)
peaks_hybrid = find_peaks(coherence_hybrid)
assert np.allclose(peaks_native, peaks_hybrid, rtol=0.05)
```

### 8.2 Acceptance Criteria

**For hybrid architecture to be production-ready:**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Coefficient correlation | >0.99 | Statistical equivalence |
| Mean relative error | <1% | Practical equivalence |
| Peak location accuracy | <5% difference | Biological interpretation preserved |
| Significance test agreement | >95% concordance | Statistical validity |
| Visual scalogram inspection | No systematic distortion | Researcher confidence |

---

## 9. Recommendations for MVP Research

### 9.1 Immediate Benchmarking Priorities

**Phase 1 (Week 1): Baseline characterization**

1. **Profile pycwt performance** across dataset sizes
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   WCT, aWCT, coi, freq, sig = pycwt.wct(signal1, signal2, dt)
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions
   ```

2. **Identify bottleneck breakdown**
   - Time spent in `cwt()` vs `smooth()` vs `wct_significance()`
   - Memory allocation patterns
   - FFT backend verification

3. **Establish performance curves**
   - Plot: log(N) vs log(time) for N = [100, 1k, 10k, 100k, 1M]
   - Separate curves for: CWT alone, WCT without sig, WCT with sig
   - Determine empirical scaling (should be ~O(N log N))

**Phase 2 (Week 2): Alternative library comparison**

4. **PyWavelets CWT benchmarks**
   ```python
   import pywt
   import time
   
   # Head-to-head comparison
   start = time.perf_counter()
   W_pywt, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=dt)
   pywt_time = time.perf_counter() - start
   
   start = time.perf_counter()
   W_pycwt, sj, freqs, coi, _, _ = pycwt.cwt(signal, dt)
   pycwt_time = time.perf_counter() - start
   
   speedup = pycwt_time / pywt_time
   print(f"PyWavelets is {speedup:.2f}× faster")
   ```

5. **Numerical validation**
   - Run equivalence tests (section 8.1)
   - Document any systematic differences
   - Determine if differences are scientifically acceptable

**Phase 3 (Week 3): Decision point**

6. **Analyze results and recommend path forward**
   - If pycwt is fast enough (N<100k in <10s): ✅ Use as-is
   - If PyWavelets shows >2× speedup + numerical equivalence: ⚠️ Consider hybrid
   - If both are too slow for target use case: 🔴 Investigate GPU path

### 9.2 Quick Optimization Wins

**Before investing in hybrid architecture:**

1. **Enable FFTW**
   ```bash
   pip install pyfftw
   # Verify: import pycwt; print(pycwt.wavelet.fft.__name__)
   ```

2. **Disable significance testing for exploratory analysis**
   ```python
   WCT, aWCT, coi, freq, _ = pycwt.wct(y1, y2, dt, sig=False)
   # 300× faster!
   ```

3. **Reduce Monte Carlo iterations**
   ```python
   sig = pycwt.wct_significance(..., mc_count=100)  # Instead of 300
   # 3× faster, still statistically valid for most purposes
   ```

4. **Use cached significance results**
   - Cache automatically stores results in `~/.cache/pycwt/`
   - Reuse for signals with similar AR(1) characteristics

5. **Parallelize multiple signal pairs**
   ```python
   from multiprocessing import Pool
   
   def compute_wct_pair(args):
       y1, y2, dt = args
       return pycwt.wct(y1, y2, dt)
   
   # Process multiple pairs in parallel
   with Pool(cpu_count()) as pool:
       results = pool.map(compute_wct_pair, signal_pairs)
   ```

### 9.3 Red Flags to Watch For

**During benchmarking, watch for:**

1. **Memory exhaustion** (OOM errors)
   - Signals >1M points may exceed RAM
   - Check with: `import psutil; psutil.Process().memory_info().rss`

2. **Super-linear scaling** (worse than O(N log N))
   - Plot log-log curves
   - If slope >1.2, investigate cause

3. **FFT backend issues**
   - Verify FFTW is actually being used
   - Check for warning messages about failed imports

4. **Cache directory permissions**
   - Ensure `~/.cache/pycwt/` is writable
   - Failed cache writes waste computation

5. **Numerical instabilities**
   - Watch for NaN or Inf in results
   - Can occur for signals with extreme values
   - May need preprocessing (detrending, normalization)

---

## 10. Conclusions

### 10.1 Key Findings Summary

1. **✅ FFT-based architecture confirmed**
   - All core operations use FFT via convolution theorem
   - Well-implemented following Torrence & Compo (1998)
   - Theoretically sound, but computationally expensive for large datasets

2. **✅ Monte Carlo testing is the primary bottleneck**
   - 300 iterations × 8 FFT operations = 2,400 FFT calls
   - Sequential implementation (no parallelization)
   - Dominates computation time for N > 10,000

3. **✅ Hybrid architecture is feasible**
   - Smoothing function can be extracted and reused
   - PyWavelets CWT coefficients can be adapted
   - Numerical validation required but straightforward

4. **✅ Multiple optimization paths available**
   - Immediate: FFTW, reduce MC iterations, skip significance
   - Medium-term: Parallelize Monte Carlo, hybrid PyWavelets
   - Long-term: GPU acceleration, adaptive algorithms

### 10.2 Answers to MVP Research Questions

**MVP Q1: Is pycwt fast enough for BioXen's target biological applications?**

**Answer:** **It depends on dataset size and significance testing requirements:**
- ✅ **YES** for circadian rhythms (<1k points): seconds
- ✅ **YES** for single-cell imaging (1k-10k points) without significance: seconds
- ⚠️ **MAYBE** for physiological signals (10k-100k points) with reduced MC: minutes
- ❌ **NO** for neural recordings (>100k points) with full significance: hours

**MVP Q2: What is the actual speedup of PyWavelets over pycwt?**

**Answer:** **Requires empirical testing, but code analysis suggests:**
- PyWavelets uses C/Cython for core operations
- pycwt uses NumPy/SciPy FFT (with optional FFTW)
- Expected speedup: **2-5× for CWT alone**
- End-to-end WCT speedup: **Less than 2×** (smoothing still needed)

**MVP Q3: Can we build a hybrid architecture?**

**Answer:** ✅ **YES, but requires custom implementation:**
- Extract smoothing logic from pycwt
- Adapt PyWavelets coefficients to pycwt conventions
- Validate numerical equivalence on biological data
- Estimated development time: 2-3 weeks

**MVP Q4: What is the production recommendation?**

**Answer:** **Tiered approach based on use case:**

```
IF dataset_size < 10,000:
    USE pycwt native  # Good enough
ELIF dataset_size < 100,000:
    IF need_significance:
        USE pycwt with mc_count=100  # Reduced iterations
    ELSE:
        USE pycwt native
ELIF dataset_size < 1,000,000:
    USE hybrid_pywt_pycwt OR pycwt with sig=False
ELSE:  # >1M points
    REQUIRE GPU acceleration OR distributed processing
```

### 10.3 Final Recommendation

**For BioXen Phase 1 MVP:**

1. **Ship with pycwt native** as baseline
   - Document performance characteristics
   - Provide guidelines for dataset size limits
   - Recommend `sig=False` for exploratory analysis

2. **Implement quick optimizations:**
   - Ensure FFTW installation in requirements
   - Add `mc_count` parameter to API
   - Enable parallel processing of multiple signal pairs

3. **Defer hybrid architecture to Phase 2:**
   - Only pursue if users report bottlenecks
   - Requires 2-3 weeks development + validation
   - Risk of introducing numerical bugs

4. **Plan GPU path for Phase 3:**
   - For users with extreme-scale datasets
   - Focus on Monte Carlo parallelization first
   - Consider ssqueezepy integration

**Success metrics:**
- 95% of target use cases complete in <10 minutes
- Clear documentation of performance limits
- No numerical accuracy regressions
- User confidence in results

---

## 11. References

### Code References

1. **pycwt/wavelet.py**
   - Lines 12-126: `cwt()` implementation
   - Lines 421-540: `wct()` implementation  
   - Lines 543-650: `wct_significance()` Monte Carlo

2. **pycwt/mothers.py**
   - Lines 15-103: Morlet wavelet class
   - Lines 64-95: Smoothing function

3. **pycwt/helpers.py**
   - Lines 6-27: FFT backend selection
   - Lines 33-102: AR(1) coefficient estimation

### Scientific References

1. Torrence, C. and Compo, G. P. (1998). A Practical Guide to Wavelet Analysis. *Bulletin of the American Meteorological Society*, 79, 61-78.

2. Torrence, C. and Webster, P. J. (1999). Interdecadal changes in the ENSO-Monsoon system. *Journal of Climate*, 12(8), 2679-2690.

3. Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear Processes in Geophysics*, 11, 561-566.

4. Liu, Y., Liang, X. S. and Weisberg, R. H. (2007). Rectification of the bias in the wavelet power spectrum. *Journal of Atmospheric and Oceanic Technology*, 24, 2093-2102.

---

**END OF REPORT**

*This analysis provides the technical foundation for informed decision-making on pycwt performance optimization strategies for biological signal analysis.*

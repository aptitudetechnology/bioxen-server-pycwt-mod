# Deep Research Prompt: FFT-Based Performance Bottlenecks in Massive Biological Dataset Analysis

**Project:** BioXen Four-Lens Signal Analysis Library  
**Version:** Performance Optimization Research  
**Date:** October 2, 2025  
**Context:** Based on "Advanced Comparative Analysis: Python Libraries for Multi-Signal Wavelet Coherence in Computational Biology"  
**Priority:** HIGH - Critical for production scalability

---

## 🎯 Research Objective

The **Advanced Comparative Analysis** document identified that `pycwt`—the mandatory primary tool for wavelet coherence analysis—uses FFT-based computation, which "presents a functional trade-off" and "may introduce performance bottlenecks when compared to libraries leveraging lower-level C/Cython implementations or GPU acceleration" for massive biological datasets.

**This research prompt investigates:**
1. **Quantifying the bottleneck:** When does FFT-based pycwt become prohibitively slow?
2. **Alternative architectures:** Can PyWavelets (C/Cython) or ssqueezepy (GPU) replace pycwt for coherence?
3. **Hybrid solutions:** Can we compute CWT with fast engines, then feed into pycwt for coherence?
4. **Production thresholds:** What dataset sizes can we realistically support?

---

## 📊 Problem Statement

### **The pycwt Performance Concern**

From the comparative analysis document:

> "The computational method employed by pycwt relies on the Fast Fourier Transform (FFT) algorithm for calculation of the wavelet transform. While this approach maintains rigorous mathematical consistency with the original geophysical methodology developed by Torrence and Compo, it presents a functional trade-off. For the analysis of extremely large or high-sampling-rate biological signals—such as massive EEG datasets—the FFT-based approach may introduce performance bottlenecks..."

### **The Biological Use Case Spectrum**

| Biological Signal Type | Typical Size | Sampling Rate | Timepoints | Challenge |
|------------------------|--------------|---------------|------------|-----------|
| **Circadian gene expression** | Small | 1 sample/hour | 168 pts (1 week) | ✅ No issue |
| **Single-cell time-lapse** | Medium | 1 frame/5min | 2,880 pts (10 days) | ⚠️ Test zone |
| **High-res physiological (ECG/BP)** | Large | 100 Hz | 8.64M pts (24 hours) | 🔴 Bottleneck likely |
| **Multi-electrode neural recordings** | Massive | 30 kHz | 2.59B pts (24 hours) | 🔴 Impossible? |
| **Whole-brain calcium imaging** | Extreme | 30 Hz × 10k neurons | 25.9M pts/day | 🔴 Multi-dimensional |

**Key Question:** At what scale does pycwt fail to meet production requirements?

---

## 🔴 CRITICAL RESEARCH QUESTIONS

### Question 1: Performance Characterization of pycwt FFT-Based CWT

**Objective:** Establish empirical performance baselines for pycwt across biological dataset scales

**Hypotheses to Test:**
1. **H1:** pycwt performance degrades super-linearly (worse than O(N log N)) for large N
2. **H2:** Wavelet coherence (WCT) is slower than individual CWT due to double computation + cross-spectrum
3. **H3:** Memory becomes limiting factor before CPU time for datasets >1M points

**Experimental Design:**

#### **1.1 Single-Signal CWT Benchmarks**

**Test Matrix:**
- **Signal lengths:** 100, 1k, 10k, 100k, 1M, 10M timepoints
- **Wavelet:** Morlet (standard biological choice)
- **Scales:** 10, 50, 100, 500 (typical biological frequency ranges)
- **Hardware:** CPU-only (baseline), multi-core (if parallelizable)

**Metrics to Measure:**
- Wall-clock time (seconds)
- CPU time (user + system)
- Peak memory usage (MB)
- Memory bandwidth utilization

**Code Template:**
```python
import pycwt as wavelet
import numpy as np
import time
import psutil
import tracemalloc

def benchmark_pycwt_cwt(signal_length, num_scales=50):
    # Generate synthetic signal
    t = np.linspace(0, 1, signal_length)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(signal_length)
    
    # Setup
    dt = t[1] - t[0]
    dj = 1/12  # Torrence & Compo standard
    s0 = 2 * dt  # Starting scale
    J = num_scales / dj  # Number of scales
    
    # Memory tracking
    tracemalloc.start()
    mem_before = psutil.Process().memory_info().rss / 1024**2
    
    # Benchmark CWT
    start_time = time.perf_counter()
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        signal, dt, dj, s0, J, wavelet='morlet'
    )
    end_time = time.perf_counter()
    
    # Memory measurement
    mem_after = psutil.Process().memory_info().rss / 1024**2
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'signal_length': signal_length,
        'num_scales': num_scales,
        'wall_time': end_time - start_time,
        'memory_delta': mem_after - mem_before,
        'peak_memory': peak_mem / 1024**2
    }

# Run benchmark suite
results = []
for N in [100, 1000, 10000, 100000, 1000000]:
    result = benchmark_pycwt_cwt(N)
    results.append(result)
    print(f"N={N:,}: {result['wall_time']:.3f}s, {result['peak_memory']:.1f}MB")
```

**Success Criteria:**
- Establish empirical scaling relationship: T(N) = ?
- Identify memory limit: Maximum processable N on typical hardware (16GB, 64GB RAM)
- Determine "acceptable latency" threshold (e.g., <10s for interactive, <5min for batch)

**Deliverable:** Performance curve plot (log-log scale) with fitted scaling model

---

#### **1.2 Multi-Signal WCT Benchmarks**

**Test Matrix:**
- **Signal pairs:** Two signals of lengths 100, 1k, 10k, 100k, 1M
- **Wavelet coherence:** Full WCT computation with significance testing
- **Monte Carlo iterations:** 0 (raw WCT), 100, 1000 surrogates
- **Hardware:** Same as 1.1

**Metrics:**
- WCT time vs. individual CWT time (overhead factor)
- Significance testing scaling with Monte Carlo iterations
- Memory requirements for cross-spectrum storage

**Code Template:**
```python
def benchmark_pycwt_wct(signal_length, mc_iterations=0):
    # Generate two correlated signals
    t = np.linspace(0, 1, signal_length)
    signal1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(signal_length)
    signal2 = np.sin(2 * np.pi * 10 * t + 0.2) + 0.5 * np.random.randn(signal_length)
    
    dt = t[1] - t[0]
    
    # Benchmark individual CWTs
    start = time.perf_counter()
    wave1, scales, freqs, coi, _, _ = wavelet.cwt(signal1, dt)
    wave2, _, _, _, _, _ = wavelet.cwt(signal2, dt)
    cwt_time = time.perf_counter() - start
    
    # Benchmark WCT
    start = time.perf_counter()
    wct, awave, coi, freq, sig = wavelet.wct(
        signal1, signal2, dt, dj=1/12, s0=2*dt, J=7/12
    )
    wct_time = time.perf_counter() - start
    
    # Benchmark with significance testing
    if mc_iterations > 0:
        start = time.perf_counter()
        # Generate surrogates (Fourier phase randomization)
        for i in range(mc_iterations):
            surrogate1 = generate_fourier_surrogate(signal1)
            surrogate2 = generate_fourier_surrogate(signal2)
            wct_surrogate, _, _, _, _ = wavelet.wct(surrogate1, surrogate2, dt)
        mc_time = time.perf_counter() - start
    else:
        mc_time = 0
    
    return {
        'signal_length': signal_length,
        'cwt_time_2x': cwt_time,
        'wct_time': wct_time,
        'overhead_factor': wct_time / cwt_time,
        'mc_iterations': mc_iterations,
        'mc_total_time': mc_time,
        'mc_time_per_iter': mc_time / mc_iterations if mc_iterations > 0 else 0
    }
```

**Critical Questions:**
1. Is WCT >> 2× CWT time? (Cross-spectrum overhead)
2. Does Monte Carlo scale linearly? (1000 iterations = 1000× time?)
3. Can we parallelize Monte Carlo? (embarrassingly parallel problem)

**Deliverable:** WCT overhead analysis report with Monte Carlo scaling

---

### Question 2: Comparative Performance Analysis - PyWavelets vs. pycwt

**Objective:** Quantify performance advantage of PyWavelets (C/Cython) over pycwt (FFT-Python) for CWT computation

**From comparative analysis:**
> "PyWavelets... architecture is built upon low-level C and Cython code, optimizing the computation of wavelet transforms for speed. This design often grants PyWavelets a significant speed advantage over purely Python or FFT-based alternatives..."

**Hypothesis:** PyWavelets is 2-10× faster than pycwt for raw CWT computation

**Experimental Design:**

#### **2.1 Head-to-Head CWT Speed Test**

**Test Matrix:**
- **Signal lengths:** 1k, 10k, 100k, 1M, 10M
- **Wavelets:** Morlet (common), db4, db8 (Daubechies family)
- **Scales:** Matched between libraries (same frequency coverage)
- **Repetitions:** 10 runs per configuration for statistical significance

**Code Template:**
```python
import pycwt as pycwt_lib
import pywt
import numpy as np
import time

def compare_cwt_performance(signal_length, wavelet_name='morl'):
    # Generate signal
    t = np.linspace(0, 1, signal_length)
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    
    # pycwt benchmark
    dt = t[1] - t[0]
    start = time.perf_counter()
    wave_pycwt, scales, _, _, _, _ = pycwt_lib.cwt(
        signal, dt, wavelet='morlet'
    )
    pycwt_time = time.perf_counter() - start
    
    # PyWavelets benchmark
    scales_pywt = np.arange(1, 128)  # Match pycwt scales
    start = time.perf_counter()
    coeffs_pywt, freqs_pywt = pywt.cwt(
        signal, scales_pywt, wavelet_name, sampling_period=dt
    )
    pywt_time = time.perf_counter() - start
    
    return {
        'signal_length': signal_length,
        'pycwt_time': pycwt_time,
        'pywt_time': pywt_time,
        'speedup_factor': pycwt_time / pywt_time,
        'pycwt_output_size': wave_pycwt.nbytes / 1024**2,  # MB
        'pywt_output_size': coeffs_pywt.nbytes / 1024**2
    }

# Run comparison
for N in [1000, 10000, 100000, 1000000]:
    result = compare_cwt_performance(N)
    print(f"N={N:,}: PyWavelets {result['speedup_factor']:.2f}× faster")
```

**Success Criteria:**
- Establish speedup factor: PyWavelets vs. pycwt for CWT
- Verify output equivalence (numerical precision check)
- Memory efficiency comparison

**Deliverable:** Speedup curve with confidence intervals

---

#### **2.2 Numerical Equivalence Validation**

**Critical Question:** Can we safely substitute PyWavelets CWT for pycwt CWT?

**Test Protocol:**
1. Compute CWT with both libraries on identical signal
2. Measure correlation between coefficient matrices
3. Check for systematic differences (amplitude, phase)
4. Validate on biological test signal (CircaDB data)

**Code Template:**
```python
from scipy.stats import pearsonr
import numpy as np

def validate_cwt_equivalence(signal, dt):
    # pycwt
    wave_pycwt, scales_pycwt, freqs_pycwt, _, _, _ = pycwt_lib.cwt(
        signal, dt, wavelet='morlet'
    )
    
    # PyWavelets (match scales to pycwt)
    scales_pywt = scales_pycwt / dt  # Convert to PyWavelets scale convention
    coeffs_pywt, freqs_pywt = pywt.cwt(signal, scales_pywt, 'morl', sampling_period=dt)
    
    # Compare magnitudes (since phase conventions may differ)
    power_pycwt = np.abs(wave_pycwt) ** 2
    power_pywt = np.abs(coeffs_pywt) ** 2
    
    # Flatten for correlation
    correlation, p_value = pearsonr(power_pycwt.flatten(), power_pywt.flatten())
    
    # Check relative error
    relative_error = np.mean(np.abs(power_pycwt - power_pywt) / (power_pycwt + 1e-10))
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'mean_relative_error': relative_error,
        'max_relative_error': np.max(np.abs(power_pycwt - power_pywt) / (power_pycwt + 1e-10))
    }
```

**Acceptance Criteria:**
- Correlation > 0.99 (strong agreement)
- Mean relative error < 1% (practical equivalence)
- Visual inspection: Scalograms look identical

**Deliverable:** Equivalence validation report with difference maps

---

### Question 3: Hybrid Architecture - Fast CWT + pycwt Coherence

**Objective:** Design and validate hybrid pipeline using PyWavelets for CWT, pycwt for WCT calculation

**From comparative analysis:**
> "For future production environments requiring maximal computational efficiency, the core CWT complex coefficients for signals X and Y could be rapidly generated using pywt.cwt, and these highly optimized coefficients could then be seamlessly integrated into pycwt's proven and statistically robust WCT calculation and significance routines."

**Research Questions:**
1. Is the hybrid approach API-compatible? (Can we feed pywt coefficients into pycwt functions?)
2. What's the end-to-end speedup for full WCT workflow?
3. Does this break any statistical assumptions in pycwt's significance testing?

**Experimental Design:**

#### **3.1 API Compatibility Testing**

**Challenge:** pycwt's WCT function expects its own CWT output format

**Investigation:**
```python
# pycwt WCT signature
def pycwt.wct(signal1, signal2, dt, ...):
    # Internally computes CWT for both signals
    # Then calculates cross-spectrum
    pass

# Question: Can we bypass internal CWT?
# Hypothesis: Need to monkey-patch or reimplement WCT core
```

**Approach 1: Direct Substitution (Ideal)**
- Check if pycwt exposes a function that accepts pre-computed CWT coefficients
- If yes: Test with PyWavelets coefficients
- If no: Proceed to Approach 2

**Approach 2: Custom WCT Implementation**
- Extract mathematical formulation from pycwt source code
- Reimplement WCT calculation accepting arbitrary CWT inputs
- Validate against pycwt's native output

**Code Template (Approach 2):**
```python
def custom_wct_from_cwt(wave1, wave2, scales, dt):
    """
    Calculate Wavelet Coherence from pre-computed CWT coefficients
    
    Parameters:
    -----------
    wave1, wave2 : complex ndarray
        CWT coefficients from any library (pycwt, PyWavelets, ssqueezepy)
    scales : array
        Wavelet scales used
    dt : float
        Sampling period
    
    Returns:
    --------
    wct : ndarray
        Wavelet coherence (0-1)
    """
    # Cross-wavelet spectrum
    cross_spectrum = wave1 * np.conj(wave2)
    
    # Smoothing (critical step - must match pycwt)
    # pycwt uses specific smoothing in time and scale
    S_cross = smooth_wavelet(cross_spectrum, dt, scales)
    S_wave1 = smooth_wavelet(np.abs(wave1)**2, dt, scales)
    S_wave2 = smooth_wavelet(np.abs(wave2)**2, dt, scales)
    
    # Wavelet coherence
    wct = np.abs(S_cross)**2 / (S_wave1 * S_wave2)
    
    return wct

def smooth_wavelet(wave, dt, scales):
    """
    Implement pycwt's smoothing operator
    Reference: Torrence & Compo (1998) Eq. 15-16
    """
    # Scale smoothing: convolution with rectangle function
    # Time smoothing: convolution with |scale|^(-1/2) * morlet
    # TODO: Implement exact pycwt smoothing algorithm
    pass
```

**Critical Task:** Reverse-engineer pycwt's smoothing implementation to ensure equivalence

**Deliverable:** Custom WCT function validated against pycwt

---

#### **3.2 End-to-End Hybrid Performance**

**Test Protocol:**
1. Baseline: Full pycwt workflow (CWT + WCT)
2. Hybrid: PyWavelets CWT → Custom WCT
3. Compare: Time, memory, numerical accuracy

**Expected Results:**
- Hybrid time = pywt_cwt_time (2 signals) + custom_wct_time
- Speedup = pycwt_cwt_time / pywt_cwt_time (WCT overhead should be small)
- Target: 2-5× speedup for large signals

**Code Template:**
```python
def benchmark_hybrid_vs_native(signal1, signal2, dt):
    # Baseline: pycwt native
    start = time.perf_counter()
    wct_native, _, _, _, _ = pycwt_lib.wct(signal1, signal2, dt)
    native_time = time.perf_counter() - start
    
    # Hybrid: PyWavelets + custom
    start = time.perf_counter()
    # Fast CWT with PyWavelets
    wave1, _ = pywt.cwt(signal1, scales, 'morl', sampling_period=dt)
    wave2, _ = pywt.cwt(signal2, scales, 'morl', sampling_period=dt)
    # Custom WCT
    wct_hybrid = custom_wct_from_cwt(wave1, wave2, scales, dt)
    hybrid_time = time.perf_counter() - start
    
    # Validation
    correlation = np.corrcoef(wct_native.flatten(), wct_hybrid.flatten())[0, 1]
    
    return {
        'native_time': native_time,
        'hybrid_time': hybrid_time,
        'speedup': native_time / hybrid_time,
        'correlation': correlation
    }
```

**Success Criteria:**
- Speedup > 1.5× (worth the complexity)
- Correlation > 0.99 (numerically equivalent)
- No degradation in significance testing

**Deliverable:** Hybrid architecture validation report

---

### Question 4: GPU Acceleration with ssqueezepy

**Objective:** Evaluate ssqueezepy (synchrosqueezing + GPU) for extreme-scale biological datasets

**From comparative analysis:**
> "ssqueezepy... specializes in the Synchrosqueezing CWT (SSQ_CWT)... benchmarks indicating that it can surpass MATLAB in computational speed, offering native support for multi-threaded CPU operation and optional acceleration via GPU libraries like CuPy and PyTorch."

**Research Questions:**
1. What's the GPU speedup for CWT on 1M-10M point signals?
2. Can GPU handle multi-dimensional datasets (e.g., 10k neurons × 1M timepoints)?
3. When do data transfer costs (CPU→GPU→CPU) negate speedup?
4. Can we integrate ssqueezepy coefficients into pycwt workflow?

**Experimental Design:**

#### **4.1 GPU Performance Characterization**

**Test Matrix:**
- **Signal lengths:** 100k, 1M, 10M, 100M (if GPU memory allows)
- **Hardware:** CPU (baseline), GPU (NVIDIA RTX 3090, A100, or available)
- **Backends:** NumPy (CPU), CuPy (GPU), PyTorch (GPU)
- **Wavelet:** Generalized Morse (ssqueezepy default)

**Code Template:**
```python
import ssqueezepy
import numpy as np
import cupy as cp  # GPU library
import time

def benchmark_ssqueezepy_gpu(signal_length, use_gpu=False):
    # Generate signal
    signal = np.random.randn(signal_length)
    
    if use_gpu:
        # Transfer to GPU
        start_transfer = time.perf_counter()
        signal_gpu = cp.asarray(signal)
        transfer_to_gpu = time.perf_counter() - start_transfer
        
        # CWT on GPU
        start = time.perf_counter()
        Wx, scales = ssqueezepy.cwt(signal_gpu, wavelet='morlet')
        cwt_time = time.perf_counter() - start
        
        # Transfer back to CPU
        start_transfer = time.perf_counter()
        Wx_cpu = cp.asnumpy(Wx)
        transfer_to_cpu = time.perf_counter() - start_transfer
        
        total_time = transfer_to_gpu + cwt_time + transfer_to_cpu
    else:
        # CPU baseline
        start = time.perf_counter()
        Wx, scales = ssqueezepy.cwt(signal, wavelet='morlet')
        cwt_time = time.perf_counter() - start
        transfer_to_gpu = 0
        transfer_to_cpu = 0
        total_time = cwt_time
    
    return {
        'signal_length': signal_length,
        'use_gpu': use_gpu,
        'cwt_time': cwt_time,
        'transfer_to_gpu': transfer_to_gpu,
        'transfer_to_cpu': transfer_to_cpu,
        'total_time': total_time
    }

# Run GPU vs CPU comparison
for N in [100000, 1000000, 10000000]:
    cpu_result = benchmark_ssqueezepy_gpu(N, use_gpu=False)
    gpu_result = benchmark_ssqueezepy_gpu(N, use_gpu=True)
    
    speedup = cpu_result['total_time'] / gpu_result['total_time']
    print(f"N={N:,}: GPU {speedup:.2f}× faster (including transfers)")
```

**Critical Questions:**
1. At what N does GPU become faster than CPU (accounting for transfers)?
2. Can we keep data on GPU for multiple operations (batch processing)?
3. Memory limits: How large a signal can GPU handle? (Typical: 12GB-80GB VRAM)

**Deliverable:** GPU acceleration report with breakeven analysis

---

#### **4.2 Multi-Dimensional Dataset Handling**

**Biological Use Case:** Whole-brain calcium imaging
- 10,000 neurons (ROIs)
- 30 Hz sampling
- 10 minutes recording
- = 10,000 × 18,000 timepoints = 180M values

**Challenge:** Can't process 10k signals sequentially—need batch parallelization

**Approach 1: Batch Processing on GPU**
```python
def batch_cwt_gpu(signals, batch_size=100):
    """
    Process multiple signals in batches on GPU
    
    Parameters:
    -----------
    signals : ndarray, shape (n_signals, n_timepoints)
        Multiple time series (e.g., neurons)
    batch_size : int
        Number of signals to process in parallel on GPU
    """
    n_signals, n_timepoints = signals.shape
    results = []
    
    for i in range(0, n_signals, batch_size):
        batch = signals[i:i+batch_size]
        batch_gpu = cp.asarray(batch)
        
        # Process batch in parallel
        Wx_batch = []
        for signal_gpu in batch_gpu:
            Wx, scales = ssqueezepy.cwt(signal_gpu)
            Wx_batch.append(Wx)
        
        Wx_batch = cp.stack(Wx_batch)
        results.append(cp.asnumpy(Wx_batch))
    
    return np.concatenate(results, axis=0)
```

**Research Question:** What's the optimal batch size for GPU memory utilization?

**Deliverable:** Multi-signal batch processing benchmark

---

### Question 5: Production Scalability Thresholds

**Objective:** Define maximum supportable dataset sizes for each architecture and establish deployment guidelines

**Decision Framework:**

| Dataset Size | Architecture | Latency | Hardware | Recommendation |
|--------------|--------------|---------|----------|----------------|
| < 10k pts | pycwt native | < 1s | CPU (any) | ✅ Use as-is |
| 10k-100k pts | pycwt or PyWavelets | 1-10s | CPU (multi-core) | ✅ Acceptable |
| 100k-1M pts | PyWavelets hybrid | 10s-1min | CPU (16+ cores) | ⚠️ Consider optimization |
| 1M-10M pts | PyWavelets or ssqueezepy | 1-10min | CPU (64GB RAM) or GPU | 🟡 Hybrid required |
| 10M-100M pts | ssqueezepy GPU | 10min-1hr | GPU (24GB+ VRAM) | 🔴 GPU mandatory |
| > 100M pts | Distributed/Dask | Hours | Cluster or cloud | 🔴 Out-of-core or downsample |

**Research Tasks:**
1. Validate each threshold empirically
2. Identify failure modes (memory exhaustion, timeout, accuracy loss)
3. Establish graceful degradation strategies (downsampling, decimation)

**Deliverable:** Production deployment guide with decision tree

---

## 📋 Experimental Validation Protocol

### **Phase 1: Baseline Performance (Week 1)**
- [ ] Install all libraries: pycwt, PyWavelets, ssqueezepy, CuPy (if GPU)
- [ ] Implement benchmarking framework (timing, memory, profiling)
- [ ] Run Question 1 experiments (pycwt characterization)
- [ ] Generate performance curves and identify bottlenecks
- **Deliverable:** Performance characterization report

### **Phase 2: Comparative Analysis (Week 2)**
- [ ] Run Question 2 experiments (PyWavelets vs. pycwt)
- [ ] Validate numerical equivalence
- [ ] Quantify speedup factors with confidence intervals
- **Deliverable:** Library comparison matrix

### **Phase 3: Hybrid Architecture (Week 3)**
- [ ] Implement custom WCT function (Question 3.1)
- [ ] Validate hybrid workflow end-to-end
- [ ] Benchmark hybrid vs. native performance
- **Deliverable:** Hybrid architecture proof-of-concept

### **Phase 4: GPU Evaluation (Week 4, Optional)**
- [ ] Run Question 4 experiments (ssqueezepy GPU)
- [ ] Characterize GPU speedup and breakeven point
- [ ] Test multi-dimensional batch processing
- **Deliverable:** GPU acceleration feasibility report

### **Phase 5: Production Guidelines (Week 5)**
- [ ] Consolidate all findings into decision framework
- [ ] Define deployment thresholds (Question 5)
- [ ] Create production implementation guide
- **Deliverable:** BioXen performance optimization recommendations

---

## 🎯 Success Criteria

### **Minimum Viable Research (MVR):**
1. ✅ Empirical performance curves for pycwt (Question 1)
2. ✅ PyWavelets speedup quantified (Question 2)
3. ✅ Hybrid architecture validated OR rejected (Question 3)
4. ✅ Production threshold recommendations (Question 5)

### **Extended Research (if time/resources allow):**
5. ✅ GPU acceleration benchmarks (Question 4)
6. ✅ Multi-dimensional dataset handling
7. ✅ Out-of-core processing strategies (Dask, Vaex)

### **Publication-Ready Outcomes:**
- Performance comparison paper: "Optimizing Wavelet Coherence for Large-Scale Biological Data"
- Technical blog post: "When Does pycwt Become Too Slow?"
- BioXen documentation: "Performance Guide for Multi-Signal Analysis"

---

## 🛠️ Implementation Roadmap

### **Immediate Actions (This Week):**
1. Set up benchmarking environment
2. Install all candidate libraries
3. Create synthetic dataset generator (100 pts → 10M pts)
4. Run pilot benchmarks on small datasets

### **Week 1 Deliverables:**
- pycwt performance characterization complete
- Identified bottleneck: Memory or CPU?
- Preliminary recommendation: Optimize or acceptable?

### **Decision Point (End Week 1):**
**IF** pycwt is fast enough for 95% of biological use cases:
- ✅ No optimization needed, proceed with pycwt
- Document performance limits in user guide

**ELSE IF** pycwt is bottleneck for common datasets:
- ⚠️ Proceed to Week 2-3 (hybrid architecture)
- Prioritize PyWavelets integration

**ELSE IF** GPU required for target datasets:
- 🔴 Proceed to Week 4 (ssqueezepy GPU)
- Evaluate hardware requirements and costs

---

## 📊 Expected Outcomes

### **Scenario 1: pycwt is Sufficient (Optimistic)**
**Finding:** pycwt handles up to 1M points in < 10 seconds
**Recommendation:** Use pycwt as-is, document limitations
**Impact:** Zero development time for optimization
**Trade-off:** May not support extreme-scale datasets (>10M pts)

### **Scenario 2: Hybrid Architecture Needed (Likely)**
**Finding:** PyWavelets is 3-5× faster, hybrid achieves 2-3× overall speedup
**Recommendation:** Implement PyWavelets CWT + custom WCT
**Impact:** 2-3 weeks development, ongoing maintenance
**Trade-off:** Added complexity, but supports 10× larger datasets

### **Scenario 3: GPU Required (Challenging)**
**Finding:** CPU insufficient for target datasets, GPU provides 10-50× speedup
**Recommendation:** Integrate ssqueezepy with GPU backend
**Impact:** 4-6 weeks development, hardware requirements (GPU)
**Trade-off:** High performance, but deployment complexity and cost

---

## 🔗 Integration with BioXen Research Prompts

### **Relationship to MVP Prompt (wavelets-deep-research-prompt2-mvp.md):**
- **MVP Q1 (Multi-Signal Analysis):** This prompt addresses "Use existing library OR justify custom implementation"
- **MVP assumes pycwt is sufficient** → This research validates or refutes that assumption
- **IF** this research shows pycwt is slow → MVP must be updated to use hybrid

### **Relationship to Phase 2 Prompt (wavelets-deep-research-prompt3-mvp.md):**
- **Phase 2 Q6 (Performance Benchmarking):** This prompt IS the deep dive into Q6
- Provides the data needed to decide: Optimize or accept?
- Informs when to trigger Phase 2 optimization

### **Relationship to Comparative Analysis Document:**
- This research **validates the performance claims** in the comparative analysis
- Tests hypothesis: "FFT-based approach may introduce performance bottlenecks"
- Provides empirical evidence for architectural recommendations

---

## 📚 Technical Resources

### **Essential Reading:**
1. Torrence & Compo (1998) - "A Practical Guide to Wavelet Analysis" (FFT-based CWT algorithm)
2. PyWavelets documentation - C/Cython implementation details
3. ssqueezepy documentation - GPU acceleration guides
4. NumPy/SciPy FFT documentation - Understanding FFT performance

### **Performance Profiling Tools:**
- `cProfile` / `line_profiler` - CPU hotspot identification
- `memory_profiler` / `tracemalloc` - Memory usage tracking
- `psutil` - System resource monitoring
- `py-spy` - Sampling profiler for production
- `nvidia-smi` - GPU utilization (if GPU benchmarking)

### **Benchmarking Best Practices:**
- Warm-up runs (discard first iteration)
- Multiple repetitions (10+ for statistical significance)
- Control for system load (close other processes)
- Monitor CPU frequency scaling (governor settings)
- Measure wall-clock and CPU time separately

---

## ⚠️ Risks & Mitigation

### **Risk 1: PyWavelets CWT is not equivalent to pycwt**
**Impact:** Cannot use hybrid architecture
**Mitigation:** Rigorous numerical validation (Question 2.2)
**Fallback:** Stick with pycwt, optimize Python code or wait for library updates

### **Risk 2: Custom WCT implementation is buggy**
**Impact:** Incorrect coherence results, failed validation
**Mitigation:** Extensive testing against pycwt, unit tests, CircaDB validation
**Fallback:** Use pycwt native, accept performance limits

### **Risk 3: GPU not available in production**
**Impact:** Cannot deploy ssqueezepy solution
**Mitigation:** Provide CPU fallback, or use cloud GPU (AWS, GCP)
**Fallback:** Downsampling or distributed processing

### **Risk 4: Development time exceeds value**
**Impact:** 6 weeks optimization work for 2× speedup
**Mitigation:** Decision point after Week 1 - is optimization worth it?
**Fallback:** Accept pycwt performance, document limitations

---

## 🎓 Learning Objectives

By completing this research, the team will:
1. ✅ Understand FFT-based CWT performance characteristics
2. ✅ Master profiling and benchmarking methodologies
3. ✅ Gain expertise in wavelet library internals
4. ✅ Learn GPU acceleration techniques for signal processing
5. ✅ Develop production scalability intuition

**Knowledge Transfer:**
- Technical blog posts on findings
- Internal documentation of benchmarking methodology
- Code templates for performance testing
- Decision framework for future optimizations

---

## 📝 Documentation Deliverables

### **1. Performance Characterization Report**
- Empirical scaling curves (pycwt, PyWavelets, ssqueezepy)
- Memory and CPU profiling data
- Bottleneck analysis with hotspot identification

### **2. Library Comparison Matrix**
- Speedup factors with confidence intervals
- Numerical equivalence validation results
- API compatibility assessment

### **3. Hybrid Architecture Specification**
- Custom WCT implementation (if needed)
- Integration guide for PyWavelets + pycwt
- Testing protocol for validation

### **4. GPU Acceleration Guide** (if pursued)
- GPU vs. CPU performance curves
- Hardware requirements and costs
- Deployment recommendations

### **5. Production Deployment Guide**
- Decision tree for architecture selection
- Dataset size thresholds
- Performance troubleshooting guide

---

## 🚀 Final Recommendations Template

At the end of this research, provide a clear recommendation:

**For BioXen wavelet coherence analysis, we recommend:**

**Architecture:** [pycwt native | PyWavelets hybrid | ssqueezepy GPU]

**Justification:**
- Performance: [X×] speedup over baseline for [typical dataset size]
- Complexity: [Low | Medium | High] implementation effort
- Scalability: Supports datasets up to [N] timepoints
- Cost: [No additional hardware | CPU cluster | GPU required]

**Implementation Priority:** [Phase 1 MVP | Phase 2 Optimization | Phase 3 Advanced]

**Trade-offs:**
- ✅ Pros: [List benefits]
- ❌ Cons: [List drawbacks]

**Next Steps:**
1. [Action 1]
2. [Action 2]
3. [Action 3]

---

## 📅 Timeline Summary

| Phase | Duration | Deliverable | Decision Point |
|-------|----------|-------------|----------------|
| Phase 1 | 1 week | pycwt baseline | Go/No-Go for optimization |
| Phase 2 | 1 week | PyWavelets comparison | Hybrid feasibility |
| Phase 3 | 1-2 weeks | Hybrid implementation | Production readiness |
| Phase 4 | 1-2 weeks | GPU evaluation (optional) | GPU recommendation |
| Phase 5 | 1 week | Final recommendations | Deploy decision |
| **TOTAL** | **5-7 weeks** | | |

**Fast Track (Minimum):** Phases 1-2 only = 2 weeks (if pycwt sufficient or hybrid clear winner)

**Full Research:** All phases = 7 weeks (comprehensive evaluation)

---

## 🎯 Success Metrics

**Quantitative:**
- ✅ Performance curves generated for 10+ configurations
- ✅ Speedup factors measured with <5% error
- ✅ Memory limits identified within 10%
- ✅ GPU breakeven point determined

**Qualitative:**
- ✅ Clear architectural recommendation
- ✅ Production deployment guide
- ✅ User-facing performance documentation
- ✅ Future optimization roadmap

**Impact:**
- ✅ BioXen can process [target dataset size] in [acceptable time]
- ✅ Users understand performance expectations
- ✅ Bottlenecks eliminated or documented
- ✅ Development team has optimization expertise

---

**END OF DEEP RESEARCH PROMPT**

*This research is critical for production readiness. Without understanding performance limits, BioXen cannot make deployment commitments or set user expectations.*

**Priority Level: HIGH** - Should be addressed in Phase 1-2 alongside MVP development.

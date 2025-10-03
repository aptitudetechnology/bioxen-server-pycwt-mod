# Deep Research Prompt: Parallel Monte Carlo Implementation for pycwt Wavelet Coherence

**Project:** BioXen Four-Lens Signal Analysis Library - Performance Optimization  
**Version:** Implementation Phase 1 (MVP)  
**Date:** October 2, 2025  
**Priority:** HIGH - Critical path optimization with highest ROI  
**Prerequisite:** Code analysis report (pwt-report.md) completed

---

## 🎯 Research Objective

Implement parallelized Monte Carlo significance testing for wavelet coherence analysis to achieve **4-8× speedup** with minimal development effort (1-2 days). This is the **highest ROI optimization** identified in the performance analysis, targeting the primary bottleneck: sequential execution of 300 Monte Carlo iterations (2,400 FFT operations).

---

## 📊 Problem Statement

### Current Bottleneck (from Code Analysis)

**Location:** `pycwt/wavelet.py`, lines 543-650 - `wct_significance()` function

**Current Implementation:**
```python
def wct_significance(al1, al2, dt, dj, s0, J, 
                     significance_level=0.95,
                     wavelet='morlet', 
                     mc_count=300,      # ⚠️ Sequential iterations
                     progress=True,
                     cache=True):
    
    # Setup (once)
    N = int(np.ceil(ms * 6))
    nbins = 1000
    wlc = np.ma.zeros([J + 1, nbins])
    
    # ❌ SEQUENTIAL LOOP - THE BOTTLENECK
    for _ in tqdm(range(mc_count), disable=not progress):
        # Generate red-noise surrogates
        noise1 = rednoise(N, al1, 1)
        noise2 = rednoise(N, al2, 1)
        
        # Compute CWT for BOTH signals (2 FFTs each)
        nW1, sj, freq, coi, _, _ = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
        nW2, sj, freq, coi, _, _ = cwt(noise2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
        
        # Compute cross-spectrum
        nW12 = nW1 * nW2.conj()
        
        # Smooth spectra (3 smoothing operations)
        S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales, dt, dj, sj)
        S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, dt, dj, sj)
        S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)
        
        # Compute coherence
        R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
        
        # Build histogram
        for s in range(maxscale):
            cd = np.floor(R2[s, :] * nbins)
            for j, t in enumerate(cd[~cd.mask]):
                wlc[s, int(t)] += 1
    
    # Compute significance threshold from histogram
    # ...
```

**Computational Cost:**
- **300 iterations** × 8 FFT operations = **2,400 FFT calls**
- Each iteration: 2 CWT + 3 smoothing + 1 coherence
- **Sequential execution** on single core
- Typical time: 15 minutes for N=100k, 2.5 hours for N=1M

**Key Observation:** Each Monte Carlo iteration is **completely independent** - textbook embarrassingly parallel problem.

---

## 🔴 Critical Research Questions

### Question 1: Parallelization Strategy - Which Approach?

**Objective:** Select optimal parallelization method for Python-based Monte Carlo

#### Approach 1.1: `multiprocessing.Pool` (Standard Library)

**Advantages:**
- ✅ No external dependencies
- ✅ True parallelism (bypasses GIL)
- ✅ Simple API
- ✅ Cross-platform

**Code Sketch:**
```python
from multiprocessing import Pool, cpu_count

def _mc_iteration(args):
    """Single Monte Carlo iteration (worker function)."""
    al1, al2, N, dt, dj, s0, J, wavelet_name, scales, outsidecoi, maxscale, nbins = args
    
    # Import within worker (necessary for multiprocessing)
    from pycwt import cwt, rednoise
    from pycwt.mothers import Morlet
    
    # Generate surrogates
    noise1 = rednoise(N, al1, 1)
    noise2 = rednoise(N, al2, 1)
    
    # Compute CWT
    wavelet = Morlet(6) if wavelet_name == 'morlet' else wavelet
    nW1, sj, freq, coi, _, _ = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    nW2, sj, freq, coi, _, _ = cwt(noise2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    
    # Compute coherence
    nW12 = nW1 * nW2.conj()
    S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, dt, dj, sj)
    S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)
    R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
    
    # Build local histogram
    wlc_local = np.zeros((maxscale, nbins))
    for s in range(maxscale):
        cd = np.floor(R2[s, :] * nbins)
        for j, t in enumerate(cd[~cd.mask]):
            wlc_local[s, int(t)] += 1
    
    return wlc_local

def wct_significance_parallel(al1, al2, dt, dj, s0, J, 
                               mc_count=300, 
                               n_workers=None):
    """Parallelized Monte Carlo significance testing."""
    
    if n_workers is None:
        n_workers = cpu_count()
    
    # Setup (unchanged)
    # ...
    
    # Prepare arguments for all iterations
    args_list = [(al1, al2, N, dt, dj, s0, J, wavelet.name, 
                  scales, outsidecoi, maxscale, nbins) 
                 for _ in range(mc_count)]
    
    # Parallel execution
    with Pool(n_workers) as pool:
        results = pool.map(_mc_iteration, args_list)
    
    # Aggregate histograms
    wlc = np.sum(results, axis=0)
    
    # Compute significance threshold (unchanged)
    # ...
```

**Research Tasks:**
1. Test on various dataset sizes (N = 1k, 10k, 100k, 1M)
2. Measure speedup vs. number of cores (1, 2, 4, 8, 16)
3. Profile overhead (process spawning, data serialization)
4. Test on Windows, Linux, macOS

**Expected Speedup:** 4-8× on modern 8-core CPU

#### Approach 1.2: `joblib.Parallel` (Scikit-learn Backend)

**Advantages:**
- ✅ Better memory sharing (uses `loky` backend)
- ✅ Progress bar support
- ✅ Automatic caching
- ✅ NumPy-friendly

**Code Sketch:**
```python
from joblib import Parallel, delayed
import numpy as np

def wct_significance_joblib(al1, al2, dt, dj, s0, J, 
                             mc_count=300, 
                             n_jobs=-1):  # -1 = all cores
    
    # Setup
    # ...
    
    # Parallel execution with joblib
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_mc_iteration)(al1, al2, N, dt, dj, s0, J, 
                               wavelet.name, scales, outsidecoi, maxscale, nbins)
        for _ in range(mc_count)
    )
    
    # Aggregate
    wlc = np.sum(results, axis=0)
    
    # Compute significance
    # ...
```

**Research Tasks:**
1. Compare memory usage vs. multiprocessing
2. Benchmark overhead for small vs. large N
3. Test with different backends (loky, threading, multiprocessing)
4. Evaluate progress bar integration

**Expected Speedup:** 4-8× with potentially lower memory overhead

#### Approach 1.3: `concurrent.futures.ProcessPoolExecutor` (Modern Approach)

**Advantages:**
- ✅ Modern Python (3.2+)
- ✅ Context manager support
- ✅ Future objects for fine control
- ✅ Exception handling

**Code Sketch:**
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def wct_significance_futures(al1, al2, dt, dj, s0, J, 
                              mc_count=300, 
                              n_workers=None):
    
    # Setup
    # ...
    
    # Submit tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_mc_iteration, al1, al2, N, dt, dj, s0, J,
                           wavelet.name, scales, outsidecoi, maxscale, nbins)
            for _ in range(mc_count)
        ]
        
        # Collect results with progress bar
        results = []
        for future in tqdm(as_completed(futures), total=mc_count):
            results.append(future.result())
    
    # Aggregate
    wlc = np.sum(results, axis=0)
    
    # Compute significance
    # ...
```

**Research Tasks:**
1. Compare exception handling vs. multiprocessing
2. Test progressive result aggregation
3. Evaluate memory management
4. Profile task scheduling overhead

**Expected Speedup:** 4-8× with better error handling

#### Approach 1.4: `Dask` (Distributed Computing)

**Advantages:**
- ✅ Scales beyond single machine
- ✅ Automatic task graph optimization
- ✅ Native NumPy integration
- ✅ Dashboard for monitoring

**Code Sketch:**
```python
import dask
from dask.distributed import Client, progress

def wct_significance_dask(al1, al2, dt, dj, s0, J, 
                          mc_count=300,
                          scheduler='processes'):  # or 'threads', 'distributed'
    
    # Setup Dask client
    client = Client(processes=True, n_workers=cpu_count())
    
    # Create lazy tasks
    tasks = [
        dask.delayed(_mc_iteration)(al1, al2, N, dt, dj, s0, J,
                                     wavelet.name, scales, outsidecoi, maxscale, nbins)
        for _ in range(mc_count)
    ]
    
    # Execute in parallel
    results = dask.compute(*tasks)
    
    # Aggregate
    wlc = np.sum(results, axis=0)
    
    client.close()
    
    # Compute significance
    # ...
```

**Research Tasks:**
1. Benchmark overhead vs. multiprocessing
2. Test distributed execution (multi-node)
3. Evaluate dashboard usefulness
4. Profile memory management

**Expected Speedup:** 4-8× local, 20-50× distributed

---

### Question 2: Data Transfer Overhead Analysis

**Objective:** Quantify serialization/deserialization costs and optimize data transfer

**Challenge:** Each worker needs:
- Function code
- Input parameters (al1, al2, N, dt, dj, s0, J, wavelet, scales, outsidecoi)
- Return values (histogram array)

**Data Volume per Iteration:**
- Input: ~1 KB (parameters)
- Output: ~800 KB (histogram array for J=100, nbins=1000)
- Total for 300 iterations: ~240 MB

**Research Tasks:**

#### 2.1: Measure Overhead
```python
import pickle
import time

def measure_serialization_overhead():
    # Create representative data
    wlc_local = np.zeros((100, 1000))  # Typical histogram size
    
    # Measure pickle overhead
    start = time.perf_counter()
    pickled = pickle.dumps(wlc_local)
    pickle_time = time.perf_counter() - start
    
    start = time.perf_counter()
    unpickled = pickle.loads(pickled)
    unpickle_time = time.perf_counter() - start
    
    return {
        'pickle_time': pickle_time,
        'unpickle_time': unpickle_time,
        'data_size': len(pickled),
        'overhead_percent': (pickle_time + unpickle_time) / compute_time * 100
    }
```

**Acceptance Criteria:**
- Serialization overhead < 10% of total compute time
- If overhead > 10%, investigate optimization strategies

#### 2.2: Optimization Strategies

**Strategy A: Shared Memory (multiprocessing.shared_memory)**
```python
from multiprocessing import shared_memory
import numpy as np

def wct_significance_shared_memory(al1, al2, dt, dj, s0, J, mc_count=300):
    # Create shared memory for accumulator
    shm = shared_memory.SharedMemory(create=True, size=wlc.nbytes)
    wlc_shared = np.ndarray(wlc.shape, dtype=wlc.dtype, buffer=shm.buf)
    wlc_shared[:] = 0
    
    # Workers write to shared memory (requires locking)
    # ...
    
    shm.close()
    shm.unlink()
```

**Research:** Does shared memory reduce overhead significantly?

**Strategy B: Compress Return Values**
```python
import blosc  # Fast compression library

def _mc_iteration_compressed(args):
    # ... compute wlc_local ...
    
    # Compress before return
    compressed = blosc.compress_ptr(wlc_local.__array_interface__['data'][0],
                                     wlc_local.size, wlc_local.dtype.itemsize)
    return compressed

# Main: decompress results
results = [blosc.decompress(r) for r in compressed_results]
```

**Research:** Does compression reduce transfer time for large histograms?

**Strategy C: Incremental Aggregation**
```python
# Aggregate in batches to reduce memory footprint
batch_size = 50
for i in range(0, mc_count, batch_size):
    batch_results = pool.map(_mc_iteration, args_list[i:i+batch_size])
    wlc += np.sum(batch_results, axis=0)
```

**Research:** Does batching reduce peak memory usage?

---

### Question 3: Progress Monitoring and User Experience

**Objective:** Maintain informative progress feedback during parallel execution

**Current Implementation:**
```python
# Sequential: tqdm works perfectly
for _ in tqdm(range(mc_count), disable=not progress):
    # ... iteration ...
```

**Challenge:** Standard progress bars don't work with multiprocessing

**Research Tasks:**

#### 3.1: Progress Bar Solutions

**Solution A: Shared Counter with multiprocessing.Manager**
```python
from multiprocessing import Manager
from tqdm import tqdm
import time

def wct_significance_with_progress(al1, al2, dt, dj, s0, J, mc_count=300):
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    def _mc_iteration_with_counter(args):
        result = _mc_iteration(args)
        with lock:
            counter.value += 1
        return result
    
    # Monitor progress in main thread
    with Pool(cpu_count()) as pool:
        async_result = pool.map_async(_mc_iteration_with_counter, args_list)
        
        # Progress bar update loop
        with tqdm(total=mc_count) as pbar:
            while not async_result.ready():
                completed = counter.value
                pbar.n = completed
                pbar.refresh()
                time.sleep(0.1)
            pbar.n = mc_count
            pbar.refresh()
        
        results = async_result.get()
```

**Research:** Does shared counter introduce significant overhead?

**Solution B: Callback-Based Progress**
```python
def update_progress(result):
    """Callback called after each task completion."""
    pbar.update(1)

with tqdm(total=mc_count) as pbar:
    with Pool(cpu_count()) as pool:
        for args in args_list:
            pool.apply_async(_mc_iteration, args=(args,), callback=update_progress)
        pool.close()
        pool.join()
```

**Research:** Is callback approach more efficient?

**Solution C: joblib's Built-in Progress**
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1, verbose=10)(  # verbose=10 for progress
    delayed(_mc_iteration)(args) for args in args_list
)
```

**Research:** Is joblib's progress reporting sufficient for users?

#### 3.2: Estimated Time Remaining

**Enhanced Progress Information:**
```python
import time

def wct_significance_with_eta(al1, al2, dt, dj, s0, J, mc_count=300):
    start_time = time.time()
    
    # ... parallel execution with progress tracking ...
    
    # Calculate ETA
    completed = counter.value
    if completed > 0:
        elapsed = time.time() - start_time
        avg_time_per_iteration = elapsed / completed
        remaining = mc_count - completed
        eta_seconds = remaining * avg_time_per_iteration
        
        print(f"Completed: {completed}/{mc_count} "
              f"({completed/mc_count*100:.1f}%) "
              f"ETA: {eta_seconds/60:.1f} minutes")
```

**Research:** Does ETA calculation add value for users?

---

### Question 4: Numerical Validation and Reproducibility

**Objective:** Ensure parallel implementation produces identical results to sequential

**Critical Concerns:**
1. Floating-point arithmetic order may differ
2. Random number generation must be controlled
3. Histogram aggregation must be exact

**Research Tasks:**

#### 4.1: Deterministic Results

**Challenge:** Random number generation in parallel

**Solution A: Seeded RNG per Worker**
```python
def _mc_iteration(args, seed=None):
    """Worker with deterministic RNG."""
    if seed is not None:
        np.random.seed(seed)
    
    # ... generate surrogates using np.random.randn() ...
```

**Call with:**
```python
# Generate unique seeds for each iteration
seeds = np.random.SeedSequence(42).spawn(mc_count)
args_list = [(al1, al2, N, dt, dj, s0, J, wavelet.name, 
              scales, outsidecoi, maxscale, nbins, seed) 
             for seed in seeds]
```

**Research:** 
- Does seeding work correctly across platforms?
- Are results bit-for-bit identical with sequential?

**Solution B: Pre-generated Random Numbers**
```python
# Generate all random numbers in advance
all_noise = np.random.randn(mc_count, 2, N + tau)

def _mc_iteration(args, noise_pair):
    noise1, noise2 = noise_pair
    # Use pre-generated noise instead of rednoise()
    # ...
```

**Research:** Does pre-generation impact memory or performance?

#### 4.2: Validation Protocol

**Test Suite:**
```python
def test_parallel_vs_sequential():
    """Verify parallel produces same results as sequential."""
    
    # Test parameters
    N = 1000
    al1, al2 = 0.5, 0.3
    dt, dj, s0, J = 0.25, 1/12, 2*0.25, 7/0.12
    mc_count = 100
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Sequential (reference)
    sig_sequential = wct_significance(al1, al2, dt, dj, s0, J, 
                                      mc_count=mc_count, 
                                      wavelet='morlet')
    
    # Parallel (test)
    np.random.seed(42)
    sig_parallel = wct_significance_parallel(al1, al2, dt, dj, s0, J,
                                              mc_count=mc_count,
                                              wavelet='morlet')
    
    # Validation
    assert np.allclose(sig_sequential, sig_parallel, rtol=1e-10), \
        "Parallel results differ from sequential!"
    
    # Check histogram distributions
    # ...
```

**Acceptance Criteria:**
- ✅ Results match sequential within floating-point precision
- ✅ Coherence thresholds match exactly
- ✅ Tests pass on Windows, Linux, macOS
- ✅ Tests pass with different core counts (1, 2, 4, 8)

#### 4.3: Statistical Validity

**Verify Monte Carlo properties:**
```python
def test_monte_carlo_properties():
    """Verify parallelization doesn't affect statistical properties."""
    
    # Run multiple times
    results_parallel = []
    for trial in range(10):
        sig = wct_significance_parallel(al1, al2, dt, dj, s0, J, mc_count=100)
        results_parallel.append(sig)
    
    # Statistical tests
    mean_sig = np.mean(results_parallel, axis=0)
    std_sig = np.std(results_parallel, axis=0)
    
    # Expected statistical variation for MC with 100 samples
    expected_cv = 1.0 / np.sqrt(100)  # Coefficient of variation
    observed_cv = std_sig / mean_sig
    
    assert observed_cv < 2 * expected_cv, \
        "Parallel MC shows higher variance than expected!"
```

---

### Question 5: Performance Benchmarking and Scalability

**Objective:** Empirically measure speedup and identify scaling limits

**Research Tasks:**

#### 5.1: Strong Scaling Analysis

**Test: Fixed problem size, varying core count**

```python
import time
import numpy as np

def benchmark_strong_scaling(signal_length, mc_count=300):
    """Measure speedup vs. number of cores."""
    
    # Generate test signal
    t = np.linspace(0, 1, signal_length)
    signal1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(signal_length)
    signal2 = np.sin(2 * np.pi * 10 * t + 0.2) + 0.5 * np.random.randn(signal_length)
    
    # Estimate AR(1) coefficients
    al1, _, _ = ar1(signal1)
    al2, _, _ = ar1(signal2)
    
    results = {}
    
    # Baseline: Sequential
    start = time.perf_counter()
    sig_seq = wct_significance(al1, al2, dt=0.01, dj=1/12, s0=0.02, J=50,
                                mc_count=mc_count, wavelet='morlet')
    seq_time = time.perf_counter() - start
    results['sequential'] = {'time': seq_time, 'speedup': 1.0}
    
    # Parallel: Varying cores
    for n_cores in [1, 2, 4, 8, 16]:
        if n_cores > cpu_count():
            break
        
        start = time.perf_counter()
        sig_par = wct_significance_parallel(al1, al2, dt=0.01, dj=1/12, s0=0.02, J=50,
                                             mc_count=mc_count, n_workers=n_cores,
                                             wavelet='morlet')
        par_time = time.perf_counter() - start
        
        results[f'{n_cores}_cores'] = {
            'time': par_time,
            'speedup': seq_time / par_time,
            'efficiency': (seq_time / par_time) / n_cores
        }
    
    return results

# Run benchmarks
for N in [1000, 10000, 100000]:
    print(f"\n=== Signal Length: {N} ===")
    results = benchmark_strong_scaling(N, mc_count=300)
    for config, metrics in results.items():
        print(f"{config}: {metrics['time']:.2f}s, "
              f"speedup={metrics.get('speedup', 1.0):.2f}×, "
              f"efficiency={metrics.get('efficiency', 1.0):.2%}")
```

**Expected Results:**
```
Signal Length: 1000
sequential: 5.2s, speedup=1.00×
1_cores: 5.3s, speedup=0.98× (overhead test)
2_cores: 2.8s, speedup=1.86×, efficiency=93%
4_cores: 1.5s, speedup=3.47×, efficiency=87%
8_cores: 0.9s, speedup=5.78×, efficiency=72%

Signal Length: 100000
sequential: 520s, speedup=1.00×
2_cores: 270s, speedup=1.93×, efficiency=96%
4_cores: 140s, speedup=3.71×, efficiency=93%
8_cores: 75s, speedup=6.93×, efficiency=87%
```

**Analysis:**
- Efficiency = speedup / n_cores
- Target efficiency: >80% up to 8 cores
- Identify: At what core count does efficiency drop below 70%?

#### 5.2: Weak Scaling Analysis

**Test: Problem size scales with core count**

```python
def benchmark_weak_scaling(base_signal_length=10000, base_mc_count=100):
    """Measure performance when work scales with cores."""
    
    results = {}
    
    for n_cores in [1, 2, 4, 8]:
        # Scale work proportionally
        mc_count = base_mc_count * n_cores
        
        start = time.perf_counter()
        sig = wct_significance_parallel(al1, al2, dt, dj, s0, J,
                                         mc_count=mc_count,
                                         n_workers=n_cores,
                                         wavelet='morlet')
        elapsed = time.perf_counter() - start
        
        results[n_cores] = {
            'mc_count': mc_count,
            'time': elapsed,
            'time_per_iteration': elapsed / mc_count
        }
    
    return results
```

**Ideal Result:** Time remains constant as cores and work scale together

#### 5.3: Overhead Profiling

**Measure sources of overhead:**
```python
def profile_overhead():
    """Break down time spent in different phases."""
    
    # Phase 1: Process spawning
    start = time.perf_counter()
    with Pool(8) as pool:
        spawn_time = time.perf_counter() - start
        
        # Phase 2: Task distribution
        start = time.perf_counter()
        async_result = pool.map_async(_mc_iteration, args_list)
        distribute_time = time.perf_counter() - start
        
        # Phase 3: Computation
        start = time.perf_counter()
        results = async_result.get()
        compute_time = time.perf_counter() - start
    
    # Phase 4: Result aggregation
    start = time.perf_counter()
    wlc = np.sum(results, axis=0)
    aggregate_time = time.perf_counter() - start
    
    total = spawn_time + distribute_time + compute_time + aggregate_time
    
    return {
        'spawn_overhead': spawn_time / total * 100,
        'distribute_overhead': distribute_time / total * 100,
        'compute_time': compute_time / total * 100,
        'aggregate_overhead': aggregate_time / total * 100
    }
```

**Target Breakdown:**
- Spawn overhead: <5%
- Distribute overhead: <5%
- Compute time: >85%
- Aggregate overhead: <5%

---

### Question 6: Error Handling and Robustness

**Objective:** Handle failures gracefully without losing all computation

**Research Tasks:**

#### 6.1: Worker Failure Handling

**Challenge:** What if one worker crashes?

**Solution: Retry Failed Tasks**
```python
def wct_significance_robust(al1, al2, dt, dj, s0, J, mc_count=300, max_retries=3):
    """Parallel MC with retry logic."""
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    failed_tasks = []
    results = []
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_mc_iteration, args): i
            for i, args in enumerate(args_list)
        }
        
        # Collect results, track failures
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result(timeout=300)  # 5 min timeout per task
                results.append(result)
            except Exception as e:
                print(f"Task {task_id} failed: {e}")
                failed_tasks.append((task_id, args_list[task_id]))
        
        # Retry failed tasks
        for retry in range(max_retries):
            if not failed_tasks:
                break
            
            print(f"Retrying {len(failed_tasks)} failed tasks (attempt {retry+1})")
            retry_futures = {
                executor.submit(_mc_iteration, args): (task_id, args)
                for task_id, args in failed_tasks
            }
            
            failed_tasks = []
            for future in as_completed(retry_futures):
                task_id, args = retry_futures[future]
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    print(f"Task {task_id} failed again: {e}")
                    failed_tasks.append((task_id, args))
    
    if failed_tasks:
        raise RuntimeError(f"{len(failed_tasks)} tasks failed after {max_retries} retries")
    
    return results
```

**Research:**
- How often do workers fail in practice?
- Is retry logic necessary or paranoid?
- What timeout is appropriate?

#### 6.2: Memory Management

**Challenge:** Prevent out-of-memory errors

**Solution: Chunked Processing**
```python
def wct_significance_chunked(al1, al2, dt, dj, s0, J, 
                              mc_count=300,
                              chunk_size=50):
    """Process in chunks to limit memory usage."""
    
    wlc_accumulated = np.zeros((J + 1, nbins))
    
    for chunk_start in range(0, mc_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, mc_count)
        chunk_args = args_list[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//chunk_size + 1}/"
              f"{(mc_count + chunk_size - 1) // chunk_size}")
        
        # Process chunk in parallel
        with Pool(cpu_count()) as pool:
            chunk_results = pool.map(_mc_iteration, chunk_args)
        
        # Aggregate immediately (frees memory)
        wlc_accumulated += np.sum(chunk_results, axis=0)
        
        # Explicit garbage collection
        del chunk_results
        import gc
        gc.collect()
    
    return wlc_accumulated
```

**Research:**
- What chunk size optimizes speed vs. memory?
- Does chunking significantly impact performance?
- Can we auto-tune chunk size based on available RAM?

#### 6.3: Checkpoint and Resume

**Solution: Save Intermediate Results**
```python
import pickle
from pathlib import Path

def wct_significance_checkpointed(al1, al2, dt, dj, s0, J, 
                                   mc_count=300,
                                   checkpoint_dir='./mc_checkpoints',
                                   checkpoint_interval=50):
    """Save progress periodically for recovery."""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_file = checkpoint_dir / f'wct_sig_{al1:.3f}_{al2:.3f}_{mc_count}.pkl'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        completed = checkpoint['completed']
        wlc_accumulated = checkpoint['wlc']
        print(f"Resuming from checkpoint: {completed}/{mc_count} iterations complete")
    else:
        completed = 0
        wlc_accumulated = np.zeros((J + 1, nbins))
    
    # Process remaining iterations
    for chunk_start in range(completed, mc_count, checkpoint_interval):
        chunk_end = min(chunk_start + checkpoint_interval, mc_count)
        chunk_args = args_list[chunk_start:chunk_end]
        
        with Pool(cpu_count()) as pool:
            chunk_results = pool.map(_mc_iteration, chunk_args)
        
        wlc_accumulated += np.sum(chunk_results, axis=0)
        completed = chunk_end
        
        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'completed': completed,
                'wlc': wlc_accumulated,
                'params': (al1, al2, dt, dj, s0, J)
            }, f)
    
    # Clean up checkpoint file
    checkpoint_file.unlink()
    
    return wlc_accumulated
```

**Research:**
- Is checkpointing useful for typical run times (<1 hour)?
- What checkpoint interval balances overhead vs. safety?

---

### Question 7: API Design and Backward Compatibility

**Objective:** Integrate parallel implementation seamlessly with existing API

**Research Tasks:**

#### 7.1: Backward-Compatible API

**Design Option A: Auto-Detect (Recommended)**
```python
def wct_significance(al1, al2, dt, dj, s0, J, 
                     significance_level=0.95,
                     wavelet='morlet', 
                     mc_count=300,
                     progress=True,
                     cache=True,
                     parallel=True,      # NEW: Enable parallelization
                     n_workers=None):    # NEW: Number of workers (None=auto)
    """
    Wavelet coherence significance testing.
    
    Parameters
    ----------
    parallel : bool, optional
        If True (default), uses parallel processing for Monte Carlo.
        Set to False to use original sequential implementation.
    n_workers : int, optional
        Number of parallel workers. If None, uses cpu_count().
        Only used if parallel=True.
    
    Notes
    -----
    Parallel processing provides 4-8× speedup on multi-core systems
    with negligible overhead for small problems.
    """
    
    if parallel and mc_count > 10:  # Parallel only worth it for >10 iterations
        return _wct_significance_parallel(al1, al2, dt, dj, s0, J,
                                           significance_level, wavelet,
                                           mc_count, progress, cache, n_workers)
    else:
        return _wct_significance_sequential(al1, al2, dt, dj, s0, J,
                                             significance_level, wavelet,
                                             mc_count, progress, cache)
```

**Design Option B: Separate Function**
```python
# Keep original function unchanged
def wct_significance(al1, al2, dt, dj, s0, J, ...):
    """Original sequential implementation."""
    pass

# New parallel function
def wct_significance_parallel(al1, al2, dt, dj, s0, J, ..., n_workers=None):
    """Parallelized Monte Carlo significance testing."""
    pass

# Convenience: High-level function chooses automatically
def wct(y1, y2, dt, ..., sig=True, parallel='auto'):
    """
    Wavelet coherence transform.
    
    Parameters
    ----------
    parallel : {'auto', True, False}, optional
        Parallelization strategy for significance testing.
        'auto' (default): Use parallel if mc_count > 50 and cpu_count() > 1
        True: Always use parallel
        False: Never use parallel
    """
    # ... CWT computation ...
    
    if sig:
        if parallel == 'auto':
            use_parallel = (mc_count > 50 and cpu_count() > 1)
        else:
            use_parallel = parallel
        
        if use_parallel:
            sig = wct_significance_parallel(a1, a2, dt, dj, s0, J, ...)
        else:
            sig = wct_significance(a1, a2, dt, dj, s0, J, ...)
```

**Research:** Which API design is most user-friendly?

#### 7.2: Configuration Management

**Global Configuration Option:**
```python
# pycwt/__init__.py
_PARALLEL_CONFIG = {
    'enabled': True,
    'n_workers': None,  # Auto-detect
    'chunk_size': None,  # Auto-tune
    'progress': True,
    'backend': 'multiprocessing'  # or 'joblib', 'dask'
}

def set_parallel_config(**kwargs):
    """
    Configure parallel processing globally.
    
    Examples
    --------
    >>> import pycwt
    >>> pycwt.set_parallel_config(enabled=True, n_workers=4)
    >>> pycwt.set_parallel_config(backend='joblib')
    """
    _PARALLEL_CONFIG.update(kwargs)

def get_parallel_config():
    """Get current parallel processing configuration."""
    return _PARALLEL_CONFIG.copy()
```

**Research:** Is global configuration useful or confusing?

#### 7.3: Performance Hints to Users

**Documentation Example:**
```python
def wct(y1, y2, dt, ...):
    """
    Wavelet coherence transform.
    
    Performance Notes
    -----------------
    For signals with N > 10,000 timepoints:
    - Enable parallel processing: parallel=True
    - Set n_workers to match your CPU cores
    - Reduce mc_count for exploratory analysis (100 vs 300)
    - Install FFTW for 2× faster FFT: pip install pyfftw
    
    Expected computation times (8-core CPU, N=100k):
    - With parallel=True, mc_count=300: ~3 minutes
    - With parallel=False, mc_count=300: ~15 minutes
    - With parallel=True, mc_count=100: ~1 minute
    
    Examples
    --------
    >>> # Fast exploratory analysis
    >>> WCT, aWCT, coi, freq, sig = wct(y1, y2, dt, sig=True,
    ...                                  parallel=True, mc_count=100)
    
    >>> # Publication-ready (slower but more robust)
    >>> WCT, aWCT, coi, freq, sig = wct(y1, y2, dt, sig=True,
    ...                                  parallel=True, mc_count=300)
    """
```

---

## 📋 Implementation Roadmap

### Phase 1: Proof of Concept (Day 1)

**Objective:** Get basic parallel working

**Tasks:**
1. Extract `_mc_iteration()` worker function
2. Implement `wct_significance_parallel()` with `multiprocessing.Pool`
3. Test on small dataset (N=1000, mc_count=10)
4. Verify numerical equivalence with sequential

**Deliverable:** Working prototype with basic parallelization

### Phase 2: Optimization and Testing (Day 2)

**Objective:** Optimize and validate thoroughly

**Tasks:**
1. Add progress monitoring (tqdm or joblib)
2. Implement error handling and retry logic
3. Run strong scaling benchmarks (1, 2, 4, 8 cores)
4. Test on representative biological datasets
5. Validate reproducibility across platforms

**Deliverable:** Production-ready implementation with tests

### Phase 3: Integration and Documentation (Day 3, Optional)

**Objective:** Integrate with main API

**Tasks:**
1. Add `parallel` parameter to `wct()` and `wct_significance()`
2. Update documentation with performance notes
3. Add configuration options
4. Write user guide for parallel processing
5. Update examples in `sample/` directory

**Deliverable:** Fully integrated feature with docs

---

## ✅ Success Criteria

### Performance Metrics

**Must Achieve:**
- ✅ 4× speedup on 4-core CPU (80% efficiency)
- ✅ 6× speedup on 8-core CPU (75% efficiency)
- ✅ <10% overhead for small problems (N<1000)
- ✅ <5% serialization overhead

**Target Goals:**
- 🎯 8× speedup on 8-core CPU (100% efficiency)
- 🎯 Linear scaling up to 16 cores
- 🎯 <5% overhead for all problem sizes

### Correctness

**Must Pass:**
- ✅ Bit-for-bit identical results with sequential (with fixed seed)
- ✅ Statistical properties preserved (variance, distribution)
- ✅ All existing unit tests pass
- ✅ Cross-platform compatibility (Windows, Linux, macOS)

### Usability

**Must Have:**
- ✅ Progress bar or status updates
- ✅ Automatic core count detection
- ✅ Backward-compatible API
- ✅ Clear error messages

**Nice to Have:**
- 🎯 ETA estimation
- 🎯 Automatic chunk size tuning
- 🎯 Checkpoint/resume capability

---

## 🎓 Expected Outcomes

### Deliverable 1: Parallel Implementation

**File:** `src/pycwt/wavelet_parallel.py` (new) or update `src/pycwt/wavelet.py`

**Key Functions:**
```python
def wct_significance_parallel(al1, al2, dt, dj, s0, J, 
                               mc_count=300,
                               n_workers=None,
                               backend='multiprocessing',
                               progress=True):
    """Parallelized Monte Carlo significance testing."""
    pass

def _mc_iteration(args):
    """Worker function for single Monte Carlo iteration."""
    pass
```

### Deliverable 2: Performance Report

**File:** `research/parallel-monte-carlo-results.md`

**Contents:**
- Strong scaling results (speedup vs. cores)
- Weak scaling results (constant time as work scales)
- Overhead analysis (breakdown by phase)
- Platform comparison (Windows, Linux, macOS)
- Recommendations for optimal configuration

### Deliverable 3: Updated Documentation

**Files:**
- `docs/tutorial/performance.md` (new)
- `docs/reference/index.md` (update)
- `README.md` (update)

**Contents:**
- Performance tuning guide
- Parallel processing best practices
- Configuration examples
- Troubleshooting common issues

### Deliverable 4: Test Suite

**File:** `src/pycwt/tests/test_parallel.py` (new)

**Tests:**
```python
def test_parallel_vs_sequential()
def test_parallel_scaling()
def test_parallel_reproducibility()
def test_parallel_error_handling()
def test_parallel_progress_monitoring()
def test_parallel_cross_platform()
```

---

## 🚀 Quick Start Guide (After Implementation)

**For Users:**
```python
import pycwt

# Enable parallel processing (4-8× speedup)
WCT, aWCT, coi, freq, sig = pycwt.wct(
    signal1, signal2, dt=0.1,
    sig=True,
    parallel=True,        # Enable parallelization
    mc_count=300,         # Full Monte Carlo
    n_workers=8           # Use 8 cores
)

# Fast exploratory analysis (reduced iterations)
WCT, aWCT, coi, freq, sig = pycwt.wct(
    signal1, signal2, dt=0.1,
    sig=True,
    parallel=True,
    mc_count=100,         # Faster, still statistically valid
    n_workers=-1          # Use all available cores
)

# Benchmark your system
import pycwt
results = pycwt.benchmark_parallel(signal_length=100000, mc_count=300)
print(f"Speedup on your system: {results['speedup']:.2f}×")
```

---

## 📊 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overhead > speedup for small N | Medium | Low | Auto-disable parallel for N<1000 |
| Platform-specific bugs | Medium | Medium | Extensive cross-platform testing |
| Floating-point non-determinism | Low | High | Seeded RNG, validation tests |
| Memory exhaustion | Low | High | Chunked processing, monitoring |
| User confusion with API | Medium | Low | Clear docs, sensible defaults |

**All risks are manageable with proper implementation and testing.**

---

## 🔍 Follow-Up Research (Future Phases)

**If parallel MC is successful, consider:**

1. **Distributed Monte Carlo (Dask/Ray)**
   - Scale beyond single machine
   - Expected: 20-50× on cluster
   - Effort: 1-2 weeks

2. **GPU Monte Carlo**
   - Use CuPy for CWT on GPU
   - Batch process MC iterations
   - Expected: 10-50× for N>100k
   - Effort: 2-3 weeks

3. **Adaptive Monte Carlo**
   - Stop early if significance is clear
   - Sequential probability ratio test (SPRT)
   - Expected: 2-5× average speedup
   - Effort: 1 week

4. **Hybrid CPU-GPU**
   - CPU for small datasets, GPU for large
   - Dynamic load balancing
   - Expected: Optimal across all scales
   - Effort: 2-3 weeks

---

## 📚 References

### Parallelization Resources

1. **Python multiprocessing**: https://docs.python.org/3/library/multiprocessing.html
2. **joblib documentation**: https://joblib.readthedocs.io/
3. **Dask parallel**: https://docs.dask.org/en/latest/
4. **Amdahl's Law**: https://en.wikipedia.org/wiki/Amdahl%27s_law

### Monte Carlo Methods

5. Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear Processes in Geophysics*, 11, 561-566.

6. Torrence, C. and Compo, G. P. (1998). A Practical Guide to Wavelet Analysis. *Bulletin of the American Meteorological Society*, 79, 61-78.

### Performance Optimization

7. pwt-report.md - pycwt code analysis report (this repository)
8. SIMD-analysis.md - SIMD optimization assessment (this repository)

---

**END OF RESEARCH PROMPT**

*This research prompt provides comprehensive guidance for implementing the highest-ROI optimization for pycwt: parallelized Monte Carlo significance testing. Expected outcome: 4-8× speedup with 1-2 days development effort.*

---

**Priority:** 🔴 **CRITICAL** - Implement immediately after infrastructure verification  
**Estimated Effort:** 2-3 days (proof of concept → production-ready)  
**Expected Impact:** 4-8× speedup for Monte Carlo significance testing  
**Dependencies:** None (uses Python standard library)

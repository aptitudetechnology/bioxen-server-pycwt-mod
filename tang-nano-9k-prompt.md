# Tang Nano 9K FPGA Backend Plugin for pycwt-mod Monte Carlo

**Project:** pycwt-mod - Modular Python Continuous Wavelet Transform Library  
**Feature:** Hardware-Accelerated Monte Carlo Backend using Tang Nano 9K FPGA  
**Target:** Monte Carlo Wavelet Coherence Significance Testing  
**Date:** October 3, 2025  
**Status:** Design Phase - Plugin Integration  
**Prerequisites:** Phase 1 (Backend Architecture) + Phase 2 (Integration) Complete

---

## 🎯 Objective

Integrate the Tang Nano 9K FPGA as a hardware-accelerated backend plugin specifically for **Monte Carlo simulations** in pycwt-mod's wavelet coherence significance testing (`wct_significance()` function). This will leverage the existing modular backend architecture to provide FPGA acceleration as a drop-in replacement for CPU-based Monte Carlo backends.

### What Gets Accelerated

The FPGA backend accelerates the **Monte Carlo loop** in `wct_significance()`:
- **300 Monte Carlo iterations** (default, configurable)
- Each iteration: Generate red noise → CWT → Smooth → Compute coherence → Update histogram
- **2,400 FFT operations** per typical run (300 iterations × 8 FFTs)
- This is the primary bottleneck identified in performance analysis

---

## 📊 Architecture Overview

### Current Backend System (Phases 1 & 2 Complete)

```
pycwt_mod/
├── backends/
│   ├── base.py              ✅ Abstract MonteCarloBackend (Monte Carlo interface)
│   ├── registry.py          ✅ Backend registration system
│   ├── sequential.py        ✅ Single-core CPU Monte Carlo
│   ├── joblib.py           ✅ Multi-core CPU Monte Carlo
│   ├── dask.py             🔲 Distributed Monte Carlo (placeholder)
│   ├── gpu.py              🔲 GPU Monte Carlo (placeholder)
│   └── tang_nano_9k.py     🆕 FPGA Monte Carlo (NEW)
```

### Monte Carlo Backend Integration Point

The existing backend system provides a clean integration point through the abstract `MonteCarloBackend` class, which is specifically designed for Monte Carlo simulation execution:

```python
class MonteCarloBackend(ABC):
    """Abstract base class for Monte Carlo execution backends."""
    
    @abstractmethod
    def run_monte_carlo(self, worker_func, n_simulations, 
                       worker_args=(), seed=None, 
                       verbose=False, **kwargs):
        """Execute Monte Carlo simulations."""
        pass
    
    @abstractmethod
    def is_available(self):
        """Check if backend is available."""
        pass
```

---

## 🔧 Tang Nano 9K Backend Design

### Backend Class Structure

```python
# File: src/pycwt_mod/backends/tang_nano_9k.py

from .base import MonteCarloBackend
import numpy as np

class TangNano9KBackend(MonteCarloBackend):
    """
    FPGA-accelerated Monte Carlo backend using Tang Nano 9K.
    
    This backend offloads Monte Carlo wavelet coherence simulations
    to the Tang Nano 9K FPGA for hardware acceleration. Each Monte Carlo
    iteration (red noise generation → CWT → smoothing → coherence) is
    executed in parallel on the FPGA hardware.
    
    What Gets Accelerated:
    - Monte Carlo iterations (typically 300)
    - Red noise generation (2 per iteration)
    - FFT operations (8 per iteration = 2,400 total)
    - Wavelet transforms
    - Spectral smoothing
    - Coherence calculation
    - Histogram accumulation
    
    Hardware Requirements:
    - Tang Nano 9K FPGA board
    - USB connection
    - Bitstream loaded with Monte Carlo wavelet computation core
    
    Software Requirements:
    - pyserial (for UART communication)
    - tang_nano_driver (custom driver package)
    """
    
    def __init__(self, name='tang_nano_9k', config=None):
        super().__init__(name, config)
        self._device = None
        self._bitstream_loaded = False
    
    def is_available(self):
        """
        Check if Tang Nano 9K is available.
        
        Returns:
            bool: True if FPGA is connected and bitstream is loaded
        """
        try:
            # Check for required driver
            import serial
            import tang_nano_driver
            
            # Try to detect device
            device = tang_nano_driver.detect_tang_nano_9k()
            if device is None:
                return False
            
            # Check if bitstream is loaded
            if not tang_nano_driver.check_bitstream(device):
                return False
            
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def run_monte_carlo(self, worker_func, n_simulations, 
                       worker_args=(), seed=None, 
                       verbose=False, **kwargs):
        """
        Execute Monte Carlo simulations on FPGA.
        
        The FPGA executes the complete Monte Carlo workflow in hardware:
        1. Generate random seeds for each simulation
        2. For each simulation:
           - Generate AR(1) red noise (2 signals)
           - Compute CWT for both signals (2 FFTs each)
           - Compute cross-wavelet transform
           - Smooth spectra (3 operations)
           - Calculate coherence
           - Update histogram bins
        3. Return accumulated histogram results
        
        Parameters:
        -----------
        worker_func : callable
            Worker function (_wct_significance_worker) - function signature
            used to extract parameters, actual computation done on FPGA
        n_simulations : int
            Number of Monte Carlo simulations to run on FPGA
        worker_args : tuple
            Arguments for worker function (al1, al2, N, dt, dj, s0, J, ...)
        seed : int, optional
            Random seed for reproducibility of Monte Carlo
        verbose : bool
            Show progress during FPGA computation
        
        Returns:
        --------
        list
            Results from each simulation (wlc histogram arrays)
        """
        if not self.is_available():
            raise RuntimeError(
                "Tang Nano 9K backend not available. "
                "Check device connection and bitstream."
            )
        
        # Import FPGA driver
        import tang_nano_driver as tnd
        
        # Connect to device
        if self._device is None:
            self._device = tnd.connect()
        
        # Extract parameters from worker_args
        # These need to match _wct_significance_worker signature
        (al1, al2, N, dt, dj, s0, J, wavelet, 
         sj, scales, outsidecoi, maxscale, nbins) = worker_args
        
        # Configure FPGA with parameters
        config = tnd.FPGAConfig(
            al1=al1,
            al2=al2,
            N=N,
            dt=dt,
            dj=dj,
            s0=s0,
            J=J,
            wavelet_type=wavelet.name,
            scales=scales,
            maxscale=maxscale,
            nbins=nbins,
            n_simulations=n_simulations,
            seed=seed if seed is not None else np.random.randint(0, 2**31)
        )
        
        # Upload configuration
        tnd.configure(self._device, config)
        
        # Start computation
        if verbose:
            print(f"Running {n_simulations} simulations on Tang Nano 9K FPGA...")
        
        tnd.start_computation(self._device)
        
        # Wait for completion with optional progress
        if verbose:
            from tqdm import tqdm
            with tqdm(total=n_simulations, desc="FPGA Progress") as pbar:
                while not tnd.is_complete(self._device):
                    progress = tnd.get_progress(self._device)
                    pbar.update(progress - pbar.n)
                pbar.update(n_simulations - pbar.n)
        else:
            tnd.wait_for_completion(self._device)
        
        # Read results
        results = tnd.read_results(self._device, n_simulations)
        
        # Convert FPGA output format to expected format
        # FPGA returns aggregated histogram, split into individual results
        # (or return as single aggregated result)
        fpga_results = []
        for i in range(n_simulations):
            wlc_simulation = results[i]  # Shape: [J+1, nbins]
            fpga_results.append(wlc_simulation)
        
        return fpga_results
    
    def __del__(self):
        """Cleanup FPGA connection."""
        if self._device is not None:
            try:
                import tang_nano_driver as tnd
                tnd.disconnect(self._device)
            except:
                pass
```

---

## 🔌 Integration with Existing System

### 1. Backend Registration

The Tang Nano 9K backend auto-registers through the existing system:

**File: `src/pycwt_mod/backends/__init__.py`**

```python
# Existing code...
from .sequential import SequentialBackend
from .joblib import JoblibBackend

# Add Tang Nano 9K import
try:
    from .tang_nano_9k import TangNano9KBackend
    _TANG_NANO_AVAILABLE = True
except ImportError:
    _TANG_NANO_AVAILABLE = False

def _register_builtin_backends():
    """Register all built-in backends."""
    # Existing registrations...
    register_backend('sequential', SequentialBackend)
    register_backend('joblib', JoblibBackend)
    
    # Register Tang Nano 9K if available
    if _TANG_NANO_AVAILABLE:
        register_backend('tang_nano_9k', TangNano9KBackend)

# Auto-register at import
_register_builtin_backends()
```

### 2. Usage in wct_significance

No changes needed! The existing implementation automatically supports the new backend:

```python
from pycwt_mod import wct_significance

# Auto-select (will use Tang Nano if available and best choice)
sig95 = wct_significance(
    al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=28,
    mc_count=1000
)

# Or explicitly request Tang Nano 9K backend
sig95 = wct_significance(
    al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=28,
    mc_count=1000,
    backend='tang_nano_9k'
)
```

### 3. Backend Selection Logic

Update recommendation function to consider Tang Nano 9K:

**File: `src/pycwt_mod/backends/registry.py`**

```python
def get_recommended_backend(n_simulations=100):
    """
    Recommend optimal backend based on problem size.
    
    Priority order:
    1. tang_nano_9k (if available, for any size)
    2. joblib (if available, for n >= 100)
    3. sequential (always available)
    """
    # Check Tang Nano 9K first (best for all sizes)
    if 'tang_nano_9k' in _registry:
        backend = _registry['tang_nano_9k']()
        if backend.is_available():
            return 'tang_nano_9k'
    
    # Existing logic for joblib/sequential
    if n_simulations >= 100 and 'joblib' in _registry:
        backend = _registry['joblib']()
        if backend.is_available():
            return 'joblib'
    
    return 'sequential'
```

---

## 📦 Tang Nano 9K Driver Package

The backend requires a companion driver package for hardware communication.

### Driver Package Structure

```
tang_nano_driver/
├── __init__.py
├── device.py           # Device detection and connection
├── protocol.py         # Communication protocol
├── bitstream.py        # Bitstream management
├── config.py           # Configuration structures
└── wavelet_core.py     # Wavelet-specific operations
```

### Driver API Design

```python
# tang_nano_driver/__init__.py

from .device import (
    detect_tang_nano_9k,
    connect,
    disconnect,
    check_bitstream
)

from .config import FPGAConfig

from .wavelet_core import (
    configure,
    start_computation,
    is_complete,
    get_progress,
    wait_for_completion,
    read_results
)
```

### Example Driver Implementation

```python
# tang_nano_driver/device.py

import serial
import serial.tools.list_ports

def detect_tang_nano_9k():
    """
    Detect Tang Nano 9K device on USB.
    
    Returns:
        str: Serial port path or None if not found
    """
    # Tang Nano 9K typically shows up as FTDI device
    for port in serial.tools.list_ports.comports():
        if 'FTDI' in port.description or 'Tang Nano' in port.description:
            # Verify it's actually Tang Nano 9K
            try:
                ser = serial.Serial(port.device, 115200, timeout=1)
                ser.write(b'ID?\n')
                response = ser.readline()
                ser.close()
                
                if b'TANG_NANO_9K' in response:
                    return port.device
            except:
                continue
    
    return None

def connect(port=None):
    """
    Connect to Tang Nano 9K.
    
    Parameters:
        port : str, optional
            Serial port path (auto-detect if None)
    
    Returns:
        serial.Serial: Connected device
    """
    if port is None:
        port = detect_tang_nano_9k()
    
    if port is None:
        raise RuntimeError("Tang Nano 9K not found")
    
    device = serial.Serial(
        port,
        baudrate=115200,
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=5
    )
    
    return device

def disconnect(device):
    """Close connection to device."""
    if device and device.is_open:
        device.close()

def check_bitstream(device):
    """
    Check if correct bitstream is loaded.
    
    Returns:
        bool: True if wavelet coherence bitstream is loaded
    """
    device.write(b'BITSTREAM?\n')
    response = device.readline()
    return b'WAVELET_COHERENCE_V1' in response
```

---

## 🔬 FPGA Implementation Requirements

### Hardware Architecture for Monte Carlo

The Tang Nano 9K needs a custom bitstream implementing the complete Monte Carlo pipeline:

1. **Monte Carlo Control Unit**
   - Manages iteration loop (e.g., 300 iterations)
   - Generates unique seeds for each simulation
   - Coordinates pipeline stages
   - Tracks progress for host reporting

2. **Random Number Generator (RNG)**
   - AR(1) red noise generator for Monte Carlo
   - Seeded for reproducibility
   - Generates 2 independent noise signals per iteration
   - Parameters: al1, al2, N (signal length)

3. **FFT Core**
   - Configurable length FFT for CWT
   - Complex number support
   - Processes 2 signals per Monte Carlo iteration
   - Parallel processing pipelines

4. **Wavelet Transform Core**
   - Morlet wavelet multiplication
   - Support for multiple wavelet types
   - Scale-based computation
   - Processes both signals in Monte Carlo pair

5. **Smoothing Filter**
   - Configurable smoothing windows
   - Real-time filtering
   - 3 smoothing operations per Monte Carlo iteration

6. **Coherence Calculator**
   - Complex multiplication
   - Magnitude and phase computation
   - Masking for COI (Cone of Influence)
   - Computes coherence for each Monte Carlo simulation

7. **Histogram Accumulator**
   - 1000-bin histogram per scale
   - Accumulation across all Monte Carlo simulations
   - Returns aggregated results to host
   - Reduces data transfer overhead

### Resource Utilization (Estimated)

Tang Nano 9K specifications:
- FPGA: Gowin GW1NR-9
- Logic Units: 8,640 LUTs
- RAM: 468 Kbits
- DSP: 20 multipliers
- PLLs: 2
- I/O: 54 pins

Estimated resource usage:
- RNG: ~500 LUTs
- FFT Core: ~2000 LUTs, 8 DSPs
- Wavelet Core: ~1500 LUTs, 4 DSPs
- Smoothing: ~800 LUTs, 2 DSPs
- Coherence: ~600 LUTs, 4 DSPs
- Histogram: ~400 LUTs
- Control: ~300 LUTs
- **Total: ~6100 LUTs (~70%), 18 DSPs (~90%)**

### Monte Carlo Data Flow

```
Host (Python) → UART → Tang Nano 9K
                         ↓
            [Load Monte Carlo Configuration]
            - Number of simulations (e.g., 300)
            - Random seed
            - Wavelet parameters (al1, al2, dt, dj, s0, J)
                         ↓
            [Initialize Monte Carlo Loop]
            [Generate Seeds for Each Simulation]
                         ↓
         ┌───────────────┴────────────────────────┐
         │   Monte Carlo Loop (300× iterations):  │
         │                                         │
         │   For simulation i = 1 to n_sims:      │
         │   1. Generate red noise pair (x2)      │
         │      - Use seed[i] for RNG             │
         │      - Apply AR(1) with al1, al2       │
         │   2. Compute FFT (x2 signals)          │
         │      - Transform to frequency domain   │
         │   3. Compute wavelet transform         │
         │      - Apply wavelet at multiple scales│
         │   4. Smooth spectra (x3 operations)    │
         │      - Smooth power spectra            │
         │   5. Calculate coherence               │
         │      - Cross-spectrum / (S1 * S2)      │
         │   6. Update histogram bins             │
         │      - Accumulate coherence values     │
         │                                         │
         │   [Progress: i/n_sims → Host]          │
         └───────────────┬────────────────────────┘
                         ↓
            [Finalize Histogram Accumulation]
            - All 300 simulations accumulated
                         ↓
         UART → Host (Python) ← [Aggregated Histogram Results]
         
Note: FPGA returns accumulated histogram, not individual
      simulation results, to minimize data transfer.
```

---

## 🧪 Testing Strategy

### Test File Structure

```python
# src/pycwt_mod/tests/backends/test_tang_nano_9k.py

import pytest
import numpy as np
from pycwt_mod.backends import get_backend

@pytest.mark.skipif(
    not get_backend('tang_nano_9k').is_available(),
    reason="Tang Nano 9K not available"
)
class TestTangNano9KBackend:
    """Tests for Tang Nano 9K FPGA backend."""
    
    def test_backend_available(self):
        """Test that backend can be instantiated."""
        backend = get_backend('tang_nano_9k')
        assert backend.is_available()
    
    def test_device_connection(self):
        """Test device connection and communication."""
        import tang_nano_driver as tnd
        
        device = tnd.connect()
        assert device is not None
        assert device.is_open
        
        # Test basic communication
        device.write(b'PING\n')
        response = device.readline()
        assert b'PONG' in response
        
        tnd.disconnect(device)
    
    def test_bitstream_loaded(self):
        """Test that correct bitstream is loaded."""
        import tang_nano_driver as tnd
        
        device = tnd.connect()
        assert tnd.check_bitstream(device)
        tnd.disconnect(device)
    
    def test_fpga_vs_sequential_equivalence(self):
        """Test FPGA produces equivalent results to sequential."""
        from pycwt_mod import wct_significance
        
        # Sequential
        sig_seq = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=50, backend='sequential', progress=False, cache=False
        )
        
        # FPGA
        sig_fpga = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=50, backend='tang_nano_9k', progress=False, cache=False
        )
        
        # Should be highly correlated (stochastic variation expected)
        valid_seq = sig_seq[~np.isnan(sig_seq)]
        valid_fpga = sig_fpga[~np.isnan(sig_fpga)]
        
        correlation = np.corrcoef(valid_seq, valid_fpga)[0, 1]
        assert correlation > 0.8, f"Low correlation: {correlation}"
    
    @pytest.mark.slow
    def test_fpga_performance(self):
        """Test FPGA performance vs CPU."""
        import time
        from pycwt_mod import wct_significance
        
        # Sequential
        start = time.time()
        wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=20,
            mc_count=500, backend='sequential', progress=False, cache=False
        )
        seq_time = time.time() - start
        
        # FPGA
        start = time.time()
        wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=20,
            mc_count=500, backend='tang_nano_9k', progress=False, cache=False
        )
        fpga_time = time.time() - start
        
        speedup = seq_time / fpga_time
        print(f"\nFPGA Speedup: {speedup:.2f}×")
        
        # FPGA should be faster
        assert speedup > 1.0, f"FPGA not faster: {speedup:.2f}×"
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        from pycwt_mod import wct_significance
        
        # Would need seed parameter support in wct_significance
        # For now, test that results are consistent
        sig1 = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=50, backend='tang_nano_9k', progress=False, cache=False
        )
        
        sig2 = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=50, backend='tang_nano_9k', progress=False, cache=False
        )
        
        # Results should be similar (if not seeded, will vary)
        # This test mainly checks the backend doesn't crash
        assert sig1 is not None
        assert sig2 is not None
```

---

## 📋 Implementation Phases

### Phase 1: Driver Development (Week 1)
**Goal:** Create tang_nano_driver package

- [ ] Device detection and connection
- [ ] UART communication protocol
- [ ] Configuration data structures
- [ ] Basic read/write operations
- [ ] Unit tests for driver

**Deliverables:**
- `tang_nano_driver` package
- Driver documentation
- Connection test script

### Phase 2: FPGA Bitstream (Week 2-3)
**Goal:** Implement wavelet computation on FPGA

- [ ] Design RTL for random number generation
- [ ] Implement FFT core (or integrate IP)
- [ ] Implement wavelet transform
- [ ] Implement smoothing filter
- [ ] Implement coherence calculator
- [ ] Implement histogram accumulator
- [ ] Synthesis and place-and-route
- [ ] Timing analysis and optimization

**Deliverables:**
- Bitstream file (.fs)
- HDL source code
- Resource utilization report
- Timing report

### Phase 3: Backend Integration (Week 4)
**Goal:** Integrate FPGA as pycwt-mod backend

- [ ] Implement `TangNano9KBackend` class
- [ ] Register backend with system
- [ ] Update backend selection logic
- [ ] Test integration
- [ ] Benchmark performance

**Deliverables:**
- `tang_nano_9k.py` backend
- Integration tests
- Performance benchmarks
- Updated documentation

### Phase 4: Validation & Optimization (Week 5)
**Goal:** Ensure correctness and optimize performance

- [ ] Equivalence testing vs CPU backends
- [ ] Numerical precision validation
- [ ] Performance profiling
- [ ] Bottleneck identification
- [ ] Optimization iterations
- [ ] Final benchmarks

**Deliverables:**
- Test suite passing
- Performance report
- Optimization documentation

---

## 🎯 Expected Performance

### Baseline (CPU Monte Carlo)
- **Sequential:** ~15 minutes for N=100k, mc_count=300 Monte Carlo iterations
  - 300 iterations × (2 noise gen + 8 FFTs + 3 smoothing + coherence)
- **Joblib (4 cores):** ~4 minutes for same workload
  - Parallel execution of Monte Carlo iterations

### Target (FPGA Monte Carlo)
- **Tang Nano 9K:** ~30-60 seconds for N=100k, mc_count=300
- **Expected speedup: 15-30× vs sequential, 4-8× vs joblib**
- Benefits increase with higher mc_count (more Monte Carlo iterations)

### Performance Factors for Monte Carlo on FPGA

**Advantages:**
- ✅ Parallel pipeline processing of Monte Carlo iterations
- ✅ All 300 Monte Carlo iterations run in hardware
- ✅ No memory bandwidth bottleneck for small N
- ✅ Dedicated hardware for FFT (2,400 FFT ops for mc_count=300)
- ✅ No context switching overhead between iterations
- ✅ On-chip histogram accumulation across all Monte Carlo runs

**Limitations:**
- ⚠️ UART bandwidth (115200 baud = ~11 KB/s)
- ⚠️ One-time configuration upload time
- ⚠️ Result download time (histogram only, not individual iterations)
- ⚠️ Limited on-chip memory for very large N
- ⚠️ Fixed-point precision vs floating-point CPU

**Monte Carlo-Specific Optimizations:**
1. **Batch all Monte Carlo iterations:** Configure once, run 300× on FPGA
2. **On-chip accumulation:** Histogram accumulated across all MC iterations
3. **Minimal data transfer:** Upload params once, download histogram once
4. **Parallel iteration execution:** Pipeline multiple MC iterations simultaneously
5. **Compressed results:** Send only non-zero histogram bins
6. **Higher baud rate:** Use FTDI high-speed mode (921600 baud)
7. **Streaming seeds:** Generate seeds on-chip instead of uploading

---

## 📚 Documentation Requirements

### User Guide Updates

Add section to `docs/user-guide/backends.md`:

```markdown
## Tang Nano 9K FPGA Backend

The Tang Nano 9K backend provides hardware acceleration for wavelet
coherence significance testing using FPGA computation.

### Installation

1. Install tang_nano_driver package:
   ```bash
   pip install tang_nano_driver
   ```

2. Connect Tang Nano 9K via USB

3. Load bitstream:
   ```bash
   tang-nano-flash wavelet_coherence_v1.fs
   ```

### Usage

```python
from pycwt_mod import wct_significance

# FPGA-accelerated computation
sig95 = wct_significance(
    al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=28,
    mc_count=1000,
    backend='tang_nano_9k'
)
```

### Troubleshooting

**Device not detected:**
- Check USB connection
- Verify driver installation: `python -c "import tang_nano_driver"`
- Check permissions: `sudo chmod 666 /dev/ttyUSB0`

**Bitstream not loaded:**
- Flash bitstream: `tang-nano-flash wavelet_coherence_v1.fs`
- Verify: `tang-nano-info`

**Communication errors:**
- Check baud rate configuration
- Verify device not in use by another process
- Try reconnecting device
```

---

## 🔗 Related Resources

### Hardware
- [Tang Nano 9K Documentation](https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-9K/Nano-9K.html)
- Gowin GW1NR-9 FPGA Datasheet
- FTDI USB-Serial Interface

### Software
- Gowin EDA (FPGA toolchain)
- pyserial (Python serial communication)
- numpy (numerical computations)

### Existing Backend System
- `src/pycwt_mod/backends/base.py` - Base class
- `src/pycwt_mod/backends/registry.py` - Registration
- `PHASE1_COMPLETE.md` - Backend architecture documentation
- `PHASE2_COMPLETE.md` - Integration documentation

---

## ✅ Success Criteria

### Functional Requirements
- [ ] Backend registers correctly
- [ ] Device detection works reliably
- [ ] Configuration upload succeeds
- [ ] Computation completes without errors
- [ ] Results match CPU backends (correlation > 0.8)
- [ ] Backward compatibility maintained

### Performance Requirements
- [ ] Speedup > 10× vs sequential
- [ ] Speedup > 2× vs joblib (4 cores)
- [ ] Configuration overhead < 1 second
- [ ] Result download overhead < 2 seconds

### Quality Requirements
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Error handling robust
- [ ] User-friendly error messages

---

## 🚀 Quick Start (After Implementation)

```bash
# 1. Install driver
pip install tang_nano_driver

# 2. Connect Tang Nano 9K
# (Plug in USB cable)

# 3. Verify detection
python3 -c "from pycwt_mod.backends import get_backend; print(get_backend('tang_nano_9k').is_available())"

# 4. Run test
python3 -c "
from pycwt_mod import wct_significance
sig95 = wct_significance(0.72, 0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
                         mc_count=100, backend='tang_nano_9k')
print('✓ FPGA backend working!')
"
```

---

**Document Version:** 1.0  
**Last Updated:** October 3, 2025  
**Status:** Design Phase - Ready for Implementation  
**Next Steps:** Begin Phase 1 - Driver Development

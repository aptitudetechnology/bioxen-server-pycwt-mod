# ELM11 Lua Backend Plugin for pycwt-mod Monte Carlo

**Project:** pycwt-mod - Modular Python Continuous Wavelet Transform Library  
**Feature:** Embedded Hardware-Accelerated Monte Carlo Backend using ELM11 with Lua  
**Target:** Monte Carlo Wavelet Coherence Significance Testing  
**Date:** October 3, 2025  
**Status:** Design Phase - Plugin Integration  
**Prerequisites:** Phase 1 (Backend Architecture) + Phase 2 (Integration) Complete

---

## 🔍 Quick Summary: ELM11 vs Tang Nano 9K

This backend differs from the Tang Nano 9K FPGA backend in key ways:

| Aspect | Tang Nano 9K (FPGA) | ELM11 (Lua) |
|--------|---------------------|-------------|
| **Approach** | Hardware pipeline (parallel) | Software scripting (sequential) |
| **Programming** | Verilog/VHDL | Lua (high-level) |
| **Development** | Complex, slow iteration | Simple, rapid prototyping |
| **Performance** | 15-30× speedup | 1.5-3× speedup |
| **Power** | Medium (1-2W) | Low (<0.5W) |
| **Best For** | Maximum performance | Embedded, low power, flexible |
| **Existing Code** | Build from scratch | **Leverage ELM11-Lua-FFT** |

**Key Advantage:** The existing `ELM11-Lua-FFT/` project provides:
- ✅ Serial communication (`elm11_interface.py`)
- ✅ Lua REPL interface
- ✅ Signal generation functions
- ✅ FFT framework
- ✅ Testing infrastructure (`shim_interface.py`)

**Strategy:** **Wrap and extend** existing infrastructure rather than building from scratch.

---

## 🎯 Objective

Integrate the ELM11 embedded device as a Lua-scriptable hardware-accelerated backend plugin specifically for **Monte Carlo simulations** in pycwt-mod's wavelet coherence significance testing (`wct_significance()` function). This will leverage the existing modular backend architecture to provide embedded hardware acceleration through Lua scripting as a drop-in replacement for CPU-based Monte Carlo backends.

### What Gets Accelerated

The ELM11 backend accelerates the **Monte Carlo loop** in `wct_significance()`:
- **300 Monte Carlo iterations** (default, configurable)
- Each iteration: Generate red noise → FFT → Wavelet transform → Smooth → Compute coherence → Update histogram
- **2,400 FFT operations** per typical run (300 iterations × 8 FFTs)
- This is the primary bottleneck identified in performance analysis

### ELM11 Advantages

- **Lua Scripting:** High-level programming on embedded hardware
- **FFT Library:** Pre-built Lua FFT functions available
- **Low Power:** Embedded device suitable for long-running Monte Carlo
- **Flexibility:** Lua scripting allows rapid prototyping and debugging
- **Python Integration:** Easy communication via serial/network protocols

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
│   ├── tang_nano_9k.py     🔲 FPGA Monte Carlo (planned)
│   └── elm11_lua.py        🆕 ELM11 Lua Monte Carlo (NEW)
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

## 🔧 ELM11 Lua Backend Design

### ELM11 Device Overview

Based on the existing ELM11-Lua-FFT project:
- **Hardware Interface:** Serial (USB/UART) at 115200 baud
- **Lua Runtime:** Native Lua REPL on microcontroller
- **FFT Support:** Lua-based FFT implementation (can be enhanced)
- **Signal Processing:** Already has signal generation and basic FFT
- **Communication:** Python PC-side interface via `elm11_interface.py`

### Backend Class Structure

```python
# File: src/pycwt_mod/backends/elm11_lua.py

from .base import MonteCarloBackend
import numpy as np
import serial
import time

class ELM11LuaBackend(MonteCarloBackend):
    """
    ELM11 Lua-scriptable embedded Monte Carlo backend.
    
    This backend offloads Monte Carlo wavelet coherence simulations
    to the ELM11 embedded device running Lua scripts. The Monte Carlo
    workflow is implemented in Lua and executed on the embedded hardware,
    with Python handling coordination and result collection.
    
    What Gets Accelerated:
    - Monte Carlo iterations (typically 300)
    - Red noise generation (2 per iteration) - Lua script
    - FFT operations (8 per iteration = 2,400 total) - Lua FFT library
    - Wavelet transforms - Lua implementation
    - Spectral smoothing - Lua convolution
    - Coherence calculation - Lua script
    - Histogram accumulation - Lua arrays
    
    Hardware Requirements:
    - ELM11 embedded device
    - Serial/Network connection (USB, UART, or Ethernet)
    - Lua runtime environment
    - Sufficient memory for signal processing
    
    Software Requirements:
    - pyserial (for serial communication)
    - elm11_lua_driver (wrapper around existing elm11_interface.py)
    - Lua scripts for Monte Carlo workflow (build on existing fourier/ code)
    """
    
    def __init__(self, name='elm11_lua', config=None):
        super().__init__(name, config)
        self._device = None  # Serial connection
        self._lua_scripts_loaded = False
        # Connection parameters from existing elm11_interface.py
        self._baud_rate = 115200
        self._timeout = 2
    
    def is_available(self):
        """
        Check if ELM11 device is available.
        
        Returns:
            bool: True if device is connected and Lua scripts are loaded
        """
        try:
            # Check for required driver
            import elm11_lua_driver
            
            # Try to detect device
            device = elm11_lua_driver.detect_elm11()
            if device is None:
                return False
            
            # Check if Lua scripts are loaded
            if not elm11_lua_driver.check_lua_scripts(device):
                return False
            
            # Verify Lua FFT library available
            if not elm11_lua_driver.verify_fft_library(device):
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
        Execute Monte Carlo simulations on ELM11 using Lua scripts.
        
        The ELM11 executes the complete Monte Carlo workflow in Lua:
        1. Initialize Lua environment with parameters
        2. Load FFT library and wavelet functions
        3. For each simulation:
           - Generate AR(1) red noise (Lua RNG)
           - Compute FFT (Lua FFT library)
           - Compute wavelet transform (Lua script)
           - Smooth spectra (Lua convolution)
           - Calculate coherence (Lua computation)
           - Update histogram bins (Lua arrays)
        4. Return accumulated histogram results
        
        Parameters:
        -----------
        worker_func : callable
            Worker function (_wct_significance_worker) - function signature
            used to extract parameters, actual computation done in Lua on ELM11
        n_simulations : int
            Number of Monte Carlo simulations to run on ELM11
        worker_args : tuple
            Arguments for worker function (al1, al2, N, dt, dj, s0, J, ...)
        seed : int, optional
            Random seed for reproducibility of Monte Carlo
        verbose : bool
            Show progress during ELM11 computation
        
        Returns:
        --------
        list
            Results from each simulation (wlc histogram arrays)
        """
        if not self.is_available():
            raise RuntimeError(
                "ELM11 Lua backend not available. "
                "Check device connection and Lua scripts."
            )
        
        # Import ELM11 driver
        import elm11_lua_driver as eld
        
        # Connect to device
        if self._device is None:
            self._device = eld.connect()
        
        # Extract parameters from worker_args
        # These need to match _wct_significance_worker signature
        (al1, al2, N, dt, dj, s0, J, wavelet, 
         sj, scales, outsidecoi, maxscale, nbins) = worker_args
        
        # Prepare Lua script configuration
        lua_config = eld.LuaMonteCarloConfig(
            al1=al1,
            al2=al2,
            N=N,
            dt=dt,
            dj=dj,
            s0=s0,
            J=J,
            wavelet_type=wavelet.name,
            scales=scales.tolist(),
            maxscale=maxscale,
            nbins=nbins,
            n_simulations=n_simulations,
            seed=seed if seed is not None else np.random.randint(0, 2**31)
        )
        
        # Upload configuration and Lua scripts
        if verbose:
            print(f"Uploading configuration to ELM11...")
        
        eld.upload_config(self._device, lua_config)
        
        # Upload Lua Monte Carlo script if not already loaded
        if not self._lua_scripts_loaded:
            if verbose:
                print("Loading Lua Monte Carlo scripts on ELM11...")
            eld.upload_lua_scripts(self._device, [
                'rednoise.lua',      # AR(1) red noise generation
                'fft_wrapper.lua',   # FFT operations
                'wavelet.lua',       # Wavelet transforms
                'smoothing.lua',     # Spectral smoothing
                'coherence.lua',     # Coherence calculation
                'monte_carlo.lua'    # Main Monte Carlo loop
            ])
            self._lua_scripts_loaded = True
        
        # Start computation
        if verbose:
            print(f"Running {n_simulations} Monte Carlo simulations on ELM11 (Lua)...")
        
        eld.start_lua_execution(self._device)
        
        # Wait for completion with optional progress
        if verbose:
            from tqdm import tqdm
            with tqdm(total=n_simulations, desc="ELM11 Lua Progress") as pbar:
                while not eld.is_complete(self._device):
                    progress = eld.get_progress(self._device)
                    pbar.update(progress - pbar.n)
                    import time
                    time.sleep(0.1)  # Poll every 100ms
                pbar.update(n_simulations - pbar.n)
        else:
            eld.wait_for_completion(self._device)
        
        # Read results from ELM11
        if verbose:
            print("Downloading results from ELM11...")
        
        results = eld.read_results(self._device, n_simulations)
        
        # Convert ELM11 Lua output format to expected format
        # ELM11 returns aggregated histogram, split into individual results
        # (or return as single aggregated result depending on implementation)
        elm11_results = []
        for i in range(n_simulations):
            wlc_simulation = results[i]  # Shape: [J+1, nbins]
            elm11_results.append(wlc_simulation)
        
        return elm11_results
    
    def __del__(self):
        """Cleanup ELM11 connection."""
        if self._device is not None:
            try:
                import elm11_lua_driver as eld
                eld.disconnect(self._device)
            except:
                pass
```

---

## 🔌 Integration with Existing System

### 1. Backend Registration

The ELM11 Lua backend auto-registers through the existing system:

**File: `src/pycwt_mod/backends/__init__.py`**

```python
# Existing code...
from .sequential import SequentialBackend
from .joblib import JoblibBackend

# Add ELM11 Lua import
try:
    from .elm11_lua import ELM11LuaBackend
    _ELM11_AVAILABLE = True
except ImportError:
    _ELM11_AVAILABLE = False

def _register_builtin_backends():
    """Register all built-in backends."""
    # Existing registrations...
    register_backend('sequential', SequentialBackend)
    register_backend('joblib', JoblibBackend)
    
    # Register ELM11 Lua if available
    if _ELM11_AVAILABLE:
        register_backend('elm11_lua', ELM11LuaBackend)

# Auto-register at import
_register_builtin_backends()
```

### 2. Usage in wct_significance

No changes needed! The existing implementation automatically supports the new backend:

```python
from pycwt_mod import wct_significance

# Auto-select (will use ELM11 if available and best choice)
sig95 = wct_significance(
    al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=28,
    mc_count=1000
)

# Or explicitly request ELM11 Lua backend
sig95 = wct_significance(
    al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=28,
    mc_count=1000,
    backend='elm11_lua'
)
```

### 3. Backend Selection Logic

Update recommendation function to consider ELM11 Lua:

**File: `src/pycwt_mod/backends/registry.py`**

```python
def get_recommended_backend(n_simulations=100):
    """
    Recommend optimal backend based on problem size.
    
    Priority order:
    1. tang_nano_9k (if available, for large sizes)
    2. elm11_lua (if available, for medium+ sizes, low power)
    3. joblib (if available, for n >= 100)
    4. sequential (always available)
    """
    # Check Tang Nano 9K first (best raw performance)
    if 'tang_nano_9k' in _registry:
        backend = _registry['tang_nano_9k']()
        if backend.is_available() and n_simulations >= 200:
            return 'tang_nano_9k'
    
    # Check ELM11 Lua (good for medium workloads, flexible)
    if 'elm11_lua' in _registry:
        backend = _registry['elm11_lua']()
        if backend.is_available() and n_simulations >= 100:
            return 'elm11_lua'
    
    # Existing logic for joblib/sequential
    if n_simulations >= 100 and 'joblib' in _registry:
        backend = _registry['joblib']()
        if backend.is_available():
            return 'joblib'
    
    return 'sequential'
```

---

## 📦 ELM11 Lua Driver Package

The backend requires a companion driver package for device communication and Lua script management.

### Driver Package Structure

```
elm11_lua_driver/
├── __init__.py
├── device.py           # Device detection and connection
│                       # (Wraps existing elm11_interface.py functionality)
├── protocol.py         # Serial communication protocol
├── lua_manager.py      # Lua script upload and management
├── config.py           # Configuration structures
├── monte_carlo.py      # Monte Carlo-specific operations
└── lua_scripts/        # Lua scripts for ELM11
    ├── init.lua        # Core functions (EXISTING from ELM11-Lua-FFT)
    ├── rednoise.lua    # AR(1) red noise generation (NEW)
    ├── fft_wrapper.lua # FFT library wrapper (BUILD ON EXISTING)
    ├── wavelet.lua     # Wavelet transform functions (NEW)
    ├── smoothing.lua   # Spectral smoothing (NEW)
    ├── coherence.lua   # Coherence calculation (NEW)
    └── monte_carlo.lua # Main Monte Carlo loop (NEW)

Note: The existing ELM11-Lua-FFT project already provides:
- Serial communication (elm11_interface.py)
- Basic signal generation functions (fourier/init.lua)
- FFT framework (fourier/fourier_main.lua)
- Testing infrastructure (shim_interface.py)
```

### Driver API Design

```python
# elm11_lua_driver/__init__.py

from .device import (
    detect_elm11,
    connect,
    disconnect,
    check_lua_scripts,
    verify_fft_library
)

from .config import LuaMonteCarloConfig

from .lua_manager import upload_lua_scripts

from .monte_carlo import (
    upload_config,
    start_lua_execution,
    is_complete,
    get_progress,
    wait_for_completion,
    read_results
)
```

### Example Driver Implementation

```python
# elm11_lua_driver/device.py
# Wraps existing elm11_interface.py functionality

import serial
import serial.tools.list_ports
import glob
import time

# Serial configuration from existing elm11_interface.py
SERIAL_PORTS = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
if not SERIAL_PORTS:
    SERIAL_PORTS = ['/dev/ttyUSB0']
BAUD_RATES = [115200, 9600, 19200, 38400, 57600]
TIMEOUT = 2

def detect_elm11():
    """
    Detect ELM11 device on serial.
    
    Based on existing elm11_interface.py connect_serial() function.
    
    Returns:
        dict: Connection info {'type': 'serial', 'address': ..., 'baud': ...}
              or None if not found
    """
    # Try each port and baud rate combination
    for port in SERIAL_PORTS:
        for baud in BAUD_RATES:
            try:
                ser = serial.Serial(port, baud, timeout=TIMEOUT)
                
                # Wait for ELM11 to be ready
                time.sleep(2)
                
                # Clear any pending data
                ser.read(1024)
                
                # Try to get into REPL mode
                ser.write(b'q\r\n')  # Exit any listing mode
                ser.flush()
                time.sleep(0.5)
                ser.read(256)
                
                # Send newline to get prompt
                ser.write(b'\r\n')
                ser.flush()
                time.sleep(0.5)
                response = ser.read(256)
                
                ser.close()
                
                # If we got any response, assume it's ELM11
                if response:
                    return {
                        'type': 'serial',
                        'address': port,
                        'baud': baud
                    }
            except Exception as e:
                continue
    
    return None

def connect(device_info=None):
    """
    Connect to ELM11.
    
    Parameters:
        device_info : dict, optional
            Device connection info (auto-detect if None)
    
    Returns:
        object: Connected device (Serial or Socket)
    """
    if device_info is None:
        device_info = detect_elm11()
    
    if device_info is None:
        raise RuntimeError("ELM11 device not found")
    
    if device_info['type'] == 'serial':
        device = serial.Serial(
            device_info['address'],
            baudrate=115200,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=5
        )
    elif device_info['type'] == 'network':
        device = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device.settimeout(5)
        device.connect((device_info['address'], device_info['port']))
    else:
        raise ValueError(f"Unknown device type: {device_info['type']}")
    
    return device

def disconnect(device):
    """Close connection to device."""
    if device:
        try:
            device.close()
        except:
            pass

def check_lua_scripts(device):
    """
    Check if Lua Monte Carlo scripts are loaded on device.
    
    Returns:
        bool: True if scripts are available
    """
    if isinstance(device, serial.Serial):
        device.write(b'lua_scripts_loaded?\n')
        response = device.readline()
        return b'YES' in response
    else:  # Socket
        device.send(b'lua_scripts_loaded?\n')
        response = device.recv(1024)
        return b'YES' in response

def verify_fft_library(device):
    """
    Verify Lua FFT library is available.
    
    Returns:
        bool: True if FFT library is functional
    """
    if isinstance(device, serial.Serial):
        device.write(b'test_fft\n')
        response = device.readline()
        return b'FFT_OK' in response
    else:  # Socket
        device.send(b'test_fft\n')
        response = device.recv(1024)
        return b'FFT_OK' in response
```

---

## � Leveraging Existing ELM11-Lua-FFT Infrastructure

The existing `ELM11-Lua-FFT/` project provides a solid foundation for this backend:

### Existing Components to Reuse

**1. Serial Communication** (`ELM11-Lua-FFT/elm11_interface.py`)
- Already handles serial port discovery (`/dev/ttyUSB*`, `/dev/ttyACM*`)
- Implements multiple baud rate attempts (115200, 9600, 19200, 38400, 57600)
- Provides REPL mode detection and interaction
- Handles Lua code upload with proper timing

**2. Testing Infrastructure** (`ELM11-Lua-FFT/shim_interface.py`)
- Dual-mode testing (Python NumPy vs Lua execution)
- Useful for validating Lua implementations against Python reference
- Can be adapted for Monte Carlo validation

**3. Signal Generation** (`ELM11-Lua-FFT/fourier/init.lua`)
- Already implements:
  - `sine()`, `square()`, `sawtooth()`, `triangle()` waveforms
  - Sample rate: 48000 Hz
  - Buffer size: 1024
  - FFT size: 512
- Can extend with red noise generation for Monte Carlo

**4. FFT Framework** (`ELM11-Lua-FFT/fourier/fourier_main.lua`)
- Basic FFT computation already in place
- LÖVE2D visualization (useful for debugging)

### Integration Strategy

Instead of building from scratch, **wrap and extend** existing functionality:

```python
# elm11_lua_driver/device.py
# Import and extend existing elm11_interface functionality

import sys
import os

# Add ELM11-Lua-FFT to path
ELM11_FFT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ELM11-Lua-FFT')
sys.path.insert(0, ELM11_FFT_PATH)

# Import existing ELM11 interface
try:
    import elm11_interface as elm11_base
    HAS_ELM11_BASE = True
except ImportError:
    HAS_ELM11_BASE = False
    print("Warning: ELM11-Lua-FFT base interface not found")

def connect(device_info=None):
    """
    Connect to ELM11 using existing elm11_interface.
    
    Wraps elm11_base.connect_serial() functionality.
    """
    if HAS_ELM11_BASE:
        # Use existing connection logic
        device = elm11_base.connect_serial()
        if device is None:
            raise RuntimeError("ELM11 device not found")
        return device
    else:
        # Fallback to standalone implementation
        if device_info is None:
            device_info = detect_elm11()
        # ... (standalone implementation as shown earlier)
```

### Lua Script Extension Plan

**Build on existing `fourier/init.lua`:**

```lua
-- Extend existing signal generation with red noise
-- Add to ELM11-Lua-FFT/fourier/init.lua or create monte_carlo/rednoise.lua

-- Existing functions in init.lua:
-- sine(freq, amplitude, phase, length)
-- square(freq, amplitude, phase, length)
-- sawtooth(freq, amplitude, phase, length)
-- triangle(freq, amplitude, phase, length)

-- NEW: Red noise generation for Monte Carlo
function rednoise(N, alpha, seed)
    -- AR(1) process: y[n] = alpha * y[n-1] + sqrt(1 - alpha^2) * noise[n]
    math.randomseed(seed or os.time())
    
    local signal = {}
    signal[1] = math.random() - 0.5
    
    local scale = math.sqrt(1 - alpha * alpha)
    for i = 2, N do
        local noise = math.random() - 0.5
        signal[i] = alpha * signal[i-1] + scale * noise
    end
    
    return signal
end
```

### Testing with Shim Interface

The existing `shim_interface.py` can validate Lua implementations:

```python
# Test red noise generation: Python vs Lua
from ELM11_Lua_FFT import shim_interface

# Test in Python mode (NumPy reference)
shim = shim_interface.ShimInterface(mode='python')
python_noise = shim.generate_rednoise(N=1000, alpha=0.72, seed=42)

# Test in Lua mode (ELM11 execution)
shim = shim_interface.ShimInterface(mode='lua')
lua_noise = shim.generate_rednoise(N=1000, alpha=0.72, seed=42)

# Compare results
import numpy as np
correlation = np.corrcoef(python_noise, lua_noise)[0, 1]
print(f"Python vs Lua correlation: {correlation:.4f}")
# Should be > 0.95 for deterministic seeded noise
```

---

## �📜 Lua Scripts for Monte Carlo

### Main Monte Carlo Loop

```lua
-- lua_scripts/monte_carlo.lua
-- Main Monte Carlo simulation loop for wavelet coherence

local fft = require("fft_wrapper")
local rednoise = require("rednoise")
local wavelet = require("wavelet")
local smoothing = require("smoothing")
local coherence = require("coherence")

-- Monte Carlo configuration (loaded from Python)
local config = {
    n_simulations = 300,
    al1 = 0.72,
    al2 = 0.72,
    N = 1000,
    dt = 0.25,
    dj = 0.25,
    s0 = 0.5,
    J = 28,
    wavelet_type = "morlet",
    scales = {},  -- Array of scales
    maxscale = 20,
    nbins = 1000,
    seed = 42
}

-- Initialize histogram accumulator
local wlc = {}
for j = 0, config.J do
    wlc[j] = {}
    for bin = 1, config.nbins do
        wlc[j][bin] = 0
    end
end

-- Set random seed for reproducibility
math.randomseed(config.seed)

-- Progress reporting function
local function report_progress(iteration)
    -- Send progress back to Python
    io.write(string.format("PROGRESS:%d/%d\n", iteration, config.n_simulations))
    io.flush()
end

-- Main Monte Carlo loop
for sim = 1, config.n_simulations do
    -- Generate unique seed for this simulation
    local sim_seed = config.seed + sim
    math.randomseed(sim_seed)
    
    -- 1. Generate red noise signals
    local noise1 = rednoise.generate(config.N, config.al1)
    local noise2 = rednoise.generate(config.N, config.al2)
    
    -- 2. Compute CWT (calls FFT internally)
    local nW1 = wavelet.cwt(noise1, config.dt, config.dj, config.s0, 
                           config.J, config.wavelet_type)
    local nW2 = wavelet.cwt(noise2, config.dt, config.dj, config.s0, 
                           config.J, config.wavelet_type)
    
    -- 3. Compute cross-spectrum
    local nW12 = wavelet.cross_spectrum(nW1, nW2)
    
    -- 4. Smooth spectra
    local S1 = smoothing.smooth(wavelet.power_spectrum(nW1), config.scales, 
                               config.dt, config.dj)
    local S2 = smoothing.smooth(wavelet.power_spectrum(nW2), config.scales, 
                               config.dt, config.dj)
    local S12 = smoothing.smooth(nW12, config.scales, config.dt, config.dj)
    
    -- 5. Calculate coherence
    local R2 = coherence.compute(S12, S1, S2)
    
    -- 6. Update histogram
    for s = 0, config.maxscale - 1 do
        for t = 1, config.N do
            if R2[s][t] then  -- Check if inside COI
                local bin = math.floor(R2[s][t] * config.nbins) + 1
                if bin >= 1 and bin <= config.nbins then
                    wlc[s][bin] = wlc[s][bin] + 1
                end
            end
        end
    end
    
    -- Report progress every 10 simulations
    if sim % 10 == 0 then
        report_progress(sim)
    end
end

-- Final progress
report_progress(config.n_simulations)

-- Return results
return wlc
```

### FFT Wrapper

```lua
-- lua_scripts/fft_wrapper.lua
-- Wrapper for Lua FFT library (assuming ELM11 has FFT support)

local fft = {}

-- FFT implementation (uses ELM11's built-in FFT if available)
function fft.fft(signal)
    -- Call native FFT function
    -- This assumes ELM11 has a C-based FFT library accessible from Lua
    return native_fft(signal)  -- Placeholder
end

function fft.ifft(spectrum)
    -- Inverse FFT
    return native_ifft(spectrum)  -- Placeholder
end

function fft.fftfreq(n, dt)
    -- Compute FFT frequencies
    local freq = {}
    local f = 1.0 / (n * dt)
    for i = 0, n/2 do
        freq[i+1] = i * f
    end
    for i = n/2+1, n-1 do
        freq[i+1] = (i - n) * f
    end
    return freq
end

return fft
```

### Red Noise Generation

```lua
-- lua_scripts/rednoise.lua
-- AR(1) red noise generation

local rednoise = {}

function rednoise.generate(N, alpha)
    -- Generate AR(1) red noise with lag-1 autocorrelation alpha
    local noise = {}
    
    -- Start with white noise
    noise[1] = math.random() - 0.5
    
    -- AR(1) process: x[n] = alpha * x[n-1] + sqrt(1-alpha^2) * white_noise
    local scale = math.sqrt(1 - alpha * alpha)
    
    for i = 2, N do
        local white = (math.random() - 0.5) * 2  -- Uniform(-1, 1)
        noise[i] = alpha * noise[i-1] + scale * white
    end
    
    return noise
end

return rednoise
```

---

## 🧪 Testing Strategy

### Leveraging Existing Test Infrastructure

The existing `ELM11-Lua-FFT/shim_interface.py` provides excellent dual-mode testing:

```python
# Validate Lua implementations against Python reference
from ELM11_Lua_FFT import shim_interface

# Strategy: Test each Lua component in isolation before integration

# 1. Test red noise generation
shim_py = shim_interface.ShimInterface(mode='python')
shim_lua = shim_interface.ShimInterface(mode='lua')

noise_py = shim_py.generate_rednoise(N=1000, alpha=0.72, seed=42)
noise_lua = shim_lua.generate_rednoise(N=1000, alpha=0.72, seed=42)

import numpy as np
correlation = np.corrcoef(noise_py, noise_lua)[0, 1]
print(f"Red noise Python vs Lua: {correlation:.4f}")  # Should be > 0.95

# 2. Test FFT computation
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
fft_py = shim_py.compute_fft(signal)
fft_lua = shim_lua.compute_fft(signal)

# Compare magnitudes
mag_error = np.mean(np.abs(np.abs(fft_py) - np.abs(fft_lua)))
print(f"FFT magnitude error: {mag_error:.6f}")  # Should be < 1e-5

# 3. Test wavelet transform
# ... similar pattern for each component
```

### Test File Structure

```python
# src/pycwt_mod/tests/backends/test_elm11_lua.py

import pytest
import numpy as np
from pycwt_mod.backends import get_backend

@pytest.mark.skipif(
    not get_backend('elm11_lua').is_available(),
    reason="ELM11 Lua device not available"
)
class TestELM11LuaBackend:
    """Tests for ELM11 Lua Monte Carlo backend."""
    
    def test_backend_available(self):
        """Test that backend can be instantiated."""
        backend = get_backend('elm11_lua')
        assert backend.is_available()
    
    def test_device_connection(self):
        """Test device connection and communication."""
        import elm11_lua_driver as eld
        
        device_info = eld.detect_elm11()
        assert device_info is not None
        
        device = eld.connect(device_info)
        assert device is not None
        
        eld.disconnect(device)
    
    def test_lua_scripts_loaded(self):
        """Test that Lua scripts are loaded on device."""
        import elm11_lua_driver as eld
        
        device = eld.connect()
        assert eld.check_lua_scripts(device)
        assert eld.verify_fft_library(device)
        eld.disconnect(device)
    
    def test_elm11_vs_sequential_equivalence(self):
        """Test ELM11 Lua produces equivalent results to sequential."""
        from pycwt_mod import wct_significance
        
        # Sequential
        sig_seq = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=50, backend='sequential', progress=False, cache=False
        )
        
        # ELM11 Lua
        sig_elm11 = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=50, backend='elm11_lua', progress=False, cache=False
        )
        
        # Should be highly correlated (stochastic variation expected)
        valid_seq = sig_seq[~np.isnan(sig_seq)]
        valid_elm11 = sig_elm11[~np.isnan(sig_elm11)]
        
        correlation = np.corrcoef(valid_seq, valid_elm11)[0, 1]
        assert correlation > 0.8, f"Low correlation: {correlation}"
    
    @pytest.mark.slow
    def test_elm11_performance(self):
        """Test ELM11 Lua performance vs CPU."""
        import time
        from pycwt_mod import wct_significance
        
        # Sequential
        start = time.time()
        wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=20,
            mc_count=300, backend='sequential', progress=False, cache=False
        )
        seq_time = time.time() - start
        
        # ELM11 Lua
        start = time.time()
        wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=20,
            mc_count=300, backend='elm11_lua', progress=False, cache=False
        )
        elm11_time = time.time() - start
        
        speedup = seq_time / elm11_time
        print(f"\nELM11 Lua Speedup: {speedup:.2f}×")
        
        # ELM11 should provide some speedup
        assert speedup > 0.5, f"ELM11 unexpectedly slow: {speedup:.2f}×"
    
    def test_lua_script_execution(self):
        """Test basic Lua script execution on device."""
        import elm11_lua_driver as eld
        
        device = eld.connect()
        
        # Test simple Lua script
        test_script = """
        return 42
        """
        result = eld.execute_lua(device, test_script)
        assert result == 42
        
        eld.disconnect(device)
    
    def test_progress_reporting(self):
        """Test that progress reporting works."""
        from pycwt_mod import wct_significance
        
        # This should report progress without crashing
        sig = wct_significance(
            al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
            mc_count=30, backend='elm11_lua', progress=True, cache=False
        )
        
        assert sig is not None
```

---

## 📋 Implementation Phases

### Phase 1: Driver Development (Week 1)
**Goal:** Create elm11_lua_driver package

- [ ] Device detection (serial and network)
- [ ] Communication protocol implementation
- [ ] Lua script upload mechanism
- [ ] Configuration data structures
- [ ] Basic read/write operations
- [ ] Unit tests for driver

**Deliverables:**
- `elm11_lua_driver` package
- Driver documentation
- Connection test script

### Phase 2: Lua Script Development (Week 2)
**Goal:** Implement Monte Carlo workflow in Lua

- [ ] Red noise generation (rednoise.lua)
- [ ] FFT wrapper (fft_wrapper.lua)
- [ ] Wavelet transform (wavelet.lua)
- [ ] Smoothing functions (smoothing.lua)
- [ ] Coherence calculator (coherence.lua)
- [ ] Main Monte Carlo loop (monte_carlo.lua)
- [ ] Test scripts on ELM11 device

**Deliverables:**
- Complete Lua script library
- Individual script tests
- Integration test

### Phase 3: Backend Integration (Week 3)
**Goal:** Integrate ELM11 Lua as pycwt-mod backend

- [ ] Implement `ELM11LuaBackend` class
- [ ] Register backend with system
- [ ] Update backend selection logic
- [ ] Test integration with wct_significance
- [ ] Benchmark performance

**Deliverables:**
- `elm11_lua.py` backend
- Integration tests
- Performance benchmarks
- Updated documentation

### Phase 4: Optimization & Validation (Week 4)
**Goal:** Ensure correctness and optimize performance

- [ ] Equivalence testing vs CPU backends
- [ ] Numerical precision validation
- [ ] Performance profiling (Python ↔ Lua overhead)
- [ ] Bottleneck identification
- [ ] Lua script optimization
- [ ] Final benchmarks

**Deliverables:**
- Test suite passing
- Performance report
- Optimization documentation

---

## 🎯 Expected Performance

### Baseline (CPU Monte Carlo)
- **Sequential:** ~15 minutes for N=100k, mc_count=300 Monte Carlo iterations
- **Joblib (4 cores):** ~4 minutes for same workload

### Target (ELM11 Lua Monte Carlo)
- **ELM11 Lua:** ~5-10 minutes for N=100k, mc_count=300
- **Expected speedup: 1.5-3× vs sequential**
- Benefits: Low power consumption, embedded deployment

### Performance Factors for Monte Carlo on ELM11

**Advantages:**
- ✅ Offload computation from host CPU
- ✅ Low power embedded execution
- ✅ Lua scripting flexibility for rapid optimization
- ✅ Native FFT library (if available on ELM11)
- ✅ Continuous operation without host intervention
- ✅ Good for long-running Monte Carlo experiments

**Limitations:**
- ⚠️ Lua interpreter overhead vs compiled code
- ⚠️ Communication bandwidth (serial/network)
- ⚠️ Configuration upload time
- ⚠️ Limited memory compared to PC
- ⚠️ Single-threaded Lua execution

**Monte Carlo-Specific Optimizations:**
1. **Batch all MC iterations:** Upload config once, run 300× on ELM11
2. **On-device accumulation:** Histogram accumulated in Lua
3. **Minimal data transfer:** Upload params once, download results once
4. **Lua JIT:** Use LuaJIT if available for better performance
5. **Compressed results:** Send only non-zero histogram bins
6. **Streaming results:** Stream partial results during computation
7. **Optimized Lua:** Profile and optimize Lua scripts

### Performance Comparison

| Backend        | Time (mc=300) | Speedup | Power  | Notes                      |
|----------------|---------------|---------|--------|----------------------------|
| Sequential CPU | 15 min        | 1.0×    | High   | Baseline                   |
| Joblib (4 CPU) | 4 min         | 3.75×   | High   | Best CPU performance       |
| ELM11 Lua      | 5-10 min      | 1.5-3×  | Low    | Embedded, scriptable       |
| Tang Nano 9K   | 0.5-1 min     | 15-30×  | Medium | Hardware pipeline (planned)|

---

## 📚 Documentation Requirements

### User Guide Updates

Add section to `docs/user-guide/backends.md`:

```markdown
## ELM11 Lua Backend

The ELM11 Lua backend provides embedded hardware acceleration for wavelet
coherence significance testing using Lua-scriptable Monte Carlo computation.

### Installation

1. Install elm11_lua_driver package:
   ```bash
   pip install elm11_lua_driver
   ```

2. Connect ELM11 device (USB serial or Ethernet)

3. Upload Lua scripts:
   ```bash
   elm11-upload-scripts
   ```

### Usage

```python
from pycwt_mod import wct_significance

# ELM11 Lua-accelerated computation
sig95 = wct_significance(
    al1=0.72, al2=0.72, dt=0.25, dj=0.25, s0=0.5, J=28,
    mc_count=1000,
    backend='elm11_lua'
)
```

### Advantages
- **Low Power:** Run long Monte Carlo simulations on embedded hardware
- **Flexible:** Modify Lua scripts for custom workflows
- **Offload:** Free up host CPU for other tasks
- **Portable:** Deploy to embedded systems

### Troubleshooting

**Device not detected:**
- Check USB/Ethernet connection
- Verify driver: `python -c "import elm11_lua_driver"`
- Check permissions (Linux): `sudo chmod 666 /dev/ttyUSB0`

**Lua scripts not loaded:**
- Upload scripts: `elm11-upload-scripts`
- Verify: `elm11-check-scripts`

**FFT library not found:**
- Verify ELM11 firmware includes FFT support
- Update firmware if needed

**Slow performance:**
- Check communication bandwidth
- Optimize Lua scripts
- Consider using LuaJIT if available
```

---

## 🔗 Related Resources

### Hardware
- ELM11 Device Documentation
- ELM11 Lua API Reference
- Serial/Network Communication Specs

### Software
- Lua 5.x Documentation
- LuaJIT (for performance)
- pyserial (Python serial communication)
- socket (Python network communication)

### Lua FFT Libraries
- luafft (Pure Lua FFT)
- lua-fftw (FFTW binding)
- ELM11 native FFT (if available)

### Existing Backend System
- `src/pycwt_mod/backends/base.py` - Base class
- `src/pycwt_mod/backends/registry.py` - Registration
- `PHASE1_COMPLETE.md` - Backend architecture documentation
- `PHASE2_COMPLETE.md` - Integration documentation
- `tang-nano-9k-prompt.md` - FPGA backend (reference)

---

## ✅ Success Criteria

### Functional Requirements
- [ ] Backend registers correctly
- [ ] Device detection works (serial and network)
- [ ] Lua scripts upload successfully
- [ ] Configuration transfer succeeds
- [ ] Monte Carlo computation completes without errors
- [ ] Results match CPU backends (correlation > 0.8)
- [ ] Progress reporting works
- [ ] Backward compatibility maintained

### Performance Requirements
- [ ] Speedup > 1.5× vs sequential
- [ ] Configuration overhead < 5 seconds
- [ ] Result download overhead < 5 seconds
- [ ] Lua script execution time acceptable

### Quality Requirements
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Error handling robust
- [ ] User-friendly error messages
- [ ] Lua scripts well-documented

---

## 🚀 Quick Start (After Implementation)

```bash
# 1. Install driver
pip install elm11_lua_driver

# 2. Connect ELM11
# (Plug in USB cable or connect to network)

# 3. Upload Lua scripts
elm11-upload-scripts

# 4. Verify detection
python3 -c "from pycwt_mod.backends import get_backend; print(get_backend('elm11_lua').is_available())"

# 5. Run test
python3 -c "
from pycwt_mod import wct_significance
sig95 = wct_significance(0.72, 0.72, dt=0.25, dj=0.25, s0=0.5, J=10,
                         mc_count=100, backend='elm11_lua')
print('✓ ELM11 Lua backend working!')
"
```

---

## 🔄 Development Workflow

### Recommended Development Approach

**Phase 1: Use Existing Infrastructure**

1. **Start with `shim_interface.py`** for dual-mode testing:
   ```python
   # Test Lua implementations against Python reference
   from ELM11_Lua_FFT import shim_interface
   
   # Python reference (NumPy)
   shim_py = shim_interface.ShimInterface(mode='python')
   result_py = shim_py.compute_something(...)
   
   # Lua implementation (ELM11)
   shim_lua = shim_interface.ShimInterface(mode='lua')
   result_lua = shim_lua.compute_something(...)
   
   # Compare
   assert np.allclose(result_py, result_lua, rtol=1e-5)
   ```

2. **Extend `fourier/init.lua`** with Monte Carlo functions:
   - Add red noise generation to existing signal functions
   - Build on existing FFT framework
   - Test incrementally with shim interface

3. **Wrap `elm11_interface.py`** for backend driver:
   - Don't reimplement serial communication
   - Use existing `connect_serial()`, `send_lua()`, `upload_code()`
   - Add Monte Carlo-specific operations

**Phase 2: Incremental Testing**

```python
# Test each component in isolation before integration
# 1. Red noise generation
from test_utils import test_rednoise
test_rednoise(mode='lua')  # ✓ Pass

# 2. FFT computation
from test_utils import test_fft
test_fft(mode='lua')  # ✓ Pass

# 3. Wavelet transform
from test_utils import test_wavelet
test_wavelet(mode='lua')  # ✓ Pass

# 4. Full Monte Carlo iteration
from test_utils import test_monte_carlo_iteration
test_monte_carlo_iteration(mode='lua')  # ✓ Pass

# 5. Full backend integration
pytest src/pycwt_mod/tests/backends/test_elm11_lua.py
```

**Phase 3: Optimization**

Profile Lua execution to find bottlenecks:
```lua
-- Add timing instrumentation
local start_time = os.clock()

-- ... computation ...

local end_time = os.clock()
print("Duration:", end_time - start_time)
```

Consider:
- LuaJIT for faster execution
- Minimize string operations
- Optimize array access patterns
- Reduce garbage collection overhead

### Testing Lua Scripts Directly on ELM11

```lua
-- Test script: test_rednoise.lua
local rednoise = require("rednoise")

-- Generate test noise
local noise = rednoise.generate(1000, 0.72)

-- Verify properties
local mean = 0
for i = 1, #noise do
    mean = mean + noise[i]
end
mean = mean / #noise

print("Mean:", mean)  -- Should be near 0
print("Length:", #noise)  -- Should be 1000

-- Verify autocorrelation
local autocorr = 0
for i = 2, #noise do
    autocorr = autocorr + noise[i] * noise[i-1]
end
autocorr = autocorr / (#noise - 1)

print("Autocorrelation:", autocorr)  -- Should be near 0.72
```

### Debugging Communication Issues

```python
# Test driver communication
import elm11_lua_driver as eld

device = eld.connect()

# Send test command
device.write(b'PING\n')
response = device.readline()
print(f"Response: {response}")

# Test Lua execution
result = eld.execute_lua(device, "return 2+2")
print(f"2+2 = {result}")

eld.disconnect(device)
```

---

**Document Version:** 1.0  
**Last Updated:** October 3, 2025  
**Status:** Design Phase - Ready for Implementation  
**Next Steps:** Begin Phase 1 - Driver Development  
**Note:** Adapt based on actual ELM11 specifications and available FFT library

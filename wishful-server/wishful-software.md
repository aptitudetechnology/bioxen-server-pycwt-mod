# BioXen Computation Server - Open Source Software Stack 📚

**Purpose:** Comprehensive list of open source libraries that can run on the BioXen computation server for signal analysis, optimization, and hardware integration.

**Date:** October 5, 2025  
**Server Architecture:** Stateless computation service (no VM management, no simulation execution)

---

## 🎯 Service Model

```
┌─────────────────┐                    ┌─────────────────┐
│  BioXen Local   │                    │  BioXen Server  │
│   (Client)      │                    │  (Computation)  │
├─────────────────┤                    ├─────────────────┤
│ • VM Management │  ──── HTTP ───>    │ • Fourier       │
│ • Simulation    │  <─── JSON ────    │ • Wavelet       │
│ • Time-series   │                    │ • Laplace       │
│   generation    │                    │ • Z-Transform   │
│ • Results       │                    │ • Validation    │
│   storage       │                    │ • Optimization  │
└─────────────────┘                    └─────────────────┘
```

---

## 🔬 Scientific Computing Core

### Signal Analysis & Transforms

#### **NumPy** - Array Operations
- **Version:** 1.24.3+
- **Purpose:** Foundation for all numerical computing
- **Use Cases:** Array operations, linear algebra, FFT
- **License:** BSD-3-Clause
- **Installation:** `pip install numpy==1.24.3`
- **Documentation:** https://numpy.org/doc/

#### **SciPy** - Scientific Algorithms
- **Version:** 1.11.3+
- **Purpose:** Scientific computing and technical computing
- **Use Cases:**
  - `scipy.fft` - Fast Fourier Transform
  - `scipy.signal` - Signal processing, filtering, spectral analysis
  - `scipy.signal.lti` - Laplace/Z-transform, pole-zero analysis
  - `scipy.optimize` - Optimization algorithms (LBFGS, Nelder-Mead, Powell)
  - `scipy.integrate` - ODE solvers, numerical integration
  - `scipy.interpolate` - Interpolation methods
- **License:** BSD-3-Clause
- **Installation:** `pip install scipy==1.11.3`
- **Documentation:** https://scipy.org/

#### **Astropy** - Astronomy Computations
- **Version:** 5.3.4+
- **Purpose:** Lomb-Scargle periodogram for irregular sampling
- **Use Cases:**
  - `astropy.timeseries.LombScargle` - Frequency analysis of unevenly sampled data
  - Astronomical time series analysis (applicable to biological rhythms)
- **License:** BSD-3-Clause
- **Installation:** `pip install astropy==5.3.4`
- **Documentation:** https://docs.astropy.org/

#### **PyWavelets** - Wavelet Transforms
- **Version:** 1.4.1+
- **Purpose:** Continuous and discrete wavelet transforms
- **Use Cases:**
  - CWT (Continuous Wavelet Transform)
  - DWT (Discrete Wavelet Transform)
  - Wavelet packet decomposition
  - Multi-resolution analysis
- **License:** MIT
- **Installation:** `pip install PyWavelets==1.4.1`
- **Documentation:** https://pywavelets.readthedocs.io/

#### **pycwt-mod** - Enhanced Wavelet Analysis (Custom Fork)
- **Version:** Custom fork
- **Purpose:** Hardware-accelerated wavelet analysis with pluggable backends
- **Use Cases:**
  - Wavelet coherence (WCT)
  - Cross-wavelet transform (XWT)
  - Monte Carlo significance testing
  - Hardware acceleration (CPU, GPU, FPGA)
- **License:** Custom (based on pycwt)
- **Repository:** https://github.com/aptitudetechnology/pycwt-mod
- **Features:**
  - Modular backend architecture
  - FPGA acceleration (Tang Nano 9K)
  - Embedded device support (ELM11)

---

## 🎯 Optimization & Parameter Tuning

### General Optimization

#### **scipy.optimize** - Standard Optimizers
- **Included with:** SciPy
- **Algorithms:**
  - `minimize()` - LBFGS, Nelder-Mead, Powell, CG, BFGS, SLSQP
  - `differential_evolution()` - Global optimization
  - `basinhopping()` - Global optimization with local refinement
  - `least_squares()` - Non-linear least squares
  - `curve_fit()` - Curve fitting wrapper
- **Use Cases:** Rate constant tuning, parameter fitting
- **Documentation:** https://docs.scipy.org/doc/scipy/reference/optimize.html

#### **scikit-optimize (skopt)** - Bayesian Optimization
- **Version:** 0.9.0+
- **Purpose:** Sequential model-based optimization
- **Algorithms:**
  - Gaussian Process optimization
  - Random Forest optimization
  - Gradient Boosting Trees
- **Use Cases:** Expensive function optimization, hyperparameter tuning
- **License:** BSD-3-Clause
- **Installation:** `pip install scikit-optimize==0.9.0`
- **Documentation:** https://scikit-optimize.github.io/

#### **NLopt** - Non-linear Optimization Library
- **Version:** Latest
- **Purpose:** Library of non-linear optimization algorithms
- **Algorithms:** 40+ algorithms including COBYLA, BOBYQA, PRAXIS, DIRECT
- **Use Cases:** Constrained/unconstrained optimization
- **License:** LGPL/MIT
- **Installation:** `pip install nlopt`
- **Documentation:** https://nlopt.readthedocs.io/

### Evolutionary & Genetic Algorithms

#### **DEAP** - Distributed Evolutionary Algorithms
- **Version:** Latest
- **Purpose:** Evolutionary computation framework
- **Features:**
  - Genetic algorithms
  - Genetic programming
  - Evolution strategies
  - Multi-objective optimization (NSGA-II, NSGA-III, SPEA2)
- **Use Cases:** Multi-objective optimization, complex fitness landscapes
- **License:** LGPL
- **Installation:** `pip install deap`
- **Documentation:** https://deap.readthedocs.io/

#### **pymoo** - Multi-objective Optimization
- **Version:** Latest
- **Purpose:** Multi-objective optimization framework
- **Algorithms:**
  - NSGA-II, NSGA-III
  - MOEA/D, RVEA
  - Decision making tools
- **Use Cases:** Pareto front generation, trade-off analysis
- **License:** Apache 2.0
- **Installation:** `pip install pymoo`
- **Documentation:** https://pymoo.org/

### Sensitivity Analysis

#### **SALib** - Sensitivity Analysis Library
- **Version:** 1.4.7+
- **Purpose:** Global sensitivity analysis
- **Methods:**
  - Sobol indices (variance-based)
  - Morris method (screening)
  - FAST (Fourier Amplitude Sensitivity Test)
  - Delta moment-independent
  - DGSM (derivative-based)
- **Use Cases:** Parameter sensitivity, uncertainty quantification
- **License:** MIT
- **Installation:** `pip install SALib==1.4.7`
- **Documentation:** https://salib.readthedocs.io/

---

## 🧬 Biological & Time Series Analysis

### Circadian & Oscillation Analysis

#### **CosinorPy** - Cosinor Analysis
- **Version:** Latest
- **Purpose:** Cosinor analysis for biological rhythms
- **Features:**
  - Single/multi-component cosinor
  - Population-mean cosinor
  - Acrophase calculation
- **Use Cases:** Circadian rhythm analysis, periodic pattern detection
- **License:** GPL
- **Installation:** `pip install cosinorpy`
- **Documentation:** https://github.com/mmoskon/CosinorPy

#### **pyboat** - Biological Oscillations Analysis Toolkit
- **Version:** Latest
- **Purpose:** Time-frequency analysis for biological oscillations
- **Features:**
  - Continuous wavelet transform
  - Ridge extraction
  - Synchronization analysis
- **Use Cases:** Cellular oscillations, developmental timing
- **License:** GPL
- **Installation:** `pip install pyboat`
- **Documentation:** https://github.com/tensionhead/pyBOAT

### Time Series Analysis

#### **statsmodels** - Statistical Models
- **Version:** Latest
- **Purpose:** Statistical modeling and econometrics
- **Features:**
  - ARIMA/SARIMA models
  - Seasonal decomposition
  - Granger causality
  - Time series regression
- **Use Cases:** Trend analysis, forecasting, causality testing
- **License:** BSD-3-Clause
- **Installation:** `pip install statsmodels`
- **Documentation:** https://www.statsmodels.org/

#### **tslearn** - Time Series Machine Learning
- **Version:** Latest
- **Purpose:** Machine learning for time series
- **Features:**
  - Dynamic Time Warping (DTW)
  - Time series clustering
  - Classification algorithms
  - Shapelet discovery
- **Use Cases:** Pattern matching, clustering, classification
- **License:** BSD-2-Clause
- **Installation:** `pip install tslearn`
- **Documentation:** https://tslearn.readthedocs.io/

---

## 🖥️ Web Framework & API

### Core Framework

#### **FastAPI** - Modern Web Framework
- **Version:** 0.104.1+
- **Purpose:** High-performance async web framework
- **Features:**
  - Automatic OpenAPI/Swagger documentation
  - Pydantic data validation
  - Async/await support
  - Dependency injection
  - WebSocket support
- **Use Cases:** REST API server, real-time communication
- **License:** MIT
- **Installation:** `pip install fastapi==0.104.1`
- **Documentation:** https://fastapi.tiangolo.com/

#### **Uvicorn** - ASGI Server
- **Version:** 0.24.0+
- **Purpose:** Lightning-fast ASGI server
- **Features:**
  - HTTP/1.1 and HTTP/2
  - WebSocket support
  - Graceful shutdown
  - Hot reload (development)
- **Use Cases:** Production server, development server
- **License:** BSD-3-Clause
- **Installation:** `pip install uvicorn[standard]==0.24.0`
- **Documentation:** https://www.uvicorn.org/

#### **Pydantic** - Data Validation
- **Version:** 2.4.2+
- **Purpose:** Data validation using Python type hints
- **Features:**
  - Automatic validation
  - JSON Schema generation
  - Custom validators
  - Dataclass support
- **Use Cases:** Request/response validation, configuration management
- **License:** MIT
- **Installation:** `pip install pydantic==2.4.2`
- **Documentation:** https://docs.pydantic.dev/

### Additional API Tools

#### **httpx** - HTTP Client
- **Version:** Latest
- **Purpose:** Modern HTTP client with async support
- **Features:**
  - Sync and async APIs
  - HTTP/1.1 and HTTP/2
  - Connection pooling
- **Use Cases:** Testing, client libraries, API calls
- **License:** BSD-3-Clause
- **Installation:** `pip install httpx`
- **Documentation:** https://www.python-httpx.org/

#### **aiohttp** - Async HTTP Client/Server
- **Version:** Latest
- **Purpose:** Asynchronous HTTP client/server framework
- **Features:**
  - Async client and server
  - WebSocket support
  - Middleware support
- **Use Cases:** Async HTTP operations, WebSocket server
- **License:** Apache 2.0
- **Installation:** `pip install aiohttp`
- **Documentation:** https://docs.aiohttp.org/

#### **python-multipart** - Multipart Form Data
- **Version:** Latest
- **Purpose:** Parse multipart/form-data
- **Features:** File upload handling
- **Use Cases:** File uploads via REST API
- **License:** Apache 2.0
- **Installation:** `pip install python-multipart`
- **Documentation:** https://github.com/andrew-d/python-multipart

---

## ⚡ Hardware Acceleration & Parallelization

### CPU Parallelization

#### **joblib** - Pipeline Parallelization
- **Version:** 1.3.2+
- **Purpose:** Lightweight pipelining and parallel computing
- **Features:**
  - Transparent disk-caching
  - Parallel for loops
  - Serialization
- **Use Cases:** Multi-core CPU parallelization, caching expensive computations
- **License:** BSD-3-Clause
- **Installation:** `pip install joblib==1.3.2`
- **Documentation:** https://joblib.readthedocs.io/

#### **numba** - JIT Compiler
- **Version:** 0.58.0+
- **Purpose:** JIT compilation for NumPy code
- **Features:**
  - LLVM-based JIT compilation
  - Automatic parallelization
  - GPU acceleration (CUDA)
- **Use Cases:** Speed up NumPy operations, tight loops
- **License:** BSD-2-Clause
- **Installation:** `pip install numba==0.58.0`
- **Documentation:** https://numba.pydata.org/

#### **multiprocessing** - Built-in Parallelism
- **Version:** Built-in
- **Purpose:** Process-based parallelism
- **Features:**
  - Process pools
  - Shared memory
  - Queues and pipes
- **Use Cases:** CPU-bound parallelization
- **Documentation:** https://docs.python.org/3/library/multiprocessing.html

#### **concurrent.futures** - High-level Parallelism
- **Version:** Built-in
- **Purpose:** High-level interface for parallel execution
- **Features:**
  - ThreadPoolExecutor
  - ProcessPoolExecutor
  - Future objects
- **Use Cases:** Simple parallel execution patterns
- **Documentation:** https://docs.python.org/3/library/concurrent.futures.html

### GPU Acceleration

#### **CuPy** - GPU-accelerated NumPy
- **Version:** Latest
- **Purpose:** NumPy-compatible array library accelerated by CUDA
- **Features:**
  - Drop-in replacement for NumPy
  - CUDA kernel interface
  - Automatic memory management
- **Use Cases:** GPU-accelerated numerical computing
- **License:** MIT
- **Requirements:** NVIDIA GPU with CUDA
- **Installation:** `pip install cupy-cuda11x` (adjust for CUDA version)
- **Documentation:** https://cupy.dev/

#### **PyCUDA** - Python CUDA Interface
- **Version:** Latest
- **Purpose:** Direct CUDA programming from Python
- **Features:**
  - Direct GPU memory access
  - Custom kernel compilation
  - GPU arrays
- **Use Cases:** Custom GPU kernels, low-level GPU programming
- **License:** MIT
- **Requirements:** NVIDIA GPU with CUDA
- **Installation:** `pip install pycuda`
- **Documentation:** https://documen.tician.de/pycuda/

### FPGA & Embedded Hardware

#### **pyserial** - Serial Communication
- **Version:** 3.5+
- **Purpose:** Python Serial Port Extension
- **Features:**
  - Cross-platform serial port access
  - Timeout handling
  - Binary/text modes
- **Use Cases:** FPGA communication, embedded device control
- **License:** BSD-3-Clause
- **Installation:** `pip install pyserial==3.5`
- **Documentation:** https://pyserial.readthedocs.io/

#### **Custom Tang Nano 9K / ELM11 Drivers** (Already Implemented!)
- **Purpose:** FPGA acceleration for wavelet analysis
- **Features:**
  - Hardware-accelerated Monte Carlo simulations
  - Serial communication protocol
  - FFT acceleration
- **Use Cases:** Hardware-accelerated wavelet transforms
- **Location:** Already working in pycwt-mod server!

---

## 🔌 Hardware Sensors

### Environmental Sensors

#### **smbus2** - I2C Communication
- **Version:** 0.4.2+
- **Purpose:** Pure Python SMBus/I2C library
- **Features:**
  - I2C read/write operations
  - Block operations
  - Cross-platform
- **Use Cases:** Sensor communication, I2C devices
- **License:** MIT
- **Installation:** `pip install smbus2==0.4.2`
- **Documentation:** https://github.com/kplindegaard/smbus2

#### **pimoroni-bme280** - Temperature/Humidity/Pressure Sensor
- **Version:** 0.1.0+
- **Purpose:** BME280 sensor driver
- **Features:**
  - Temperature reading (±1°C accuracy)
  - Humidity reading (±3% RH accuracy)
  - Pressure reading (±1 hPa accuracy)
- **Use Cases:** Environmental monitoring, temperature compensation studies
- **License:** MIT
- **Installation:** `pip install pimoroni-bme280==0.1.0`
- **Hardware:** BME280 breakout board (~$10)
- **Documentation:** https://github.com/pimoroni/bme280-python

#### **ltr559** - Light & Proximity Sensor
- **Version:** 0.1.0+
- **Purpose:** LTR-559 light and proximity sensor driver
- **Features:**
  - Light intensity (lux)
  - Proximity detection
  - Configurable gain and integration time
- **Use Cases:** Light entrainment studies, day/night cycle monitoring
- **License:** MIT
- **Installation:** `pip install ltr559==0.1.0`
- **Hardware:** LTR-559 breakout board (~$15)
- **Documentation:** https://github.com/pimoroni/ltr559-python

### General Hardware Libraries

#### **Adafruit CircuitPython Libraries**
- **Purpose:** Extensive collection of sensor drivers
- **Features:**
  - 300+ sensor libraries
  - Unified API design
  - Well-documented examples
- **Use Cases:** Additional sensor support
- **License:** MIT
- **Installation:** `pip install adafruit-circuitpython-*`
- **Documentation:** https://circuitpython.org/libraries

#### **RPi.GPIO** - Raspberry Pi GPIO
- **Purpose:** Raspberry Pi GPIO control
- **Features:**
  - GPIO pin control
  - PWM support
  - Event detection
- **Use Cases:** Raspberry Pi hardware control
- **License:** MIT
- **Requirements:** Raspberry Pi
- **Installation:** `pip install RPi.GPIO`
- **Documentation:** https://sourceforge.net/p/raspberry-gpio-python/wiki/

---

## 🧪 Testing & Quality Assurance

### Testing Frameworks

#### **pytest** - Testing Framework
- **Version:** 7.4.3+
- **Purpose:** Comprehensive testing framework
- **Features:**
  - Simple assertion syntax
  - Fixtures
  - Parameterized tests
  - Plugin ecosystem
- **Use Cases:** Unit tests, integration tests
- **License:** MIT
- **Installation:** `pip install pytest==7.4.3`
- **Documentation:** https://docs.pytest.org/

#### **pytest-asyncio** - Async Test Support
- **Version:** 0.21.1+
- **Purpose:** Pytest support for async functions
- **Features:**
  - Async test fixtures
  - Event loop management
- **Use Cases:** Testing async endpoints
- **License:** Apache 2.0
- **Installation:** `pip install pytest-asyncio==0.21.1`
- **Documentation:** https://pytest-asyncio.readthedocs.io/

#### **hypothesis** - Property-Based Testing
- **Version:** Latest
- **Purpose:** Property-based testing framework
- **Features:**
  - Automatic test case generation
  - Shrinking failed examples
  - Stateful testing
- **Use Cases:** Finding edge cases, robust validation
- **License:** MPL 2.0
- **Installation:** `pip install hypothesis`
- **Documentation:** https://hypothesis.readthedocs.io/

---

## 📈 Monitoring & Performance

### Profiling Tools

#### **cProfile** - Built-in Profiler
- **Version:** Built-in
- **Purpose:** Deterministic profiling
- **Features:**
  - Function call counts
  - Execution time per function
  - Call graph generation
- **Use Cases:** Performance bottleneck identification
- **Documentation:** https://docs.python.org/3/library/profile.html

#### **line_profiler** - Line-by-Line Profiling
- **Version:** Latest
- **Purpose:** Line-by-line execution time profiling
- **Features:**
  - Detailed line timings
  - Decorator-based profiling
- **Use Cases:** Fine-grained performance analysis
- **License:** BSD-3-Clause
- **Installation:** `pip install line_profiler`
- **Documentation:** https://github.com/pyutils/line_profiler

#### **memory_profiler** - Memory Usage Profiling
- **Version:** Latest
- **Purpose:** Monitor memory consumption
- **Features:**
  - Line-by-line memory usage
  - Memory leak detection
- **Use Cases:** Memory optimization, leak detection
- **License:** BSD-3-Clause
- **Installation:** `pip install memory_profiler`
- **Documentation:** https://pypi.org/project/memory-profiler/

#### **py-spy** - Sampling Profiler
- **Version:** Latest
- **Purpose:** Low-overhead sampling profiler
- **Features:**
  - No code modification needed
  - Flame graph generation
  - Minimal performance impact
- **Use Cases:** Production profiling
- **License:** MIT
- **Installation:** `pip install py-spy`
- **Documentation:** https://github.com/benfred/py-spy

### System Monitoring

#### **prometheus-client** - Metrics Export
- **Version:** 0.18.0+
- **Purpose:** Prometheus instrumentation library
- **Features:**
  - Counter, Gauge, Histogram, Summary metrics
  - HTTP exposition
  - Multi-process support
- **Use Cases:** Production monitoring, alerting
- **License:** Apache 2.0
- **Installation:** `pip install prometheus-client==0.18.0`
- **Documentation:** https://github.com/prometheus/client_python

#### **psutil** - System Resource Monitoring
- **Version:** 5.9.6+
- **Purpose:** Cross-platform system and process utilities
- **Features:**
  - CPU, memory, disk, network monitoring
  - Process management
  - System information
- **Use Cases:** Resource monitoring, auto-scaling
- **License:** BSD-3-Clause
- **Installation:** `pip install psutil==5.9.6`
- **Documentation:** https://psutil.readthedocs.io/

---

## 🗄️ Caching & Storage (Optional)

### Caching

#### **redis-py** - Redis Client
- **Version:** Latest
- **Purpose:** Python interface to Redis
- **Features:**
  - Key-value store
  - Pub/sub messaging
  - Transactions
- **Use Cases:** Caching expensive computations, session storage
- **License:** MIT
- **Requirements:** Redis server
- **Installation:** `pip install redis`
- **Documentation:** https://redis-py.readthedocs.io/

#### **diskcache** - Disk-based Cache
- **Version:** Latest
- **Purpose:** Disk and file-based cache
- **Features:**
  - Persistent cache
  - LRU/LFU eviction
  - Thread-safe
- **Use Cases:** Caching without Redis, persistent cache
- **License:** Apache 2.0
- **Installation:** `pip install diskcache`
- **Documentation:** https://grantjenks.com/docs/diskcache/

#### **joblib.Memory** - Function Memoization
- **Included with:** joblib
- **Purpose:** Transparent disk-caching of functions
- **Features:**
  - Automatic caching based on inputs
  - Cache invalidation
- **Use Cases:** Memoization of expensive functions
- **Documentation:** https://joblib.readthedocs.io/en/latest/memory.html

### Database (Optional)

#### **SQLAlchemy** - SQL ORM
- **Version:** Latest
- **Purpose:** SQL toolkit and ORM
- **Features:**
  - Multiple database backends
  - Query builder
  - Migration support (with Alembic)
- **Use Cases:** Structured data storage, result persistence
- **License:** MIT
- **Installation:** `pip install sqlalchemy`
- **Documentation:** https://www.sqlalchemy.org/

#### **TinyDB** - Lightweight JSON Database
- **Version:** Latest
- **Purpose:** Document-oriented database
- **Features:**
  - Pure Python
  - No external dependencies
  - Query API
- **Use Cases:** Lightweight data storage, configuration
- **License:** MIT
- **Installation:** `pip install tinydb`
- **Documentation:** https://tinydb.readthedocs.io/

---

## 📊 Data Processing (Optional - Client Side Recommended)

### Data Structures

#### **pandas** - DataFrames
- **Version:** Latest
- **Purpose:** Data manipulation and analysis
- **Features:**
  - DataFrame and Series structures
  - Time series functionality
  - CSV/Excel/SQL I/O
- **Use Cases:** Data preprocessing, result formatting
- **License:** BSD-3-Clause
- **Installation:** `pip install pandas`
- **Documentation:** https://pandas.pydata.org/

#### **xarray** - Multi-dimensional Arrays
- **Version:** Latest
- **Purpose:** N-dimensional labeled arrays
- **Features:**
  - NetCDF/HDF5 support
  - Labeled dimensions
  - Built-in parallel computing
- **Use Cases:** Multi-dimensional data, climate/geo data
- **License:** Apache 2.0
- **Installation:** `pip install xarray`
- **Documentation:** https://xarray.pydata.org/

#### **h5py** - HDF5 File Format
- **Version:** Latest
- **Purpose:** Python interface to HDF5
- **Features:**
  - Large dataset storage
  - Compression
  - Hierarchical structure
- **Use Cases:** Large array storage, scientific data
- **License:** BSD-3-Clause
- **Installation:** `pip install h5py`
- **Documentation:** https://www.h5py.org/

---

## 🎨 Visualization (Client Side - Not for Server)

**Note:** Visualization libraries should typically run on the client side, not the computation server. Listed here for completeness.

### Plotting Libraries

#### **matplotlib** - 2D Plotting
- **License:** PSF-based
- **Installation:** `pip install matplotlib`
- **Documentation:** https://matplotlib.org/

#### **seaborn** - Statistical Visualization
- **License:** BSD-3-Clause
- **Installation:** `pip install seaborn`
- **Documentation:** https://seaborn.pydata.org/

#### **plotly** - Interactive Plots
- **License:** MIT
- **Installation:** `pip install plotly`
- **Documentation:** https://plotly.com/python/

---

## 📦 Complete Requirements Files

### Minimal Setup (Phase 1: Analysis APIs)

```txt
# requirements-minimal.txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# Scientific Computing
numpy==1.24.3
scipy==1.11.3
astropy==5.3.4
PyWavelets==1.4.1

# Testing
pytest==7.4.3
httpx==0.25.1
```

### Full Server Stack (All Phases)

```txt
# requirements-full.txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# Scientific Computing (Core)
numpy==1.24.3
scipy==1.11.3
astropy==5.3.4
PyWavelets==1.4.1

# Optimization & Tuning
scikit-optimize==0.9.0
SALib==1.4.7
nlopt

# Time Series Analysis
statsmodels
cosinorpy
pyboat

# Parallelization
joblib==1.3.2
numba==0.58.0

# Hardware (Optional - if using sensors)
smbus2==0.4.2
pimoroni-bme280==0.1.0
ltr559==0.1.0
pyserial==3.5

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.1
hypothesis

# Monitoring (Optional)
prometheus-client==0.18.0
psutil==5.9.6

# Caching (Optional)
diskcache
redis

# Data Processing (Optional)
pandas
xarray
```

### GPU-Accelerated Version

```txt
# requirements-gpu.txt
# Include all from requirements-full.txt, plus:

# GPU Acceleration (requires NVIDIA GPU + CUDA)
cupy-cuda11x  # Adjust for your CUDA version
pycuda
```

---

## 🚀 Installation Guide

### Step 1: Basic Setup

```bash
cd wishful-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install minimal requirements
pip install -r requirements-minimal.txt
```

### Step 2: Add Optimization (Phase 3)

```bash
pip install scikit-optimize==0.9.0 SALib==1.4.7
```

### Step 3: Add Hardware Support (Phase 4)

```bash
# For Raspberry Pi or similar with I2C
pip install smbus2 pimoroni-bme280 ltr559

# Enable I2C on Raspberry Pi
sudo raspi-config  # Interface Options -> I2C -> Enable
```

### Step 4: GPU Acceleration (Optional)

```bash
# Check CUDA version first
nvidia-smi

# Install CuPy (adjust for your CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
# or: cupy-cuda12x for CUDA 12.x
```

---

## 🎯 Use Case Matrix

| Library | Fourier | Wavelet | Laplace | Z-Transform | Validation | Optimization | Hardware |
|---------|---------|---------|---------|-------------|------------|--------------|----------|
| NumPy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| SciPy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| Astropy | ✅ | - | - | - | - | - | - |
| PyWavelets | - | ✅ | - | - | - | - | - |
| pycwt-mod | - | ✅ | - | - | - | - | ✅ |
| scikit-optimize | - | - | - | - | - | ✅ | - |
| SALib | - | - | - | - | - | ✅ | - |
| joblib | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| numba | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| BME280 | - | - | - | - | - | - | ✅ |
| LTR-559 | - | - | - | - | - | - | ✅ |

---

## 🎓 Learning Resources

### SciPy Ecosystem
- **SciPy Lecture Notes:** https://scipy-lectures.org/
- **NumPy Tutorial:** https://numpy.org/doc/stable/user/quickstart.html
- **SciPy Cookbook:** https://scipy-cookbook.readthedocs.io/

### Optimization
- **scipy.optimize Tutorial:** https://docs.scipy.org/doc/scipy/tutorial/optimize.html
- **SALib Examples:** https://salib.readthedocs.io/en/latest/basics.html
- **skopt Examples:** https://scikit-optimize.github.io/stable/auto_examples/

### FastAPI
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/tutorial/
- **Real Python FastAPI Guide:** https://realpython.com/fastapi-python-web-apis/

---

## 📝 Notes

1. **Server vs Client Responsibilities:**
   - **Server:** Computation, analysis, optimization (stateless)
   - **Client:** VM management, simulation, visualization, data storage

2. **Hardware Requirements:**
   - **Minimal:** CPU with 4+ cores, 8GB RAM
   - **Recommended:** CPU with 8+ cores, 16GB RAM
   - **GPU (Optional):** NVIDIA GPU with CUDA support
   - **FPGA (Optional):** Tang Nano 9K (already supported!)
   - **Sensors (Optional):** Raspberry Pi with I2C sensors

3. **License Compatibility:**
   - All listed libraries are compatible with commercial use
   - Most use permissive licenses (MIT, BSD, Apache 2.0)
   - Some use copyleft licenses (GPL, LGPL) - check before redistribution

4. **Version Pinning:**
   - Pin versions in production for reproducibility
   - Use `pip freeze > requirements.txt` after testing
   - Regularly update for security patches

---

**Last Updated:** October 5, 2025  
**Maintained By:** BioXen Development Team  
**Related Documents:**
- `wishful-api-specification-document.md` - API specification
- `IMPLEMENTATION_CHECKLIST.md` - Implementation guide
- `QUICK_START.md` - Getting started tutorial

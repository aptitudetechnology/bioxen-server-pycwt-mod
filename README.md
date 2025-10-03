PyCWT-mod
=========

[![ReadTHeDocs](https://readthedocs.org/projects/pycwt-mod/badge/?version=latest)](https://pycwt-mod.readthedocs.io/en/latest/?badge=latest)

[![PyPI version](https://badge.fury.io/py/pycwt-mod.svg)](https://badge.fury.io/py/pycwt-mod)

A modular fork of the continuous wavelet transform module for Python. This fork
includes a collection of routines for wavelet transform and statistical analysis
via FFT algorithm. In addition, the module also includes cross-wavelet transforms,
wavelet coherence tests and sample scripts.

Please read the documentation [here](https://pycwt-mod.readthedocs.io/en/latest/).

This module requires ``NumPy``, ``SciPy``, ``tqdm``. In addition, you will 
also need ``matplotlib`` to run the examples.

**New in pycwt-mod:** Hardware-accelerated Monte Carlo backends for improved
performance in wavelet coherence significance testing. Choose from CPU (sequential,
parallel), embedded systems (ELM11), or FPGA (Tang Nano 9K) acceleration.

The sample scripts (`sample.py`, `sample_xwt.py`) illustrate the use of
the wavelet and inverse wavelet transforms, cross-wavelet transform and
wavelet transform coherence. Results are plotted in figures similar to the
sample images.


## ✨ Key Features

- **🚀 Hardware Acceleration**: FPGA (Tang Nano 9K) and embedded system support
- **⚡ Multiple Backends**: Sequential, parallel (joblib), and hardware-accelerated options  
- **🔌 Plug-and-Play**: Automatic hardware detection with fallback to CPU
- **🧪 Production Ready**: Comprehensive test suite (23+ tests) with hardware validation
- **📊 Backward Compatible**: Drop-in replacement for existing PyCWT workflows
- **🛠️ Developer Friendly**: Interactive detection scripts and detailed setup guides


## 🚀 Development Roadmap

### ✅ Completed Phases

**Phase 1: Backend Architecture (Complete)**
- ✅ Abstract `MonteCarloBackend` class for plugin system
- ✅ Backend registry with auto-discovery
- ✅ Sequential CPU backend (single-core)
- ✅ Joblib parallel backend (multi-core)
- ✅ Comprehensive test suites for all backends
- ✅ Integration with `wct_significance()` function

**Phase 2: Core Integration (Complete)**
- ✅ Modified `wct_significance()` to use backend system
- ✅ Added backend parameter (`backend='sequential'|'joblib'|'elm11'|...`)
- ✅ Maintained backward compatibility
- ✅ Performance validation and benchmarking
- ✅ User testing instructions (`REMOTE-TESTING.md`)

**Phase 3: Hardware Acceleration (Complete)**
- ✅ **Tang Nano 9K FPGA Backend**: Fully implemented and tested
  - SIPEED JTAG Debugger detection and communication
  - Serial protocol at 9600 baud with interactive interface
  - Hardware availability detection and multi-baudrate support
  - All 23 backend tests passing with hardware connected
- ✅ **ELM11 Backend**: Production-ready implementation
  - Serial communication with ELM11 microcontroller
  - Multi-device detection (Tang Nano, JTAG Debugger, SIPEED)
  - Lua-scriptable Monte Carlo execution
  - Integrated hardware detection and availability checks

### � Current Status

**Phase 4: Documentation & Testing (Complete)**
- ✅ Complete hardware setup guides (`TANG-NANO-9K-TESTING.md`)
- ✅ Interactive detection scripts (`test-tang-nano-9k.py`)
- ✅ Performance benchmarking and validation
- ✅ Backend selection documentation and troubleshooting

### 🔮 Future Phases

**Phase 5: Additional Hardware Backends**
- GPU acceleration (CUDA/OpenCL)
- Distributed computing (Dask clusters)
- Custom FPGA implementations beyond Tang Nano 9K
- ARM-based embedded systems

**Phase 6: Advanced Features**
- Dynamic backend selection based on workload
- Backend-specific optimizations
- Real-time monitoring and profiling
- Automatic hardware discovery and configuration

### 📊 Backend Performance Comparison

| Backend | Status | Performance | Use Case |
|---------|--------|-------------|----------|
| Sequential | ✅ Production | 1.0× baseline | Reference implementation |
| Joblib | ✅ Production | 3-4× speedup | Multi-core CPU systems |
| ELM11 | ✅ Production | 1.5-3× speedup | Embedded systems, low power |
| Tang Nano 9K | ✅ Production | Variable speedup | FPGA acceleration |
| GPU | 🔲 Planned | Variable | GPU-accelerated systems |
| Dask | 🔲 Planned | Variable | Distributed computing |

**Hardware Requirements:**
- **Sequential/Joblib**: Any CPU (Python environment)
- **ELM11**: Serial port, compatible microcontroller
- **Tang Nano 9K**: SIPEED Tang Nano 9K FPGA board via USB

For detailed setup instructions, see:
- [`TANG-NANO-9K-TESTING.md`](TANG-NANO-9K-TESTING.md) - Complete hardware testing guide
- [`test-tang-nano-9k.py`](test-tang-nano-9k.py) - Interactive detection script
- [`research/`](research/) - Design documents and analysis


### How to cite

Sebastian Krieger, Nabil Freij, and contributors. _PyCWT-mod: modular wavelet 
spectral analysis in Python with FPGA and embedded hardware acceleration_. Python. 2025. <https://github.com/aptitudetechnology/pycwt-mod>.


Disclaimer
----------

This module is based on routines provided by C. Torrence and G. P. Compo
available at http://paos.colorado.edu/research/wavelets/, on routines
provided by A. Grinsted, J. Moore and S. Jevrejeva available at
http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and
on routines provided by A. Brazhe available at
http://cell.biophys.msu.ru/static/swan/.

This software is released under a BSD-style open source license. Please read
the license file for further information. This routine is provided as is
without any express or implied warranties whatsoever.


Installation
------------

We recommend using PyPI to install this package.

```commandline
$ pip install pycwt-mod
```

However, if you want to install directly from GitHub, use:

```commandline
$ pip install git+https://github.com/aptitudetechnology/pycwt-mod
```

**Development Installation:**

For development and testing the latest features:

```commandline
$ git clone https://github.com/aptitudetechnology/pycwt-mod
$ cd pycwt-mod
$ pip install -e .
```

**Hardware Backend Dependencies:**

For hardware acceleration backends:
- `joblib` - Multi-core CPU acceleration (auto-installed)
- `pyserial>=3.5` - Serial communication for FPGA/embedded devices
- Hardware-specific setup (see [`TANG-NANO-9K-TESTING.md`](TANG-NANO-9K-TESTING.md))

**Hardware Setup:**
- **Tang Nano 9K**: USB connection, SIPEED JTAG Debugger detection
- **ELM11**: Serial port configuration, device permissions


Quick Start
-----------

```python
import pycwt_mod as pycwt

# Basic wavelet transform
dat = pycwt.load_sample('NINO3')
wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(dat, 1/12, dj=1/12)

# Wavelet coherence with hardware acceleration
# Choose backend: 'sequential', 'joblib', 'elm11'
WCT, aWCT, coi, freqs, sig95 = pycwt.wct(
    dat1, dat2, 
    dt=1/12, 
    dj=1/12, 
    sig=True, 
    significance_level=0.95,
    backend='elm11'  # Use FPGA acceleration (Tang Nano 9K)
)

# Test hardware backend availability
from pycwt_mod.backends import list_backends, get_backend
print("Available backends:", list_backends())

# Check if Tang Nano 9K is connected
backend = get_backend('elm11')
if backend.is_available():
    print("Tang Nano 9K FPGA detected and operational!")
else:
    print("Hardware not available, using CPU backend")
```

For detailed examples, see the `sample.py` and `sample_xwt.py` scripts.


Testing
-------

**Quick Hardware Test:**
```bash
# Test Tang Nano 9K FPGA detection and communication
python3 test-tang-nano-9k.py

# Run comprehensive backend test suite
pytest src/pycwt_mod/tests/backends/test_elm11.py -v

# Test all backends
python3 run_phase2_tests.py
```

**Test Coverage:**
- ✅ 23 backend tests covering initialization, detection, execution
- ✅ Hardware detection for SIPEED JTAG Debugger devices
- ✅ Serial communication validation at multiple baud rates
- ✅ Monte Carlo execution with deterministic behavior
- ✅ Integration tests with `wct_significance()` function
- ✅ Performance benchmarking and comparison

See [`TANG-NANO-9K-TESTING.md`](TANG-NANO-9K-TESTING.md) for complete testing instructions.


Project Structure
-----------------

```
pycwt-mod/
├── src/pycwt_mod/          # Main package
│   ├── backends/           # Backend implementations
│   │   ├── elm11.py       # Tang Nano 9K / ELM11 backend
│   │   ├── joblib.py      # Multi-core CPU backend  
│   │   └── sequential.py   # Single-core reference
│   ├── tests/             # Comprehensive test suite
│   └── sample/            # Example datasets
├── docs/                  # Documentation
├── research/              # Design documents and analysis
├── test-tang-nano-9k.py  # Hardware detection script
├── TANG-NANO-9K-TESTING.md # Hardware setup guide
└── run_phase2_tests.py   # Backend test runner
```


Acknowledgements
----------------

We would like to thank Christopher Torrence, Gilbert P. Compo, Aslak Grinsted,
John Moore, Svetlana Jevrejevaand and Alexey Brazhe for their code and also
Jack Ireland and Renaud Dussurget for their attentive eyes, feedback and
debugging.


Contributors
------------

- Sebastian Krieger
- Nabil Freij
- Ken Mankoff
- Aaron Nielsen
- Rodrigo Nemmen
- Ondrej Grover
- Joscelin Rocha Hidalgo
- Stuart Mumford
- ymarcon1
- Tariq Hassan


References
----------

1. Torrence, C. and Compo, G. P.. A Practical Guide to Wavelet
   Analysis. Bulletin of the American Meteorological Society, *American
   Meteorological Society*, **1998**, 79, 61-78.
2. Torrence, C. and Webster, P. J.. Interdecadal changes in the
   ENSO-Monsoon system, *Journal of Climate*, **1999**, 12(8),
   2679-2690.
3. Grinsted, A.; Moore, J. C. & Jevrejeva, S. Application of the cross
   wavelet transform and wavelet coherence to geophysical time series.
   *Nonlinear Processes in Geophysics*, **2004**, 11, 561-566.
4. Mallat, S.. A wavelet tour of signal processing: The sparse way.
   *Academic Press*, **2008**, 805.
5. Addison, P. S. The illustrated wavelet transform handbook:
   introductory theory and applications in science, engineering,
   medicine and finance. *IOP Publishing*, **2002**.
6. Liu, Y., Liang, X. S. and Weisberg, R. H. Rectification of the bias
   in the wavelet power spectrum. *Journal of Atmospheric and Oceanic
   Technology*, **2007**, 24, 2093-2102.

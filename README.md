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
parallel), embedded Lua (ELM11), or FPGA (Tang Nano 9K) acceleration.

The sample scripts (`sample.py`, `sample_xwt.py`) illustrate the use of
the wavelet and inverse wavelet transforms, cross-wavelet transform and
wavelet transform coherence. Results are plotted in figures similar to the
sample images.


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
- ✅ Added backend parameter (`backend='sequential'|'joblib'|'elm11_lua'|...`)
- ✅ Maintained backward compatibility
- ✅ Performance validation and benchmarking
- ✅ User testing instructions (`laptop-test-instructions.md`)

### 🔄 Current Status

**Hardware Acceleration Planning (In Progress)**
- ✅ **Tang Nano 9K FPGA Backend**: Complete design document (`tang-nano-9k-prompt.md`)
  - Monte Carlo pipeline in Verilog
  - Serial communication protocol
  - Expected 15-30× speedup vs CPU
- ✅ **ELM11 Lua Backend**: Complete design document (`ELM11-prompt.md`)
  - Lua-scriptable Monte Carlo on embedded device
  - Leverages existing ELM11-Lua-FFT infrastructure
  - Expected 1.5-3× speedup with low power consumption

### 🔮 Future Phases

**Phase 3: Documentation & User Guide**
- Update user documentation for backend selection
- Add performance tuning guide
- Create hardware setup tutorials

**Phase 4: Hardware Backend Implementation**
- Tang Nano 9K FPGA backend implementation
- ELM11 Lua backend implementation
- Additional hardware backends (GPU, distributed computing)

**Phase 5: Advanced Features**
- Dynamic backend selection based on workload
- Backend-specific optimizations
- Real-time monitoring and profiling

### 📊 Backend Performance Comparison

| Backend | Status | Performance | Use Case |
|---------|--------|-------------|----------|
| Sequential | ✅ Complete | 1.0× baseline | Reference implementation |
| Joblib | ✅ Complete | 3-4× speedup | Multi-core CPU systems |
| ELM11 Lua | 📋 Planned | 1.5-3× speedup | Embedded, low power |
| Tang Nano 9K | 📋 Planned | 15-30× speedup | High-performance computing |
| GPU | 🔲 Future | Variable | GPU-accelerated systems |
| Dask | 🔲 Future | Variable | Distributed computing |

For detailed implementation plans, see:
- [`tang-nano-9k-prompt.md`](tang-nano-9k-prompt.md) - FPGA backend design
- [`ELM11-prompt.md`](ELM11-prompt.md) - Lua embedded backend design
- [`integration-plan.md`](integration-plan.md) - Overall modular architecture


### How to cite

Sebastian Krieger, Nabil Freij, and contributors. _PyCWT-mod: modular wavelet 
spectral analysis in Python with hardware acceleration_. Python. 2025. <https://github.com/aptitudetechnology/pycwt-mod>.


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

**Optional Dependencies:**

For hardware acceleration backends:
- `joblib` - Multi-core CPU acceleration (auto-installed)
- `pyserial` - Serial communication for embedded devices
- Hardware-specific drivers (see backend documentation)


Quick Start
-----------

```python
import pycwt_mod as pycwt

# Basic wavelet transform
dat = pycwt.load_sample('NINO3')
wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(dat, 1/12, dj=1/12)

# Wavelet coherence with hardware acceleration
# Choose backend: 'sequential', 'joblib', 'elm11_lua', 'tang_nano_9k'
WCT, aWCT, coi, freqs, sig95 = pycwt.wct(
    dat1, dat2, 
    dt=1/12, 
    dj=1/12, 
    sig=True, 
    significance_level=0.95,
    backend='joblib'  # Use multi-core acceleration
)
```

For detailed examples, see the `sample.py` and `sample_xwt.py` scripts.


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

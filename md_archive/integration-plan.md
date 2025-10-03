# Parallel Monte Carlo Integration Plan (Plugin Architecture)
## From Prototype (85%) to Production (100%)

**Project:** BioXen Four-Lens Signal Analysis Library - pycwt Performance Optimization  
**Date:** October 2, 2025  
**Status:** Google Canvas prototype received - integration planning phase  
**Architecture:** Plugin-style modular backends (Option B)  
**Target:** Production-ready parallel Monte Carlo with extensible backend system

---

## 📋 Executive Summary

**What We Have (85% Complete):**
- ✅ Working prototype with joblib parallelization
- ✅ Deterministic RNG using `np.random.SeedSequence`
- ✅ Self-contained worker function
- ✅ Benchmarking infrastructure
- ✅ Progress monitoring via joblib verbose
- ✅ Backward-compatible API design

**What We Need (15% Remaining):**
- 🔧 **Plugin architecture** - Modular backend system for extensibility
- 🔧 Integration with actual pycwt codebase
- 🔧 Replace mocks with real functions
- 🔧 Comprehensive validation testing
- 🔧 Error handling and edge cases
- 🔧 Documentation and examples
- 🔧 Cross-platform testing

**Architecture Decision: Plugin-Style Backends**
- 🏗️ Modular backend system for multiple execution strategies
- 🔌 Easy to add Dask, GPU, or other backends in future
- 🎯 Clean separation: core logic vs. execution strategy
- 🧪 Each backend independently testable

**Estimated Effort:** 2-3 days to complete remaining work (includes plugin architecture)

---

## 🎯 Phase 1: Plugin Architecture Setup (5-7 hours)

### Task 1.0: Create Backend Directory Structure

**Objective:** Set up modular plugin architecture for extensible backends

**Actions:**
```bash
# Create backend plugin directory
mkdir -p src/pycwt/backends

# Create backend files
touch src/pycwt/backends/__init__.py
touch src/pycwt/backends/base.py          # Abstract base class
touch src/pycwt/backends/sequential.py    # Default backend
touch src/pycwt/backends/joblib.py        # Parallel backend (from bench.md)
touch src/pycwt/backends/registry.py      # Backend registration system

# Future backends (placeholders)
touch src/pycwt/backends/dask.py          # Future: Distributed computing
touch src/pycwt/backends/gpu.py           # Future: GPU acceleration
```

**Directory Structure:**
```
src/pycwt/
├── __init__.py
├── wavelet.py                    # Core functions (minimal changes)
├── mothers.py                    # Wavelet definitions
├── helpers.py                    # Utilities
├── backends/                     # NEW: Plugin architecture
│   ├── __init__.py              # Backend registry exports
│   ├── base.py                  # Abstract base class
│   ├── registry.py              # Backend registration system
│   ├── sequential.py            # Default sequential backend
│   ├── joblib.py                # Parallel joblib backend
│   ├── dask.py                  # Future: Dask backend (stub)
│   └── gpu.py                   # Future: GPU backend (stub)
└── tests/
    ├── test_wavelet.py
    └── backends/                 # NEW: Backend tests
        ├── __init__.py
        ├── test_base.py
        ├── test_sequential.py
        ├── test_joblib.py
        └── test_registry.py
```

**Deliverable:** Backend directory structure created

---

### Task 1.1: Define Abstract Base Backend

**Objective:** Create abstract interface that all backends must implement

**File:** `src/pycwt/backends/base.py`

```python
"""
Abstract base class for Monte Carlo significance testing backends.

All backend implementations must inherit from MCBackend and implement
the execute() method.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class MCBackend(ABC):
    """
    Abstract base class for Monte Carlo execution backends.
    
    This defines the interface that all backends (sequential, parallel,
    distributed, GPU) must implement.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backend.
        
        Parameters
        ----------
        name : str
            Human-readable backend name
        config : dict, optional
            Backend-specific configuration
        """
        self.name = name
        self.config = config or {}
        self._available = None
    
    @abstractmethod
    def execute(
        self,
        al1: float,
        al2: float,
        dt: float,
        dj: float,
        s0: float,
        J: int,
        scales: np.ndarray,
        outsidecoi: np.ndarray,
        wavelet_name: str,
        mother_param: float,
        mc_count: int = 300,
        rng_seed: Optional[int] = None,
        progress: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Execute Monte Carlo significance testing.
        
        Parameters
        ----------
        al1, al2 : float
            Lag-1 autocorrelation coefficients
        dt, dj, s0, J : float/int
            Wavelet transform parameters
        scales : np.ndarray
            Wavelet scales
        outsidecoi : np.ndarray
            Cone of influence mask
        wavelet_name : str
            Wavelet name ('morlet', 'paul', 'dog', 'mexicanhat')
        mother_param : float
            Wavelet-specific parameter
        mc_count : int
            Number of Monte Carlo iterations
        rng_seed : int, optional
            Random seed for reproducibility
        progress : bool
            Show progress information
        **kwargs
            Backend-specific parameters
        
        Returns
        -------
        wlc : np.ndarray
            Accumulated histogram for significance calculation
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if backend is available (dependencies installed, etc.).
        
        Returns
        -------
        available : bool
            True if backend can be used
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get backend information and capabilities.
        
        Returns
        -------
        info : dict
            Backend metadata (version, capabilities, etc.)
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate backend configuration.
        
        Returns
        -------
        valid : bool
            True if configuration is valid
        """
        return True
    
    def __str__(self) -> str:
        return f"{self.name} backend"
    
    def __repr__(self) -> str:
        available = "available" if self.is_available() else "unavailable"
        return f"<{self.name} backend ({available})>"
```

**Deliverable:** Abstract base class for all backends

---

### Task 1.2: Create Backend Registry System

**Objective:** Implement plugin registration and selection system

**File:** `src/pycwt/backends/registry.py`

```python
"""
Backend registry for Monte Carlo significance testing.

Provides centralized registration and selection of execution backends.
"""

from typing import Dict, Optional, Type, List
from pycwt.backends.base import MCBackend
import warnings


class BackendRegistry:
    """
    Registry for Monte Carlo execution backends.
    
    Manages registration, discovery, and selection of backends.
    """
    
    def __init__(self):
        self._backends: Dict[str, Type[MCBackend]] = {}
        self._default_backend: Optional[str] = None
    
    def register(
        self,
        name: str,
        backend_class: Type[MCBackend],
        default: bool = False
    ):
        """
        Register a backend.
        
        Parameters
        ----------
        name : str
            Unique backend identifier
        backend_class : Type[MCBackend]
            Backend class (must inherit from MCBackend)
        default : bool
            Set as default backend
        """
        if not issubclass(backend_class, MCBackend):
            raise TypeError(
                f"Backend must inherit from MCBackend, got {backend_class}"
            )
        
        self._backends[name] = backend_class
        
        if default or self._default_backend is None:
            self._default_backend = name
    
    def get(self, name: Optional[str] = None) -> MCBackend:
        """
        Get backend instance.
        
        Parameters
        ----------
        name : str, optional
            Backend name. If None, returns default or best available.
        
        Returns
        -------
        backend : MCBackend
            Backend instance
        """
        if name is None:
            name = self.get_default()
        
        if name not in self._backends:
            raise ValueError(
                f"Unknown backend '{name}'. "
                f"Available: {self.list_backends()}"
            )
        
        backend_class = self._backends[name]
        backend = backend_class()
        
        if not backend.is_available():
            warnings.warn(
                f"Backend '{name}' is registered but not available. "
                f"Check dependencies. Falling back to default.",
                UserWarning
            )
            # Try to fall back to sequential
            if name != 'sequential' and 'sequential' in self._backends:
                return self.get('sequential')
            else:
                raise RuntimeError(
                    f"Backend '{name}' unavailable and no fallback found"
                )
        
        return backend
    
    def get_default(self) -> str:
        """
        Get default backend name.
        
        Returns best available backend in order of preference:
        1. User-set default
        2. First available parallel backend
        3. Sequential backend
        """
        if self._default_backend and self._backends[self._default_backend]().is_available():
            return self._default_backend
        
        # Prefer parallel backends
        for name in ['joblib', 'dask', 'gpu']:
            if name in self._backends and self._backends[name]().is_available():
                return name
        
        # Fall back to sequential
        return 'sequential'
    
    def set_default(self, name: str):
        """Set default backend."""
        if name not in self._backends:
            raise ValueError(f"Unknown backend '{name}'")
        self._default_backend = name
    
    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self._backends.keys())
    
    def list_available(self) -> List[str]:
        """List available backend names (dependencies satisfied)."""
        return [
            name for name, cls in self._backends.items()
            if cls().is_available()
        ]
    
    def info(self) -> Dict[str, Dict]:
        """Get information about all registered backends."""
        return {
            name: {
                'class': cls.__name__,
                'available': cls().is_available(),
                'info': cls().get_info() if cls().is_available() else {}
            }
            for name, cls in self._backends.items()
        }


# Global registry instance
_registry = BackendRegistry()


# Public API
def register_backend(name: str, backend_class: Type[MCBackend], default: bool = False):
    """Register a Monte Carlo backend."""
    _registry.register(name, backend_class, default)


def get_backend(name: Optional[str] = None) -> MCBackend:
    """Get backend instance by name."""
    return _registry.get(name)


def list_backends() -> List[str]:
    """List all registered backends."""
    return _registry.list_backends()


def list_available_backends() -> List[str]:
    """List available backends."""
    return _registry.list_available()


def set_default_backend(name: str):
    """Set default backend."""
    _registry.set_default(name)


def get_backend_info() -> Dict[str, Dict]:
    """Get information about all backends."""
    return _registry.info()
```

**Deliverable:** Backend registry system

---

### Task 1.3: Analyze Current pycwt Implementation

**Objective:** Understand exact structure of `wct_significance()` to integrate cleanly

**Actions:**
```bash
# Read the actual implementation
cat src/pycwt/wavelet.py | grep -A 100 "def wct_significance"

# Identify dependencies
grep -r "from.*import" src/pycwt/wavelet.py
```

**Key Questions to Answer:**
1. What is the exact signature of `wct_significance()`?
2. How are `scales`, `outsidecoi`, `maxscale`, `nbins` calculated?
3. Where does `N` (signal length) come from?
4. How is the wavelet object passed/created?
5. What is the exact return format?

**Deliverable:** Document current implementation structure

---

### Task 1.4: Implement Sequential Backend

**Objective:** Create sequential backend from existing pycwt code

**File:** `src/pycwt/backends/sequential.py`

```python
"""
Sequential Monte Carlo backend.

This is the default backend that executes Monte Carlo iterations
sequentially. It wraps the existing pycwt implementation.
"""

import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm

from pycwt.backends.base import MCBackend
from pycwt import cwt, rednoise
from pycwt.mothers import Morlet, Paul, DOG, MexicanHat


class SequentialBackend(MCBackend):
    """
    Sequential Monte Carlo execution backend.
    
    This is the reference implementation and fallback when parallel
    backends are unavailable.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Sequential", config)
    
    def execute(
        self,
        al1: float,
        al2: float,
        dt: float,
        dj: float,
        s0: float,
        J: int,
        scales: np.ndarray,
        outsidecoi: np.ndarray,
        wavelet_name: str,
        mother_param: float,
        mc_count: int = 300,
        rng_seed: Optional[int] = None,
        progress: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Execute Monte Carlo sequentially (existing pycwt logic)."""
        
        # Set seed if provided
        if rng_seed is not None:
            np.random.seed(rng_seed)
        
        # Get wavelet object
        wavelet = self._get_wavelet(wavelet_name, mother_param)
        
        # Calculate parameters
        ms = wavelet.flambda()
        N = int(np.ceil(ms * 6))
        nbins = 1000
        maxscale = len(scales)
        
        # Initialize histogram
        wlc = np.zeros((maxscale, nbins))
        
        # Sequential Monte Carlo loop (EXISTING PYCWT CODE)
        for _ in tqdm(range(mc_count), disable=not progress, desc="MC iterations"):
            # Generate red-noise surrogates
            noise1 = rednoise(N, al1, 1)
            noise2 = rednoise(N, al2, 1)
            
            # Compute CWT for both signals
            nW1, sj, freq, coi, _, _ = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
            nW2, sj, freq, coi, _, _ = cwt(noise2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
            
            # Compute cross-spectrum
            nW12 = nW1 * nW2.conj()
            
            # Smooth spectra
            S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales[:, None], dt, dj, sj)
            S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales[:, None], dt, dj, sj)
            S12 = wavelet.smooth(nW12 / scales[:, None], dt, dj, sj)
            
            # Compute coherence
            R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
            
            # Build histogram
            for s in range(maxscale):
                cd = np.floor(R2[s, :] * nbins)
                for j, t in enumerate(cd[~cd.mask]):
                    if 0 <= int(t) < nbins:
                        wlc[s, int(t)] += 1
        
        return wlc
    
    def is_available(self) -> bool:
        """Sequential backend always available."""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            'name': 'Sequential',
            'description': 'Sequential Monte Carlo (reference implementation)',
            'parallel': False,
            'requires': [],
            'version': '1.0.0'
        }
    
    @staticmethod
    def _get_wavelet(wavelet_name: str, mother_param: float):
        """Get wavelet object."""
        if wavelet_name == 'morlet':
            return Morlet(mother_param)
        elif wavelet_name == 'paul':
            return Paul(mother_param)
        elif wavelet_name == 'dog':
            return DOG(mother_param)
        elif wavelet_name == 'mexicanhat':
            return MexicanHat()
        else:
            raise ValueError(f"Unknown wavelet: {wavelet_name}")
```

**Deliverable:** Sequential backend implementation

---

### Task 1.5: Implement Joblib Parallel Backend

**Objective:** Adapt bench.md code into joblib backend plugin

**File:** `src/pycwt/backends/joblib.py`

**Current Code (bench.md):**
```python
# MOCK - To be replaced
def _get_mock_wavelet(wavelet_name: str):
    """Mocks the pycwt Wavelet object structure."""
    if wavelet_name == 'morlet':
        wavelet = Morlet(6)
    else:
        class MockWavelet:
            # ... placeholder ...
```

**Production Code (target):**
```python
# NO MOCK - Use actual pycwt imports
from pycwt.mothers import Morlet, Paul, DOG, MexicanHat

def _get_wavelet(wavelet_name: str, mother_param: float = None):
    """Get actual pycwt wavelet object."""
    if wavelet_name == 'morlet':
        return Morlet(mother_param or 6)
    elif wavelet_name == 'paul':
        return Paul(mother_param or 4)
    elif wavelet_name == 'dog':
        return DOG(mother_param or 2)
    elif wavelet_name == 'mexicanhat':
        return MexicanHat()
    else:
        raise ValueError(f"Unknown wavelet: {wavelet_name}")
```

**Changes Required:**

1. **Worker Function - Replace Mock Wavelet:**
```python
# OLD (bench.md line ~44)
wavelet = _get_mock_wavelet(wavelet_name)

# NEW (production)
# Import at module level (not in worker - multiprocessing issue)
# Pass wavelet parameters instead of object
from pycwt.mothers import Morlet, Paul, DOG, MexicanHat

def _mc_iteration_worker(
    al1: float, al2: float, N: int, dt: float, dj: float, s0: float, J: int,
    wavelet_name: str, mother_param: float,  # ADD: wavelet parameter
    scales: np.ndarray, outsidecoi: np.ndarray, 
    maxscale: int, nbins: int, seed: int
) -> np.ndarray:
    
    # Create wavelet within worker (avoid pickling issues)
    wavelet = _get_wavelet(wavelet_name, mother_param)
    
    # Rest of code unchanged...
```

2. **Remove Mock Parameters:**
```python
# OLD (bench.md lines ~119-125)
N = 10000  # Placeholder length for surrogate generation
s0 = MOCK_S0
J1 = J
scales = s0 * 2**(np.arange(J1 + 1) * dj)
maxscale = len(scales)
nbins = 1000
outsidecoi = np.ones((maxscale, N), dtype=bool)

# NEW (production) - Get from actual pycwt context
# These should be passed from the calling wct_significance() function
# which already has them calculated from the input signals
```

3. **Import Real pycwt Functions:**
```python
# At top of src/pycwt/wavelet.py
from pycwt import cwt, rednoise
from pycwt.helpers import ar1
from pycwt.mothers import Morlet, Paul, DOG, MexicanHat
```

**Production Code (joblib backend):**
```python
"""
Joblib parallel Monte Carlo backend.

Provides parallelized Monte Carlo using joblib for multi-core execution.
"""

import numpy as np
from typing import Optional, Dict, Any
import warnings

from pycwt.backends.base import MCBackend

# Check for joblib availability
try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    cpu_count = lambda: 1


class JoblibBackend(MCBackend):
    """
    Parallel Monte Carlo backend using joblib.
    
    Provides 4-8× speedup on multi-core systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Joblib", config)
        
        # Default configuration
        self.config.setdefault('backend', 'loky')  # joblib backend
        self.config.setdefault('verbose', 10)       # progress level
    
    def execute(
        self,
        al1: float,
        al2: float,
        dt: float,
        dj: float,
        s0: float,
        J: int,
        scales: np.ndarray,
        outsidecoi: np.ndarray,
        wavelet_name: str,
        mother_param: float,
        mc_count: int = 300,
        rng_seed: Optional[int] = None,
        progress: bool = True,
        n_workers: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Execute Monte Carlo in parallel using joblib."""
        
        if not self.is_available():
            raise RuntimeError(
                "Joblib backend not available. Install with: pip install joblib"
            )
        
        # Determine worker count
        if n_workers is None:
            n_workers = cpu_count()
        elif n_workers == -1:
            n_workers = cpu_count()
        
        # Calculate parameters
        from pycwt.mothers import Morlet, Paul, DOG, MexicanHat
        wavelet = self._get_wavelet(wavelet_name, mother_param)
        ms = wavelet.flambda()
        N = int(np.ceil(ms * 6))
        nbins = 1000
        maxscale = len(scales)
        
        # Generate deterministic seeds
        if rng_seed is not None:
            rng_sequence = np.random.SeedSequence(rng_seed)
            seeds = [s.entropy for s in rng_sequence.spawn(mc_count)]
        else:
            seeds = np.random.randint(0, 2**31, size=mc_count)
        
        # Prepare arguments
        args_list = [
            (al1, al2, N, dt, dj, s0, J, wavelet_name, mother_param,
             scales, outsidecoi, maxscale, nbins, seed)
            for seed in seeds
        ]
        
        # Progress information
        if progress:
            print(f"\n{'='*50}")
            print(f"Parallel Monte Carlo (Joblib Backend)")
            print(f"{'='*50}")
            print(f"  Iterations:  {mc_count}")
            print(f"  Workers:     {n_workers}")
            print(f"  Wavelet:     {wavelet_name}")
            print(f"{'='*50}\n")
        
        # Execute in parallel
        verbose_level = self.config['verbose'] if progress else 0
        
        results = Parallel(
            n_jobs=n_workers,
            verbose=verbose_level,
            backend=self.config['backend']
        )(
            delayed(_mc_iteration_worker)(*args)
            for args in args_list
        )
        
        # Aggregate histograms
        wlc = np.sum(results, axis=0)
        
        if progress:
            print(f"\n{'='*50}")
            print(f"  Monte Carlo completed")
            print(f"{'='*50}\n")
        
        return wlc
    
    def is_available(self) -> bool:
        """Check if joblib is installed."""
        return JOBLIB_AVAILABLE
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = {
            'name': 'Joblib',
            'description': 'Parallel Monte Carlo using joblib',
            'parallel': True,
            'requires': ['joblib>=1.0'],
        }
        
        if JOBLIB_AVAILABLE:
            import joblib
            info['version'] = joblib.__version__
            info['cores'] = cpu_count()
            info['backend'] = self.config.get('backend', 'loky')
        
        return info
    
    @staticmethod
    def _get_wavelet(wavelet_name: str, mother_param: float):
        """Get wavelet object."""
        from pycwt.mothers import Morlet, Paul, DOG, MexicanHat
        
        if wavelet_name == 'morlet':
            return Morlet(mother_param)
        elif wavelet_name == 'paul':
            return Paul(mother_param)
        elif wavelet_name == 'dog':
            return DOG(mother_param)
        elif wavelet_name == 'mexicanhat':
            return MexicanHat()
        else:
            raise ValueError(f"Unknown wavelet: {wavelet_name}")


# Worker function (from bench.md)
def _mc_iteration_worker(
    al1: float, al2: float, N: int, dt: float, dj: float, s0: float, J: int,
    wavelet_name: str, mother_param: float,
    scales: np.ndarray, outsidecoi: np.ndarray, 
    maxscale: int, nbins: int, seed: int
) -> np.ndarray:
    """
    Worker function for single Monte Carlo iteration.
    
    This is self-contained for multiprocessing safety.
    """
    
    # Import within worker
    import numpy as np
    from pycwt import cwt, rednoise
    from pycwt.mothers import Morlet, Paul, DOG, MexicanHat
    
    # Deterministic seeding
    np.random.seed(seed)
    
    # Get wavelet
    if wavelet_name == 'morlet':
        wavelet = Morlet(mother_param)
    elif wavelet_name == 'paul':
        wavelet = Paul(mother_param)
    elif wavelet_name == 'dog':
        wavelet = DOG(mother_param)
    elif wavelet_name == 'mexicanhat':
        wavelet = MexicanHat()
    else:
        raise ValueError(f"Unknown wavelet: {wavelet_name}")
    
    # Generate surrogates
    noise1 = rednoise(N, al1, 1)
    noise2 = rednoise(N, al2, 1)
    
    # Compute CWT
    nW1, sj, freq, coi, _, _ = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    nW2, sj, freq, coi, _, _ = cwt(noise2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
    
    # Ensure maxscale is within bounds
    maxscale = min(maxscale, nW1.shape[0])
    
    # Compute cross-spectrum
    nW12 = nW1 * nW2.conj()
    
    # Smooth spectra
    S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales[:maxscale, None], dt, dj, sj)
    S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales[:maxscale, None], dt, dj, sj)
    S12 = wavelet.smooth(nW12 / scales[:maxscale, None], dt, dj, sj)
    
    # Compute coherence
    R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi[:maxscale, :])
    
    # Build local histogram
    wlc_local = np.zeros((maxscale, nbins))
    
    for s in range(maxscale):
        cd = np.floor(R2[s, :].data * nbins)
        valid_cd = cd[~R2[s, :].mask]
        
        for t in valid_cd:
            idx = int(t)
            if 0 <= idx < nbins:
                wlc_local[s, idx] += 1
    
    return wlc_local
```

**Deliverable:** Joblib parallel backend implementation

---

### Task 1.6: Create Backend Initialization and Registration

**Objective:** Set up automatic backend registration

**File:** `src/pycwt/backends/__init__.py`

```python
"""
pycwt backends for Monte Carlo execution.

Provides pluggable backends for different execution strategies.
"""

from pycwt.backends.base import MCBackend
from pycwt.backends.registry import (
    register_backend,
    get_backend,
    list_backends,
    list_available_backends,
    set_default_backend,
    get_backend_info
)

# Import and register available backends
from pycwt.backends.sequential import SequentialBackend

# Always register sequential (always available)
register_backend('sequential', SequentialBackend, default=True)

# Try to register joblib backend
try:
    from pycwt.backends.joblib import JoblibBackend
    register_backend('joblib', JoblibBackend, default=True)  # Prefer parallel
except ImportError:
    pass

# Future backends (registered when available)
# try:
#     from pycwt.backends.dask import DaskBackend
#     register_backend('dask', DaskBackend)
# except ImportError:
#     pass
#
# try:
#     from pycwt.backends.gpu import GPUBackend
#     register_backend('gpu', GPUBackend)
# except ImportError:
#     pass


# Public API
__all__ = [
    'MCBackend',
    'register_backend',
    'get_backend',
    'list_backends',
    'list_available_backends',
    'set_default_backend',
    'get_backend_info',
    'SequentialBackend',
]
```

**Deliverable:** Backend initialization and registration

---

### Task 1.7: Create Future Backend Stubs

**Objective:** Provide placeholders for future backends

**File:** `src/pycwt/backends/dask.py` (stub)

```python
"""
Dask distributed Monte Carlo backend (FUTURE).

This is a placeholder for future distributed computing support.
"""

import numpy as np
from typing import Optional, Dict, Any
from pycwt.backends.base import MCBackend


class DaskBackend(MCBackend):
    """
    Dask distributed Monte Carlo backend (FUTURE IMPLEMENTATION).
    
    Will provide distributed Monte Carlo across multiple nodes.
    Expected speedup: 20-50× on cluster.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Dask", config)
    
    def execute(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "Dask backend not yet implemented. "
            "This is a placeholder for future development."
        )
    
    def is_available(self) -> bool:
        """Dask backend not yet implemented."""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'Dask',
            'description': 'Distributed Monte Carlo using Dask (planned)',
            'status': 'Not implemented',
            'parallel': True,
            'distributed': True,
            'requires': ['dask', 'distributed'],
        }
```

**File:** `src/pycwt/backends/gpu.py` (stub)

```python
"""
GPU-accelerated Monte Carlo backend (FUTURE).

This is a placeholder for future GPU acceleration support.
"""

import numpy as np
from typing import Optional, Dict, Any
from pycwt.backends.base import MCBackend


class GPUBackend(MCBackend):
    """
    GPU-accelerated Monte Carlo backend (FUTURE IMPLEMENTATION).
    
    Will provide GPU-accelerated Monte Carlo using CuPy.
    Expected speedup: 10-50× for large datasets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("GPU", config)
    
    def execute(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "GPU backend not yet implemented. "
            "This is a placeholder for future development."
        )
    
    def is_available(self) -> bool:
        """GPU backend not yet implemented."""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'GPU',
            'description': 'GPU-accelerated Monte Carlo using CuPy (planned)',
            'status': 'Not implemented',
            'parallel': True,
            'requires': ['cupy', 'cuda'],
        }
```

**Deliverable:** Future backend stubs

---

### Task 1.8: Integrate Backends into wavelet.py

**Strategy:** Minimal changes to existing `wct_significance()` function

**Strategy:** Refactor current function to call parallel version

**Current Structure (from pwt-report.md analysis):**
```python
def wct_significance(al1, al2, dt, dj, s0, J, 
                     significance_level=0.95,
                     wavelet='morlet', 
                     mc_count=300,
                     progress=True,
                     cache=True):
    
    # Setup phase (lines 543-570)
    # - Calculate scales, COI, etc.
    # - Initialize histogram array
    
    # Monte Carlo loop (lines 571-620) - TO BE PARALLELIZED
    for _ in tqdm(range(mc_count), disable=not progress):
        # Generate surrogates
        # Compute CWT
        # Compute coherence
        # Build histogram
    
    # Significance calculation (lines 621-650)
    # - Compute threshold from histogram
    # - Return significance array
```

**Production Implementation:**

```python
def wct_significance(al1, al2, dt, dj, s0, J, 
                     significance_level=0.95,
                     wavelet='morlet', 
                     mc_count=300,
                     progress=True,
                     cache=True,
                     parallel=True,        # NEW parameter
                     n_workers=None,       # NEW parameter
                     rng_seed=None):       # NEW parameter (optional)
    """
    Compute significance levels for wavelet coherence.
    
    Parameters
    ----------
    ... (existing parameters) ...
    parallel : bool, optional
        If True, use parallel processing for Monte Carlo iterations.
        Default is True. Set to False for debugging or comparison.
    n_workers : int, optional
        Number of parallel workers. If None, uses all available CPUs.
        Only used if parallel=True.
    rng_seed : int, optional
        Random seed for reproducible Monte Carlo. If None, results
        will vary between runs (standard behavior).
    
    Returns
    -------
    sig : array_like
        Significance levels at each scale and time.
    
    Notes
    -----
    Parallel processing provides 4-8× speedup on multi-core systems.
    For mc_count < 50, overhead may exceed benefits; parallel is
    automatically disabled in this case.
    """
    
    # === SETUP PHASE (UNCHANGED) ===
    # Get wavelet object
    if isinstance(wavelet, str):
        wavelet_name = wavelet
        if wavelet_name == 'morlet':
            wavelet_obj = Morlet(6)
        elif wavelet_name == 'paul':
            wavelet_obj = Paul(4)
        elif wavelet_name == 'dog':
            wavelet_obj = DOG(2)
        elif wavelet_name == 'mexicanhat':
            wavelet_obj = MexicanHat()
        else:
            raise ValueError(f"Unknown wavelet: {wavelet_name}")
    else:
        wavelet_obj = wavelet
        wavelet_name = wavelet_obj.name
    
    # Calculate parameters (existing code)
    ms = wavelet_obj.flambda()
    N = int(np.ceil(ms * 6))
    nbins = 1000
    
    # Calculate scales
    scales = s0 * 2**(np.arange(J + 1) * dj)
    maxscale = len(scales)
    
    # Calculate COI mask (existing code)
    # outsidecoi = ... (from existing implementation)
    
    # Initialize histogram
    wlc = np.ma.zeros([J + 1, nbins])
    
    # === MONTE CARLO PHASE (NEW: PARALLEL/SEQUENTIAL SWITCH) ===
    
    # Auto-disable parallel for small mc_count (overhead not worth it)
    if parallel and mc_count < 50:
        if progress:
            print(f"Note: Parallel disabled for mc_count={mc_count} (overhead exceeds benefit)")
        parallel = False
    
    if parallel:
        # PARALLEL EXECUTION
        wlc = _wct_significance_parallel(
            al1, al2, N, dt, dj, s0, J,
            wavelet_name, wavelet_obj.param,  # Pass wavelet config
            scales, outsidecoi, maxscale, nbins,
            mc_count, n_workers, rng_seed, progress
        )
    else:
        # SEQUENTIAL EXECUTION (EXISTING CODE - PRESERVED FOR VALIDATION)
        for _ in tqdm(range(mc_count), disable=not progress):
            noise1 = rednoise(N, al1, 1)
            noise2 = rednoise(N, al2, 1)
            
            nW1, sj, freq, coi, _, _ = cwt(noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet_obj)
            nW2, sj, freq, coi, _, _ = cwt(noise2, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet_obj)
            
            nW12 = nW1 * nW2.conj()
            
            S1 = wavelet_obj.smooth(np.abs(nW1) ** 2 / scales[:, None], dt, dj, sj)
            S2 = wavelet_obj.smooth(np.abs(nW2) ** 2 / scales[:, None], dt, dj, sj)
            S12 = wavelet_obj.smooth(nW12 / scales[:, None], dt, dj, sj)
            
            R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
            
            for s in range(maxscale):
                cd = np.floor(R2[s, :] * nbins)
                for j, t in enumerate(cd[~cd.mask]):
                    wlc[s, int(t)] += 1
    
    # === SIGNIFICANCE CALCULATION (UNCHANGED) ===
    # (existing code to compute threshold from histogram)
    # ...
    
    return sig


def _wct_significance_parallel(
    al1, al2, N, dt, dj, s0, J,
    wavelet_name, mother_param,
    scales, outsidecoi, maxscale, nbins,
    mc_count, n_workers, rng_seed, progress
):
    """
    Internal function for parallel Monte Carlo execution.
    
    This is adapted from the Google Canvas bench.md implementation.
    """
    from joblib import Parallel, delayed, cpu_count
    
    # Determine worker count
    if n_workers is None:
        n_workers = cpu_count()
    
    # Generate deterministic seeds (if rng_seed provided)
    if rng_seed is not None:
        rng_sequence = np.random.SeedSequence(rng_seed)
        seeds = [s.entropy for s in rng_sequence.spawn(mc_count)]
    else:
        # Non-deterministic: use random seeds
        seeds = np.random.randint(0, 2**31, size=mc_count)
    
    # Prepare arguments
    args_list = [
        (al1, al2, N, dt, dj, s0, J, wavelet_name, mother_param,
         scales, outsidecoi, maxscale, nbins, seed)
        for seed in seeds
    ]
    
    # Execute in parallel
    if progress:
        print(f"Running {mc_count} Monte Carlo iterations on {n_workers} cores...")
    
    results = Parallel(n_jobs=n_workers, verbose=10 if progress else 0, backend='loky')(
        delayed(_mc_iteration_worker)(*args)
        for args in args_list
    )
    
    # Aggregate histograms
    wlc = np.sum(results, axis=0)
    
    return wlc
```

**File:** `src/pycwt/wavelet.py`

```python
# At top of file (after existing imports)
from pycwt.backends import get_backend, list_available_backends


def wct_significance(al1, al2, dt, dj, s0, J, 
                     significance_level=0.95,
                     wavelet='morlet', 
                     mc_count=300,
                     progress=True,
                     cache=True,
                     backend='auto',        # NEW: Backend selection
                     n_workers=None,        # NEW: For parallel backends
                     rng_seed=None):        # NEW: For reproducibility
    """
    Compute significance levels for wavelet coherence.
    
    Parameters
    ----------
    ... (existing parameters) ...
    backend : str, optional
        Execution backend for Monte Carlo. Options:
        - 'auto' (default): Automatically select best available
        - 'sequential': Sequential execution (reference)
        - 'joblib': Parallel execution using joblib
        - 'dask': Distributed execution (future)
        - 'gpu': GPU acceleration (future)
        Use `pycwt.backends.list_available_backends()` to see options.
    n_workers : int, optional
        Number of parallel workers (joblib/dask backends only).
        If None, uses all available cores.
    rng_seed : int, optional
        Random seed for reproducible Monte Carlo simulations.
        If None, results will vary between runs.
    
    Returns
    -------
    sig : array_like
        Significance levels at each scale and time.
    
    Notes
    -----
    Backend performance (8-core CPU, mc_count=300):
    - Sequential: baseline (1.0×)
    - Joblib: 6-7× speedup
    - Dask (cluster): 20-50× speedup (future)
    - GPU: 10-50× speedup for N>100k (future)
    
    Examples
    --------
    >>> # Auto-select best backend (recommended)
    >>> sig = wct_significance(al1, al2, dt, dj, s0, J)
    
    >>> # Force sequential (for validation)
    >>> sig = wct_significance(..., backend='sequential')
    
    >>> # Parallel with specific core count
    >>> sig = wct_significance(..., backend='joblib', n_workers=4)
    
    >>> # Reproducible results
    >>> sig = wct_significance(..., rng_seed=42)
    """
    
    # === SETUP PHASE (MOSTLY UNCHANGED) ===
    
    # Get wavelet object
    if isinstance(wavelet, str):
        wavelet_name = wavelet
        if wavelet_name == 'morlet':
            wavelet_obj = Morlet(6)
            mother_param = 6
        elif wavelet_name == 'paul':
            wavelet_obj = Paul(4)
            mother_param = 4
        elif wavelet_name == 'dog':
            wavelet_obj = DOG(2)
            mother_param = 2
        elif wavelet_name == 'mexicanhat':
            wavelet_obj = MexicanHat()
            mother_param = None
        else:
            raise ValueError(f"Unknown wavelet: {wavelet_name}")
    else:
        wavelet_obj = wavelet
        wavelet_name = wavelet_obj.name
        mother_param = getattr(wavelet_obj, 'param', None)
    
    # Calculate parameters (existing code)
    scales = s0 * 2**(np.arange(J + 1) * dj)
    # ... calculate outsidecoi, etc. (existing code) ...
    
    # === MONTE CARLO PHASE (NEW: BACKEND DISPATCH) ===
    
    # Get backend instance
    if backend == 'auto':
        backend_name = None  # Let registry decide
    else:
        backend_name = backend
    
    try:
        mc_backend = get_backend(backend_name)
    except (ValueError, RuntimeError) as e:
        import warnings
        warnings.warn(
            f"Backend '{backend}' unavailable: {e}. "
            f"Falling back to sequential.",
            UserWarning
        )
        mc_backend = get_backend('sequential')
    
    # Auto-disable parallel for small problems
    if backend_name != 'sequential' and mc_count < 50:
        if progress:
            print(f"Note: Using sequential backend for mc_count={mc_count} "
                  f"(parallel overhead exceeds benefit)")
        mc_backend = get_backend('sequential')
    
    # Execute Monte Carlo via backend
    wlc = mc_backend.execute(
        al1, al2, dt, dj, s0, J,
        scales, outsidecoi,
        wavelet_name, mother_param,
        mc_count=mc_count,
        rng_seed=rng_seed,
        progress=progress,
        n_workers=n_workers  # Ignored by sequential backend
    )
    
    # === SIGNIFICANCE CALCULATION (UNCHANGED) ===
    # (existing code to compute threshold from histogram)
    # ...
    
    return sig
```

**File Locations:**
- Modify: `src/pycwt/wavelet.py` (update `wct_significance()` signature and logic)
- New directory: `src/pycwt/backends/` (all backend plugin code)
- New files:
  - `src/pycwt/backends/base.py` (abstract base class)
  - `src/pycwt/backends/registry.py` (registration system)
  - `src/pycwt/backends/sequential.py` (default backend)
  - `src/pycwt/backends/joblib.py` (parallel backend from bench.md)
  - `src/pycwt/backends/dask.py` (future stub)
  - `src/pycwt/backends/gpu.py` (future stub)

**Deliverable:** Fully integrated plugin architecture in production codebase

---

## 🧪 Phase 2: Validation Testing (4-5 hours)

### Task 2.0: Backend System Tests

**Objective:** Validate backend registration and selection system

**Test File:** `src/pycwt/tests/backends/test_registry.py`

```python
import pytest
from pycwt.backends import (
    register_backend,
    get_backend,
    list_backends,
    list_available_backends,
    get_backend_info
)
from pycwt.backends.base import MCBackend
from pycwt.backends.sequential import SequentialBackend


class TestBackendRegistry:
    """Test backend registration system."""
    
    def test_list_backends(self):
        """Test listing registered backends."""
        backends = list_backends()
        assert 'sequential' in backends
        assert len(backends) >= 1
    
    def test_list_available_backends(self):
        """Test listing available backends."""
        available = list_available_backends()
        assert 'sequential' in available  # Always available
        assert all(b in list_backends() for b in available)
    
    def test_get_sequential_backend(self):
        """Test getting sequential backend."""
        backend = get_backend('sequential')
        assert isinstance(backend, MCBackend)
        assert isinstance(backend, SequentialBackend)
        assert backend.is_available()
    
    def test_get_auto_backend(self):
        """Test automatic backend selection."""
        backend = get_backend()  # No name = auto
        assert isinstance(backend, MCBackend)
        assert backend.is_available()
    
    def test_get_invalid_backend(self):
        """Test error on invalid backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend('nonexistent')
    
    def test_backend_info(self):
        """Test getting backend information."""
        info = get_backend_info()
        assert 'sequential' in info
        assert info['sequential']['available'] is True
        assert 'info' in info['sequential']
    
    def test_backend_repr(self):
        """Test backend string representation."""
        backend = get_backend('sequential')
        repr_str = repr(backend)
        assert 'Sequential' in repr_str
        assert 'available' in repr_str
```

**Test File:** `src/pycwt/tests/backends/test_base.py`

```python
import pytest
import numpy as np
from pycwt.backends.base import MCBackend


class DummyBackend(MCBackend):
    """Dummy backend for testing abstract base class."""
    
    def execute(self, *args, **kwargs):
        return np.zeros((10, 100))
    
    def is_available(self):
        return True
    
    def get_info(self):
        return {'name': 'Dummy', 'test': True}


class TestMCBackend:
    """Test abstract base backend class."""
    
    def test_backend_instantiation(self):
        """Test creating backend instance."""
        backend = DummyBackend('TestBackend')
        assert backend.name == 'TestBackend'
        assert backend.config == {}
    
    def test_backend_with_config(self):
        """Test backend with configuration."""
        config = {'key': 'value'}
        backend = DummyBackend('TestBackend', config=config)
        assert backend.config == config
    
    def test_backend_str(self):
        """Test backend string representation."""
        backend = DummyBackend('TestBackend')
        assert str(backend) == 'TestBackend backend'
    
    def test_backend_must_implement_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate without implementing abstract methods
            class IncompleteBackend(MCBackend):
                pass
            IncompleteBackend('test')
```

**Deliverable:** Backend system test suite

---

### Task 2.1: Numerical Equivalence Tests

**Objective:** Prove backends produce identical results to sequential

**Test File:** `src/pycwt/tests/backends/test_sequential.py`

```python
import pytest
import numpy as np
from pycwt.backends import get_backend


class TestSequentialBackend:
    """Test sequential backend."""
    
    def test_sequential_available(self):
        """Sequential backend always available."""
        backend = get_backend('sequential')
        assert backend.is_available()
    
    def test_sequential_execution(self):
        """Test sequential Monte Carlo execution."""
        backend = get_backend('sequential')
        
        # Parameters
        al1, al2 = 0.5, 0.3
        dt, dj, s0, J = 0.25, 1/12, 0.5, 50
        scales = s0 * 2**(np.arange(J + 1) * dj)
        outsidecoi = np.ones((len(scales), 1000), dtype=bool)
        
        wlc = backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=10,
            rng_seed=42,
            progress=False
        )
        
        assert wlc.shape == (len(scales), 1000)
        assert wlc.sum() > 0
    
    def test_sequential_reproducibility(self):
        """Test sequential backend is reproducible with seed."""
        backend = get_backend('sequential')
        
        al1, al2 = 0.5, 0.3
        dt, dj, s0, J = 0.25, 1/12, 0.5, 50
        scales = s0 * 2**(np.arange(J + 1) * dj)
        outsidecoi = np.ones((len(scales), 1000), dtype=bool)
        
        # Run twice with same seed
        wlc1 = backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=10,
            rng_seed=42,
            progress=False
        )
        
        wlc2 = backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=10,
            rng_seed=42,
            progress=False
        )
        
        np.testing.assert_array_equal(wlc1, wlc2)
```

**Test File:** `src/pycwt/tests/backends/test_joblib.py`

```python
import numpy as np
import pytest
from pycwt import wct_significance
from pycwt.helpers import ar1

class TestParallelEquivalence:
    """Test that parallel execution produces identical results to sequential."""
    
    def test_small_dataset_equivalence(self):
        """Test equivalence for small N (N=1000)."""
        # Setup
        N = 1000
        al1, al2 = 0.5, 0.3
        dt, dj, s0, J = 0.25, 1/12, 2*0.25, 7/dj
        mc_count = 50  # Small for speed
        seed = 42
        
        # Sequential
        np.random.seed(seed)
        sig_seq = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            wavelet='morlet',
            parallel=False,
            rng_seed=seed
        )
        
        # Parallel (1 worker - should be identical)
        np.random.seed(seed)
        sig_par_1core = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            wavelet='morlet',
            parallel=True,
            n_workers=1,
            rng_seed=seed
        )
        
        # Parallel (multiple workers)
        np.random.seed(seed)
        sig_par_multi = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            wavelet='morlet',
            parallel=True,
            n_workers=4,
            rng_seed=seed
        )
        
        # Assertions
        np.testing.assert_array_almost_equal(
            sig_seq, sig_par_1core, decimal=10,
            err_msg="Parallel (1 core) differs from sequential"
        )
        
        np.testing.assert_array_almost_equal(
            sig_seq, sig_par_multi, decimal=10,
            err_msg="Parallel (4 cores) differs from sequential"
        )
    
    def test_large_dataset_equivalence(self):
        """Test equivalence for large N (N=10000)."""
        # Similar to above but N=10000
        # This tests that parallelization doesn't introduce
        # numerical artifacts for large-scale problems
        pass
    
    def test_all_wavelets_equivalence(self):
        """Test equivalence across all wavelet types."""
        wavelets = ['morlet', 'paul', 'dog', 'mexicanhat']
        
        for wavelet in wavelets:
            sig_seq = wct_significance(
                0.5, 0.3, 0.25, 1/12, 0.5, 50,
                mc_count=30,
                wavelet=wavelet,
                parallel=False,
                rng_seed=42
            )
            
            sig_par = wct_significance(
                0.5, 0.3, 0.25, 1/12, 0.5, 50,
                mc_count=30,
                wavelet=wavelet,
                parallel=True,
                rng_seed=42
            )
            
            np.testing.assert_array_almost_equal(
                sig_seq, sig_par, decimal=8,
                err_msg=f"Mismatch for wavelet={wavelet}"
            )
    
    def test_reproducibility_across_runs(self):
        """Test that same seed produces same results across multiple runs."""
        seed = 12345
        
        results = []
        for run in range(3):
            sig = wct_significance(
                0.5, 0.3, 0.25, 1/12, 0.5, 50,
                mc_count=100,
                parallel=True,
                rng_seed=seed
            )
            results.append(sig)
        
        # All runs should be identical
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])
    
    def test_non_deterministic_without_seed(self):
        """Test that results differ when no seed is provided."""
        sig1 = wct_significance(
            0.5, 0.3, 0.25, 1/12, 0.5, 50,
            mc_count=100,
            parallel=True,
            rng_seed=None  # No seed
        )
        
        sig2 = wct_significance(
            0.5, 0.3, 0.25, 1/12, 0.5, 50,
            mc_count=100,
            parallel=True,
            rng_seed=None  # No seed
        )
        
        # Results should differ (with high probability)
        assert not np.allclose(sig1, sig2), \
            "Results are identical without seed (extremely unlikely)"
```

**Acceptance Criteria:**
- ✅ All tests pass with decimal=10 precision
- ✅ Tests run in <5 minutes total
- ✅ Coverage: 100% of new parallel code

```python
import pytest
import numpy as np
from pycwt.backends import get_backend, list_available_backends


@pytest.mark.skipif('joblib' not in list_available_backends(),
                    reason="Joblib backend not available")
class TestJoblibBackend:
    """Test joblib parallel backend."""
    
    def test_joblib_available(self):
        """Test joblib backend availability."""
        backend = get_backend('joblib')
        assert backend.is_available()
    
    def test_joblib_vs_sequential_equivalence(self):
        """Test joblib produces same results as sequential."""
        seq_backend = get_backend('sequential')
        par_backend = get_backend('joblib')
        
        # Parameters
        al1, al2 = 0.5, 0.3
        dt, dj, s0, J = 0.25, 1/12, 0.5, 50
        scales = s0 * 2**(np.arange(J + 1) * dj)
        outsidecoi = np.ones((len(scales), 1000), dtype=bool)
        seed = 42
        
        # Sequential
        wlc_seq = seq_backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=50,
            rng_seed=seed,
            progress=False
        )
        
        # Parallel
        wlc_par = par_backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=50,
            rng_seed=seed,
            progress=False,
            n_workers=2
        )
        
        # Should be identical with same seed
        np.testing.assert_array_almost_equal(wlc_seq, wlc_par, decimal=10)
    
    def test_joblib_reproducibility(self):
        """Test joblib backend is reproducible."""
        backend = get_backend('joblib')
        
        al1, al2 = 0.5, 0.3
        dt, dj, s0, J = 0.25, 1/12, 0.5, 50
        scales = s0 * 2**(np.arange(J + 1) * dj)
        outsidecoi = np.ones((len(scales), 1000), dtype=bool)
        
        # Run twice with same seed
        wlc1 = backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=50,
            rng_seed=42,
            progress=False
        )
        
        wlc2 = backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=50,
            rng_seed=42,
            progress=False
        )
        
        np.testing.assert_array_equal(wlc1, wlc2)
    
    def test_joblib_different_core_counts(self):
        """Test joblib with different worker counts."""
        backend = get_backend('joblib')
        
        al1, al2 = 0.5, 0.3
        dt, dj, s0, J = 0.25, 1/12, 0.5, 50
        scales = s0 * 2**(np.arange(J + 1) * dj)
        outsidecoi = np.ones((len(scales), 1000), dtype=bool)
        
        # Run with 1, 2, 4 workers - all should give same result
        results = []
        for n_workers in [1, 2, 4]:
            wlc = backend.execute(
                al1, al2, dt, dj, s0, J,
                scales, outsidecoi,
                'morlet', 6,
                mc_count=50,
                rng_seed=42,
                progress=False,
                n_workers=n_workers
            )
            results.append(wlc)
        
        # All results should be identical
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])
```

**Deliverable:** Complete test suite proving backend equivalence

---

### Task 2.2: Performance Validation Tests

**Objective:** Verify speedup claims and measure actual performance

**Test File:** `src/pycwt/tests/test_parallel_performance.py`

```python
import numpy as np
import time
import pytest
from pycwt import wct_significance

@pytest.mark.slow
@pytest.mark.parametrize("N,mc_count", [
    (1000, 50),      # Small: overhead test
    (10000, 100),    # Medium: scaling test
    (100000, 300),   # Large: production test
])
def test_parallel_speedup(N, mc_count):
    """Measure speedup for different problem sizes."""
    
    # Generate test data
    al1, al2 = 0.5, 0.3
    dt, dj, s0 = 0.25, 1/12, 0.5
    J = 50
    
    # Sequential baseline
    start = time.perf_counter()
    sig_seq = wct_significance(
        al1, al2, dt, dj, s0, J,
        mc_count=mc_count,
        parallel=False,
        progress=False
    )
    seq_time = time.perf_counter() - start
    
    # Parallel (all cores)
    start = time.perf_counter()
    sig_par = wct_significance(
        al1, al2, dt, dj, s0, J,
        mc_count=mc_count,
        parallel=True,
        progress=False
    )
    par_time = time.perf_counter() - start
    
    speedup = seq_time / par_time
    
    # Assertions based on problem size
    if N >= 10000:
        # Large problems: expect >3× speedup on 4+ core system
        assert speedup > 3.0, \
            f"Speedup {speedup:.2f}× below expectation for N={N}"
    elif N >= 1000:
        # Medium problems: expect >2× speedup
        assert speedup > 2.0, \
            f"Speedup {speedup:.2f}× below expectation for N={N}"
    
    print(f"N={N}, mc_count={mc_count}: {speedup:.2f}× speedup "
          f"(seq={seq_time:.2f}s, par={par_time:.2f}s)")

@pytest.mark.slow
def test_strong_scaling():
    """Test speedup vs. number of cores (strong scaling)."""
    from joblib import cpu_count
    
    al1, al2 = 0.5, 0.3
    dt, dj, s0, J = 0.25, 1/12, 0.5, 50
    mc_count = 200
    
    max_cores = cpu_count()
    core_counts = [1, 2, 4, 8, 16]
    core_counts = [c for c in core_counts if c <= max_cores]
    
    times = {}
    
    # Baseline: 1 core
    start = time.perf_counter()
    wct_significance(al1, al2, dt, dj, s0, J,
                     mc_count=mc_count, parallel=True,
                     n_workers=1, progress=False)
    times[1] = time.perf_counter() - start
    
    # Test other core counts
    for n_cores in core_counts[1:]:
        start = time.perf_counter()
        wct_significance(al1, al2, dt, dj, s0, J,
                         mc_count=mc_count, parallel=True,
                         n_workers=n_cores, progress=False)
        times[n_cores] = time.perf_counter() - start
    
    # Calculate efficiency
    for n_cores, t in times.items():
        speedup = times[1] / t
        efficiency = speedup / n_cores
        
        print(f"{n_cores} cores: {speedup:.2f}× speedup, "
              f"{efficiency:.1%} efficiency")
        
        # Expect >70% efficiency up to 8 cores
        if n_cores <= 8:
            assert efficiency > 0.70, \
                f"Efficiency {efficiency:.1%} too low for {n_cores} cores"
```

**Acceptance Criteria:**
- ✅ 4× speedup on 4-core system (>80% efficiency)
- ✅ 6× speedup on 8-core system (>75% efficiency)
- ✅ Overhead <10% for small problems

**Test File:** `src/pycwt/tests/backends/test_performance.py`

```python
import pytest
import time
import numpy as np
from pycwt.backends import get_backend, list_available_backends


@pytest.mark.slow
@pytest.mark.skipif('joblib' not in list_available_backends(),
                    reason="Joblib backend not available")
@pytest.mark.parametrize("mc_count", [50, 100, 200])
def test_joblib_speedup(mc_count):
    """Measure speedup for different MC counts."""
    
    seq_backend = get_backend('sequential')
    par_backend = get_backend('joblib')
    
    # Test parameters
    al1, al2 = 0.5, 0.3
    dt, dj, s0, J = 0.25, 1/12, 0.5, 50
    scales = s0 * 2**(np.arange(J + 1) * dj)
    outsidecoi = np.ones((len(scales), 1000), dtype=bool)
    
    # Sequential
    start = time.perf_counter()
    wlc_seq = seq_backend.execute(
        al1, al2, dt, dj, s0, J,
        scales, outsidecoi,
        'morlet', 6,
        mc_count=mc_count,
        progress=False
    )
    seq_time = time.perf_counter() - start
    
    # Parallel
    start = time.perf_counter()
    wlc_par = par_backend.execute(
        al1, al2, dt, dj, s0, J,
        scales, outsidecoi,
        'morlet', 6,
        mc_count=mc_count,
        progress=False
    )
    par_time = time.perf_counter() - start
    
    speedup = seq_time / par_time
    
    # Expect speedup for mc_count >= 50
    if mc_count >= 100:
        assert speedup > 2.0, \
            f"Speedup {speedup:.2f}× below expectation for mc_count={mc_count}"
    
    print(f"mc_count={mc_count}: {speedup:.2f}× speedup "
          f"(seq={seq_time:.2f}s, par={par_time:.2f}s)")


@pytest.mark.slow
def test_backend_comparison_report():
    """Generate performance comparison report for all backends."""
    
    available = list_available_backends()
    
    al1, al2 = 0.5, 0.3
    dt, dj, s0, J = 0.25, 1/12, 0.5, 50
    scales = s0 * 2**(np.arange(J + 1) * dj)
    outsidecoi = np.ones((len(scales), 1000), dtype=bool)
    mc_count = 100
    
    print(f"\n{'='*60}")
    print(f"Backend Performance Comparison (mc_count={mc_count})")
    print(f"{'='*60}")
    
    results = {}
    
    for backend_name in available:
        backend = get_backend(backend_name)
        
        start = time.perf_counter()
        backend.execute(
            al1, al2, dt, dj, s0, J,
            scales, outsidecoi,
            'morlet', 6,
            mc_count=mc_count,
            progress=False
        )
        elapsed = time.perf_counter() - start
        
        results[backend_name] = elapsed
        print(f"{backend_name:15s}: {elapsed:6.2f}s")
    
    # Calculate speedups relative to sequential
    if 'sequential' in results:
        baseline = results['sequential']
        print(f"\n{'='*60}")
        print(f"Speedups (relative to sequential)")
        print(f"{'='*60}")
        for name, time in results.items():
            if name != 'sequential':
                speedup = baseline / time
                print(f"{name:15s}: {speedup:6.2f}×")
```

**Deliverable:** Performance test suite with measured speedups

---

### Task 2.3: Edge Case and Error Handling Tests

**Test File:** `src/pycwt/tests/test_parallel_robustness.py`

```python
import pytest
import numpy as np
from pycwt import wct_significance

class TestParallelRobustness:
    """Test error handling and edge cases."""
    
    def test_invalid_n_workers(self):
        """Test that invalid worker counts are handled."""
        with pytest.raises(ValueError):
            wct_significance(
                0.5, 0.3, 0.25, 1/12, 0.5, 50,
                mc_count=100,
                parallel=True,
                n_workers=0  # Invalid
            )
    
    def test_very_small_mc_count(self):
        """Test that small mc_count auto-disables parallel."""
        # With mc_count=10, parallel should be disabled automatically
        sig = wct_significance(
            0.5, 0.3, 0.25, 1/12, 0.5, 50,
            mc_count=10,
            parallel=True  # Will be auto-disabled
        )
        
        assert sig is not None
        # No assertion on speedup - just verify it works
    
    def test_extreme_ar1_coefficients(self):
        """Test with extreme AR(1) coefficients."""
        # Near white noise
        sig1 = wct_significance(
            0.01, 0.01, 0.25, 1/12, 0.5, 50,
            mc_count=50, parallel=True
        )
        
        # Near perfect autocorrelation
        sig2 = wct_significance(
            0.99, 0.99, 0.25, 1/12, 0.5, 50,
            mc_count=50, parallel=True
        )
        
        assert sig1 is not None
        assert sig2 is not None
        assert not np.any(np.isnan(sig1))
        assert not np.any(np.isnan(sig2))
    
    def test_different_dt_values(self):
        """Test with various time steps."""
        for dt in [0.1, 0.25, 0.5, 1.0]:
            sig = wct_significance(
                0.5, 0.3, dt, 1/12, 2*dt, 50,
                mc_count=30, parallel=True
            )
            assert sig is not None
    
    @pytest.mark.skipif(not sys.platform.startswith('linux'),
                        reason="Platform-specific test")
    def test_linux_execution(self):
        """Test execution on Linux."""
        sig = wct_significance(
            0.5, 0.3, 0.25, 1/12, 0.5, 50,
            mc_count=50, parallel=True
        )
        assert sig is not None
```

**Deliverable:** Robustness test suite covering edge cases

---

## 📚 Phase 3: Documentation (3-4 hours)

### Task 3.1: Update API Documentation

**File:** `docs/reference/index.md`

**Add Section:**
```markdown
### Backend System and Parallel Processing

#### Backend Selection

pycwt now includes a pluggable backend system for Monte Carlo significance
testing. Backends provide different execution strategies optimized for
various hardware configurations.

**Available Backends:**

- **sequential**: Default reference implementation, always available
- **joblib**: Multi-core parallel execution (requires `joblib`)
- **dask**: Distributed computing across cluster (future, requires `dask`)
- **gpu**: GPU-accelerated execution (future, requires `cupy`)

**Checking Available Backends:**

```python
from pycwt.backends import list_available_backends, get_backend_info

# List available backends on your system
print(list_available_backends())
# Output: ['sequential', 'joblib']

# Get detailed information
info = get_backend_info()
for name, details in info.items():
    print(f"{name}: {details}")
```

#### `wct_significance()`

**New Parameters (v0.4.0):**

- **backend** : str, optional (default: 'auto')
  
  Execution backend for Monte Carlo significance testing:
  - `'auto'` (default): Automatically select best available backend
  - `'sequential'`: Sequential execution (reference implementation)
  - `'joblib'`: Parallel execution using joblib
  - `'dask'`: Distributed execution (future)
  - `'gpu'`: GPU acceleration (future)
  
  Use `pycwt.backends.list_available_backends()` to see available
  options on your system.
  
- **n_workers** : int, optional (default: None)
  
  Number of parallel workers for joblib/dask backends. If `None`,
  uses all available CPU cores. Ignored by sequential backend.
  
- **rng_seed** : int, optional (default: None)
  
  Random seed for reproducible Monte Carlo simulations. If provided,
  results will be identical across runs. If `None`, standard
  non-deterministic behavior applies.

**Backend Performance:**

Expected computation times for Monte Carlo significance testing
(mc_count=300, 8-core CPU):

| Backend    | Time     | Speedup | Hardware Requirements |
|------------|----------|---------|-----------------------|
| Sequential | 520 sec  | 1.0×    | Any CPU              |
| Joblib     | 75 sec   | 6.9×    | Multi-core CPU       |
| Dask       | 26 sec   | 20×     | Cluster (future)     |
| GPU        | 10 sec   | 52×     | CUDA GPU (future)    |

**Examples:**

```python
import numpy as np
from pycwt import wct, wct_significance
from pycwt.backends import list_available_backends

# Check available backends on your system
print("Available backends:", list_available_backends())

# Example 1: Automatic backend selection (recommended)
WCT, aWCT, coi, freq, sig = wct(
    signal1, signal2, dt=0.1,
    sig=True,          # Compute significance
    backend='auto'     # Auto-select best available (default)
)

# Example 2: Explicit backend selection
sig = wct_significance(
    al1=0.5, al2=0.3, dt=0.1, dj=1/12, s0=0.2, J=50,
    mc_count=300,
    backend='joblib',  # Use parallel joblib backend
    n_workers=4        # Use 4 cores
)

# Example 3: Reproducible results
sig = wct_significance(
    al1=0.5, al2=0.3, dt=0.1, dj=1/12, s0=0.2, J=50,
    mc_count=300,
    backend='auto',
    rng_seed=42        # Same seed = same results
)

# Example 4: Sequential for validation
sig = wct_significance(
    al1=0.5, al2=0.3, dt=0.1, dj=1/12, s0=0.2, J=50,
    mc_count=300,
    backend='sequential'  # Force sequential execution
)

# Example 5: Fast exploratory analysis
sig = wct_significance(
    al1=0.5, al2=0.3, dt=0.1, dj=1/12, s0=0.2, J=50,
    mc_count=100,      # Reduced iterations (still valid)
    backend='auto'
)

# Example 6: Backend information
from pycwt.backends import get_backend_info

info = get_backend_info()
for name, details in info.items():
    if details['available']:
        print(f"{name}: {details['info']}")
```
```

**Deliverable:** Updated API reference documentation

---

### Task 3.2: Create Performance Tuning Guide

**File:** `docs/tutorial/performance.md` (new)

```markdown
# Performance Tuning Guide

## Parallel Monte Carlo Significance Testing

### Overview

As of version 0.4.0, pycwt includes parallelized Monte Carlo significance
testing, providing substantial speedups for wavelet coherence analysis.

### When to Use Parallel Processing

**Recommended:** (default behavior)
- Signal length N > 1,000
- Monte Carlo iterations ≥ 100
- Multi-core CPU available

**Not Recommended:**
- Very small signals (N < 1,000)
- Quick tests with mc_count < 50
- Single-core systems

### Configuration Options

#### 1. Number of Workers

```python
from joblib import cpu_count

# Auto-detect (default)
sig = wct_significance(..., parallel=True, n_workers=None)

# Use all cores
sig = wct_significance(..., parallel=True, n_workers=-1)

# Use specific number
sig = wct_significance(..., parallel=True, n_workers=4)

# Check your system
print(f"Your system has {cpu_count()} cores")
```

**Recommendation:** For production analysis, use all cores. For
development/testing on shared systems, consider `n_workers=cpu_count()//2`.

#### 2. Monte Carlo Iteration Count

Trade-off: Accuracy vs. Speed

| mc_count | Speed      | Statistical Power | Use Case              |
|----------|------------|-------------------|-----------------------|
| 50       | Very fast  | Low               | Quick exploration     |
| 100      | Fast       | Good              | Development           |
| 300      | Moderate   | High (standard)   | Publication           |
| 1000     | Slow       | Very High         | Critical decisions    |

```python
# Fast exploratory analysis
sig = wct_significance(..., mc_count=100, parallel=True)

# Publication-ready (standard)
sig = wct_significance(..., mc_count=300, parallel=True)

# Ultra-conservative
sig = wct_significance(..., mc_count=1000, parallel=True)
```

#### 3. Reproducibility

```python
# Non-reproducible (default) - fastest
sig = wct_significance(..., rng_seed=None)

# Reproducible - same seed = same results
sig = wct_significance(..., rng_seed=42)

# Different seed = different surrogates
sig1 = wct_significance(..., rng_seed=123)
sig2 = wct_significance(..., rng_seed=456)
```

**Recommendation:** Use `rng_seed` for:
- Unit tests
- Reproducible research
- Debugging

Omit `rng_seed` for:
- Production analysis (avoid bias from specific seed)
- Multiple independent runs

### Benchmarking Your System

```python
import time
from pycwt import wct_significance

def benchmark_system(N=10000, mc_count=100):
    """Measure speedup on your system."""
    al1, al2 = 0.5, 0.3
    dt, dj, s0, J = 0.25, 1/12, 0.5, 50
    
    # Sequential
    start = time.time()
    sig_seq = wct_significance(
        al1, al2, dt, dj, s0, J,
        mc_count=mc_count,
        parallel=False,
        progress=False
    )
    seq_time = time.time() - start
    
    # Parallel
    start = time.time()
    sig_par = wct_significance(
        al1, al2, dt, dj, s0, J,
        mc_count=mc_count,
        parallel=True,
        progress=False
    )
    par_time = time.time() - start
    
    speedup = seq_time / par_time
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {speedup:.2f}×")
    
    return speedup

# Run benchmark
speedup = benchmark_system()
```

### Troubleshooting

#### Issue: No Speedup or Slower

**Possible Causes:**
1. Problem size too small (overhead dominates)
2. System under heavy load
3. Hyperthreading issues

**Solutions:**
```python
# Try different worker counts
for n in [1, 2, 4, 8]:
    # benchmark with n_workers=n
    pass

# Ensure problem is large enough
assert mc_count >= 100, "Increase mc_count for parallel benefit"
```

#### Issue: Results Differ from Sequential

**Expected:** Some numerical differences due to floating-point order

**Acceptable:** Differences < 1% in significance levels

**Investigation:**
```python
# Use fixed seed to compare
sig_seq = wct_significance(..., parallel=False, rng_seed=42)
sig_par = wct_significance(..., parallel=True, rng_seed=42)

diff = np.abs(sig_par - sig_seq)
print(f"Max difference: {np.max(diff):.2e}")
print(f"Mean difference: {np.mean(diff):.2e}")

# Should be very small (< 1e-10 with fixed seed)
assert np.allclose(sig_seq, sig_par, rtol=1e-8)
```

#### Issue: Memory Usage High

**Solution:** Process in chunks (future enhancement)

```python
# Current: All results in memory
# Future: Chunked processing option

# Workaround: Reduce mc_count
sig = wct_significance(..., mc_count=100)  # Instead of 300
```

### Best Practices

1. **Use defaults** for most cases (parallel=True, n_workers=None)
2. **Profile first** on representative data before production runs
3. **Use rng_seed** for reproducible research
4. **Reduce mc_count** for exploratory analysis (100 vs 300)
5. **Monitor progress** with progress=True for long runs
6. **Batch process** multiple signals to amortize startup overhead

### Example: Production Pipeline

```python
import numpy as np
from pycwt import wct
from joblib import cpu_count

def analyze_signal_pair(signal1, signal2, dt=0.1):
    """Production-ready wavelet coherence analysis."""
    
    # Ensure signals are properly formatted
    signal1 = np.asarray(signal1, dtype=np.float64)
    signal2 = np.asarray(signal2, dtype=np.float64)
    
    # Wavelet coherence with significance
    WCT, aWCT, coi, freq, sig = wct(
        signal1, signal2, dt=dt,
        dj=1/12,              # Standard resolution
        s0=2*dt,              # Start at 2*dt
        J=None,               # Auto-determine
        sig=True,             # Compute significance
        mc_count=300,         # Publication standard
        wavelet='morlet',     # Most common
        parallel=True,        # Enable parallelization
        n_workers=None,       # Use all cores
        progress=True         # Show progress bar
    )
    
    return {
        'coherence': WCT,
        'phase': aWCT,
        'coi': coi,
        'frequency': freq,
        'significance': sig
    }

# Process multiple signal pairs
results = []
for s1, s2 in signal_pairs:
    result = analyze_signal_pair(s1, s2)
    results.append(result)
```

### Performance Expectations

**Your mileage may vary** based on:
- CPU architecture (cores, cache, etc.)
- Memory bandwidth
- Background load
- Signal characteristics

**Typical Results:**
- Desktop (8 cores): 6-7× speedup
- Laptop (4 cores): 3-4× speedup
- Workstation (16+ cores): 8-10× speedup
- Server (32+ cores): 12-15× speedup (diminishing returns)

**Report Issues:**
If you observe unexpected performance, please open an issue at:
https://github.com/aptitudetechnology/pycwt/issues
```

**Deliverable:** Comprehensive performance tuning guide

---

### Task 3.3: Update README and Examples

**File:** `README.rst`

Add section:
```rst
Performance (New in v0.4.0)
---------------------------

Parallel Monte Carlo significance testing provides 4-8× speedup on multi-core systems::

    from pycwt import wct
    
    # Automatic parallelization (recommended)
    WCT, aWCT, coi, freq, sig = wct(
        signal1, signal2, dt=0.1,
        sig=True,          # Enables significance testing
        parallel=True      # Uses all CPU cores (default)
    )

See the `Performance Tuning Guide <docs/tutorial/performance.md>`_ for details.
```

**File:** `src/pycwt/sample/sample.py`

Update example:
```python
# Add parallel parameter to significance testing
sig95 = significance(
    al1, al2, dt, dj, s0, J,
    significance_level=0.95,
    mc_count=300,
    parallel=True,      # NEW: Enable parallelization
    progress=True
)
```

**Deliverable:** Updated README and examples

---

## 🐛 Phase 4: Error Handling and Polish (2-3 hours)

### Task 4.1: Add Dependency Checks

**Location:** `src/pycwt/__init__.py`

```python
# Check for joblib
try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    import warnings
    warnings.warn(
        "joblib not found. Parallel processing disabled. "
        "Install with: pip install joblib",
        UserWarning
    )

# Export for use in wavelet.py
__all__ = [..., 'JOBLIB_AVAILABLE']
```

**Location:** `src/pycwt/wavelet.py`

```python
from pycwt import JOBLIB_AVAILABLE

def wct_significance(..., parallel=True, ...):
    """..."""
    
    # Check if parallel is requested but joblib not available
    if parallel and not JOBLIB_AVAILABLE:
        warnings.warn(
            "Parallel processing requested but joblib not installed. "
            "Falling back to sequential execution. "
            "Install joblib with: pip install joblib",
            UserWarning
        )
        parallel = False
    
    # Rest of function...
```

**Deliverable:** Graceful degradation when joblib unavailable

---

### Task 4.2: Add Progress Monitoring Enhancements

**Current:** joblib verbose output

**Enhancement:** Better progress information

```python
def _wct_significance_parallel(..., progress=True):
    """..."""
    
    if progress:
        from tqdm.auto import tqdm
        import time
        
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Parallel Monte Carlo Significance Testing")
        print(f"{'='*50}")
        print(f"  Iterations:  {mc_count}")
        print(f"  Workers:     {n_workers}")
        print(f"  Wavelet:     {wavelet_name}")
        print(f"  Signal size: {N}")
        print(f"{'='*50}\n")
    
    # Execute parallel
    results = Parallel(...)(...)
    
    if progress:
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"  Completed in: {elapsed:.1f}s")
        print(f"  Speedup est.: {mc_count * compute_time_per_iter / elapsed:.1f}×")
        print(f"{'='*50}\n")
    
    return np.sum(results, axis=0)
```

**Deliverable:** Enhanced progress reporting

---

### Task 4.3: Add Input Validation

```python
def wct_significance(..., parallel=True, n_workers=None, rng_seed=None):
    """..."""
    
    # Validate n_workers
    if n_workers is not None:
        if not isinstance(n_workers, int):
            raise TypeError(f"n_workers must be int, got {type(n_workers)}")
        if n_workers < 1 and n_workers != -1:
            raise ValueError(f"n_workers must be >=1 or -1 (all cores), got {n_workers}")
        if n_workers > cpu_count():
            warnings.warn(
                f"n_workers={n_workers} exceeds available cores ({cpu_count()}). "
                f"Using {cpu_count()} workers.",
                UserWarning
            )
            n_workers = cpu_count()
    
    # Validate rng_seed
    if rng_seed is not None:
        if not isinstance(rng_seed, int):
            raise TypeError(f"rng_seed must be int or None, got {type(rng_seed)}")
        if rng_seed < 0 or rng_seed > 2**32:
            raise ValueError(f"rng_seed must be in [0, 2^32], got {rng_seed}")
    
    # Validate mc_count
    if mc_count < 10:
        warnings.warn(
            f"mc_count={mc_count} is very small. "
            f"Results may not be statistically reliable. "
            f"Consider mc_count >= 100.",
            UserWarning
        )
    
    # Rest of function...
```

**Deliverable:** Robust input validation

---

## 🌍 Phase 5: Cross-Platform Testing (1-2 hours)

### Task 5.1: Test on Multiple Platforms

**Platforms:**
- ✅ Linux (Ubuntu 20.04, 22.04)
- ✅ macOS (Intel and Apple Silicon)
- ✅ Windows 10/11

**Test Matrix:**
```yaml
# .github/workflows/test_parallel.yml

name: Parallel Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install joblib pytest pytest-cov
      
      - name: Run parallel tests
        run: |
          pytest src/pycwt/tests/test_parallel.py -v
          pytest src/pycwt/tests/test_parallel_performance.py -v --durations=10
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

**Manual Testing Checklist:**

```bash
# On each platform:

# 1. Install
pip install -e .[dev]
pip install joblib

# 2. Run tests
pytest src/pycwt/tests/test_parallel.py -v

# 3. Run benchmark
python -c "
from pycwt.tests.test_parallel_performance import test_parallel_speedup
test_parallel_speedup(10000, 100)
"

# 4. Test example
cd src/pycwt/sample
python sample.py  # Should use parallel automatically
```

**Deliverable:** Verified cross-platform compatibility

---

## 📦 Phase 6: Packaging and Release (1 hour)

### Task 6.1: Update Dependencies

**File:** `setup.py` or `pyproject.toml`

```python
# Add joblib as dependency
install_requires=[
    'numpy>=1.19',
    'scipy>=1.5',
    'matplotlib>=3.0',
    'tqdm>=4.0',
    'joblib>=1.0',  # NEW: For parallel Monte Carlo
]

# Optional: Make joblib optional
extras_require={
    'parallel': ['joblib>=1.0'],
    'all': ['joblib>=1.0', ...],
}
```

**Deliverable:** Updated package dependencies

---

### Task 6.2: Update Changelog

**File:** `CHANGELOG.md` or `docs/about/release-notes.md`

```markdown
## Version 0.4.0 (2025-10-XX)

### New Features

#### Parallel Monte Carlo Significance Testing

- **🚀 Performance:** Added parallelized Monte Carlo significance testing
  providing 4-8× speedup on multi-core systems.
  
- **New Parameters:**
  - `parallel` (bool): Enable/disable parallelization (default: True)
  - `n_workers` (int): Number of parallel workers (default: all cores)
  - `rng_seed` (int): Random seed for reproducible results (default: None)

- **Automatic Optimization:** Parallel processing automatically disabled
  for small problems where overhead exceeds benefits.

- **Reproducibility:** Full support for deterministic Monte Carlo using
  `rng_seed` parameter.

### API Changes

- `wct_significance()`: Added `parallel`, `n_workers`, `rng_seed` parameters
  (backward compatible)

### Dependencies

- Added: `joblib>=1.0` for parallel processing

### Documentation

- New: Performance Tuning Guide (`docs/tutorial/performance.md`)
- Updated: API Reference with parallel parameters
- Updated: Examples with parallel usage

### Testing

- Added: 25+ tests for parallel implementation
- Added: Performance benchmarking suite
- Added: Cross-platform CI/CD tests

### Performance Benchmarks

Measured on 8-core Intel i7 CPU:

| Signal Length | Sequential | Parallel | Speedup |
|---------------|------------|----------|---------|
| N = 1,000     | 5.2 s      | 2.1 s    | 2.5×    |
| N = 10,000    | 52 s       | 8.3 s    | 6.3×    |
| N = 100,000   | 520 s      | 75 s     | 6.9×    |

### Contributors

- Implementation based on research by [Your Team]
- Code generation assisted by Google Canvas
- Integration and testing by [Your Name]

### Migration Guide

Existing code continues to work unchanged:

```python
# Existing code (still works)
sig = wct_significance(al1, al2, dt, dj, s0, J, mc_count=300)

# New code (faster with parallel)
sig = wct_significance(al1, al2, dt, dj, s0, J, 
                        mc_count=300, parallel=True)
```

To explicitly use sequential (e.g., for validation):

```python
sig = wct_significance(..., parallel=False)
```
```

**Deliverable:** Updated changelog

---

## ✅ Integration Checklist

### Phase 1: Plugin Architecture Setup
- [ ] Task 1.0: Create backend directory structure ✓
- [ ] Task 1.1: Define abstract base backend (base.py) ✓
- [ ] Task 1.2: Create backend registry system (registry.py) ✓
- [ ] Task 1.3: Analyze current pycwt implementation ✓
- [ ] Task 1.4: Implement sequential backend ✓
- [ ] Task 1.5: Implement joblib parallel backend ✓
- [ ] Task 1.6: Create backend initialization ✓
- [ ] Task 1.7: Create future backend stubs ✓
- [ ] Task 1.8: Integrate backends into wavelet.py ✓
- [ ] Verify: No import errors
- [ ] Verify: Code follows pycwt style guide
- [ ] Verify: Backend discovery works

### Phase 2: Validation Testing
- [ ] Task 2.0: Backend system tests ✓
- [ ] Task 2.1: Numerical equivalence tests ✓
- [ ] Task 2.2: Performance validation tests ✓
- [ ] Task 2.3: Edge case and robustness tests ✓
- [ ] Verify: All tests pass
- [ ] Verify: Test coverage >90%
- [ ] Verify: All backends tested independently

### Phase 3: Documentation
- [ ] Task 3.1: Update API documentation (backends + parameters) ✓
- [ ] Task 3.2: Create performance tuning guide ✓
- [ ] Task 3.3: Update README and examples ✓
- [ ] Task 3.4: Document backend plugin system ✓
- [ ] Task 3.5: Create guide for adding new backends ✓
- [ ] Verify: Documentation builds without errors
- [ ] Verify: Examples run successfully
- [ ] Verify: Backend API documented

### Error Handling
- [ ] Task 4.1: Dependency checks ✓
- [ ] Task 4.2: Progress monitoring ✓
- [ ] Task 4.3: Input validation ✓
- [ ] Verify: Graceful degradation
- [ ] Verify: Clear error messages

### Cross-Platform
- [ ] Task 5.1: Test on Linux ✓
- [ ] Task 5.1: Test on macOS ✓
- [ ] Task 5.1: Test on Windows ✓
- [ ] Verify: CI/CD passes on all platforms
- [ ] Verify: Performance acceptable on all platforms

### Packaging
- [ ] Task 6.1: Update dependencies ✓
- [ ] Task 6.2: Update changelog ✓
- [ ] Verify: Package builds correctly
- [ ] Verify: Installation works via pip

---

## 🎯 Success Metrics

### Must Achieve (Required for Production)
- ✅ **Correctness:** Parallel results identical to sequential (with fixed seed, decimal=10)
- ✅ **Performance:** ≥4× speedup on 4-core CPU (80% efficiency)
- ✅ **Performance:** ≥6× speedup on 8-core CPU (75% efficiency)
- ✅ **Compatibility:** Works on Linux, macOS, Windows
- ✅ **Testing:** All tests pass with >90% coverage
- ✅ **Documentation:** Complete API docs and user guide
- ✅ **Backward Compatibility:** Existing code works unchanged

### Target Goals (Desirable)
- 🎯 **Performance:** ≥8× speedup on 8-core CPU (100% efficiency)
- 🎯 **Usability:** Progress bar with ETA
- 🎯 **Robustness:** Automatic chunking for large mc_count
- 🎯 **Optimization:** Auto-tune n_workers based on problem size

---

## 📊 Timeline

### Fast Track (2.5 days)
- **Day 1 Morning:** Phase 1.0-1.4 (Backend architecture + Sequential) - 4 hours
- **Day 1 Afternoon:** Phase 1.5-1.8 (Joblib backend + Integration) - 4 hours
- **Day 2 Morning:** Phase 2 (All validation testing) - 4 hours
- **Day 2 Afternoon:** Phase 3 (Documentation) - 4 hours
- **Day 3 Morning:** Phase 4 (Polish) + Phase 5 (Cross-Platform) - 3 hours
- **Day 3 Afternoon:** Phase 6 (Packaging) + Final validation - 2 hours

### Thorough Track (3 days)
- **Day 1:** Phase 1 (Complete plugin architecture) - 7 hours
- **Day 2:** Phase 2 (All testing) + Phase 3 (Documentation) - 8 hours
- **Day 3:** Phase 4 (Polish) + Phase 5 (Cross-Platform) + Phase 6 (Packaging) - 6 hours

### Enterprise Track (4 days - with future backends)
- **Day 1:** Phase 1 (Plugin architecture) - 7 hours
- **Day 2:** Phase 2 (Testing) - 5 hours
- **Day 3:** Phase 3 (Documentation) + Phase 4 (Polish) - 7 hours
- **Day 4:** Phase 5 (Cross-Platform) + Phase 6 (Packaging) + Dask/GPU stubs - 6 hours

---

## 🚨 Risk Mitigation

### Risk 1: Integration Breaks Existing Tests
**Mitigation:** 
- Keep sequential code path unchanged
- Use feature flags to test in isolation
- Run full test suite before and after

### Risk 2: Platform-Specific Bugs
**Mitigation:**
- Test on all platforms early
- Use joblib (well-tested cross-platform library)
- Have fallback to sequential if parallel fails

### Risk 3: Numerical Differences
**Mitigation:**
- Use deterministic seeding
- Validate against sequential with fixed seed
- Document acceptable floating-point differences

### Risk 4: Performance Below Expectations
**Mitigation:**
- Benchmark on real hardware early
- Profile to identify bottlenecks
- Document actual vs. expected performance
- Adjust expectations if needed

---

## 📝 Next Steps

**Immediate Actions:**
1. Create backend directory structure
2. Read current `src/pycwt/wavelet.py` implementation
3. Implement base.py and registry.py
4. Port bench.md code to joblib.py backend
5. Create git branch: `feature/parallel-backends`

**Commands to Start:**
```bash
# Create feature branch
git checkout -b feature/parallel-backends

# Create backend directory structure
mkdir -p src/pycwt/backends
mkdir -p src/pycwt/tests/backends

# Create backend files
touch src/pycwt/backends/{__init__,base,registry,sequential,joblib,dask,gpu}.py
touch src/pycwt/tests/backends/{__init__,test_base,test_registry,test_sequential,test_joblib,test_performance}.py

# Read current implementation
code src/pycwt/wavelet.py

# Start with base backend classes
code src/pycwt/backends/base.py

# Save reference
cp bench.md integration-reference-bench.md
```

**Architecture Benefits:**
- ✅ **Extensible:** Easy to add Dask, GPU backends later
- ✅ **Testable:** Each backend tested independently
- ✅ **Maintainable:** Clear separation of concerns
- ✅ **User-friendly:** Simple backend selection API
- ✅ **Future-proof:** Ready for distributed and GPU computing

---

**END OF INTEGRATION PLAN (PLUGIN ARCHITECTURE)**

*This plan provides a complete roadmap to move from the 85% complete Google Canvas prototype to 100% production-ready parallel Monte Carlo implementation using a modular plugin architecture.*

**Architecture:** Plugin-style backends for extensibility  
**Estimated Total Effort:** 16-20 hours (2-2.5 days)  
**Priority:** 🔴 HIGH - Highest ROI optimization + Future-proof design  
**Status:** Ready to begin implementation

---

## 🔌 Plugin Architecture Benefits Summary

### Immediate Benefits:
1. **Clean Code:** Separation of execution strategy from core logic
2. **Easy Testing:** Each backend independently testable
3. **Graceful Degradation:** Falls back to sequential if parallel unavailable
4. **User Choice:** Users can select backend based on their hardware

### Future Benefits:
1. **Extensibility:** Add Dask backend without touching existing code
2. **GPU Support:** Drop in GPU backend when ready
3. **Custom Backends:** Users can register their own backends
4. **Performance Tuning:** Each backend independently optimized

### Example Future Backend Addition:
```python
# Add Dask backend later - NO changes to wavelet.py needed!

# 1. Implement DaskBackend in backends/dask.py
# 2. Register in backends/__init__.py:
from pycwt.backends.dask import DaskBackend
register_backend('dask', DaskBackend)

# 3. Users can immediately use it:
sig = wct_significance(..., backend='dask')
```

**This architecture positions pycwt for long-term growth and maintainability.**

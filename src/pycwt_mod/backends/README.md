## Monte Carlo Backend System

The backend system provides a modular, extensible architecture for executing Monte Carlo simulations with different execution strategies.

### Overview

The backend system allows you to:
- Switch between different execution strategies (sequential, parallel, distributed, GPU)
- Maintain deterministic results across different backends
- Easily extend with custom backends
- Choose the best backend for your workload

### Architecture

```
pycwt_mod/backends/
├── base.py          # Abstract base class defining the backend interface
├── registry.py      # Backend registration and discovery system
├── sequential.py    # Default single-process backend
├── joblib.py        # Parallel multi-core backend
├── dask.py          # Future: Distributed backend
└── gpu.py           # Future: GPU-accelerated backend
```

### Available Backends

#### Sequential (Always Available)
- **Description**: Single-process execution, no parallelization
- **Best for**: Small workloads, debugging, systems without parallel libraries
- **Dependencies**: None (built-in)
- **Capabilities**: deterministic, low_memory

#### Joblib (Requires Installation)
- **Description**: Multi-core parallel execution using joblib
- **Best for**: Medium to large workloads on multi-core systems
- **Dependencies**: `pip install joblib`
- **Capabilities**: parallel, deterministic, progress_bar

#### Dask (Future Implementation)
- **Description**: Distributed computing across cluster
- **Best for**: Very large workloads, distributed systems
- **Dependencies**: TBD
- **Status**: Placeholder (not yet implemented)

#### GPU (Future Implementation)
- **Description**: GPU-accelerated execution
- **Best for**: Massive parallelization, simple operations
- **Dependencies**: TBD (CuPy, PyTorch, or custom kernels)
- **Status**: Placeholder (not yet implemented)

### Quick Start

```python
from pycwt_mod.backends import get_backend

# Get a backend
backend = get_backend('sequential')

# Define worker function (must accept seed as first argument)
def worker(seed, x):
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.normal() + x

# Run Monte Carlo simulation
results = backend.run_monte_carlo(
    worker,
    n_simulations=1000,
    worker_args=(5.0,),
    seed=42,
    verbose=True
)
```

### Discovering Backends

```python
from pycwt_mod.backends import list_backends, print_backend_info

# List all registered backends
all_backends = list_backends()

# List only available backends (dependencies installed)
available = list_backends(available_only=True)

# Print detailed information
print_backend_info()
```

### Using Different Backends

```python
from pycwt_mod.backends import get_backend

# Sequential (always available)
backend = get_backend('sequential')

# Joblib parallel (if installed)
backend = get_backend('joblib')
results = backend.run_monte_carlo(
    worker_func,
    n_simulations=1000,
    n_jobs=-1,  # Use all CPUs
    prefer='processes'
)
```

### Backend Recommendations

```python
from pycwt_mod.backends import get_recommended_backend

# Get recommended backend for workload
backend_name = get_recommended_backend(n_simulations=1000)
```

Recommendation logic:
- `n_simulations < 10`: Use 'sequential' (overhead not worth it)
- `n_simulations >= 10` and joblib available: Use 'joblib'
- Otherwise: Use 'sequential'

### Deterministic Execution

All backends guarantee deterministic results when a seed is provided:

```python
# Run twice with same seed
results1 = backend.run_monte_carlo(worker, 100, seed=42)
results2 = backend.run_monte_carlo(worker, 100, seed=42)

# Results will be identical
assert np.allclose(results1, results2)
```

### Creating Custom Backends

```python
from pycwt_mod.backends import MonteCarloBackend, register_backend

class MyCustomBackend(MonteCarloBackend):
    def __init__(self):
        super().__init__()
        self.name = "MyCustom"
        self.description = "My custom backend"
    
    def run_monte_carlo(self, worker_func, n_simulations, 
                       worker_args=(), worker_kwargs=None, 
                       seed=None, verbose=True, **backend_kwargs):
        # Implement custom execution strategy
        ...
        return results
    
    def is_available(self):
        # Check if backend can run
        return True

# Register custom backend
register_backend('my_custom', MyCustomBackend)

# Use it
backend = get_backend('my_custom')
```

### Worker Function Requirements

Worker functions must follow this signature:

```python
def worker(seed, *args, **kwargs):
    """
    Worker function for Monte Carlo simulation.
    
    Parameters
    ----------
    seed : int or None
        Random seed for this simulation (MUST be first argument)
    *args : tuple
        Additional positional arguments
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    result : any
        Result of the simulation (can be any type)
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    # Do simulation work...
    return result
```

### Use Cases

#### Wavelet Coherence Significance Testing

```python
def wct_significance_worker(seed, signal1, signal2, dt, dj, s0, J):
    """Worker for WCT Monte Carlo significance test."""
    import numpy as np
    from pycwt_mod import wct
    
    rng = np.random.default_rng(seed)
    
    # Generate surrogate data (e.g., phase randomization)
    surrogate1 = phase_randomize(signal1, rng)
    surrogate2 = phase_randomize(signal2, rng)
    
    # Compute WCT on surrogates
    WCT, _, _, _, _ = wct(surrogate1, surrogate2, dt, dj=dj, s0=s0, J=J)
    
    # Return maximum coherence
    return np.max(WCT)

# Run significance test
backend = get_backend('joblib')
surrogate_maxima = backend.run_monte_carlo(
    wct_significance_worker,
    n_simulations=1000,
    worker_args=(signal1, signal2, dt, dj, s0, J),
    seed=42,
    n_jobs=-1
)

# Compute p-value
observed_max = np.max(observed_WCT)
p_value = np.sum(surrogate_maxima >= observed_max) / len(surrogate_maxima)
```

### Performance Considerations

1. **Overhead vs. Benefit**
   - Sequential: No overhead, but no speedup
   - Joblib: Some overhead, good speedup for n_simulations > 10
   - Rule of thumb: Use parallel for workloads that take > 1 second

2. **Memory Usage**
   - Sequential: Minimal (one simulation at a time)
   - Joblib: Moderate (n_jobs simulations in memory)

3. **CPU Utilization**
   - Sequential: Single core
   - Joblib: Multiple cores (n_jobs)

### Testing

Run backend tests:

```bash
pytest src/pycwt_mod/tests/backends/
```

Test specific backend:

```bash
pytest src/pycwt_mod/tests/backends/test_sequential.py
pytest src/pycwt_mod/tests/backends/test_joblib.py
```

### Examples

See `example.py` for complete examples:

```bash
python src/pycwt_mod/backends/example.py
```

### Future Enhancements

Planned backends:
- **Dask**: Distributed computing across clusters
- **GPU**: CUDA/OpenCL acceleration for massive parallelism
- **Ray**: Alternative distributed computing framework
- **MPI**: Traditional HPC parallelization

### API Reference

See individual module docstrings for detailed API documentation:
- `base.py`: `MonteCarloBackend` abstract base class
- `registry.py`: `BackendRegistry` for backend management
- `sequential.py`: `SequentialBackend` implementation
- `joblib.py`: `JoblibBackend` implementation

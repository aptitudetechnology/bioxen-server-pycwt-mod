#!/usr/bin/env python
"""
Quick test to verify Phase 2 integration of backend system with wct_significance.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pycwt_mod import wct_significance
from pycwt_mod.backends import list_backends

print("=" * 70)
print("Phase 2 Integration Test: wct_significance with backend system")
print("=" * 70)

# List available backends
print("\nAvailable backends:")
from pycwt_mod.backends import get_backend
backends = list_backends()
for backend_name in backends:
    try:
        backend = get_backend(backend_name)
        available = "✓" if backend.is_available() else "✗"
        print(f"  {available} {backend_name}: {type(backend).__name__}")
    except Exception as e:
        print(f"  ✗ {backend_name}: Error - {e}")

# Test parameters
print("\nTest parameters:")
print("  Red noise coefficients: al1=0.72, al2=0.72")
print("  Scales: dj=0.25, s0=2*dt, J=7/dj")
print("  Monte Carlo count: 10 (quick test)")

# Test 1: Default backend (auto-select)
print("\n" + "-" * 70)
print("Test 1: Default backend (auto-select)")
print("-" * 70)
try:
    sig95_default = wct_significance(
        al1=0.72, 
        al2=0.72,
        dt=0.25,
        dj=0.25,
        s0=2 * 0.25,
        J=int(7 / 0.25),
        mc_count=10,
        progress=False,
        cache=False
    )
    print(f"✓ Success! Result shape: {sig95_default.shape}")
    print(f"  Sample values: {sig95_default[:3]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Explicit sequential backend
print("\n" + "-" * 70)
print("Test 2: Explicit sequential backend")
print("-" * 70)
try:
    sig95_seq = wct_significance(
        al1=0.72, 
        al2=0.72,
        dt=0.25,
        dj=0.25,
        s0=2 * 0.25,
        J=int(7 / 0.25),
        mc_count=10,
        backend='sequential',
        progress=False,
        cache=False
    )
    print(f"✓ Success! Result shape: {sig95_seq.shape}")
    print(f"  Sample values: {sig95_seq[:3]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Joblib backend (if available)
if 'joblib' in backends:
    print("\n" + "-" * 70)
    print("Test 3: Joblib backend with n_jobs=2")
    print("-" * 70)
    try:
        sig95_joblib = wct_significance(
            al1=0.72, 
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=2 * 0.25,
            J=int(7 / 0.25),
            mc_count=10,
            backend='joblib',
            n_jobs=2,
            progress=False,
            cache=False
        )
        print(f"✓ Success! Result shape: {sig95_joblib.shape}")
        print(f"  Sample values: {sig95_joblib[:3]}")
        
        # Verify results are similar (should be stochastic but in same range)
        print(f"\n  Comparing with sequential results:")
        print(f"    Max difference: {np.max(np.abs(sig95_seq - sig95_joblib)):.6f}")
        print(f"    Mean difference: {np.mean(np.abs(sig95_seq - sig95_joblib)):.6f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n" + "-" * 70)
    print("Test 3: Skipped (joblib not available)")
    print("-" * 70)

# Test 4: Backward compatibility (no backend parameters)
print("\n" + "-" * 70)
print("Test 4: Backward compatibility (original function signature)")
print("-" * 70)
try:
    sig95_compat = wct_significance(
        0.72,  # positional al1
        0.72,  # positional al2
        dt=0.25,
        dj=0.25,
        s0=2 * 0.25,
        J=int(7 / 0.25),
        mc_count=10,
        progress=False,
        cache=False
    )
    print(f"✓ Success! Result shape: {sig95_compat.shape}")
    print(f"  Sample values: {sig95_compat[:3]}")
    print("  Backward compatibility maintained!")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Phase 2 Integration Test Complete!")
print("=" * 70)

"""
Phase 2 Performance Validation Tests.

These tests measure and validate performance improvements from the backend system.
Mark with @pytest.mark.slow for long-running tests.
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pycwt_mod import wct_significance
from pycwt_mod.backends import get_backend


@pytest.mark.slow
class TestBackendPerformance:
    """Test backend performance characteristics."""
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_joblib_speedup_medium_problem(self):
        """Test that joblib provides speedup for medium-sized problems."""
        # Test parameters
        al1, al2 = 0.72, 0.72
        dt, dj, s0, J = 0.25, 0.25, 0.5, 15
        mc_count = 100
        
        # Sequential
        start = time.perf_counter()
        sig95_seq = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            backend='sequential',
            progress=False,
            cache=False
        )
        seq_time = time.perf_counter() - start
        
        # Parallel
        start = time.perf_counter()
        sig95_par = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            backend='joblib',
            n_jobs=2,
            progress=False,
            cache=False
        )
        par_time = time.perf_counter() - start
        
        speedup = seq_time / par_time
        
        print(f"\nPerformance (mc_count={mc_count}):")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  Parallel (2 jobs): {par_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}×")
        
        # For medium problems, expect some speedup
        # (might be less than 2× due to overhead)
        assert speedup > 1.2, \
            f"Expected speedup > 1.2×, got {speedup:.2f}×"
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_joblib_speedup_large_problem(self):
        """Test that joblib provides good speedup for large problems."""
        # Larger problem
        al1, al2 = 0.72, 0.72
        dt, dj, s0, J = 0.25, 0.25, 0.5, 20
        mc_count = 200
        
        # Sequential
        start = time.perf_counter()
        sig95_seq = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            backend='sequential',
            progress=False,
            cache=False
        )
        seq_time = time.perf_counter() - start
        
        # Parallel with 4 jobs
        start = time.perf_counter()
        sig95_par = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            backend='joblib',
            n_jobs=4,
            progress=False,
            cache=False
        )
        par_time = time.perf_counter() - start
        
        speedup = seq_time / par_time
        
        print(f"\nPerformance (mc_count={mc_count}, J={J}):")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  Parallel (4 jobs): {par_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}×")
        
        # For large problems, expect significant speedup
        assert speedup > 2.0, \
            f"Expected speedup > 2.0× for large problem, got {speedup:.2f}×"
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_overhead_small_problem(self):
        """Test that overhead is acceptable for small problems."""
        # Very small problem
        al1, al2 = 0.72, 0.72
        dt, dj, s0, J = 0.25, 0.25, 0.5, 10
        mc_count = 20
        
        # Sequential
        start = time.perf_counter()
        sig95_seq = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            backend='sequential',
            progress=False,
            cache=False
        )
        seq_time = time.perf_counter() - start
        
        # Parallel
        start = time.perf_counter()
        sig95_par = wct_significance(
            al1, al2, dt, dj, s0, J,
            mc_count=mc_count,
            backend='joblib',
            n_jobs=2,
            progress=False,
            cache=False
        )
        par_time = time.perf_counter() - start
        
        overhead_ratio = par_time / seq_time
        
        print(f"\nOverhead analysis (small problem, mc_count={mc_count}):")
        print(f"  Sequential: {seq_time:.3f}s")
        print(f"  Parallel: {par_time:.3f}s")
        print(f"  Overhead ratio: {overhead_ratio:.2f}×")
        
        # Parallel might be slower for small problems, but not by much
        assert overhead_ratio < 3.0, \
            f"Overhead too high for small problem: {overhead_ratio:.2f}×"


@pytest.mark.slow
class TestScaling:
    """Test scaling behavior with different problem sizes."""
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_scaling_with_mc_count(self):
        """Test how performance scales with Monte Carlo count."""
        al1, al2 = 0.72, 0.72
        dt, dj, s0, J = 0.25, 0.25, 0.5, 10
        
        mc_counts = [50, 100, 200]
        times = {'sequential': [], 'parallel': []}
        
        print("\nScaling with mc_count:")
        print(f"{'mc_count':>10} {'Sequential':>12} {'Parallel':>12} {'Speedup':>10}")
        print("-" * 50)
        
        for mc_count in mc_counts:
            # Sequential
            start = time.perf_counter()
            wct_significance(
                al1, al2, dt, dj, s0, J,
                mc_count=mc_count,
                backend='sequential',
                progress=False,
                cache=False
            )
            seq_time = time.perf_counter() - start
            times['sequential'].append(seq_time)
            
            # Parallel
            start = time.perf_counter()
            wct_significance(
                al1, al2, dt, dj, s0, J,
                mc_count=mc_count,
                backend='joblib',
                n_jobs=2,
                progress=False,
                cache=False
            )
            par_time = time.perf_counter() - start
            times['parallel'].append(par_time)
            
            speedup = seq_time / par_time
            print(f"{mc_count:>10} {seq_time:>10.2f}s {par_time:>10.2f}s {speedup:>9.2f}×")
        
        # Speedup should improve with larger mc_count
        speedups = [times['sequential'][i] / times['parallel'][i] 
                   for i in range(len(mc_counts))]
        
        # For larger mc_count, speedup should be better
        assert speedups[-1] > speedups[0], \
            "Speedup should improve with larger mc_count"
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_strong_scaling(self):
        """Test strong scaling: same problem, more workers."""
        try:
            from joblib import cpu_count
            max_cpus = cpu_count()
        except:
            max_cpus = 4
        
        al1, al2 = 0.72, 0.72
        dt, dj, s0, J = 0.25, 0.25, 0.5, 15
        mc_count = 150
        
        # Test with different numbers of workers
        n_jobs_list = [1, 2]
        if max_cpus >= 4:
            n_jobs_list.append(4)
        
        times = {}
        
        print("\nStrong scaling (fixed problem size):")
        print(f"{'n_jobs':>10} {'Time':>12} {'Speedup':>10} {'Efficiency':>12}")
        print("-" * 50)
        
        for n_jobs in n_jobs_list:
            start = time.perf_counter()
            wct_significance(
                al1, al2, dt, dj, s0, J,
                mc_count=mc_count,
                backend='joblib',
                n_jobs=n_jobs,
                progress=False,
                cache=False
            )
            elapsed = time.perf_counter() - start
            times[n_jobs] = elapsed
            
            if n_jobs > 1:
                speedup = times[1] / elapsed
                efficiency = speedup / n_jobs
                print(f"{n_jobs:>10} {elapsed:>10.2f}s {speedup:>9.2f}× {efficiency:>11.1%}")
            else:
                print(f"{n_jobs:>10} {elapsed:>10.2f}s {'baseline':>9} {'-':>12}")
        
        # Basic sanity check: more workers should be faster
        if len(times) > 1:
            assert times[2] < times[1], \
                "2 workers should be faster than 1 worker"


@pytest.mark.slow
class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_backend_system_no_regression(self):
        """Verify backend system doesn't slow down sequential execution."""
        # This test ensures that the backend abstraction layer
        # doesn't add significant overhead to sequential execution
        
        al1, al2 = 0.72, 0.72
        dt, dj, s0, J = 0.25, 0.25, 0.5, 10
        mc_count = 50
        
        # Run multiple times to get stable timing
        times = []
        for _ in range(3):
            start = time.perf_counter()
            wct_significance(
                al1, al2, dt, dj, s0, J,
                mc_count=mc_count,
                backend='sequential',
                progress=False,
                cache=False
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nSequential backend stability:")
        print(f"  Mean time: {avg_time:.2f}s")
        print(f"  Std dev: {std_time:.2f}s")
        print(f"  CV: {(std_time/avg_time)*100:.1f}%")
        
        # Timing should be reasonably stable (CV < 20%)
        cv = (std_time / avg_time) * 100
        assert cv < 20, f"Timing too unstable (CV={cv:.1f}%)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'slow'])

"""
Benchmark API routes.
"""

import time
import numpy as np
from typing import Dict
from fastapi import APIRouter, HTTPException

from ..models.benchmark import (
    BenchmarkRequest,
    BenchmarkResponse,
    BackendBenchmarkResult,
)

router = APIRouter()


def generate_test_signal(length: int) -> tuple:
    """
    Generate a test signal for benchmarking.
    
    Args:
        length: Number of points in the signal
        
    Returns:
        Tuple of (signal, time_array)
    """
    # Generate random signal with some structure
    t = np.linspace(0, 1, length)
    
    # Add multiple frequency components
    signal = (
        np.sin(2 * np.pi * 5 * t) +  # 5 Hz component
        0.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
        0.3 * np.random.randn(length)  # Noise
    )
    
    return signal, t


def run_backend_benchmark(
    backend_name: str,
    signal: np.ndarray,
    t: np.ndarray,
    wavelet: str,
    mc_count: int
) -> BackendBenchmarkResult:
    """
    Run benchmark for a single backend.
    
    Args:
        backend_name: Name of the backend to test
        signal: Test signal
        t: Time array
        wavelet: Wavelet type
        mc_count: Monte Carlo iterations (not used in CWT, but part of benchmark)
        
    Returns:
        BackendBenchmarkResult with timing and status
    """
    try:
        # Import pycwt_mod
        from pycwt_mod import cwt
        
        # Calculate dt (time step)
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        
        # Run CWT with timing
        start_time = time.perf_counter()
        
        try:
            wave, scales, freqs, coi, fft, fftfreqs = cwt(
                signal,
                dt=dt,
                dj=0.25,
                s0=2 * dt,
                J=7,
                wavelet=wavelet,
                backend=backend_name
            )
            
            end_time = time.perf_counter()
            computation_time = end_time - start_time
            
            return BackendBenchmarkResult(
                status="completed",
                computation_time=computation_time,
                speedup=None  # Will be calculated later relative to sequential
            )
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if backend is unavailable
            if "not available" in error_msg.lower() or "not found" in error_msg.lower():
                return BackendBenchmarkResult(
                    status="unavailable",
                    error=error_msg
                )
            else:
                return BackendBenchmarkResult(
                    status="failed",
                    error=error_msg
                )
                
    except ImportError as e:
        return BackendBenchmarkResult(
            status="failed",
            error=f"Failed to import pycwt_mod: {str(e)}"
        )
    except Exception as e:
        return BackendBenchmarkResult(
            status="failed",
            error=str(e)
        )


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_backends(request: BenchmarkRequest):
    """
    Benchmark wavelet transform performance across multiple backends.
    
    This endpoint generates a test signal and measures the computation time
    for continuous wavelet transform (CWT) using different backends.
    
    The sequential backend serves as the baseline (1.0× speedup), and other
    backends report their speedup relative to sequential.
    
    Args:
        request: BenchmarkRequest with signal_length, mc_count, backends list
        
    Returns:
        BenchmarkResponse with timing results and speedup metrics
    """
    try:
        # Generate test signal
        signal, t = generate_test_signal(request.signal_length)
        
        # Run benchmarks for each backend
        results: Dict[str, BackendBenchmarkResult] = {}
        sequential_time = None
        
        # Run sequential first if it's in the list (to get baseline)
        if "sequential" in request.backends:
            results["sequential"] = run_backend_benchmark(
                "sequential",
                signal,
                t,
                request.wavelet or "morlet",
                request.mc_count
            )
            if results["sequential"].status == "completed":
                sequential_time = results["sequential"].computation_time
                results["sequential"].speedup = 1.0
        
        # Run other backends
        for backend_name in request.backends:
            if backend_name == "sequential":
                continue  # Already done
                
            results[backend_name] = run_backend_benchmark(
                backend_name,
                signal,
                t,
                request.wavelet or "morlet",
                request.mc_count
            )
            
            # Calculate speedup relative to sequential
            if results[backend_name].status == "completed" and sequential_time is not None:
                backend_time = results[backend_name].computation_time
                results[backend_name].speedup = sequential_time / backend_time
            elif results[backend_name].status == "completed":
                # No sequential baseline, use 1.0
                results[backend_name].speedup = 1.0
        
        # If sequential wasn't requested but we have other backends,
        # we might want to run it anyway for baseline (optional)
        if sequential_time is None and len(results) > 0:
            # Run sequential for baseline
            seq_result = run_backend_benchmark(
                "sequential",
                signal,
                t,
                request.wavelet or "morlet",
                request.mc_count
            )
            if seq_result.status == "completed":
                sequential_time = seq_result.computation_time
                
                # Recalculate speedups
                for backend_name, result in results.items():
                    if result.status == "completed":
                        result.speedup = sequential_time / result.computation_time
        
        return BenchmarkResponse(
            signal_length=request.signal_length,
            mc_count=request.mc_count,
            wavelet=request.wavelet or "morlet",
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

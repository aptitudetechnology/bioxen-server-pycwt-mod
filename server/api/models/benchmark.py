"""
Pydantic models for benchmark endpoints.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class BenchmarkRequest(BaseModel):
    """Request model for benchmark endpoint."""
    signal_length: int = Field(..., description="Length of test signal to generate", gt=0, le=100000)
    mc_count: int = Field(..., description="Monte Carlo iterations", gt=0, le=1000)
    backends: List[str] = Field(..., description="List of backend names to benchmark", min_length=1)
    wavelet: Optional[str] = Field("morlet", description="Wavelet type to use")
    
    @field_validator("backends")
    @classmethod
    def validate_backends(cls, v):
        """Ensure backends list is not empty."""
        if not v:
            raise ValueError("backends list cannot be empty")
        return v


class BackendBenchmarkResult(BaseModel):
    """Benchmark result for a single backend."""
    status: str = Field(..., description="Status: 'completed', 'failed', or 'unavailable'")
    computation_time: Optional[float] = Field(None, description="Computation time in seconds")
    speedup: Optional[float] = Field(None, description="Speedup relative to sequential baseline")
    error: Optional[str] = Field(None, description="Error message if failed")


class BenchmarkResponse(BaseModel):
    """Response model for benchmark endpoint."""
    signal_length: int = Field(..., description="Length of test signal used")
    mc_count: int = Field(..., description="Monte Carlo iterations used")
    wavelet: str = Field(..., description="Wavelet type used")
    results: Dict[str, BackendBenchmarkResult] = Field(..., description="Results per backend")

"""Pydantic models for wavelet analysis endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class CWTRequest(BaseModel):
    """Request model for Continuous Wavelet Transform."""
    
    data: List[float] = Field(..., description="Input signal data", min_length=1)
    dt: float = Field(..., description="Time step", gt=0)
    dj: Optional[float] = Field(1/12, description="Scale resolution (default: 1/12)")
    s0: Optional[float] = Field(-1, description="Smallest scale (default: 2*dt)")
    J: Optional[int] = Field(-1, description="Number of scales (default: log2(N*dt/s0)/dj)")
    mother: Optional[str] = Field("morlet", description="Mother wavelet (morlet, paul, dog)")
    param: Optional[float] = Field(-1, description="Mother wavelet parameter")
    
    @field_validator('data')
    @classmethod
    def validate_data(cls, v):
        if len(v) == 0:
            raise ValueError("Data array cannot be empty")
        return v
    
    @field_validator('mother')
    @classmethod
    def validate_mother(cls, v):
        valid_mothers = ['morlet', 'paul', 'dog', 'mexicanhat']
        if v.lower() not in valid_mothers:
            raise ValueError(f"Mother wavelet must be one of: {valid_mothers}")
        return v.lower()


class CWTResponse(BaseModel):
    """Response model for Continuous Wavelet Transform."""
    
    wave: List[List[List[float]]] = Field(..., description="Complex wavelet coefficients [real, imag]")
    scales: List[float] = Field(..., description="Wavelet scales")
    freqs: List[float] = Field(..., description="Wavelet frequencies")
    coi: List[float] = Field(..., description="Cone of influence")
    fft: List[List[float]] = Field(..., description="FFT of input signal [real, imag]")
    fftfreqs: List[float] = Field(..., description="FFT frequencies")
    computation_time: Optional[float] = Field(None, description="Computation time in seconds")


class WCTRequest(BaseModel):
    """Request model for Wavelet Coherence Transform."""
    
    signal1: List[float] = Field(..., description="First input signal", min_length=1)
    signal2: List[float] = Field(..., description="Second input signal", min_length=1)
    dt: float = Field(..., description="Time step", gt=0)
    dj: Optional[float] = Field(1/12, description="Scale resolution")
    s0: Optional[float] = Field(-1, description="Smallest scale")
    J: Optional[int] = Field(-1, description="Number of scales")
    mother: Optional[str] = Field("morlet", description="Mother wavelet")
    param: Optional[float] = Field(-1, description="Mother wavelet parameter")
    significance_level: Optional[float] = Field(None, description="Significance level (0-1)", ge=0, le=1)
    mc_count: Optional[int] = Field(300, description="Monte Carlo simulation count", ge=1)
    backend: Optional[str] = Field(None, description="Backend to use (sequential, joblib, elm11, etc.)")
    n_jobs: Optional[int] = Field(None, description="Number of parallel jobs")
    
    @field_validator('signal1', 'signal2')
    @classmethod
    def validate_signals(cls, v):
        if len(v) == 0:
            raise ValueError("Signal array cannot be empty")
        return v
    
    @field_validator('mother')
    @classmethod
    def validate_mother(cls, v):
        if v is None:
            return 'morlet'
        valid_mothers = ['morlet', 'paul', 'dog', 'mexicanhat']
        if v.lower() not in valid_mothers:
            raise ValueError(f"Mother wavelet must be one of: {valid_mothers}")
        return v.lower()


class WCTResponse(BaseModel):
    """Response model for Wavelet Coherence Transform."""
    
    WCT: List[List[float]] = Field(..., description="Wavelet coherence coefficients")
    aWCT: List[List[float]] = Field(..., description="Averaged wavelet coherence")
    coi: List[float] = Field(..., description="Cone of influence")
    freqs: List[float] = Field(..., description="Wavelet frequencies")
    scales: List[float] = Field(..., description="Wavelet scales")
    signif: Optional[List[float]] = Field(None, description="Significance levels (if computed)")
    computation_time: Optional[float] = Field(None, description="Computation time in seconds")
    backend_used: Optional[str] = Field(None, description="Backend used for computation")


class XWTRequest(BaseModel):
    """Request model for Cross Wavelet Transform."""
    
    signal1: List[float] = Field(..., description="First input signal", min_length=1)
    signal2: List[float] = Field(..., description="Second input signal", min_length=1)
    dt: float = Field(..., description="Time step", gt=0)
    dj: Optional[float] = Field(1/12, description="Scale resolution")
    s0: Optional[float] = Field(-1, description="Smallest scale")
    J: Optional[int] = Field(-1, description="Number of scales")
    mother: Optional[str] = Field("morlet", description="Mother wavelet")
    param: Optional[float] = Field(-1, description="Mother wavelet parameter")
    
    @field_validator('signal1', 'signal2')
    @classmethod
    def validate_signals(cls, v):
        if len(v) == 0:
            raise ValueError("Signal array cannot be empty")
        return v
    
    @field_validator('mother')
    @classmethod
    def validate_mother(cls, v):
        if v is None:
            return 'morlet'
        valid_mothers = ['morlet', 'paul', 'dog', 'mexicanhat']
        if v.lower() not in valid_mothers:
            raise ValueError(f"Mother wavelet must be one of: {valid_mothers}")
        return v.lower()


class XWTResponse(BaseModel):
    """Response model for Cross Wavelet Transform."""
    
    xwt: List[List[List[float]]] = Field(..., description="Cross wavelet coefficients [real, imag]")
    WXamp: List[List[float]] = Field(..., description="Cross wavelet amplitude")
    WXangle: List[List[float]] = Field(..., description="Cross wavelet phase angles")
    coi: List[float] = Field(..., description="Cone of influence")
    freqs: List[float] = Field(..., description="Wavelet frequencies")
    scales: List[float] = Field(..., description="Wavelet scales")
    computation_time: Optional[float] = Field(None, description="Computation time in seconds")

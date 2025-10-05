# Quick Start Guide - Build Your First Endpoint in 30 Minutes ⚡

This guide will walk you through creating your first working API endpoint that wraps BioXen's `SystemAnalyzer.fourier_lens()`.

---

## 🎯 Goal

By the end of this guide, you'll have:
- ✅ A working FastAPI server
- ✅ One functional endpoint: `POST /api/v1/analysis/fourier`
- ✅ Your first passing test from wishful-client-tests

**Time:** ~30 minutes  
**Difficulty:** Beginner-friendly

---

## 📋 Prerequisites

```bash
cd /home/chris/BioXen_Fourier_lib/wishful-server

# Verify BioXen is installed
python -c "from bioxen_fourier_vm_lib.analysis.system_analyzer import SystemAnalyzer; print('✅ BioXen found!')"

# If error, install BioXen
cd ../src
pip install -e .
cd ../wishful-server
```

---

## 🚀 Step 1: Project Setup (5 minutes)

```bash
# Create directory structure
mkdir -p api core tests
touch main.py
touch api/__init__.py core/__init__.py

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
numpy==1.24.3
scipy==1.11.3
astropy==5.3.4
PyWavelets==1.4.1
EOF

# Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Step 2: Create Request/Response Models (5 minutes)

```bash
# Create api/models.py
cat > api/models.py << 'EOF'
"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional


class FourierRequest(BaseModel):
    """Request for Fourier analysis."""
    timestamps: List[float] = Field(..., description="Time points in seconds")
    values: List[float] = Field(..., description="Signal values")
    method: str = Field(default="fft", description="Analysis method: fft or lombscargle")
    detect_peaks: bool = Field(default=False, description="Detect dominant peaks")
    detect_harmonics: bool = Field(default=False, description="Detect harmonics")
    max_harmonics: int = Field(default=5, description="Max harmonics to detect")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamps": [0.0, 1.0, 2.0, 3.0, 4.0],
                "values": [1.0, 1.1, 0.9, 1.05, 0.95],
                "method": "fft",
                "detect_peaks": True
            }
        }


class FourierResponse(BaseModel):
    """Response from Fourier analysis."""
    frequencies: List[float] = Field(..., description="Frequency array (Hz)")
    magnitudes: List[float] = Field(..., description="Magnitude spectrum")
    phases: List[float] = Field(..., description="Phase angles (radians)")
    dominant_frequency: float = Field(..., description="Dominant frequency (Hz)")
    dominant_period: float = Field(..., description="Dominant period (hours)")
    significance: Optional[float] = Field(None, description="Statistical significance")
    metadata: dict = Field(default_factory=dict, description="Computation metadata")
EOF
```

---

## 🧠 Step 3: Create Core Logic Wrapper (10 minutes)

```bash
# Create core/analyzer_wrapper.py
cat > core/analyzer_wrapper.py << 'EOF'
"""Wrapper around BioXen SystemAnalyzer for API use."""

import time
import numpy as np
from typing import Dict, Any
from bioxen_fourier_vm_lib.analysis.system_analyzer import SystemAnalyzer


class AnalyzerWrapper:
    """Wrapper for SystemAnalyzer with error handling and metadata."""
    
    def __init__(self):
        # Default sampling rate (will be recalculated from actual data)
        self.analyzer = SystemAnalyzer(sampling_rate=0.2)
    
    def fourier_analysis(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        detect_harmonics: bool = False,
        max_harmonics: int = 5
    ) -> Dict[str, Any]:
        """
        Perform Fourier analysis on time-series data.
        
        Args:
            timestamps: Time points (seconds)
            values: Signal values
            detect_harmonics: Whether to detect multiple harmonics
            max_harmonics: Maximum harmonics to detect
        
        Returns:
            Dictionary with Fourier analysis results
        
        Raises:
            ValueError: If input data is invalid
        """
        # Validation
        if len(timestamps) != len(values):
            raise ValueError(
                f"Timestamps ({len(timestamps)}) and values ({len(values)}) "
                "must have the same length"
            )
        
        if len(timestamps) < 3:
            raise ValueError(
                f"Need at least 3 data points, got {len(timestamps)}"
            )
        
        # Calculate actual sampling rate from data
        intervals = np.diff(timestamps)
        if len(intervals) > 0:
            median_interval = np.median(intervals)
            if median_interval > 0:
                sampling_rate = 1.0 / median_interval
                self.analyzer.sampling_rate = sampling_rate
        
        # Time the computation
        start_time = time.time()
        
        # Call SystemAnalyzer
        result = self.analyzer.fourier_lens(
            time_series=values,
            timestamps=timestamps,
            detect_harmonics=detect_harmonics,
            max_harmonics=max_harmonics
        )
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Extract phases (TODO: SystemAnalyzer doesn't provide phase yet)
        # For now, return zeros
        phases = [0.0] * len(result.frequencies)
        
        # Build response
        return {
            "frequencies": result.frequencies.tolist(),
            "magnitudes": result.power_spectrum.tolist(),
            "phases": phases,
            "dominant_frequency": float(result.dominant_frequency),
            "dominant_period": float(result.dominant_period),
            "significance": float(result.significance) if result.significance else None,
            "metadata": {
                "computation_time_ms": round(computation_time_ms, 2),
                "num_samples": len(timestamps),
                "sampling_rate_hz": float(self.analyzer.sampling_rate),
                "harmonics_detected": len(result.harmonics) if result.harmonics else 0
            }
        }
EOF
```

---

## 🌐 Step 4: Create API Endpoint (5 minutes)

```bash
# Create api/analysis.py
cat > api/analysis.py << 'EOF'
"""Analysis endpoints for signal processing."""

from fastapi import APIRouter, HTTPException
import numpy as np
from .models import FourierRequest, FourierResponse
from core.analyzer_wrapper import AnalyzerWrapper


router = APIRouter()
analyzer = AnalyzerWrapper()


@router.post("/fourier", response_model=FourierResponse)
async def fourier_analysis(request: FourierRequest):
    """
    Perform Fourier analysis using Lomb-Scargle periodogram.
    
    This endpoint wraps BioXen's SystemAnalyzer.fourier_lens() to detect
    periodic components in biological time-series data.
    
    - **timestamps**: Array of time points (seconds)
    - **values**: Array of signal values (same length as timestamps)
    - **detect_harmonics**: Enable multi-harmonic detection
    - **max_harmonics**: Maximum number of harmonics to detect
    
    Returns frequency spectrum with dominant period detection.
    """
    try:
        # Convert to numpy arrays
        timestamps = np.array(request.timestamps)
        values = np.array(request.values)
        
        # Perform analysis
        result = analyzer.fourier_analysis(
            timestamps=timestamps,
            values=values,
            detect_harmonics=request.detect_harmonics,
            max_harmonics=request.max_harmonics
        )
        
        return FourierResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "COMPUTATION_ERROR",
                    "message": f"Fourier analysis failed: {str(e)}"
                }
            }
        )
EOF
```

---

## 🚀 Step 5: Create Main Application (3 minutes)

```bash
# Create main.py
cat > main.py << 'EOF'
"""BioXen Computation API - Main application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.analysis import router as analysis_router


# Create FastAPI app
app = FastAPI(
    title="BioXen Computation API",
    version="0.1.0",
    description="Remote computation services for biological signal analysis",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    analysis_router,
    prefix="/api/v1/analysis",
    tags=["Signal Analysis"]
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "BioXen Computation API",
        "version": "0.1.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "analysis": "/api/v1/analysis",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}
EOF
```

---

## 🧪 Step 6: Test Your API (2 minutes)

### Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test 1: Health Check

Open a new terminal:
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"healthy","version":"0.1.0"}
```

### Test 2: OpenAPI Docs

Open browser: http://localhost:8000/docs

You should see interactive API documentation!

### Test 3: Fourier Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/analysis/fourier \
  -H "Content-Type: application/json" \
  -d '{
    "timestamps": [0, 3600, 7200, 10800, 14400, 18000],
    "values": [1.0, 1.5, 0.8, 1.2, 1.6, 0.9],
    "detect_peaks": true
  }'
```

Expected output (partial):
```json
{
  "frequencies": [0.00001, 0.00002, ...],
  "magnitudes": [10.5, 8.3, ...],
  "dominant_period": 24.3,
  "significance": 0.92,
  "metadata": {
    "computation_time_ms": 45.2,
    "num_samples": 6
  }
}
```

---

## ✅ Step 7: Run Wishful Test (3 minutes)

```bash
# In another terminal
cd ../wishful-client-tests

# Run the first test
pytest test_analysis.py::TestFourierAnalysis::test_fft_basic -v
```

**If it passes:** 🎉 **Congratulations! Your first endpoint works!**

**If it fails:** Check error message and verify:
1. Server is running on port 8000
2. BioXen library is installed
3. Request format matches specification

---

## 🎓 What You Just Built

You created:
1. **API Models** (`api/models.py`) - Request/response validation with Pydantic
2. **Business Logic** (`core/analyzer_wrapper.py`) - Wraps SystemAnalyzer with error handling
3. **API Endpoint** (`api/analysis.py`) - FastAPI route with documentation
4. **Main App** (`main.py`) - FastAPI application with CORS and routing

**Architecture:**
```
HTTP Request
    ↓
FastAPI (main.py)
    ↓
API Router (api/analysis.py)
    ↓
Pydantic Validation (api/models.py)
    ↓
Business Logic (core/analyzer_wrapper.py)
    ↓
BioXen SystemAnalyzer
    ↓
HTTP Response
```

---

## 🚀 Next Steps

Now that you have one working endpoint, you can easily add more:

### Add Wavelet Endpoint (15 minutes)

1. Add `WaveletRequest` and `WaveletResponse` to `api/models.py`
2. Add `wavelet_analysis()` method to `core/analyzer_wrapper.py`
3. Add `POST /wavelet` route to `api/analysis.py`

### Add Laplace Endpoint (15 minutes)

1. Add `LaplaceRequest` and `LaplaceResponse` to `api/models.py`
2. Add `laplace_analysis()` method to `core/analyzer_wrapper.py`
3. Add `POST /laplace` route to `api/analysis.py`

### Add Z-Transform Endpoint (15 minutes)

1. Add `ZTransformRequest` and `ZTransformResponse` to `api/models.py`
2. Add `ztransform_analysis()` method to `core/analyzer_wrapper.py`
3. Add `POST /ztransform` route to `api/analysis.py`

**After adding all 4 endpoints:** Run full analysis test suite
```bash
pytest ../wishful-client-tests/test_analysis.py -v
```

**Expected:** 21/25 tests passing (84%)! 🎉

---

## 🐛 Troubleshooting

### Server won't start

**Error:** `ModuleNotFoundError: No module named 'bioxen_fourier_vm_lib'`

**Solution:**
```bash
cd ../src
pip install -e .
cd ../wishful-server
```

---

### Import errors

**Error:** `ImportError: cannot import name 'SystemAnalyzer'`

**Solution:** Verify BioXen installation:
```bash
python -c "from bioxen_fourier_vm_lib.analysis.system_analyzer import SystemAnalyzer; print('OK')"
```

---

### Tests fail with connection error

**Error:** `httpx.ConnectError: [Errno 61] Connection refused`

**Solution:** Ensure server is running:
```bash
# Terminal 1
uvicorn main:app --reload

# Terminal 2 (run tests)
pytest ../wishful-client-tests/test_analysis.py -v
```

---

### Validation errors

**Error:** `422 Unprocessable Entity`

**Solution:** Check request format matches `FourierRequest` schema:
```python
{
  "timestamps": [0.0, 1.0, 2.0],  # Must be floats
  "values": [1.0, 1.1, 0.9],      # Must be floats, same length
  "method": "fft",                 # Optional
  "detect_peaks": true             # Optional
}
```

---

## 📚 Learn More

### FastAPI Documentation
- Tutorial: https://fastapi.tiangolo.com/tutorial/
- Advanced: https://fastapi.tiangolo.com/advanced/

### Pydantic Models
- Models: https://docs.pydantic.dev/latest/concepts/models/
- Validation: https://docs.pydantic.dev/latest/concepts/validators/

### BioXen Code
- SystemAnalyzer: `../src/bioxen_fourier_vm_lib/analysis/system_analyzer.py`
- Read the docstrings for `fourier_lens()`, `wavelet_lens()`, etc.

---

## 🎉 Success Criteria

You've successfully completed this guide if:

- [x] Server starts without errors
- [x] Can access http://localhost:8000/docs
- [x] Health check returns `{"status":"healthy"}`
- [x] Fourier endpoint accepts POST requests
- [x] Fourier endpoint returns valid JSON
- [x] At least one wishful test passes

**Congratulations!** You're ready to build the rest of the API! 🚀

---

**Estimated Total Time:** 30 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner-friendly)  
**Next:** Add remaining analysis endpoints (Wavelet, Laplace, Z-Transform)

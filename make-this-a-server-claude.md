# PyCWT-mod REST API Server - Claude's Implementation Plan

**Project:** pycwt-mod REST API Server  
**Goal:** Leverage existing backend architecture to build MVP REST API  
**Status:** Implementation Ready  
**Date:** October 3, 2025

---

## 🎯 Strategic Overview

### Key Insight: 80% Already Complete

Your project has a **major advantage** - the hardest architectural work is done:
- ✅ **Complete backend plugin system** with abstract base class
- ✅ **Backend registry** with auto-discovery
- ✅ **Multiple backends** implemented (sequential, joblib, dask, gpu)
- ✅ **Core functions** already support backend parameter
- ✅ **Comprehensive test coverage** for backends

**What this means:** The server becomes a **thin wrapper** over existing functionality, not a complete rewrite.

---

## 📊 Repository Structure Analysis

### Current State
```
pycwt-mod/
├── src/pycwt_mod/           ✅ Complete library
│   ├── backends/            ✅ Plugin architecture (Phase 1 complete)
│   │   ├── base.py         ✅ MonteCarloBackend interface
│   │   ├── registry.py     ✅ get_backend(), list_backends()
│   │   ├── sequential.py   ✅ Baseline implementation
│   │   ├── joblib.py       ✅ Parallel implementation
│   │   ├── dask.py         ✅ Distributed (stub)
│   │   └── gpu.py          ✅ GPU acceleration (stub)
│   ├── wavelet.py          ✅ cwt(), wct(), wct_significance()
│   ├── helpers.py          ✅ Utility functions
│   └── tests/              ✅ Backend tests
├── docs/                    ✅ Documentation
├── requirements.txt         ✅ Core dependencies
└── README.md               ✅ Project overview
```

### Proposed Server Addition
```
pycwt-mod/
├── src/pycwt_mod/          # Existing (unchanged)
├── server/                 # NEW - API server
│   ├── main.py            # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py      # Wavelet analysis endpoints
│   │   │   ├── backends.py      # Backend discovery endpoints
│   │   │   └── jobs.py          # Job management endpoints
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── requests.py      # Pydantic request models
│   │       └── responses.py     # Pydantic response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Server configuration
│   │   ├── job_manager.py       # Async job queue
│   │   └── wavelet_service.py   # Thin wrapper over pycwt_mod
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py          # Pytest fixtures
│   │   ├── test_api.py          # Endpoint tests
│   │   ├── test_job_manager.py  # Job queue tests
│   │   └── test_integration.py  # Full workflow tests
│   ├── requirements.txt         # FastAPI + server deps
│   ├── README.md               # Server documentation
│   └── .env.example            # Configuration template
└── README.md                    # Update to mention server
```

---

## 🚀 Implementation Plan - 4 Phases

### Phase 1: Foundation (Week 1)
**Goal:** Working FastAPI server with backend discovery

#### 1.1 Directory Setup
```bash
mkdir -p server/api/routes server/api/models server/core server/tests
touch server/__init__.py
touch server/api/__init__.py
touch server/api/routes/__init__.py
touch server/api/models/__init__.py
touch server/core/__init__.py
touch server/tests/__init__.py
```

#### 1.2 Core Configuration
**File: `server/core/config.py`**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Job management
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT_SECONDS: int = 3600
    
    # Backends
    DEFAULT_BACKEND: str = "sequential"
    AVAILABLE_BACKENDS: list[str] = ["sequential", "joblib"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### 1.3 Basic FastAPI Application
**File: `server/main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analysis, backends, jobs
from core.config import settings

app = FastAPI(
    title="PyCWT-mod API",
    description="Wavelet analysis with hardware acceleration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (basic for MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "backends_available": len(settings.AVAILABLE_BACKENDS)
    }

# Include routers
app.include_router(backends.router, prefix="/api/v1/backends", tags=["backends"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
```

#### 1.4 Backend Discovery Endpoints
**File: `server/api/routes/backends.py`**
```python
from fastapi import APIRouter, HTTPException
from pycwt_mod.backends import list_backends, get_backend

router = APIRouter()

@router.get("/")
async def list_available_backends():
    """List all available computation backends."""
    backend_names = list_backends()
    backend_info = []
    
    for name in backend_names:
        try:
            backend = get_backend(name)
            backend_info.append({
                "name": name,
                "available": backend.is_available(),
                "description": getattr(backend, "__doc__", "").split("\n")[0]
            })
        except Exception as e:
            backend_info.append({
                "name": name,
                "available": False,
                "error": str(e)
            })
    
    return {"backends": backend_info}

@router.get("/{backend_name}")
async def get_backend_info(backend_name: str):
    """Get detailed information about a specific backend."""
    try:
        backend = get_backend(backend_name)
        return {
            "name": backend_name,
            "available": backend.is_available(),
            "description": getattr(backend, "__doc__", ""),
            "type": type(backend).__name__
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

#### 1.5 Dependencies File
**File: `server/requirements.txt`**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
```

#### 1.6 Phase 1 Testing
```bash
# Install dependencies
cd server
pip install -r requirements.txt

# Run server
uvicorn main:app --reload

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/backends
curl http://localhost:8000/api/v1/backends/sequential
```

**Phase 1 Success Criteria:**
- [ ] Server starts without errors
- [ ] `/health` returns status
- [ ] `/api/v1/backends` lists all backends
- [ ] `/api/v1/backends/sequential` returns backend details
- [ ] OpenAPI docs accessible at `/docs`

---

### Phase 2: Analysis Endpoints (Week 2)
**Goal:** Wrap core wavelet functions with REST API

#### 2.1 Request/Response Models
**File: `server/api/models/requests.py`**
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class CWTRequest(BaseModel):
    data: List[float] = Field(..., description="Time series data")
    dt: float = Field(..., ge=0, description="Sample spacing")
    dj: float = Field(0.25, ge=0.01, le=1.0, description="Scale resolution")
    s0: float = Field(..., gt=0, description="Smallest scale")
    J: int = Field(..., ge=1, le=100, description="Number of scales")
    wavelet: Literal["morlet", "paul", "dog"] = "morlet"
    normalize: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "data": [1.0, 2.0, 3.0, 4.0, 5.0],
                "dt": 1.0,
                "dj": 0.25,
                "s0": 2.0,
                "J": 10,
                "wavelet": "morlet",
                "normalize": True
            }
        }

class WCTRequest(BaseModel):
    data1: List[float] = Field(..., description="First time series")
    data2: List[float] = Field(..., description="Second time series")
    dt: float = Field(..., ge=0)
    dj: float = Field(0.25, ge=0.01, le=1.0)
    s0: float = Field(..., gt=0)
    J: int = Field(..., ge=1, le=100)
    wavelet: Literal["morlet", "paul", "dog"] = "morlet"
    normalize: bool = True
    significance_test: bool = False
    backend: Optional[str] = None
    mc_count: int = Field(300, ge=10, le=10000)

class SignificanceRequest(WCTRequest):
    """Extended WCT request for async significance testing."""
    significance_test: bool = True
```

**File: `server/api/models/responses.py`**
```python
from pydantic import BaseModel
from typing import List, Any, Optional
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class CWTResponse(BaseModel):
    wave: List[List[Any]]  # Complex numbers serialized
    scales: List[float]
    freqs: List[float]
    coi: List[float]
    computation_time: float

class WCTResponse(BaseModel):
    WCT: List[List[Any]]
    aWCT: List[List[Any]]
    coi: List[float]
    freqs: List[float]
    sig95: Optional[List[List[float]]] = None
    computation_time: float
    backend_used: str

class JobSubmittedResponse(BaseModel):
    job_id: str
    status: JobStatus
    estimated_time: Optional[str] = None
    message: str = "Job submitted successfully"

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
```

#### 2.2 Wavelet Service Layer
**File: `server/core/wavelet_service.py`**
```python
import numpy as np
import time
from typing import Tuple, Optional, List
import pycwt_mod as pycwt
from pycwt_mod.backends import get_backend

class WaveletService:
    """Thin wrapper around pycwt_mod functions."""
    
    @staticmethod
    def cwt(data: List[float], dt: float, dj: float, s0: float, 
            J: int, wavelet: str = "morlet", normalize: bool = True) -> dict:
        """Compute continuous wavelet transform."""
        start_time = time.time()
        
        # Convert to numpy array
        data_array = np.array(data)
        
        # Call pycwt_mod
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
            data_array, dt, dj, s0, J, wavelet=wavelet
        )
        
        computation_time = time.time() - start_time
        
        return {
            "wave": wave.tolist(),  # Convert complex to list
            "scales": scales.tolist(),
            "freqs": freqs.tolist(),
            "coi": coi.tolist(),
            "computation_time": computation_time
        }
    
    @staticmethod
    def wct(data1: List[float], data2: List[float], dt: float, 
            dj: float, s0: float, J: int, wavelet: str = "morlet",
            normalize: bool = True, significance_test: bool = False,
            backend: Optional[str] = None, mc_count: int = 300) -> dict:
        """Compute wavelet coherence transform."""
        start_time = time.time()
        
        # Convert to numpy arrays
        d1 = np.array(data1)
        d2 = np.array(data2)
        
        # Validate backend
        backend_name = backend or "sequential"
        try:
            backend_obj = get_backend(backend_name)
            if not backend_obj.is_available():
                backend_name = "sequential"
        except ValueError:
            backend_name = "sequential"
        
        # Call pycwt_mod with backend
        if significance_test:
            WCT, aWCT, coi, freqs, sig95 = pycwt.wct(
                d1, d2, dt, dj, s0, J,
                sig=True,
                significance_level=0.95,
                wavelet=wavelet,
                backend=backend_name,
                mc_count=mc_count
            )
        else:
            WCT, aWCT, coi, freqs = pycwt.wct(
                d1, d2, dt, dj, s0, J,
                sig=False,
                wavelet=wavelet
            )
            sig95 = None
        
        computation_time = time.time() - start_time
        
        return {
            "WCT": WCT.tolist(),
            "aWCT": aWCT.tolist(),
            "coi": coi.tolist(),
            "freqs": freqs.tolist(),
            "sig95": sig95.tolist() if sig95 is not None else None,
            "computation_time": computation_time,
            "backend_used": backend_name
        }
```

#### 2.3 Analysis Routes
**File: `server/api/routes/analysis.py`**
```python
from fastapi import APIRouter, HTTPException
from api.models.requests import CWTRequest, WCTRequest, SignificanceRequest
from api.models.responses import CWTResponse, WCTResponse, JobSubmittedResponse
from core.wavelet_service import WaveletService
from core.job_manager import job_manager

router = APIRouter()
wavelet_service = WaveletService()

@router.post("/cwt", response_model=CWTResponse)
async def compute_cwt(request: CWTRequest):
    """Compute continuous wavelet transform (synchronous)."""
    try:
        result = wavelet_service.cwt(
            data=request.data,
            dt=request.dt,
            dj=request.dj,
            s0=request.s0,
            J=request.J,
            wavelet=request.wavelet,
            normalize=request.normalize
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/wct", response_model=WCTResponse)
async def compute_wct(request: WCTRequest):
    """Compute wavelet coherence transform (synchronous for small datasets)."""
    try:
        # For sync endpoint, limit mc_count to avoid long delays
        if request.significance_test and request.mc_count > 100:
            raise HTTPException(
                status_code=400,
                detail="For synchronous WCT with significance, mc_count must be ≤ 100. Use /significance for larger counts."
            )
        
        result = wavelet_service.wct(
            data1=request.data1,
            data2=request.data2,
            dt=request.dt,
            dj=request.dj,
            s0=request.s0,
            J=request.J,
            wavelet=request.wavelet,
            normalize=request.normalize,
            significance_test=request.significance_test,
            backend=request.backend,
            mc_count=request.mc_count
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/significance", response_model=JobSubmittedResponse)
async def compute_significance_async(request: SignificanceRequest):
    """Submit async job for wavelet coherence significance testing."""
    try:
        # Submit to job manager (implemented in Phase 3)
        job_id = await job_manager.submit_job(
            wavelet_service.wct,
            data1=request.data1,
            data2=request.data2,
            dt=request.dt,
            dj=request.dj,
            s0=request.s0,
            J=request.J,
            wavelet=request.wavelet,
            normalize=request.normalize,
            significance_test=True,
            backend=request.backend,
            mc_count=request.mc_count
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "estimated_time": f"{request.mc_count * 0.1:.0f}s",
            "message": "Job submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**Phase 2 Success Criteria:**
- [ ] `/api/v1/analysis/cwt` returns valid wavelet transform
- [ ] `/api/v1/analysis/wct` works with backend parameter
- [ ] `/api/v1/analysis/significance` returns job_id
- [ ] Pydantic validation catches invalid inputs
- [ ] OpenAPI docs show all request/response schemas

---

### Phase 3: Job Management (Week 3)
**Goal:** Async job queue for long-running computations

#### 3.1 Job Manager Implementation
**File: `server/core/job_manager.py`**
```python
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    job_id: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class JobManager:
    """Manage async job execution with progress tracking."""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, Job] = {}
        self.running_jobs = 0
        self._lock = asyncio.Lock()
    
    async def submit_job(self, func: Callable, *args, **kwargs) -> str:
        """Submit a job for async execution."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED
        )
        
        async with self._lock:
            self.jobs[job_id] = job
        
        # Start execution task
        asyncio.create_task(self._execute_job(job, func, *args, **kwargs))
        
        return job_id
    
    async def _execute_job(self, job: Job, func: Callable, *args, **kwargs):
        """Execute a job with progress tracking."""
        # Wait if max concurrent jobs reached
        while self.running_jobs >= self.max_concurrent_jobs:
            await asyncio.sleep(1)
        
        async with self._lock:
            self.running_jobs += 1
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
        
        try:
            # Execute function (run in thread pool for blocking operations)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = 1.0
            job.completed_at = datetime.now()
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
        
        finally:
            async with self._lock:
                self.running_jobs -= 1
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> list[Job]:
        """List all jobs, optionally filtered by status."""
        if status:
            return [j for j in self.jobs.values() if j.status == status]
        return list(self.jobs.values())

# Global job manager instance
job_manager = JobManager()
```

#### 3.2 Job Routes
**File: `server/api/routes/jobs.py`**
```python
from fastapi import APIRouter, HTTPException
from api.models.responses import JobStatusResponse
from core.job_manager import job_manager, JobStatus

router = APIRouter()

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status and progress."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error
    }

@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result (only when completed)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job.status.value}, not completed"
        )
    
    return {
        "job_id": job.job_id,
        "status": "completed",
        "result": job.result,
        "computation_time": (
            job.completed_at - job.started_at
        ).total_seconds() if job.completed_at and job.started_at else None
    }

@router.get("/")
async def list_jobs(status: Optional[str] = None):
    """List all jobs, optionally filtered by status."""
    if status:
        try:
            status_enum = JobStatus(status)
            jobs = job_manager.list_jobs(status=status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status")
    else:
        jobs = job_manager.list_jobs()
    
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "progress": j.progress,
                "created_at": j.created_at.isoformat()
            }
            for j in jobs
        ]
    }
```

**Phase 3 Success Criteria:**
- [ ] Jobs execute asynchronously without blocking API
- [ ] Job status updates correctly (queued → running → completed)
- [ ] Multiple jobs can run concurrently (up to max)
- [ ] Results retrievable after completion
- [ ] Failed jobs report errors properly

---

### Phase 4: Testing & Documentation (Week 4)
**Goal:** Comprehensive testing and production-ready documentation

#### 4.1 Test Setup
**File: `server/tests/conftest.py`**
```python
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_data():
    """Sample time series data for testing."""
    return [float(i) for i in range(100)]
```

#### 4.2 API Tests
**File: `server/tests/test_api.py`**
```python
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_backends(client):
    response = client.get("/api/v1/backends")
    assert response.status_code == 200
    assert "backends" in response.json()
    assert len(response.json()["backends"]) > 0

def test_cwt_endpoint(client, sample_data):
    request_data = {
        "data": sample_data,
        "dt": 1.0,
        "dj": 0.25,
        "s0": 2.0,
        "J": 5,
        "wavelet": "morlet"
    }
    
    response = client.post("/api/v1/analysis/cwt", json=request_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "wave" in result
    assert "scales" in result
    assert "freqs" in result
    assert "coi" in result
    assert result["computation_time"] > 0

def test_wct_with_backend(client, sample_data):
    request_data = {
        "data1": sample_data,
        "data2": sample_data,
        "dt": 1.0,
        "dj": 0.25,
        "s0": 2.0,
        "J": 5,
        "significance_test": False,
        "backend": "sequential"
    }
    
    response = client.post("/api/v1/analysis/wct", json=request_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["backend_used"] == "sequential"

def test_async_significance(client, sample_data):
    request_data = {
        "data1": sample_data[:50],
        "data2": sample_data[:50],
        "dt": 1.0,
        "dj": 0.25,
        "s0": 2.0,
        "J": 3,
        "significance_test": True,
        "backend": "sequential",
        "mc_count": 50
    }
    
    # Submit job
    response = client.post("/api/v1/analysis/significance", json=request_data)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # Check status
    import time
    time.sleep(2)  # Give job time to start
    
    status_response = client.get(f"/api/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] in ["queued", "running", "completed"]
```

#### 4.3 Integration Tests
**File: `server/tests/test_integration.py`**
```python
import time

def test_full_workflow(client, sample_data):
    """Test complete workflow: submit job → check status → get results."""
    
    # 1. Submit async significance job
    request_data = {
        "data1": sample_data[:50],
        "data2": sample_data[:50],
        "dt": 1.0,
        "dj": 0.25,
        "s0": 2.0,
        "J": 3,
        "significance_test": True,
        "backend": "sequential",
        "mc_count": 30
    }
    
    response = client.post("/api/v1/analysis/significance", json=request_data)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # 2. Poll for completion
    max_wait = 60  # seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        status = status_response.json()["status"]
        
        if status == "completed":
            break
        elif status == "failed":
            pytest.fail("Job failed")
        
        time.sleep(1)
    
    # 3. Get results
    result_response = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result_response.status_code == 200
    
    result = result_response.json()["result"]
    assert "sig95" in result
    assert result["backend_used"] == "sequential"
```

#### 4.4 Server README
**File: `server/README.md`**
```markdown
# PyCWT-mod REST API Server

REST API server for wavelet analysis with hardware-accelerated Monte Carlo backends.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install pycwt_mod library
cd ..
pip install -e .
```

### Run Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once running, visit:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### List Backends
```bash
curl http://localhost:8000/api/v1/backends
```

### Continuous Wavelet Transform
```bash
curl -X POST http://localhost:8000/api/v1/analysis/cwt \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.0, 2.0, 3.0, 4.0, 5.0],
    "dt": 1.0,
    "dj": 0.25,
    "s0": 2.0,
    "J": 5
  }'
```

### Wavelet Coherence (with backend selection)
```bash
curl -X POST http://localhost:8000/api/v1/analysis/wct \
  -H "Content-Type: application/json" \
  -d '{
    "data1": [1.0, 2.0, ...],
    "data2": [1.5, 2.5, ...],
    "dt": 1.0,
    "dj": 0.25,
    "s0": 2.0,
    "J": 5,
    "backend": "joblib",
    "significance_test": true,
    "mc_count": 100
  }'
```

### Async Significance Testing
```bash
# Submit job
curl -X POST http://localhost:8000/api/v1/analysis/significance \
  -H "Content-Type: application/json" \
  -d '{
    "data1": [...],
    "data2": [...],
    "dt": 1.0,
    "backend": "joblib",
    "mc_count": 300
  }'
# Returns: {"job_id": "job_...", "status": "queued"}

# Check status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Get results
curl http://localhost:8000/api/v1/jobs/{job_id}/result
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Configuration

Create `.env` file:
```bash
HOST=0.0.0.0
PORT=8000
MAX_CONCURRENT_JOBS=10
DEFAULT_BACKEND=sequential
```

## Production Deployment

See deployment documentation for:
- SystemD service setup
- Nginx reverse proxy configuration
- Multi-server clustering
- Monitoring and logging
```

**Phase 4 Success Criteria:**
- [ ] All tests pass
- [ ] 80%+ code coverage
- [ ] README with complete examples
- [ ] API documentation accurate and complete
- [ ] Integration tests validate full workflows

---

## 🎯 Implementation Summary

### What Makes This Plan Different

1. **Leverages Existing Architecture**: 
   - Your backend system is already complete
   - Server is a thin wrapper, not a rewrite
   - Focus on API layer, not business logic

2. **Clean Separation**:
   - `src/pycwt_mod/` remains unchanged
   - `server/` is independent and testable
   - Easy to deploy separately or together

3. **Incremental Validation**:
   - Each phase has clear success criteria
   - Can validate before moving to next phase
   - Easy to roll back if issues arise

4. **Production Ready Path**:
   - MVP gets core functionality working
   - Future phases add auth, monitoring, scaling
   - Clear upgrade path from MVP to production

### Timeline Estimate

- **Phase 1**: 3-5 days (foundation + backend discovery)
- **Phase 2**: 5-7 days (analysis endpoints + service layer)
- **Phase 3**: 4-6 days (job management + async processing)
- **Phase 4**: 3-5 days (testing + documentation)

**Total**: 2-3 weeks for working MVP

### Next Steps

1. **Start with Phase 1**: Get basic server running with backend discovery
2. **Validate early**: Test each endpoint as you build it
3. **Iterate based on feedback**: Adjust based on actual usage patterns
4. **Document as you go**: Keep README updated with working examples

### Success Metrics

**MVP Complete When:**
- [ ] All core endpoints functional
- [ ] Backend selection works correctly
- [ ] Async jobs execute without blocking
- [ ] Tests pass with good coverage
- [ ] Documentation complete and accurate
- [ ] Can handle 10+ concurrent requests
- [ ] Simple operations complete in < 1s
- [ ] Can deploy with single command

---

## 🚀 Beyond MVP

Once MVP is validated and deployed:

### Phase 5: Authentication & Security
- JWT authentication
- API key management
- Rate limiting per user
- CORS configuration

### Phase 6: Data Management
- Upload/download endpoints
- Persistent storage
- Result caching
- Data validation

### Phase 7: Production Hardening
- Nginx reverse proxy
- SystemD service configuration
- Log aggregation
- Monitoring dashboards
- Backup/recovery

### Phase 8: Advanced Features
- Multi-server clustering
- Redis job queue
- WebSocket progress streaming
- Batch processing
- Result visualization

---

**This plan provides a clear, actionable path from your current state (complete backend architecture) to a working REST API server in 2-3 weeks. The key insight is that you're 80% done - the server is just exposing what you already have built.**

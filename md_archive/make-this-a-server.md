# Transform PyCWT-mod into a Bare Metal Server with External API

**Project:** pycwt-mod - Modular Python Continuous Wavelet Transform Library  
**Feature:** REST API Server for Wavelet Analysis Services  
**Target:** Bare Metal Web Service for Scientific Computing and Hardware Acceleration  
**Date:** October 3, 2025  
**Status:** Design Phase - Bare Metal Server Architecture Planning  
**Prerequisites:** Phase 1 (Backend Architecture) + Phase 2 (Integration) Complete

---

## 🎯 Objective

Transform pycwt-mod from a Python library into a production-ready **bare metal server** application with a REST API that exposes wavelet analysis capabilities, including the modular backend system for hardware-accelerated Monte Carlo computations. The server runs directly on physical hardware without virtualization layers for maximum performance and hardware access.

The server should provide:

- **RESTful API** for all major wavelet analysis functions
- **Backend Selection** via API parameters (sequential, joblib, elm11_lua, tang_nano_9k)
- **Data Management** for time series upload/download
- **Job Management** for long-running Monte Carlo simulations
- **Monitoring & Status** endpoints for server health and job progress
- **Authentication & Security** for production deployment
- **Batch Processing** capabilities
- **Documentation** via OpenAPI/Swagger
- **Direct Hardware Access** for optimal performance

---

## 📊 Server Architecture Overview

### Current Library Structure (Completed)

```
pycwt_mod/
├── wavelet.py              ✅ Core wavelet functions (cwt, wct, wct_significance)
├── backends/               ✅ Modular backend system
│   ├── base.py            ✅ Abstract MonteCarloBackend
│   ├── registry.py        ✅ Backend auto-discovery
│   ├── sequential.py      ✅ CPU single-core
│   ├── joblib.py          ✅ CPU multi-core
│   ├── elm11_lua.py       🔲 Lua embedded (planned)
│   └── tang_nano_9k.py    🔲 FPGA (planned)
├── sample/                 ✅ Sample datasets
└── tests/                  ✅ Comprehensive test suite
```

### Target Server Structure

```
pycwt_server/
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── analysis.py     # /api/v1/analysis/*
│   │   ├── backends.py     # /api/v1/backends/*
│   │   ├── jobs.py         # /api/v1/jobs/*
│   │   └── data.py         # /api/v1/data/*
│   ├── models/
│   │   ├── requests.py     # Pydantic request models
│   │   ├── responses.py    # Pydantic response models
│   │   └── jobs.py         # Job status models
│   └── middleware/
│       ├── auth.py         # Authentication middleware
│       ├── cors.py         # CORS handling
│       └── logging.py      # Request logging
├── core/
│   ├── job_manager.py      # Async job management
│   ├── data_manager.py     # Data storage/retrieval
│   ├── backend_manager.py  # Backend coordination
│   └── config.py           # Server configuration
├── services/
│   ├── wavelet_service.py  # Wavelet computation wrapper
│   ├── backend_service.py  # Backend availability checks
│   └── validation_service.py # Input validation
├── utils/
│   ├── serialization.py    # NumPy array JSON serialization
│   ├── file_handling.py    # Data file I/O
│   └── async_utils.py      # Async computation helpers
├── scripts/
│   ├── install.sh          # Bare metal installation script
│   ├── systemd.service     # SystemD service file
│   ├── nginx.conf          # Nginx reverse proxy config
│   └── logrotate.conf      # Log rotation configuration
├── main.py                 # FastAPI application entry point
├── requirements-server.txt # Server-specific dependencies
└── README.md               # Bare metal deployment guide
```

---

## 🔧 Server Implementation Plan

### Phase 1: Core API Framework (Week 1-2)

**Goal:** Establish FastAPI server with basic wavelet endpoints

#### Dependencies to Add
```txt
# requirements-server.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
aiofiles==23.2.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
```

#### Basic Server Setup
```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import analysis, backends, jobs, data
from core.config import settings

app = FastAPI(
    title="PyCWT-mod Server",
    description="Modular Wavelet Analysis Server with Hardware Acceleration",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(backends.router, prefix="/api/v1/backends", tags=["backends"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

#### Configuration Management
```python
# core/config.py
from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Data storage
    DATA_DIR: str = "./data"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Job management
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT_MINUTES: int = 60
    
    # Hardware backends
    ENABLE_ELM11: bool = False
    ENABLE_TANG_NANO: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Phase 2: Analysis Endpoints (Week 3-4)

**Goal:** Implement core wavelet analysis API endpoints

#### API Endpoints Design

**1. Continuous Wavelet Transform**
```python
# POST /api/v1/analysis/cwt
# Request body:
{
  "data": [1.0, 2.0, 3.0, ...],  # Time series data
  "dt": 1.0,                      # Time step
  "dj": 0.25,                     # Scale resolution
  "s0": 2.0,                      # Smallest scale
  "J": 10,                        # Number of scales
  "wavelet": "morlet",            # Wavelet type
  "normalize": true               # Normalize results
}

# Response:
{
  "wave": [[complex, complex, ...], ...],  # Wavelet coefficients
  "scales": [2.0, 2.25, 2.5, ...],        # Scale values
  "freqs": [0.5, 0.44, 0.4, ...],         # Frequency values
  "coi": [1.0, 1.5, 2.0, ...],            # Cone of influence
  "fft": {"power": [...], "freqs": [...]}, # FFT results
  "computation_time": 0.123
}
```

**2. Wavelet Coherence**
```python
# POST /api/v1/analysis/wct
# Request body:
{
  "data1": [1.0, 2.0, 3.0, ...],   # First time series
  "data2": [0.5, 1.5, 2.5, ...],   # Second time series
  "dt": 1.0,
  "dj": 0.25,
  "s0": 2.0,
  "J": 10,
  "wavelet": "morlet",
  "normalize": true,
  "significance_test": true,       # Enable Monte Carlo significance
  "significance_level": 0.95,
  "backend": "joblib",             # Backend selection
  "mc_count": 300                  # Monte Carlo iterations
}

# Response:
{
  "WCT": [[complex, complex, ...], ...],    # Cross-wavelet transform
  "aWCT": [[float, float, ...], ...],       # Anti-phase cross-wavelet
  "coi": [1.0, 1.5, 2.0, ...],             # Cone of influence
  "freqs": [0.5, 0.44, 0.4, ...],          # Frequencies
  "sig95": [[float, float, ...], ...],      # Significance levels
  "computation_time": 45.67,
  "backend_used": "joblib"
}
```

**3. Significance Testing (Async)**
```python
# POST /api/v1/analysis/significance
# Request body: (same as wct above)
# Response: Job ID for async processing

{
  "job_id": "wct_sig_1234567890",
  "status": "queued",
  "estimated_time": "45-60 seconds",
  "backend": "joblib"
}

# GET /api/v1/jobs/{job_id}
{
  "job_id": "wct_sig_1234567890",
  "status": "running",
  "progress": 0.67,  # 67% complete
  "eta": "15 seconds",
  "backend": "joblib"
}

# GET /api/v1/jobs/{job_id}/result (when complete)
{
  "job_id": "wct_sig_1234567890",
  "status": "completed",
  "result": {
    "sig95": [[float, float, ...], ...],
    "computation_time": 52.34,
    "backend_used": "joblib",
    "mc_iterations": 300
  }
}
```

#### Request/Response Models
```python
# api/models/requests.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import numpy as np

class CWTRequest(BaseModel):
    data: List[float] = Field(..., description="Time series data")
    dt: float = Field(..., ge=0, description="Time step")
    dj: float = Field(0.25, ge=0.01, le=1.0, description="Scale resolution")
    s0: float = Field(..., gt=0, description="Smallest scale")
    J: int = Field(..., ge=1, le=100, description="Number of scales")
    wavelet: Literal["morlet", "paul", "dog"] = "morlet"
    normalize: bool = True

class WCTRequest(BaseModel):
    data1: List[float] = Field(..., description="First time series")
    data2: List[float] = Field(..., description="Second time series")
    dt: float = Field(..., ge=0, description="Time step")
    dj: float = Field(0.25, ge=0.01, le=1.0, description="Scale resolution")
    s0: float = Field(..., gt=0, description="Smallest scale")
    J: int = Field(..., ge=1, le=100, description="Number of scales")
    wavelet: Literal["morlet", "paul", "dog"] = "morlet"
    normalize: bool = True
    significance_test: bool = False
    significance_level: float = Field(0.95, ge=0, le=1)
    backend: Optional[str] = None
    mc_count: int = Field(300, ge=10, le=10000)

class SignificanceRequest(WCTRequest):
    # Inherits from WCTRequest, adds async processing
    pass
```

### Phase 3: Job Management & Async Processing (Week 5-6)

**Goal:** Handle long-running Monte Carlo simulations asynchronously

#### Job Manager Implementation
```python
# core/job_manager.py
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    estimated_time: Optional[int] = None  # seconds

class JobManager:
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, Job] = {}
        self.running_jobs = 0
        self._lock = asyncio.Lock()
    
    async def submit_job(self, func, *args, **kwargs) -> str:
        """Submit a job for async execution."""
        job_id = f"{func.__name__}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            estimated_time=kwargs.pop('estimated_time', None)
        )
        
        async with self._lock:
            self.jobs[job_id] = job
        
        # Start job if we have capacity
        asyncio.create_task(self._execute_job(job, func, *args, **kwargs))
        
        return job_id
    
    async def _execute_job(self, job: Job, func, *args, **kwargs):
        """Execute a job with progress tracking."""
        async with self._lock:
            if self.running_jobs >= self.max_concurrent_jobs:
                return  # Stay queued
            
            self.running_jobs += 1
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
        
        try:
            # Execute with progress callback
            def progress_callback(progress: float):
                job.progress = progress
            
            kwargs['progress_callback'] = progress_callback
            result = await func(*args, **kwargs)
            
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now()
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            
        finally:
            async with self._lock:
                self.running_jobs -= 1
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job status and result."""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if job and job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove completed jobs older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            job_id for job_id, job in self.jobs.items()
            if job.completed_at and job.completed_at < cutoff
        ]
        for job_id in to_remove:
            del self.jobs[job_id]

# Global job manager instance
job_manager = JobManager()
```

#### Job Routes
```python
# api/routes/jobs.py
from fastapi import APIRouter, HTTPException
from core.job_manager import job_manager

router = APIRouter()

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and progress."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "progress": job.progress,
    }
    
    if job.started_at:
        response["started_at"] = job.started_at.isoformat()
    
    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
    
    if job.estimated_time:
        response["estimated_time_seconds"] = job.estimated_time
    
    if job.error:
        response["error"] = job.error
    
    return response

@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result (only when completed)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != job_manager.JobStatus.COMPLETED:
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

@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_manager.cancel_job(job_id):
        return {"message": f"Job {job_id} cancelled"}
    else:
        raise HTTPException(status_code=400, detail="Job could not be cancelled")
```

### Phase 4: Data Management & File Upload (Week 7-8)

**Goal:** Support data upload, storage, and retrieval

#### Data Manager
```python
# core/data_manager.py
import os
import uuid
import aiofiles
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, data_dir: str = "./data", max_file_size: int = 100*1024*1024):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_file_size = max_file_size
        self.metadata_file = self.data_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load data file metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save data file metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    async def save_data(self, data: List[float], filename: str = None, 
                       metadata: Dict[str, Any] = None) -> str:
        """Save time series data and return data ID."""
        if not filename:
            filename = f"data_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}.json"
        
        data_id = str(uuid.uuid4())
        file_path = self.data_dir / f"{data_id}.json"
        
        # Validate data
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if len(data) > 1000000:  # 1M points max
            raise ValueError("Data too large (max 1M points)")
        
        # Save data
        data_record = {
            "data": data,
            "created_at": datetime.now().isoformat(),
            "filename": filename,
            "length": len(data),
            "metadata": metadata or {}
        }
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data_record, indent=2))
        
        # Update metadata
        self.metadata[data_id] = {
            "filename": filename,
            "created_at": datetime.now().isoformat(),
            "size": len(data),
            "file_size_bytes": file_path.stat().st_size
        }
        self._save_metadata()
        
        return data_id
    
    def load_data(self, data_id: str) -> Dict[str, Any]:
        """Load time series data by ID."""
        file_path = self.data_dir / f"{data_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Data {data_id} not found")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def list_data(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List available data files."""
        return list(self.metadata.values())[-limit:]
    
    def delete_data(self, data_id: str) -> bool:
        """Delete data file."""
        file_path = self.data_dir / f"{data_id}.json"
        if file_path.exists():
            file_path.unlink()
            if data_id in self.metadata:
                del self.metadata[data_id]
                self._save_metadata()
            return True
        return False
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """Remove data files older than max_age_days."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        to_remove = []
        
        for data_id, meta in self.metadata.items():
            created_at = datetime.fromisoformat(meta["created_at"])
            if created_at < cutoff:
                to_remove.append(data_id)
        
        for data_id in to_remove:
            self.delete_data(data_id)

# Global data manager instance
data_manager = DataManager()
```

#### Data Routes
```python
# api/routes/data.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from core.data_manager import data_manager
import json
import numpy as np

router = APIRouter()

@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    metadata: str = Form("{}")  # JSON string
):
    """Upload time series data file."""
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    # Read file content
    content = await file.read()
    if len(content) > data_manager.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Parse data (assume JSON array or CSV)
    try:
        if file.filename.endswith('.json'):
            data = json.loads(content.decode())
            if isinstance(data, list):
                data_array = data
            else:
                raise ValueError("JSON must contain an array")
        elif file.filename.endswith('.csv'):
            # Simple CSV parsing (first column only)
            lines = content.decode().strip().split('\n')
            data_array = []
            for line in lines:
                if line.strip():
                    try:
                        value = float(line.split(',')[0].strip())
                        data_array.append(value)
                    except (ValueError, IndexError):
                        continue
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        if len(data_array) == 0:
            raise HTTPException(status_code=400, detail="No valid data found")
        
        data_id = await data_manager.save_data(
            data_array, 
            filename=file.filename,
            metadata=metadata_dict
        )
        
        return {
            "data_id": data_id,
            "filename": file.filename,
            "length": len(data_array),
            "message": "Data uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@router.get("/{data_id}")
async def get_data_info(data_id: str):
    """Get data file information."""
    try:
        data_record = data_manager.load_data(data_id)
        return {
            "data_id": data_id,
            "filename": data_record["filename"],
            "length": data_record["length"],
            "created_at": data_record["created_at"],
            "metadata": data_record["metadata"]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data not found")

@router.get("/{data_id}/download")
async def download_data(data_id: str):
    """Download data file."""
    try:
        data_record = data_manager.load_data(data_id)
        # Return as JSON file
        return FileResponse(
            path=str(data_manager.data_dir / f"{data_id}.json"),
            filename=data_record["filename"],
            media_type="application/json"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data not found")

@router.get("/")
async def list_data(limit: int = 50):
    """List available data files."""
    return {"data_files": data_manager.list_data(limit)}

@router.delete("/{data_id}")
async def delete_data(data_id: str):
    """Delete data file."""
    if data_manager.delete_data(data_id):
        return {"message": f"Data {data_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Data not found")
```

### Phase 5: Backend Management & Monitoring (Week 9-10)

**Goal:** Backend status monitoring and dynamic selection

#### Backend Routes
```python
# api/routes/backends.py
from fastapi import APIRouter
from core.backend_manager import backend_manager

router = APIRouter()

@router.get("/")
async def list_backends():
    """List all available backends."""
    backends = []
    for name in backend_manager.list_backends():
        backend = backend_manager.get_backend(name)
        backends.append({
            "name": name,
            "available": backend.is_available(),
            "description": getattr(backend, 'description', ''),
            "estimated_speedup": getattr(backend, 'estimated_speedup', 1.0)
        })
    
    return {"backends": backends}

@router.get("/recommend")
async def recommend_backend(
    n_simulations: int = 100,
    data_size: int = 1000,
    priority: str = "speed"  # speed, power, reliability
):
    """Recommend optimal backend for given parameters."""
    recommendation = backend_manager.recommend_backend(
        n_simulations=n_simulations,
        data_size=data_size,
        priority=priority
    )
    
    return {
        "recommended_backend": recommendation["name"],
        "reason": recommendation["reason"],
        "estimated_time": recommendation["estimated_time"],
        "alternatives": recommendation["alternatives"]
    }

@router.get("/{backend_name}/status")
async def backend_status(backend_name: str):
    """Get detailed backend status."""
    try:
        backend = backend_manager.get_backend(backend_name)
        return {
            "name": backend_name,
            "available": backend.is_available(),
            "status": "available" if backend.is_available() else "unavailable",
            "description": getattr(backend, 'description', ''),
            "capabilities": getattr(backend, 'capabilities', {}),
            "performance_metrics": getattr(backend, 'performance_metrics', {})
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="Backend not found")
```

#### Monitoring & Health Checks
```python
# Additional health check endpoint
@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    """Detailed health check with backend status."""
    backends_status = {}
    for name in backend_manager.list_backends():
        backend = backend_manager.get_backend(name)
        backends_status[name] = {
            "available": backend.is_available(),
            "healthy": True  # Add more detailed health checks
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "backends": backends_status,
        "jobs": {
            "queued": len([j for j in job_manager.jobs.values() if j.status.name == "QUEUED"]),
            "running": len([j for j in job_manager.jobs.values() if j.status.name == "RUNNING"]),
            "completed": len([j for j in job_manager.jobs.values() if j.status.name == "COMPLETED"])
        },
        "system": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    }
```

### Phase 6: Security, Authentication & Production (Week 11-12)

**Goal:** Production-ready security and deployment

#### Authentication System
```python
# api/middleware/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from core.config import settings

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Auth routes
@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Simple token endpoint (for demo - use proper auth in production)."""
    # In production, validate against user database
    if form_data.username == "demo" and form_data.password == "demo":
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
```

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

def test_cwt_endpoint():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    request_data = {
        "data": data,
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

def test_async_significance():
    # Test async job submission
    data1 = [1.0, 2.0, 3.0] * 100  # Longer time series
    data2 = [0.5, 1.5, 2.5] * 100
    
    request_data = {
        "data1": data1,
        "data2": data2,
        "dt": 1.0,
        "dj": 0.25,
        "s0": 2.0,
        "J": 5,
        "significance_test": True,
        "backend": "sequential",
        "mc_count": 50
    }
    
    # Submit job
    response = client.post("/api/v1/analysis/significance", json=request_data)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # Check status
    status_response = client.get(f"/api/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] in ["queued", "running", "completed"]
```

### Integration Tests
- Test full analysis workflows
- Test backend switching
- Test file upload/download
- Test concurrent job handling
- Test error handling and recovery

### Load Testing
- Simulate multiple concurrent users
- Test memory usage with large datasets
- Test job queue performance
- Monitor backend resource usage

---

## 📚 API Documentation

### OpenAPI/Swagger
The server automatically generates OpenAPI documentation at `/docs` and `/redoc`.

### API Versioning
- Current: `/api/v1/*`
- Future versions will be added as `/api/v2/*` with backward compatibility

### Rate Limiting
```python
# Add rate limiting middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
```

### Error Handling
```python
# Global exception handler
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "path": str(request.url)
            }
        }
    )
```

---

```bash
# scripts/install.sh - Bare Metal Installation Script
#!/bin/bash

# PyCWT-mod Server Bare Metal Installation
# This script sets up the server on a clean Ubuntu/Debian system

set -e

echo "🚀 Installing PyCWT-mod Server on Bare Metal"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    nginx \
    supervisor \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    postgresql-client \
    redis-server \
    htop \
    iotop \
    sysstat \
    logrotate

# Create application user
echo "👤 Creating application user..."
sudo useradd -r -s /bin/false pycwt || true

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p /opt/pycwt-server
sudo chown pycwt:pycwt /opt/pycwt-server

# Clone or copy application code
echo "📋 Installing application code..."
# Assuming code is already in current directory
sudo cp -r . /opt/pycwt-server/
sudo chown -R pycwt:pycwt /opt/pycwt-server

# Create data directory
sudo mkdir -p /opt/pycwt-server/data
sudo chown pycwt:pycwt /opt/pycwt-server/data

# Create logs directory
sudo mkdir -p /var/log/pycwt-server
sudo chown pycwt:pycwt /var/log/pycwt-server

# Setup Python virtual environment
echo "🐍 Setting up Python virtual environment..."
sudo -u pycwt bash -c "
cd /opt/pycwt-server
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-server.txt
"

# Install systemd service
echo "⚙️ Installing systemd service..."
sudo cp scripts/systemd.service /etc/systemd/system/pycwt-server.service
sudo systemctl daemon-reload
sudo systemctl enable pycwt-server

# Configure Nginx
echo "🌐 Configuring Nginx reverse proxy..."
sudo cp scripts/nginx.conf /etc/nginx/sites-available/pycwt-server
sudo ln -sf /etc/nginx/sites-available/pycwt-server /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Configure logrotate
echo "📝 Setting up log rotation..."
sudo cp scripts/logrotate.conf /etc/logrotate.d/pycwt-server

# Setup firewall (optional)
echo "🔥 Configuring firewall..."
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# Create environment file
echo "🔐 Creating environment configuration..."
sudo tee /opt/pycwt-server/.env > /dev/null <<EOF
# Server Configuration
HOST=127.0.0.1
PORT=8000
WORKERS=4

# Security
SECRET_KEY=$(openssl rand -hex 32)
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Data storage
DATA_DIR=/opt/pycwt-server/data
MAX_FILE_SIZE=104857600

# Job management
MAX_CONCURRENT_JOBS=20
JOB_TIMEOUT_MINUTES=60

# Hardware backends
ENABLE_ELM11=false
ENABLE_TANG_NANO=false
EOF

sudo chown pycwt:pycwt /opt/pycwt-server/.env
sudo chmod 600 /opt/pycwt-server/.env

echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit /opt/pycwt-server/.env with your configuration"
echo "2. Start the server: sudo systemctl start pycwt-server"
echo "3. Check status: sudo systemctl status pycwt-server"
echo "4. View logs: sudo journalctl -u pycwt-server -f"
echo "5. Access API at: http://your-server-ip"
echo ""
echo "📚 API Documentation: http://your-server-ip/docs"
```

#### SystemD Service File
```ini
# scripts/systemd.service
[Unit]
Description=PyCWT-mod Wavelet Analysis Server
After=network.target
Requires=redis-server.service

[Service]
Type=simple
User=pycwt
Group=pycwt
WorkingDirectory=/opt/pycwt-server
Environment=PATH=/opt/pycwt-server/venv/bin
ExecStart=/opt/pycwt-server/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=pycwt-server

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ReadWritePaths=/opt/pycwt-server/data /var/log/pycwt-server
ProtectSystem=strict
ProtectKernelTunables=yes
ProtectControlGroups=yes

# Resource limits
LimitNOFILE=65536
MemoryLimit=4G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

#### Nginx Reverse Proxy Configuration
```nginx
# scripts/nginx.conf
server {
    listen 80;
    server_name your-server-ip-or-domain;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # Static files (if serving frontend)
    location /static/ {
        alias /opt/pycwt-server/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 300s;  # Longer for analysis jobs

        # Buffer settings for large responses
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;

        # WebSocket support (for future real-time features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

#### Log Rotation Configuration
```bash
# scripts/logrotate.conf
/var/log/pycwt-server/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 pycwt pycwt
    postrotate
        systemctl reload pycwt-server
    endscript
}
```

### 2. Cloud Bare Metal Deployment

For cloud providers offering bare metal instances:

#### AWS Bare Metal (i3.metal, c5n.metal, etc.)
```bash
# AWS EC2 bare metal instance setup
sudo yum update -y
sudo yum install -y python311 python311-pip nginx git

# Install application (similar to local install.sh)
git clone https://github.com/aptitudetechnology/pycwt-mod
cd pycwt-mod
sudo bash scripts/install.sh

# Configure security groups for ports 80, 443
# Attach IAM roles for any cloud services needed
```

#### Google Cloud Bare Metal
```bash
# Google Cloud bare metal setup
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip nginx

# Install application
git clone https://github.com/aptitudetechnology/pycwt-mod
cd pycwt-mod
sudo bash scripts/install.sh

# Configure firewall rules
gcloud compute firewall-rules create pycwt-server \
    --allow tcp:80,tcp:443 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow HTTP/HTTPS traffic to PyCWT server"
```

#### Hetzner Bare Metal
```bash
# Hetzner dedicated server setup
apt-get update
apt-get install -y python3.11 python3-pip nginx git

# Install application
git clone https://github.com/aptitudetechnology/pycwt-mod
cd pycwt-mod
sudo bash scripts/install.sh

# Configure firewall
ufw allow 80
ufw allow 443
ufw --force enable
```

### 3. Multi-Server Bare Metal Cluster

For high-availability deployments across multiple bare metal servers:

#### Load Balancer Setup (Nginx)
```nginx
# load-balancer.conf (on dedicated load balancer server)
upstream pycwt_backend {
    server 10.0.0.11:8000;
    server 10.0.0.12:8000;
    server 10.0.0.13:8000;
    server 10.0.0.14:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://pycwt_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Session persistence for job management
        proxy_cookie_path / "/; SameSite=Lax";

        # Health checks
        health_check interval=10s;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://pycwt_backend/health;
    }
}
```

#### Shared Storage Setup (NFS)
```bash
# On storage server (10.0.0.10)
sudo apt-get install -y nfs-kernel-server
sudo mkdir -p /shared/pycwt-data
sudo chown pycwt:pycwt /shared/pycwt-data

# /etc/exports
/shared/pycwt-data 10.0.0.11(rw,sync,no_subtree_check)
/shared/pycwt-data 10.0.0.12(rw,sync,no_subtree_check)
/shared/pycwt-data 10.0.0.13(rw,sync,no_subtree_check)
/shared/pycwt-data 10.0.0.14(rw,sync,no_subtree_check)

sudo systemctl restart nfs-kernel-server

# On each application server
sudo apt-get install -y nfs-common
sudo mount 10.0.0.10:/shared/pycwt-data /opt/pycwt-server/data
# Add to /etc/fstab for persistence:
# 10.0.0.10:/shared/pycwt-data /opt/pycwt-server/data nfs defaults 0 0
```

#### Redis for Job Queue (Optional)
```bash
# On dedicated Redis server
sudo apt-get install -y redis-server
sudo sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
sudo systemctl restart redis-server

# Configure application servers to use Redis
# In .env: REDIS_URL=redis://10.0.0.15:6379
```

### 4. Hardware-Specific Optimizations

#### CPU Affinity and Performance Tuning
```bash
# Pin server processes to specific CPU cores
# In systemd service file, add:
# CPUAffinity=2-15  # Use cores 2-15, leave 0-1 for system

# Disable CPU frequency scaling for consistent performance
sudo cpupower frequency-set -g performance

# Optimize network stack
sudo tee -a /etc/sysctl.conf > /dev/null <<EOF
# Network optimizations for high-throughput API
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 65536
net.ipv4.ip_local_port_range = 1024 65535
net.core.netdev_max_backlog = 5000
EOF
sudo sysctl -p
```

#### Storage Optimization
```bash
# For high-I/O workloads, use SSD storage
# Configure data directory on fast storage
# In .env: DATA_DIR=/mnt/fast-ssd/pycwt-data

# Optimize filesystem for many small files
sudo tune2fs -o journal_data_writeback /dev/sda1
sudo tune2fs -O ^has_journal /dev/sda1  # Disable journaling for data volume
```

#### Memory Management
```bash
# Configure huge pages for better memory performance
sudo tee -a /etc/sysctl.conf > /dev/null <<EOF
vm.nr_hugepages = 1024
vm.hugetlb_shm_group = $(id -g pycwt)
EOF
sudo sysctl -p

# Disable swap for deterministic performance
sudo swapoff -a
# Remove swap from /etc/fstab if needed
```

### 5. Monitoring and Maintenance

#### System Monitoring Setup
```bash
# Install monitoring tools
sudo apt-get install -y prometheus-node-exporter grafana

# Configure Prometheus node exporter
sudo systemctl enable prometheus-node-exporter
sudo systemctl start prometheus-node-exporter

# Application-specific monitoring
sudo tee /opt/pycwt-server/monitoring.sh > /dev/null <<EOF
#!/bin/bash
# Custom monitoring script
echo "=== PyCWT Server Status ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.2f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df /opt/pycwt-server | tail -1 | awk '{print $5}')"
echo "Active Jobs: $(curl -s http://localhost:8000/api/v1/health/detailed | jq '.jobs.running')"
echo "Uptime: $(uptime -p)"
EOF

sudo chmod +x /opt/pycwt-server/monitoring.sh
# Add to cron: */5 * * * * /opt/pycwt-server/monitoring.sh >> /var/log/pycwt-server/monitoring.log
```

#### Backup Strategy
```bash
# scripts/backup.sh
#!/bin/bash

BACKUP_DIR="/opt/pycwt-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/pycwt_backup_$TIMESTAMP.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Stop service for consistent backup
sudo systemctl stop pycwt-server

# Create backup
sudo tar -czf $BACKUP_FILE \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.log' \
    /opt/pycwt-server/

# Restart service
sudo systemctl start pycwt-server

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "pycwt_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
```

#### Automated Updates
```bash
# scripts/update.sh
#!/bin/bash

cd /opt/pycwt-server

# Pull latest changes
sudo -u pycwt git pull origin main

# Update dependencies
sudo -u pycwt bash -c "
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-server.txt
"

# Run database migrations if any
# python manage.py migrate

# Restart service
sudo systemctl restart pycwt-server

echo "Update completed successfully"
```

---

## 🔧 Development Workflow

### Local Development
```bash
# Clone and setup
git clone https://github.com/aptitudetechnology/pycwt-mod
cd pycwt-mod

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-server.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access documentation
open http://localhost:8000/docs
```

### Bare Metal Development Setup
```bash
# For development on the target bare metal server
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip git nginx

# Clone repository
git clone https://github.com/aptitudetechnology/pycwt-mod
cd pycwt-mod

# Setup development environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-server.txt

# Install development tools
pip install pytest pytest-asyncio httpx locust black isort mypy

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
```

### Testing
```bash
# Run API tests
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=api --cov-report=html

# Load testing
locust -f tests/load_test.py

# Performance testing on bare metal
# Test with different backend configurations
pytest tests/test_performance.py --backend=sequential
pytest tests/test_performance.py --backend=joblib
```

### Bare Metal Testing Environment
```bash
# Setup test environment on bare metal server
sudo mkdir -p /opt/pycwt-test
sudo chown $USER:$USER /opt/pycwt-test

# Copy test configuration
cp .env.example .env
# Edit .env with test settings

# Run integration tests against running server
pytest tests/test_integration.py --server-url=http://localhost:8000

# Load testing with Locust
locust -f tests/load_test.py --host=http://localhost:8000
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new endpoints
4. Update API documentation
5. Test on bare metal hardware when possible
6. Submit a pull request

### Bare Metal CI/CD Pipeline
```yaml
# .github/workflows/bare-metal-deploy.yml
name: Bare Metal Deployment

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-server.txt
    - name: Run tests
      run: pytest

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to staging server
      run: |
        echo "Deploying to bare metal staging server..."
        # SSH to staging server and run deployment script
        ssh user@staging-server "cd /opt/pycwt-server && sudo bash scripts/update.sh"

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to bare metal production servers..."
        # Deploy to production cluster
        for server in prod-server-01 prod-server-02 prod-server-03; do
          ssh user@$server "cd /opt/pycwt-server && sudo bash scripts/update.sh"
        done
```

---

## 📈 Performance Considerations

### Optimization Strategies
1. **Async Processing**: Use async/await for I/O operations
2. **Job Queue**: Limit concurrent jobs to prevent resource exhaustion
3. **Data Caching**: Cache frequently used datasets
4. **Result Compression**: Compress large result arrays
5. **Backend Selection**: Automatically choose optimal backend
6. **Horizontal Scaling**: Deploy multiple server instances

### Monitoring
- **Application Metrics**: Response times, error rates, throughput
- **System Metrics**: CPU, memory, disk usage
- **Job Metrics**: Queue length, completion times, failure rates
- **Backend Metrics**: Availability, performance per backend

### Scaling
- **Vertical Scaling**: Increase server resources
- **Horizontal Scaling**: Multiple server instances with load balancer
- **Backend Scaling**: Distribute jobs across multiple backend instances
- **Caching Layer**: Redis for session and result caching

---

## 🔮 Future Enhancements

### Advanced Features
- **Real-time Streaming**: WebSocket support for live progress updates
- **Batch Processing**: Submit multiple analyses in one request
- **Result Visualization**: Built-in plotting and chart generation
- **Plugin System**: Allow custom analysis functions
- **Federated Computing**: Distribute jobs across multiple servers

### Integration Options
- **Jupyter Integration**: Jupyter kernel for interactive analysis
- **Web Dashboard**: React/Vue.js frontend for the API
- **Database Integration**: Store results in PostgreSQL/MongoDB
- **Message Queue**: RabbitMQ/Kafka for job distribution

### Enterprise Features
- **User Management**: Multi-user support with permissions
- **Audit Logging**: Track all API usage
- **API Rate Limiting**: Per-user and per-endpoint limits
- **Data Encryption**: Encrypt sensitive data at rest and in transit

---

---

## 🖥️ Bare Metal Advantages & Considerations

### Why Bare Metal for Scientific Computing

**Performance Benefits:**
- ✅ **Zero virtualization overhead** - Direct hardware access for maximum performance
- ✅ **Full CPU core utilization** - No hypervisor scheduling delays
- ✅ **Direct memory access** - No memory ballooning or paging issues
- ✅ **Hardware-specific optimizations** - CPU pinning, huge pages, NUMA tuning
- ✅ **Predictable performance** - No noisy neighbor effects from other VMs

**Hardware Acceleration Integration:**
- ✅ **Direct GPIO/serial access** for embedded backends (ELM11 Lua)
- ✅ **FPGA programming** without virtualization layers (Tang Nano 9K)
- ✅ **GPU passthrough** for maximum performance
- ✅ **High-speed networking** without virtual switches
- ✅ **Storage optimization** with direct SSD/NVMe access

**Scientific Computing Advantages:**
- ✅ **Deterministic timing** - Critical for real-time signal processing
- ✅ **Large memory support** - Handle massive datasets without limits
- ✅ **Low-latency operations** - Essential for wavelet transforms
- ✅ **Hardware math acceleration** - Direct use of CPU vector instructions
- ✅ **Custom kernel modules** - Optimize for specific computational patterns

### Bare Metal vs Container/Cloud Trade-offs

| Aspect | Bare Metal | Containers/Cloud |
|--------|------------|------------------|
| **Performance** | Maximum (0% overhead) | 5-15% virtualization overhead |
| **Hardware Access** | Full direct access | Limited by hypervisor |
| **Cost Efficiency** | High for stable workloads | Variable, auto-scaling |
| **Management** | Manual provisioning | Automated orchestration |
| **Scalability** | Physical server addition | Dynamic scaling |
| **Development** | Hardware-specific testing | Environment consistency |
| **Security** | Physical access control | Cloud security features |

### Bare Metal Server Specifications

**Development Workstation:**
- **CPU:** 8-16 cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM:** 32-64GB DDR4
- **Storage:** 1TB NVMe SSD
- **Network:** 2.5Gbps Ethernet
- **GPU:** Optional NVIDIA/AMD for GPU backend testing
- **Cost:** $1,500-3,000

**Production Single Server:**
- **CPU:** 32-64 cores (Intel Xeon or AMD EPYC)
- **RAM:** 128-512GB DDR4/ECC
- **Storage:** 2-4TB NVMe SSD + 10-20TB HDD
- **Network:** 10-25Gbps Ethernet
- **Power:** Redundant PSUs, IPMI management
- **Cost:** $5,000-15,000

**High-Performance Cluster:**
- **Nodes:** 4-16 identical servers
- **CPU:** 64 cores per node (AMD EPYC)
- **RAM:** 1TB per node
- **Storage:** Shared 100-500TB NVMe storage
- **Network:** 100Gbps Infiniband/Ethernet
- **Management:** Centralized management node
- **Cost:** $50,000-200,000+

**Embedded Development Server:**
- **Form factor:** Rackmount or tower
- **Interfaces:** USB 3.0, serial ports, GPIO headers
- **Expansion:** PCIe slots for FPGA cards
- **Power:** 750W+ PSU for hardware acceleration
- **Cooling:** Adequate for continuous high utilization

### Bare Metal Optimization Strategies

**CPU Optimization:**
```bash
# Pin server processes to specific CPU cores
taskset -c 0-15 uvicorn main:app --workers 4

# Disable CPU frequency scaling
cpupower frequency-set -g performance

# Configure CPU governor for consistent performance
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Memory Optimization:**
```bash
# Enable huge pages for better memory performance
echo 1024 > /proc/sys/vm/nr_hugepages

# Disable memory overcommit (prevent OOM kills)
echo 0 > /proc/sys/vm/overcommit_memory

# Optimize swappiness for compute workloads
echo 1 > /proc/sys/vm/swappiness
```

**Network Optimization:**
```bash
# Increase network buffer sizes
echo 8388608 > /proc/sys/net/core/rmem_max
echo 8388608 > /proc/sys/net/core/wmem_max

# Optimize TCP settings for high-throughput
echo 4096 87380 8388608 > /proc/sys/net/ipv4/tcp_rmem
echo 4096 65536 8388608 > /proc/sys/net/ipv4/tcp_wmem
```

**Storage Optimization:**
```bash
# Use SSD for data directory
# Configure I/O scheduler for SSD
echo noop > /sys/block/sda/queue/scheduler

# Disable access time updates
mount -o noatime,nodiratime /dev/sda1 /opt/pycwt-server/data
```

### Bare Metal Monitoring & Maintenance

**System Monitoring Setup:**
```bash
# Install comprehensive monitoring
apt-get install -y htop iotop sysstat prometheus-node-exporter

# Custom monitoring script
cat > /opt/pycwt-server/monitor.sh << 'EOF'
#!/bin/bash
echo "=== PyCWT Bare Metal Server Status ==="
echo "CPU Usage: $(mpstat 1 1 | awk '$12 ~ /[0-9.]+/ { print 100 - $12"%" }')"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print "Used: "$3"/"$2" ("int($3/$2*100)"%)"}')"
echo "Disk I/O: $(iostat -x 1 1 | grep sda | awk '{print "Read: "$6" MB/s, Write: "$7" MB/s"}')"
echo "Network: $(ip -s link show eth0 | grep -A 1 "RX:" | tail -1 | awk '{print "RX: "$1" MB, TX: "$5" MB"}')"
echo "Active Jobs: $(curl -s http://localhost:8000/api/v1/health/detailed | jq '.jobs.running')"
echo "Temperature: $(sensors | grep 'Core 0' | awk '{print $3}')"
EOF

chmod +x /opt/pycwt-server/monitor.sh
# Add to cron: */5 * * * * /opt/pycwt-server/monitor.sh >> /var/log/pycwt-server/monitor.log
```

**Automated Maintenance:**
```bash
# Update script
cat > /opt/pycwt-server/update.sh << 'EOF'
#!/bin/bash
cd /opt/pycwt-server

# Backup current version
cp -r . ../backup-$(date +%Y%m%d-%H%M%S)

# Update code
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt -r requirements-server.txt

# Run migrations if needed
# python manage.py migrate

# Restart service
sudo systemctl restart pycwt-server

# Verify health
sleep 10
curl -f http://localhost:8000/health || echo "Health check failed!"
EOF
```

## ✅ Success Criteria

### Functional Requirements
- [ ] All core wavelet functions available via REST API
- [ ] Backend selection working for all supported backends
- [ ] Async job processing for long-running computations
- [ ] Data upload/download functionality
- [ ] Comprehensive error handling and validation

### Performance Requirements
- [ ] API response time < 500ms for simple operations
- [ ] Job throughput > 10 concurrent analyses
- [ ] Memory usage < 2GB per server instance
- [ ] Support for datasets up to 1M data points

### Quality Requirements
- [ ] 95%+ test coverage for API endpoints
- [ ] OpenAPI documentation automatically generated
- [ ] Comprehensive error messages and status codes
- [ ] Backward compatibility maintained across versions

### Deployment Requirements
- [ ] Bare metal installation script (`scripts/install.sh`) functional
- [ ] SystemD service file configured and working
- [ ] Nginx reverse proxy properly configured
- [ ] Log rotation configured
- [ ] Environment configuration management working
- [ ] Security hardening applied (firewall, user permissions)
- [ ] Monitoring and health checks operational
- [ ] Backup and recovery procedures documented
- [ ] Multi-server cluster support (optional)</content>
<parameter name="filePath">/home/chris/pycwt-mod/make-this-a-server.md

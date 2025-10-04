"""Main FastAPI application for pycwt REST API."""

import sys
from pathlib import Path

# Add src directory to Python path for pycwt_mod imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import backends, wavelet
from server.core.config import Settings

# Initialize settings
settings = Settings()

# Create FastAPI application
app = FastAPI(
    title="PyCWT REST API",
    description="REST API for continuous wavelet transform analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    backends.router,
    prefix="/api/v1/backends",
    tags=["backends"]
)

app.include_router(
    wavelet.router,
    prefix="/api/v1/wavelet",
    tags=["wavelet"]
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PyCWT REST API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API server.
    """
    return {
        "status": "healthy",
        "api_version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

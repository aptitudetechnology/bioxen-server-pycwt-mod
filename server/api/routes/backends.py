"""Backend discovery and information endpoints."""

import sys
from pathlib import Path

# Add src directory to Python path for pycwt_mod imports
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import APIRouter, HTTPException
from pycwt_mod.backends import list_backends, get_backend

router = APIRouter()


def get_backend_type(backend) -> str:
    """
    Determine the type of backend (cpu, fpga, gpu, embedded).
    
    Args:
        backend: Backend instance
        
    Returns:
        Backend type string
    """
    backend_class_name = type(backend).__name__.lower()
    
    # FPGA backends
    if 'elm11' in backend_class_name or 'fpga' in backend_class_name or 'tang' in backend_class_name:
        return 'fpga'
    
    # GPU backends
    if 'gpu' in backend_class_name or 'cuda' in backend_class_name:
        return 'gpu'
    
    # Embedded/microcontroller backends
    if 'embedded' in backend_class_name or 'micro' in backend_class_name:
        return 'embedded'
    
    # Default to CPU for sequential, joblib, dask, etc.
    return 'cpu'


@router.get("/")
async def list_available_backends():
    """
    List all available computation backends.
    
    Returns information about each registered backend including
    availability status and description.
    """
    backend_names = list_backends()
    backend_info = []
    
    for name in backend_names:
        try:
            backend = get_backend(name)
            backend_info.append({
                "name": name,
                "type": get_backend_type(backend),
                "available": backend.is_available(),
                "description": getattr(backend, "__doc__", "").split("\n")[0]
            })
        except Exception as e:
            backend_info.append({
                "name": name,
                "type": "unknown",
                "available": False,
                "error": str(e)
            })
    
    return {"backends": backend_info}


@router.get("/{backend_name}")
async def get_backend_info(backend_name: str):
    """
    Get detailed information about a specific backend.
    
    Args:
        backend_name: Name of the backend to query
        
    Returns:
        Detailed backend information including availability and capabilities
    """
    try:
        backend = get_backend(backend_name)
        return {
            "name": backend_name,
            "type": get_backend_type(backend),
            "available": backend.is_available(),
            "description": getattr(backend, "__doc__", "")
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

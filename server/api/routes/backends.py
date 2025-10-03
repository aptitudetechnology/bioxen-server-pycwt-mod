"""Backend discovery and information endpoints."""

from fastapi import APIRouter, HTTPException
from pycwt_mod.backends import list_backends, get_backend

router = APIRouter()


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
            "available": backend.is_available(),
            "description": getattr(backend, "__doc__", ""),
            "type": type(backend).__name__
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

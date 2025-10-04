"""
Pydantic models for hardware detection endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FPGAInfo(BaseModel):
    """FPGA hardware information."""
    available: bool = Field(..., description="Whether FPGA hardware is available")
    device: Optional[str] = Field(None, description="FPGA device name/model")
    status: Optional[str] = Field(None, description="Status message or error")
    port: Optional[str] = Field(None, description="Serial port if applicable")


class EmbeddedDeviceInfo(BaseModel):
    """Embedded device information."""
    port: str = Field(..., description="Serial port path")
    description: str = Field(..., description="Device description")
    vendor: Optional[str] = Field(None, description="Vendor ID or name")


class EmbeddedInfo(BaseModel):
    """Embedded systems information."""
    available: bool = Field(..., description="Whether embedded devices are available")
    devices: List[EmbeddedDeviceInfo] = Field(default_factory=list, description="List of detected devices")


class GPUInfo(BaseModel):
    """GPU hardware information."""
    available: bool = Field(..., description="Whether GPU hardware is available")
    type: str = Field(..., description="GPU type (NVIDIA CUDA, AMD ROCm, None, etc)")
    devices: Optional[List[str]] = Field(None, description="List of GPU devices")


class CPUInfo(BaseModel):
    """CPU information."""
    available: bool = Field(True, description="CPU is always available")
    cores: int = Field(..., description="Number of CPU cores")
    model: Optional[str] = Field(None, description="CPU model name")


class HardwareDetectionResponse(BaseModel):
    """Complete hardware detection response."""
    fpga: FPGAInfo = Field(..., description="FPGA hardware information")
    embedded: EmbeddedInfo = Field(..., description="Embedded systems information")
    gpu: GPUInfo = Field(..., description="GPU hardware information")
    cpu: CPUInfo = Field(..., description="CPU information")

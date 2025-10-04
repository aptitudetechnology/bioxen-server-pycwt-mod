"""
Hardware detection API routes.
"""

import os
import multiprocessing
from typing import List
from fastapi import APIRouter, HTTPException

from ..models.hardware import (
    HardwareDetectionResponse,
    FPGAInfo,
    EmbeddedInfo,
    EmbeddedDeviceInfo,
    GPUInfo,
    CPUInfo,
)

router = APIRouter()


def detect_fpga() -> FPGAInfo:
    """
    Detect FPGA hardware (primarily Tang Nano 9K via ELM11).
    
    Returns:
        FPGAInfo with availability and device information
    """
    try:
        # Check if ELM11 backend is configured
        import serial.tools.list_ports
        
        # Look for SIPEED JTAG debugger (Tang Nano 9K)
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            description = port.description.upper()
            # SIPEED JTAG debugger for Tang Nano 9K
            if "SIPEED" in description or "JTAG" in description:
                return FPGAInfo(
                    available=True,
                    device="Tang Nano 9K",
                    status="Connected via SIPEED JTAG",
                    port=port.device
                )
        
        # Check for USB-Serial device that might be FPGA
        for port in ports:
            description = port.description.upper()
            if "USB" in description and "SERIAL" in description:
                # Possible FPGA device
                return FPGAInfo(
                    available=True,
                    device="Unknown FPGA",
                    status="Possible FPGA device detected",
                    port=port.device
                )
        
        return FPGAInfo(
            available=False,
            status="No FPGA devices detected"
        )
        
    except Exception as e:
        return FPGAInfo(
            available=False,
            status=f"Detection error: {str(e)}"
        )


def detect_embedded() -> EmbeddedInfo:
    """
    Detect embedded systems via serial ports.
    
    Returns:
        EmbeddedInfo with list of detected devices
    """
    try:
        import serial.tools.list_ports
        
        ports = serial.tools.list_ports.comports()
        devices: List[EmbeddedDeviceInfo] = []
        
        for port in ports:
            # Filter out obvious non-embedded devices
            description = port.description
            
            # Add all serial devices
            device_info = EmbeddedDeviceInfo(
                port=port.device,
                description=description,
                vendor=f"{port.vid:04X}" if port.vid else None
            )
            devices.append(device_info)
        
        return EmbeddedInfo(
            available=len(devices) > 0,
            devices=devices
        )
        
    except Exception as e:
        return EmbeddedInfo(
            available=False,
            devices=[]
        )


def detect_gpu() -> GPUInfo:
    """
    Detect GPU hardware (NVIDIA CUDA, AMD ROCm, etc).
    
    Returns:
        GPUInfo with availability and type
    """
    # Check for NVIDIA CUDA
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        
        if device_count > 0:
            devices = []
            for i in range(device_count):
                device = cp.cuda.Device(i)
                devices.append(f"GPU {i}: {device.compute_capability}")
            
            return GPUInfo(
                available=True,
                type="NVIDIA CUDA",
                devices=devices
            )
    except ImportError:
        pass
    except Exception:
        pass
    
    # Check for AMD ROCm
    try:
        import os
        if os.path.exists("/opt/rocm"):
            return GPUInfo(
                available=True,
                type="AMD ROCm",
                devices=["ROCm detected"]
            )
    except Exception:
        pass
    
    # No GPU detected
    return GPUInfo(
        available=False,
        type="None"
    )


def detect_cpu() -> CPUInfo:
    """
    Detect CPU information.
    
    Returns:
        CPUInfo with core count and model
    """
    cores = multiprocessing.cpu_count()
    
    # Try to get CPU model name on Linux
    model = None
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        model = line.split(":")[1].strip()
                        break
    except Exception:
        pass
    
    return CPUInfo(
        available=True,
        cores=cores,
        model=model
    )


@router.get("/detect", response_model=HardwareDetectionResponse)
async def detect_hardware():
    """
    Detect available hardware (FPGA, GPU, CPU, embedded devices).
    
    This endpoint never returns an error - if detection fails for any
    component, it returns unavailable status with error message.
    
    Returns:
        HardwareDetectionResponse with all hardware categories
    """
    try:
        fpga = detect_fpga()
        embedded = detect_embedded()
        gpu = detect_gpu()
        cpu = detect_cpu()
        
        return HardwareDetectionResponse(
            fpga=fpga,
            embedded=embedded,
            gpu=gpu,
            cpu=cpu
        )
    except Exception as e:
        # Even if something goes catastrophically wrong, return safe defaults
        return HardwareDetectionResponse(
            fpga=FPGAInfo(available=False, status=f"Error: {str(e)}"),
            embedded=EmbeddedInfo(available=False, devices=[]),
            gpu=GPUInfo(available=False, type="None"),
            cpu=CPUInfo(available=True, cores=1)  # Assume at least 1 core
        )

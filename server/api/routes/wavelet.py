"""Wavelet analysis endpoints."""

import sys
import time
import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Optional

# Add src directory to Python path for pycwt_mod imports
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pycwt_mod import cwt, wct, xwt, wct_significance
from server.api.models.wavelet import (
    CWTRequest, CWTResponse,
    WCTRequest, WCTResponse,
    XWTRequest, XWTResponse
)

router = APIRouter()


@router.post("/cwt", response_model=CWTResponse)
async def continuous_wavelet_transform(request: CWTRequest):
    """
    Perform Continuous Wavelet Transform (CWT) on a signal.
    
    Args:
        request: CWT parameters including signal data and transform settings
        
    Returns:
        Wavelet coefficients, scales, frequencies, and cone of influence
    """
    try:
        start_time = time.time()
        
        # Convert input data to numpy array
        signal = np.array(request.data)
        
        # Perform CWT
        wave, scales, freqs, coi, fft, fftfreqs = cwt(
            signal,
            dt=request.dt,
            dj=request.dj if request.dj else 1/12,
            s0=request.s0 if request.s0 and request.s0 > 0 else -1,
            J=request.J if request.J and request.J > 0 else -1,
            wavelet=request.mother
        )
        
        computation_time = time.time() - start_time
        
        # Convert complex arrays to lists of [real, imag] pairs
        wave_list = [
            [[float(val.real), float(val.imag)] for val in row]
            for row in wave
        ]
        fft_list = [[float(val.real), float(val.imag)] for val in fft]
        
        return CWTResponse(
            wave=wave_list,
            scales=scales.tolist(),
            freqs=freqs.tolist(),
            coi=coi.tolist(),
            fft=fft_list,
            fftfreqs=fftfreqs.tolist(),
            computation_time=computation_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CWT computation failed: {str(e)}"
        )


@router.post("/wct", response_model=WCTResponse)
async def wavelet_coherence_transform(request: WCTRequest):
    """
    Perform Wavelet Coherence Transform (WCT) between two signals.
    
    Args:
        request: WCT parameters including two signals and transform settings
        
    Returns:
        Wavelet coherence, scales, frequencies, and optional significance levels
    """
    try:
        start_time = time.time()
        
        # Convert input data to numpy arrays
        signal1 = np.array(request.signal1)
        signal2 = np.array(request.signal2)
        
        # Validate signal lengths match
        if len(signal1) != len(signal2):
            raise HTTPException(
                status_code=400,
                detail=f"Signal lengths must match: {len(signal1)} != {len(signal2)}"
            )
        
        # Perform WCT
        WCT, aWCT, coi, freqs, signif = wct(
            signal1,
            signal2,
            dt=request.dt,
            dj=request.dj if request.dj else 1/12,
            s0=request.s0 if request.s0 and request.s0 > 0 else -1,
            J=request.J if request.J and request.J > 0 else -1,
            wavelet=request.mother
        )
        
        # Compute significance if requested
        # Significance is computed only if sig=True or significance_level is explicitly provided (not None)
        signif_list = None
        if request.sig or request.significance_level is not None:
            # Calculate lag-1 autocorrelation for red noise
            al1 = np.corrcoef(signal1[:-1], signal1[1:])[0, 1]
            al2 = np.corrcoef(signal2[:-1], signal2[1:])[0, 1]
            
            # Use provided significance_level or default to 0.95
            sig_level = request.significance_level if request.significance_level is not None else 0.95
            
            # Compute significance levels
            sig95 = wct_significance(
                al1=al1,
                al2=al2,
                dt=request.dt,
                dj=request.dj if request.dj else 1/12,
                s0=request.s0 if request.s0 and request.s0 > 0 else 2 * request.dt,
                J=request.J if request.J and request.J > 0 else int(np.log2(len(signal1) * request.dt / (2 * request.dt)) / (request.dj if request.dj else 1/12)),
                significance_level=sig_level,
                mc_count=request.mc_count if request.mc_count else 30,
                backend=request.backend,
                n_jobs=request.n_jobs,
                progress=False,
                cache=False
            )
            signif_list = sig95.tolist()
        
        computation_time = time.time() - start_time
        
        # Get scales from freqs
        scales = (1.0 / freqs).tolist()
        
        return WCTResponse(
            WCT=WCT.tolist(),
            aWCT=aWCT.tolist(),
            coi=coi.tolist(),
            freqs=freqs.tolist(),
            scales=scales,
            signif=signif_list,
            computation_time=computation_time,
            backend_used=request.backend if request.backend else "auto"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"WCT computation failed: {str(e)}"
        )


@router.post("/xwt", response_model=XWTResponse)
async def cross_wavelet_transform(request: XWTRequest):
    """
    Perform Cross Wavelet Transform (XWT) between two signals.
    
    Args:
        request: XWT parameters including two signals and transform settings
        
    Returns:
        Cross wavelet coefficients, amplitude, phase angles, and scales
    """
    try:
        start_time = time.time()
        
        # Convert input data to numpy arrays
        signal1 = np.array(request.signal1)
        signal2 = np.array(request.signal2)
        
        # Validate signal lengths match
        if len(signal1) != len(signal2):
            raise HTTPException(
                status_code=400,
                detail=f"Signal lengths must match: {len(signal1)} != {len(signal2)}"
            )
        
        # Perform XWT
        # xwt() returns: (W12, coi, freq, signif)
        W12, coi, freqs, signif = xwt(
            signal1,
            signal2,
            dt=request.dt,
            dj=request.dj if request.dj else 1/12,
            s0=request.s0 if request.s0 and request.s0 > 0 else -1,
            J=request.J if request.J and request.J > 0 else -1,
            wavelet=request.mother
        )
        
        computation_time = time.time() - start_time
        
        # Calculate amplitude and phase from complex W12
        WXamp = np.abs(W12)
        WXangle = np.angle(W12)
        
        # Convert complex XWT array to lists of [real, imag] pairs
        xwt_list = [
            [[float(val.real), float(val.imag)] for val in row]
            for row in W12
        ]
        
        # Get scales from freqs
        scales = (1.0 / freqs).tolist()
        
        return XWTResponse(
            xwt=xwt_list,
            WXamp=WXamp.tolist(),
            WXangle=WXangle.tolist(),
            coi=coi.tolist(),
            freqs=freqs.tolist(),
            scales=scales,
            computation_time=computation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"XWT computation failed: {str(e)}"
        )

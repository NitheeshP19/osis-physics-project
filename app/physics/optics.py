# app/physics/optics.py
import math
import numpy as np
from typing import Dict, Any, List
from app.utils.constants import MAX_POINTS

def estimate_spot_size_nm(wavelength_nm: float, numerical_aperture: float) -> float:
    return (0.61 * wavelength_nm) / numerical_aperture

def estimate_crosstalk(track_pitch_nm: float, spot_size_nm: float, alpha: float = 0.002) -> float:
    return math.exp(-alpha * max((track_pitch_nm - spot_size_nm), 0))

def approximate_reflectivity_spectrum(stack_config: List[Dict[str, Any]]) -> Dict[str, list]:
    """
    Approximates Transfer Matrix Method (TMM) for reflectivity based on stack materials.
    Using vectorized numpy arrays restricted to MAX_POINTS for safety and speed.
    """
    wavelengths = np.linspace(300, 900, min(MAX_POINTS, 200))
    reflectivity = np.zeros_like(wavelengths)
    
    # Heuristic approach:
    # 1. Base metal reflection (Ag -> highly reflective visible, Al -> UV drop-off)
    # 2. Add interference modulation based on dielectric thickness
    total_dielectric_nm = 0
    has_metal = False
    metal_type = ""
    phase_change_k = 0

    for layer in stack_config:
        mat = layer.get('material', '').lower()
        if "ag" in mat or "al" in mat or 'au' in mat:
            has_metal = True
            metal_type = mat
        elif "dielectric" in mat or "zns-sio2" in mat:
            total_dielectric_nm += layer.get('thicknessNm', 0)
        elif "gst" in mat or "phasechange" in mat:
            phase_change_k = layer.get('extinctionCoefficientK', 2.0)

    # Base metal reflectivity curve
    if has_metal:
        if "ag" in metal_type:
            base_curve = 0.95 - np.exp(-wavelengths / 100) * 0.2
        elif "au" in metal_type:
            base_curve = 0.90 / (1 + np.exp(-(wavelengths - 500) / 20))
        else:
            base_curve = np.full_like(wavelengths, 0.85)
    else:
        base_curve = np.full_like(wavelengths, 0.40)

    # Apply interference ripples caused by dielectric layer (Fabry-Perot heuristic)
    if total_dielectric_nm > 0:
        optical_thickness = total_dielectric_nm * 2.1 # n approx 2.1
        interference = 0.15 * np.cos(4 * np.pi * optical_thickness / wavelengths)
        base_curve += interference

    # Absorption drop from active layer
    absorption = np.exp(-phase_change_k * 10 / wavelengths)
    
    final_curve = np.clip(base_curve * absorption, 0.05, 0.99) * 100 # percentage

    return {
        "wavelengths": wavelengths.round(1).tolist(),
        "reflectivityPercentage": final_curve.round(2).tolist()
    }

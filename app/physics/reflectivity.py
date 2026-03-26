# app/physics/reflectivity.py
import numpy as np
import math

def compute_reflectivity(wavelength_nm: float, n1: float, n2: float, k2: float, thickness_nm: float) -> float:
    """
    Computes single-point base reflectivity using a simplified Fresnel + thin-film interference approximation.
    """
    # Base reflection (Fresnel approx for normal incidence)
    R0 = ((n1 - n2)**2 + k2**2) / ((n1 + n2)**2 + k2**2)
    
    # Thin-film interference term
    phase = 4 * math.pi * n2 * thickness_nm / wavelength_nm
    R = R0 * (1 + 0.3 * math.cos(phase))
    
    # Clamp to physical boundaries
    return float(max(0.0, min(1.0, R)))

def compute_radial_reflectivity(radius_array: np.ndarray, base_thickness: float, variation_scale: float, wavelength_nm: float, n1: float, n2: float, k2: float) -> np.ndarray:
    """
    Vectorized computation of reflectivity across a disc radius, incorporating sputtering thickness variations.
    """
    r_max = np.max(radius_array) if len(radius_array) > 0 else 60.0
    
    # Simulate sputtering variation (dome or wedge profile)
    thickness_r = base_thickness * (1 + variation_scale * (radius_array / r_max))
    
    # Vectorized Fresnel + Interference
    R0 = ((n1 - n2)**2 + k2**2) / ((n1 + n2)**2 + k2**2)
    phase_r = 4 * np.pi * n2 * thickness_r / wavelength_nm
    
    R_r = R0 * (1 + 0.3 * np.cos(phase_r))
    return np.clip(R_r, 0.0, 1.0)

# app/physics/manufacturing.py
import numpy as np
from typing import Dict, Any
from app.utils.constants import MAX_POINTS
from app.physics.reflectivity import compute_radial_reflectivity

def evaluate_monte_carlo_yield(mfg_config: Dict[str, Any], base_snr: float, peak_temp: float, opt_config: Dict[str, Any], samples: int = 100) -> Dict[str, Any]:
    """
    Vectorized Monte Carlo simulation of optical storage manufacturing variations 
    and their cross-coupling effect on optical SNR and Reflectivity.
    """
    molding_temp = mfg_config.get('moldingTempC', 350.0)
    mold_pressure = mfg_config.get('moldPressureTons', 50.0)
    cooling_time = mfg_config.get('coolingTimeS', 2.5)
    sputtering_rate = mfg_config.get('sputteringRateNmS', 5.0)
    defect_density = mfg_config.get('defectDensity', 0.05)
    
    base_thickness = mfg_config.get('baseThicknessNm', 15.0)
    n1 = mfg_config.get('refractiveIndexN1', 1.5)
    n2 = mfg_config.get('refractiveIndexN2', 4.1)
    k2 = mfg_config.get('refractiveIndexK2', 2.1)
    variation_scale = mfg_config.get('thicknessVariationScale', 0.05)
    
    wavelength = opt_config.get('wavelengthNm', 405.0)
    
    radius_points = min(MAX_POINTS, 100)
    radius_mm = np.linspace(20, 60, radius_points) # typical disc radius
    
    # 1. Base Deterministic Physics (Warp across radius)
    cooling_factor = max(0.1, (10 - cooling_time) / 10.0)
    temp_stress = max(0.1, (380 - molding_temp) / 100.0)
    base_biref = 10 + (radius_mm * cooling_factor * temp_stress * 0.5)
    
    # 2. Vectorized Monte Carlo (Shape: [samples, radius_points])
    biref_noise = np.random.normal(0, 5.0, (samples, radius_points))
    thick_noise = np.random.normal(0, base_thickness * 0.05, (samples, radius_points))    
    defect_noise = np.random.normal(defect_density, defect_density * 0.2, (samples, 1))

    mc_biref = np.clip(base_biref + biref_noise, 0, None)
    mc_thick_base = base_thickness + thick_noise
    mc_defect = np.clip(defect_noise, 0, None)
    
    # 3. Simulate sputtering variation radially
    r_max = 60.0
    r_ratio = radius_mm / r_max  # Shape [radius_points]
    mc_thick_r = mc_thick_base * (1 + variation_scale * r_ratio) # Shape [samples, radius_points]
    mc_thick_r = np.clip(mc_thick_r, 0, None)
    
    # 4. Compute Base Reflectivity matrix [samples, radius_points]
    R0 = ((n1 - n2)**2 + k2**2) / ((n1 + n2)**2 + k2**2)
    phase_r = 4 * np.pi * n2 * mc_thick_r / wavelength
    R_matrix = R0 * (1 + 0.3 * np.cos(phase_r))
    R_matrix = np.clip(R_matrix, 0.0, 1.0)
    
    # 5. Manufacturing & Thermal Coupling on Reflectivity
    reflectivity_loss = mc_defect * 0.1 # defect density coupling
    thermal_factor = max(0.0, 1.0 - 0.0005 * (peak_temp - 300))
    R_effective = R_matrix * (1 - reflectivity_loss) * thermal_factor
    R_effective = np.clip(R_effective, 0.0, 1.0)
    
    # 6. Cross-Coupling Domain: SNR Degradation
    # snr_db = base_snr_db + 10 * log10(reflectivity + 1e-6)
    mc_snr = base_snr + 10 * np.log10(R_effective + 1e-6)
    
    # Add Birefringence stress penalty
    mc_snr -= (mc_biref / 10.0) * 0.1
    
    # 7. Statistical Aggregation
    snr_profile_mean = np.mean(mc_snr, axis=0)
    snr_profile_std = np.std(mc_snr, axis=0)
    
    ref_profile_mean = np.mean(R_effective, axis=0)
    ref_profile_std = np.std(R_effective, axis=0)
    
    snr_center_mean = snr_profile_mean[0]   # r=20mm
    snr_edge_mean = snr_profile_mean[-1]    # r=60mm
    overall_snr_mean = np.mean(snr_profile_mean)
    overall_snr_std = np.mean(snr_profile_std)
    
    overall_ref_mean = np.mean(ref_profile_mean)
    overall_ref_std = np.mean(ref_profile_std)
    
    # 8. Yield Confidence Calculation 
    min_snr_per_sample = np.min(mc_snr, axis=1) # Shape: [samples]
    successful_samples = np.sum(min_snr_per_sample >= 20.0)
    yield_confidence = successful_samples / samples
    
    # Downsample profiles to Max Points
    if radius_points > 100:
        indices = np.linspace(0, radius_points - 1, 100, dtype=int)
        snr_profile_mean = snr_profile_mean[indices]
        ref_profile_mean = ref_profile_mean[indices]
        radius_mm_out = radius_mm[indices]
    else:
        radius_mm_out = radius_mm
        
    return {
        "snr_center": round(float(snr_center_mean), 2),
        "snr_edge": round(float(snr_edge_mean), 2),
        "snr_mean": round(float(overall_snr_mean), 2),
        "snr_std": round(float(overall_snr_std), 2),
        "yield_confidence": round(float(yield_confidence), 3),
        "variance_profile": snr_profile_mean.round(2).tolist(),
        "reflectivity_profile": ref_profile_mean.round(4).tolist(),
        "reflectivity_mean": round(float(overall_ref_mean), 4),
        "reflectivity_std": round(float(overall_ref_std), 4),
        "radiusMm": radius_mm_out.round(1).tolist(),
        "birefringenceNm": np.mean(mc_biref, axis=0).round(2).tolist(),
        "thicknessVariancePct": ((np.mean(mc_thick_r, axis=0) - base_thickness) / base_thickness * 100).round(2).tolist(),
        "estimatedYieldPct": round(float(yield_confidence * 100), 2)
    }

def simulate_manufacturing_process(mfg_config: Dict[str, Any], base_snr: float, peak_temp: float, opt_config: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility pass-through for fast deterministic mode (1 sample)"""
    res = evaluate_monte_carlo_yield(mfg_config, base_snr, peak_temp, opt_config, samples=1)
    res["snr_mean"] = res["snr_center"]
    res["snr_std"] = 0.0
    res["yield_confidence"] = 1.0 if res["estimatedYieldPct"] > 90 else 0.5
    return res

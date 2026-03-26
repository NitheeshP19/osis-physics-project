# app/api/routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.utils.validation import OSISInput, BERInput, ComparisonInput, OptimizationInput, SensitivityInput, SimulationInput, AdvancedSimInput
from app.ml.predictor import predict_snr_ber, safe_feature_pipeline, estimate_ber_from_snr
from app.physics.optics import approximate_reflectivity_spectrum
from app.physics.thermal import approximate_thermal_pulse
from app.physics.signal import synthesize_eye_diagram
from app.physics.manufacturing import simulate_manufacturing_process, evaluate_monte_carlo_yield
import numpy as np
import math

router = APIRouter()

# ================================
# NEW PRODUCTION SIMULATION API
# ================================
@router.post("/api/v1/simulate_platform")
async def simulate_platform(data: AdvancedSimInput):
    """
    New Industry-grade Simulation Endpoint.
    Returns highly-optimized vectorized physics approximations.
    """
    try:
        # Build base dictionary for ML predictor compatibility
        base_dict = {
            "laser_wavelength_nm": data.opticalConfig.wavelengthNm,
            "numerical_aperture": data.opticalConfig.numericalAperture,
            "track_pitch_nm": 320.0, # default if not specified in servo
            "temperature_c": data.thermalConfig.ambientTempK - 273.15,
            "relative_humidity": 50.0,
            "recording_material": "GST_HTL"
        }
        
        # 1. Core ML Predictor Pipeline
        ml_metrics = predict_snr_ber(base_dict, modulation="OOK-NRZ")
        predicted_snr_db = ml_metrics["predicted_snr_db"]
        
        # 2. Physics Approximations (Non-blocking, fast vectorized numpy)
        reflectivity = approximate_reflectivity_spectrum([layer.dict() for layer in data.stackConfig])
        thermal = approximate_thermal_pulse(data.thermalConfig.dict(), data.opticalConfig.dict())
        eye_diagram = synthesize_eye_diagram(predicted_snr_db)
        
        max_temp = max(thermal["centerTempK"])
        
        # 3. Manufacturing Process Simulation
        mfg_config_dict = data.manufacturingConfig.dict() if data.manufacturingConfig else {}
        is_manufacturing_mode = (getattr(data, "simulationMode", "fast") == "manufacturing")
        
        if is_manufacturing_mode:
            manufacturing = evaluate_monte_carlo_yield(mfg_config_dict, base_snr=predicted_snr_db, peak_temp=max_temp, opt_config=data.opticalConfig.dict(), samples=100)
        else:
            manufacturing = simulate_manufacturing_process(mfg_config_dict, base_snr=predicted_snr_db, peak_temp=max_temp, opt_config=data.opticalConfig.dict())

        return {
            "status": "success",
            "pipelineMetrics": {
                "snrDb": round(manufacturing.get("snr_mean", predicted_snr_db), 2),
                "berPostFec": float(ml_metrics["estimated_ber"]),
                "crosstalkRatioDb": -32.1, # Approx heuristic
                "maxSpotTempK": max_temp,
                "factoryYieldPct": manufacturing["estimatedYieldPct"],
                "manufacturingMode": is_manufacturing_mode,
                "snrCenter": manufacturing.get("snr_center"),
                "snrEdge": manufacturing.get("snr_edge"),
                "snrMean": manufacturing.get("snr_mean"),
                "snrStd": manufacturing.get("snr_std"),
                "yieldConfidence": manufacturing.get("yield_confidence"),
                "reflectivityMean": manufacturing.get("reflectivity_mean"),
                "reflectivityStd": manufacturing.get("reflectivity_std")
            },
            "visualizations": {
                "eyeDiagramData": eye_diagram,
                "reflectivitySpectrum": reflectivity,
                "thermalProfile": thermal,
                "manufacturingProcess": manufacturing
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# ================================
# LEGACY ENDPOINTS (Preserved)
# ================================
@router.post("/predict_snr")
def predict_snr(data: OSISInput):
    try:
        metrics = predict_snr_ber(data.dict(), modulation="OOK-NRZ")
        return {
            "physics_snr_db": round(metrics["physics_snr_db"], 2),
            "ml_residual_db": round(metrics["ml_residual_db"], 2),
            "predicted_snr_db": round(metrics["predicted_snr_db"], 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_ber")
def predict_ber(data: BERInput):
    try:
        metrics = predict_snr_ber(data.dict(), modulation=data.modulation)
        return {
            "predicted_snr_db": round(metrics["predicted_snr_db"], 3),
            "estimated_ber": float(metrics["estimated_ber"]),
            "modulation": data.modulation.upper()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare_models")
def compare_models(data: ComparisonInput):
    try:
        metrics = predict_snr_ber(data.dict(), modulation=data.modulation)
        analytical_ber = estimate_ber_from_snr(metrics["physics_snr_db"], modulation=data.modulation)
        
        response = {
            "analytical_physics_snr_db": round(metrics["physics_snr_db"], 3),
            "ml_hybrid_snr_db": round(metrics["predicted_snr_db"], 3),
            "snr_gain_over_analytical_db": round(metrics["predicted_snr_db"] - metrics["physics_snr_db"], 3),
            "analytical_ber": float(analytical_ber),
            "ml_hybrid_ber": float(metrics["estimated_ber"]),
            "ber_reduction_ratio": float(analytical_ber / max(metrics["estimated_ber"], 1e-15)),
            "modulation": data.modulation.upper()
        }
        if data.measured_snr_db is not None:
            response["measured_snr_db"] = round(data.measured_snr_db, 3)
            response["abs_error_analytical_db"] = round(abs(data.measured_snr_db - metrics["physics_snr_db"]), 3)
            response["abs_error_ml_hybrid_db"] = round(abs(data.measured_snr_db - metrics["predicted_snr_db"]), 3)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize_parameters")
def optimize_parameters(data: OptimizationInput):
    # Simplified optimization loop for stability
    try:
        base = data.base_config.dict()
        top_k = data.top_k
        
        nas = np.linspace(max(0.4, base["numerical_aperture"] - 0.1), min(0.95, base["numerical_aperture"] + 0.1), 3)
        temps = [base["temperature_c"], base["temperature_c"] + 10]
        
        ranked = []
        for na in nas:
            for t in temps:
                cand = dict(base)
                cand["numerical_aperture"] = float(na)
                cand["temperature_c"] = float(t)
                metrics = predict_snr_ber(cand, modulation=data.modulation)
                
                objective = metrics["predicted_snr_db"] - (10 * math.log10(max(metrics["estimated_ber"], 1e-15)))
                ranked.append({
                    "objective_score": objective,
                    "predicted_snr_db": metrics["predicted_snr_db"],
                    "estimated_ber": metrics["estimated_ber"],
                    "numerical_aperture": float(na),
                    "temperature_c": float(t)
                })
                
        ranked.sort(key=lambda x: x["objective_score"], reverse=True)
        return {
            "optimization_goal": "maximize_snr_and_minimize_ber",
            "evaluated_candidates": len(ranked),
            "top_recommendations": ranked[:top_k]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sensitivity_analysis")
def sensitivity_analysis(data: SensitivityInput):
    # Pass through to legacy logic wrapper mapping
    try:
        return {"status": "Not completely ported to safe backend constraints. Use POST /api/v1/simulate_platform for production."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate_dashboard")
def simulate_dashboard(data: SimulationInput):
    try:
        return {"status": "Legacy dashboard. Use /api/v1/simulate_platform for production."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

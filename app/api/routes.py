# app/api/routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.utils.validation import (
    OSISInput,
    BERInput,
    ComparisonInput,
    OptimizationInput,
    SensitivityInput,
    SimulationInput,
    AdvancedSimInput,
    BatchSimulationRequest,
)
from app.ml.predictor import (
    predict_snr_ber,
    predict_snr_interval,
    explain_prediction,
    estimate_ber_from_snr,
)
from app.services.batch_simulation import run_batch_simulation
from app.physics.optics import (
    approximate_reflectivity_spectrum,
    estimate_spot_size_nm,
    estimate_crosstalk,
)
from app.physics.thermal import approximate_thermal_pulse
from app.physics.signal import synthesize_eye_diagram
from app.physics.manufacturing import simulate_manufacturing_process, evaluate_monte_carlo_yield
from app.utils.constants import (
    NA_MIN,
    NA_MAX,
    TRACK_PITCH_MIN,
    TRACK_PITCH_MAX,
    TEMP_MIN,
    TEMP_MAX,
    HUMIDITY_MIN,
    HUMIDITY_MAX,
    MAX_TOP_K,
)
import numpy as np
import math

router = APIRouter()


def _prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(config)
    wavelength = float(prepared.get("laser_wavelength_nm", 405.0))
    numerical_aperture = float(prepared.get("numerical_aperture", 0.85))
    track_pitch = float(prepared.get("track_pitch_nm", 320.0))

    spot_size = estimate_spot_size_nm(wavelength, numerical_aperture) if numerical_aperture > 0 else 0.0
    prepared["spot_size_nm"] = float(spot_size)
    prepared["isi_factor"] = float(spot_size / track_pitch) if track_pitch > 0 else 0.0
    prepared["crosstalk_factor"] = float(estimate_crosstalk(track_pitch, spot_size))
    return prepared


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


@router.post("/api/v1/batch_simulations")
async def batch_simulations(data: BatchSimulationRequest):
    try:
        return await run_batch_simulation(data.batchConfigs, default_modulation=data.modulation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch simulation error: {str(e)}")

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
        reflectivity = approximate_reflectivity_spectrum([layer.model_dump() for layer in data.stackConfig])
        thermal = approximate_thermal_pulse(data.thermalConfig.model_dump(), data.opticalConfig.model_dump())
        eye_diagram = synthesize_eye_diagram(predicted_snr_db)
        
        max_temp = max(thermal["centerTempK"])
        
        # 3. Manufacturing Process Simulation
        mfg_config_dict = data.manufacturingConfig.model_dump() if data.manufacturingConfig else {}
        is_manufacturing_mode = (getattr(data, "simulationMode", "fast") == "manufacturing")
        
        if is_manufacturing_mode:
            manufacturing = evaluate_monte_carlo_yield(mfg_config_dict, base_snr=predicted_snr_db, peak_temp=max_temp, opt_config=data.opticalConfig.model_dump(), samples=100)
        else:
            manufacturing = simulate_manufacturing_process(mfg_config_dict, base_snr=predicted_snr_db, peak_temp=max_temp, opt_config=data.opticalConfig.model_dump())

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
        payload = _prepare_config(data.model_dump())
        metrics = predict_snr_ber(payload, modulation="OOK-NRZ")
        interval = predict_snr_interval(payload)
        return {
            "physics_snr_db": round(metrics["physics_snr_db"], 2),
            "ml_residual_db": round(metrics["ml_residual_db"], 2),
            "predicted_snr_db": round(metrics["predicted_snr_db"], 2),
            "estimated_ber": float(metrics["estimated_ber"]),
            "snr_lower_bound_db": round(interval["snr_lower_bound_db"], 2),
            "snr_upper_bound_db": round(interval["snr_upper_bound_db"], 2),
            "shap_explanations": explain_prediction(payload, max_features=5),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_ber")
def predict_ber(data: BERInput):
    try:
        metrics = predict_snr_ber(_prepare_config(data.model_dump()), modulation=data.modulation)
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
        metrics = predict_snr_ber(_prepare_config(data.model_dump()), modulation=data.modulation)
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
    try:
        base = data.base_config.model_dump()
        top_k = min(int(data.top_k), MAX_TOP_K)

        nas = np.linspace(
            max(NA_MIN, base["numerical_aperture"] - 0.08),
            min(NA_MAX, base["numerical_aperture"] + 0.08),
            3,
        )
        track_pitches = np.linspace(
            max(TRACK_PITCH_MIN, base["track_pitch_nm"] * 0.9),
            min(TRACK_PITCH_MAX, base["track_pitch_nm"] * 1.1),
            3,
        )
        temps = np.linspace(
            max(TEMP_MIN, base["temperature_c"] - 8.0),
            min(TEMP_MAX, base["temperature_c"] + 8.0),
            2,
        )
        humidities = np.linspace(
            max(HUMIDITY_MIN, base["relative_humidity"] - 12.0),
            min(HUMIDITY_MAX, base["relative_humidity"] + 12.0),
            2,
        )

        ranked = []
        for na in nas:
            for pitch in track_pitches:
                for temp in temps:
                    for humidity in humidities:
                        candidate = dict(base)
                        candidate["numerical_aperture"] = float(na)
                        candidate["track_pitch_nm"] = float(pitch)
                        candidate["temperature_c"] = float(temp)
                        candidate["relative_humidity"] = float(humidity)
                        metrics = predict_snr_ber(_prepare_config(candidate), modulation=data.modulation)

                        objective = metrics["predicted_snr_db"] - (10 * math.log10(max(metrics["estimated_ber"], 1e-15)))
                        ranked.append(
                            {
                                "objective_score": float(objective),
                                "predicted_snr_db": float(metrics["predicted_snr_db"]),
                                "estimated_ber": float(metrics["estimated_ber"]),
                                "numerical_aperture": float(na),
                                "track_pitch_nm": float(pitch),
                                "temperature_c": float(temp),
                                "relative_humidity": float(humidity),
                            }
                        )

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
    try:
        base = _prepare_config(data.model_dump())
        base_metrics = predict_snr_ber(base, modulation=data.modulation)

        parameter_specs = {
            "numerical_aperture": (NA_MIN, NA_MAX, 0.01),
            "track_pitch_nm": (TRACK_PITCH_MIN, TRACK_PITCH_MAX, 5.0),
            "temperature_c": (TEMP_MIN, TEMP_MAX, 1.0),
            "relative_humidity": (HUMIDITY_MIN, HUMIDITY_MAX, 1.0),
            "laser_wavelength_nm": (300.0, 900.0, 2.0),
        }

        ranked = []
        for parameter, (minimum, maximum, min_delta) in parameter_specs.items():
            base_value = float(base.get(parameter, 0.0))
            delta = max(abs(base_value) * data.delta_fraction, min_delta)
            low_value = _clamp(base_value - delta, minimum, maximum)
            high_value = _clamp(base_value + delta, minimum, maximum)

            if math.isclose(low_value, high_value):
                continue

            low_metrics = predict_snr_ber(_prepare_config({**base, parameter: low_value}), modulation=data.modulation)
            high_metrics = predict_snr_ber(_prepare_config({**base, parameter: high_value}), modulation=data.modulation)
            sensitivity = abs(high_metrics["predicted_snr_db"] - low_metrics["predicted_snr_db"]) / max(high_value - low_value, 1e-9)

            ranked.append(
                {
                    "parameter": parameter,
                    "normalized_sensitivity": float(sensitivity),
                    "low_value": float(low_value),
                    "high_value": float(high_value),
                    "low_snr_db": float(low_metrics["predicted_snr_db"]),
                    "high_snr_db": float(high_metrics["predicted_snr_db"]),
                }
            )

        if ranked:
            max_sensitivity = max(item["normalized_sensitivity"] for item in ranked) or 1.0
            for item in ranked:
                item["normalized_sensitivity"] = round(item["normalized_sensitivity"] / max_sensitivity, 4)

        ranked.sort(key=lambda item: item["normalized_sensitivity"], reverse=True)

        return {
            "base_predicted_snr_db": round(float(base_metrics["predicted_snr_db"]), 3),
            "delta_fraction": float(data.delta_fraction),
            "ranked_sensitivity": ranked,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate_dashboard")
def simulate_dashboard(data: SimulationInput):
    try:
        values = np.linspace(float(data.start), float(data.end), int(data.steps))
        frames = []

        for value in values:
            candidate = data.base_config.model_dump()
            candidate[data.sweep_parameter] = float(value)
            metrics = predict_snr_ber(_prepare_config(candidate), modulation=data.modulation)
            frames.append(
                {
                    "value": round(float(value), 4),
                    "predicted_snr_db": round(float(metrics["predicted_snr_db"]), 4),
                    "estimated_ber": float(metrics["estimated_ber"]),
                }
            )

        return {
            "status": "success",
            "sweep_parameter": data.sweep_parameter,
            "frames": frames,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

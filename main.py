from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import math
from typing import Any, Dict, List, Optional

# Load trained models, Quantiles, and Explainer
model = joblib.load("osis_snr_model.pkl")
model_lower = joblib.load("osis_snr_model_lower.pkl")
model_upper = joblib.load("osis_snr_model_upper.pkl")
feature_columns = joblib.load("osis_features.pkl")
explainer = joblib.load("osis_explainer.pkl")

app = FastAPI(title="OSIS Hybrid SNR Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# -------------------------
# PHYSICS CONSTANTS
# -------------------------
K_BOLTZMANN = 8.617e-5

# Parameter ranges used in optimization sweeps
NA_MIN, NA_MAX = 0.40, 0.95
TRACK_PITCH_MIN, TRACK_PITCH_MAX = 180.0, 1800.0
TEMP_MIN, TEMP_MAX = 20.0, 80.0
HUMIDITY_MIN, HUMIDITY_MAX = 10.0, 90.0

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def calculate_physics_snr(wavelength, NA, isi, crosstalk, thermal_factor):
    """
    Deterministic Physics Baseline SNR (Same as training)
    """
    return (85 
            + 30 * NA 
            - 0.02 * wavelength 
            - 15 * isi 
            - 10 * crosstalk 
            + 5 * thermal_factor)


def estimate_spot_size_nm(wavelength_nm: float, numerical_aperture: float) -> float:
    return (0.61 * wavelength_nm) / numerical_aperture


def estimate_crosstalk(track_pitch_nm: float, spot_size_nm: float, alpha: float = 0.002) -> float:
    return math.exp(-alpha * (track_pitch_nm - spot_size_nm))


def standardize_physical_inputs(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recompute dependent optical terms so physics consistency is preserved.
    """
    updated = dict(input_dict)

    spot_size = estimate_spot_size_nm(
        updated['laser_wavelength_nm'],
        updated['numerical_aperture']
    )
    updated['spot_size_nm'] = spot_size
    updated['isi_factor'] = spot_size / updated['track_pitch_nm']
    updated['crosstalk_factor'] = estimate_crosstalk(updated['track_pitch_nm'], spot_size)

    return updated


def snr_db_to_linear(snr_db: float) -> float:
    return 10 ** (snr_db / 10)


def estimate_ber_from_snr(snr_db: float, modulation: str = "OOK-NRZ") -> float:
    """
    BER approximations for AWGN channels.
    """
    snr_linear = max(snr_db_to_linear(snr_db), 1e-12)
    mode = modulation.upper().strip()

    if mode in {"BPSK", "QPSK"}:
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))
    else:
        ber = 0.5 * math.erfc(math.sqrt(snr_linear / 2.0))

    return max(min(ber, 0.5), 1e-15)


def build_model_features(input_dict: Dict[str, Any]):
    input_dict = standardize_physical_inputs(input_dict)

    temp_k = input_dict['temperature_c'] + 273.15
    thermal_factor = math.exp(-input_dict['activation_energy_ev'] / (K_BOLTZMANN * temp_k))
    input_dict['thermal_factor'] = thermal_factor

    physics_snr = calculate_physics_snr(
        input_dict['laser_wavelength_nm'],
        input_dict['numerical_aperture'],
        input_dict['isi_factor'],
        input_dict['crosstalk_factor'],
        thermal_factor
    )
    input_dict['physics_snr_db'] = physics_snr

    input_dict['NA_sq'] = input_dict['numerical_aperture'] ** 2
    input_dict['wavelength_div_NA'] = input_dict['laser_wavelength_nm'] / input_dict['numerical_aperture']
    input_dict['spot_div_pitch'] = input_dict['spot_size_nm'] / input_dict['track_pitch_nm']
    input_dict['temp_x_humidity'] = input_dict['temperature_c'] * input_dict['relative_humidity']

    material = input_dict.pop("recording_material")
    input_dict['recording_material_GST_HTL'] = 1 if material == "GST_HTL" else 0
    input_dict['recording_material_MDISC'] = 1 if material == "MDISC" else 0

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=feature_columns, fill_value=0)

    return physics_snr, df


def predict_full_metrics(input_dict: Dict[str, Any], modulation: str = "OOK-NRZ") -> Dict[str, Any]:
    physics_snr, df = build_model_features(input_dict)
    
    # 1. Main Ensemble Prediction
    ml_residual = float(model.predict(df)[0])
    final_snr = float(physics_snr + ml_residual)
    
    # 2. Uncertainty Quantification (Quantile Regression)
    lower_res = float(model_lower.predict(df)[0])
    upper_res = float(model_upper.predict(df)[0])
    snr_lower = float(physics_snr + lower_res)
    snr_upper = float(physics_snr + upper_res)
    
    # 3. Explainable AI (SHAP)
    # The explainer returns shap_values -> base_values + values
    shap_vals = explainer(df)
    # Get the top 5 most impactful features for this specific prediction
    feature_importances = list(zip(feature_columns, shap_vals.values[0]))
    feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)
    top_explanations = [{"feature": f, "impact": float(val)} for f, val in feature_importances[:5]]

    ber = estimate_ber_from_snr(final_snr, modulation=modulation)

    return {
        "physics_snr_db": physics_snr,
        "ml_residual_db": ml_residual,
        "predicted_snr_db": final_snr,
        "snr_lower_bound_db": snr_lower,
        "snr_upper_bound_db": snr_upper,
        "estimated_ber": ber,
        "shap_explanations": top_explanations
    }

def predict_batch_metrics(candidates: List[Dict[str, Any]], modulation: str = "OOK-NRZ") -> List[Dict[str, float]]:
    if not candidates:
        return []

    # Standardize inputs
    standardized = [standardize_physical_inputs(c) for c in candidates]
    df = pd.DataFrame(standardized)

    temp_k = df['temperature_c'] + 273.15
    df['thermal_factor'] = np.exp(-df['activation_energy_ev'] / (K_BOLTZMANN * temp_k))

    df['physics_snr_db'] = (85 
            + 30 * df['numerical_aperture'] 
            - 0.02 * df['laser_wavelength_nm'] 
            - 15 * df['isi_factor'] 
            - 10 * df['crosstalk_factor'] 
            + 5 * df['thermal_factor'])

    df['NA_sq'] = df['numerical_aperture'] ** 2
    df['wavelength_div_NA'] = df['laser_wavelength_nm'] / df['numerical_aperture']
    df['spot_div_pitch'] = df['spot_size_nm'] / df['track_pitch_nm']
    df['temp_x_humidity'] = df['temperature_c'] * df['relative_humidity']

    materials = df['recording_material']
    df['recording_material_GST_HTL'] = (materials == "GST_HTL").astype(int)
    df['recording_material_MDISC'] = (materials == "MDISC").astype(int)

    X = df.reindex(columns=feature_columns, fill_value=0)

    ml_residuals = model.predict(X)
    final_snr = df['physics_snr_db'] + ml_residuals

    results = []
    for i in range(len(df)):
        snr = float(final_snr.iloc[i])
        results.append({
            "physics_snr_db": float(df['physics_snr_db'].iloc[i]),
            "ml_residual_db": float(ml_residuals[i]),
            "predicted_snr_db": snr,
            "estimated_ber": estimate_ber_from_snr(snr, modulation=modulation)
        })
    return results

# -------------------------
# INPUT SCHEMA
# -------------------------
class OSISInput(BaseModel):
    laser_wavelength_nm: int
    numerical_aperture: float
    spot_size_nm: float
    track_pitch_nm: float
    layer_count: int
    layer_spacing_nm: float
    isi_factor: float
    crosstalk_factor: float
    recording_material: str  # GST_HTL, DYE_LTH, MDISC
    thermal_conductivity_w_mk: float
    activation_energy_ev: float
    temperature_c: float
    relative_humidity: float
    prml_enabled: int
    ctc_enabled: int


class BERInput(OSISInput):
    modulation: str = "OOK-NRZ"


class ComparisonInput(OSISInput):
    modulation: str = "OOK-NRZ"
    measured_snr_db: Optional[float] = None


class OptimizationInput(BaseModel):
    base_config: OSISInput
    modulation: str = "OOK-NRZ"
    top_k: int = 5


class SensitivityInput(OSISInput):
    delta_fraction: float = 0.05
    modulation: str = "OOK-NRZ"


class SimulationInput(BaseModel):
    base_config: OSISInput
    sweep_parameter: str = "numerical_aperture"
    start: float
    end: float
    steps: int = 20
    modulation: str = "OOK-NRZ"

# -------------------------
# PREDICTION API
# -------------------------
@app.post("/predict_snr")
def predict_snr(data: OSISInput):
    metrics = predict_full_metrics(data.model_dump(), modulation="OOK-NRZ")
    return {
        "physics_snr_db": round(metrics["physics_snr_db"], 2),
        "ml_residual_db": round(metrics["ml_residual_db"], 2),
        "predicted_snr_db": round(metrics["predicted_snr_db"], 2)
    }


@app.post("/predict_ber")
def predict_ber(data: BERInput):
    metrics = predict_full_metrics(data.model_dump(), modulation=data.modulation)
    return {
        "predicted_snr_db": round(metrics["predicted_snr_db"], 3),
        "estimated_ber": float(metrics["estimated_ber"]),
        "modulation": data.modulation.upper()
    }


@app.post("/compare_models")
def compare_models(data: ComparisonInput):
    payload = data.model_dump()
    modulation = payload.pop("modulation")
    measured = payload.pop("measured_snr_db", None)

    metrics = predict_full_metrics(payload, modulation=modulation)
    analytical_ber = estimate_ber_from_snr(metrics["physics_snr_db"], modulation=modulation)

    response = {
        "analytical_physics_snr_db": round(metrics["physics_snr_db"], 3),
        "ml_hybrid_snr_db": round(metrics["predicted_snr_db"], 3),
        "snr_gain_over_analytical_db": round(metrics["predicted_snr_db"] - metrics["physics_snr_db"], 3),
        "analytical_ber": float(analytical_ber),
        "ml_hybrid_ber": float(metrics["estimated_ber"]),
        "ber_reduction_ratio": float(analytical_ber / max(metrics["estimated_ber"], 1e-15)),
        "modulation": modulation.upper()
    }

    if measured is not None:
        response["measured_snr_db"] = round(float(measured), 3)
        response["abs_error_analytical_db"] = round(abs(float(measured) - metrics["physics_snr_db"]), 3)
        response["abs_error_ml_hybrid_db"] = round(abs(float(measured) - metrics["predicted_snr_db"]), 3)

    return response


@app.post("/optimize_parameters")
def optimize_parameters(data: OptimizationInput):
    base = standardize_physical_inputs(data.base_config.model_dump())

    na_candidates = np.linspace(
        max(NA_MIN, base["numerical_aperture"] - 0.1),
        min(NA_MAX, base["numerical_aperture"] + 0.1),
        7
    )
    pitch_candidates = np.linspace(
        max(TRACK_PITCH_MIN, base["track_pitch_nm"] * 0.85),
        min(TRACK_PITCH_MAX, base["track_pitch_nm"] * 1.15),
        7
    )
    temp_candidates = np.linspace(
        max(TEMP_MIN, base["temperature_c"] - 15),
        min(TEMP_MAX, base["temperature_c"] + 15),
        5
    )
    humidity_candidates = np.linspace(
        max(HUMIDITY_MIN, base["relative_humidity"] - 30),
        min(HUMIDITY_MAX, base["relative_humidity"] + 30),
        5
    )
    material_candidates = ["GST_HTL", "DYE_LTH", "MDISC"]

    candidates = []

    for na in na_candidates:
        for pitch in pitch_candidates:
            for temp in temp_candidates:
                for humidity in humidity_candidates:
                    for material in material_candidates:
                        for prml in [0, 1]:
                            for ctc in [0, 1]:
                                candidate = dict(base)
                                candidate["numerical_aperture"] = float(na)
                                candidate["track_pitch_nm"] = float(pitch)
                                candidate["temperature_c"] = float(temp)
                                candidate["relative_humidity"] = float(humidity)
                                candidate["recording_material"] = material
                                candidate["prml_enabled"] = prml
                                candidate["ctc_enabled"] = ctc
                                candidates.append(candidate)

    batch_metrics = predict_batch_metrics(candidates, modulation=data.modulation)

    ranked: List[Dict[str, Any]] = []

    for candidate, metrics in zip(candidates, batch_metrics):
        objective = metrics["predicted_snr_db"] - (10 * math.log10(max(metrics["estimated_ber"], 1e-15)))

        ranked.append({
            "objective_score": objective,
            "predicted_snr_db": metrics["predicted_snr_db"],
            "estimated_ber": metrics["estimated_ber"],
            "numerical_aperture": candidate["numerical_aperture"],
            "track_pitch_nm": candidate["track_pitch_nm"],
            "temperature_c": candidate["temperature_c"],
            "relative_humidity": candidate["relative_humidity"],
            "recording_material": candidate["recording_material"],
            "prml_enabled": candidate["prml_enabled"],
            "ctc_enabled": candidate["ctc_enabled"]
        })

    ranked.sort(key=lambda x: x["objective_score"], reverse=True)
    top_k = max(1, min(data.top_k, 20))
    best = ranked[:top_k]

    return {
        "optimization_goal": "maximize_snr_and_minimize_ber",
        "modulation": data.modulation.upper(),
        "evaluated_candidates": len(ranked),
        "top_recommendations": best
    }


@app.post("/sensitivity_analysis")
def sensitivity_analysis(data: SensitivityInput):
    payload = data.model_dump()
    delta_fraction = min(max(payload.pop("delta_fraction", 0.05), 0.01), 0.2)
    modulation = payload.pop("modulation", "OOK-NRZ")

    baseline = predict_full_metrics(payload, modulation=modulation)
    baseline_snr = baseline["predicted_snr_db"]

    tunable = [
        "laser_wavelength_nm",
        "numerical_aperture",
        "track_pitch_nm",
        "layer_count",
        "layer_spacing_nm",
        "thermal_conductivity_w_mk",
        "activation_energy_ev",
        "temperature_c",
        "relative_humidity"
    ]

    candidates = []
    for param in tunable:
        current = float(payload[param])
        delta = max(abs(current) * delta_fraction, 1e-6)

        plus_case = dict(payload)
        minus_case = dict(payload)
        plus_case[param] = current + delta
        minus_case[param] = current - delta

        if param == "numerical_aperture":
            plus_case[param] = min(max(plus_case[param], NA_MIN), NA_MAX)
            minus_case[param] = min(max(minus_case[param], NA_MIN), NA_MAX)
        elif param == "track_pitch_nm":
            plus_case[param] = min(max(plus_case[param], TRACK_PITCH_MIN), TRACK_PITCH_MAX)
            minus_case[param] = min(max(minus_case[param], TRACK_PITCH_MIN), TRACK_PITCH_MAX)
        elif param == "temperature_c":
            plus_case[param] = min(max(plus_case[param], TEMP_MIN), TEMP_MAX)
            minus_case[param] = min(max(minus_case[param], TEMP_MIN), TEMP_MAX)
        elif param == "relative_humidity":
            plus_case[param] = min(max(plus_case[param], HUMIDITY_MIN), HUMIDITY_MAX)
            minus_case[param] = min(max(minus_case[param], HUMIDITY_MIN), HUMIDITY_MAX)
        elif param == "layer_count":
            plus_case[param] = max(1, int(round(plus_case[param])))
            minus_case[param] = max(1, int(round(minus_case[param])))

        candidates.append(plus_case)
        candidates.append(minus_case)

    batch_metrics = predict_batch_metrics(candidates, modulation=modulation)

    scores = []
    for i, param in enumerate(tunable):
        plus_case = candidates[2*i]
        minus_case = candidates[2*i + 1]
        snr_plus = batch_metrics[2*i]["predicted_snr_db"]
        snr_minus = batch_metrics[2*i + 1]["predicted_snr_db"]
        
        current = float(payload[param])
        gradient = (snr_plus - snr_minus) / max((plus_case[param] - minus_case[param]), 1e-9)
        normalized_score = abs(gradient * (current / max(abs(baseline_snr), 1e-6)))

        scores.append({
            "parameter": param,
            "local_gradient": gradient,
            "normalized_sensitivity": normalized_score
        })

    scores.sort(key=lambda x: x["normalized_sensitivity"], reverse=True)

    return {
        "baseline_snr_db": round(baseline_snr, 3),
        "baseline_ber": baseline["estimated_ber"],
        "delta_fraction_used": delta_fraction,
        "ranked_sensitivity": scores
    }


@app.post("/simulate_dashboard")
def simulate_dashboard(data: SimulationInput):
    base = data.base_config.model_dump()
    steps = max(5, min(int(data.steps), 200))
    sweep_parameter = data.sweep_parameter

    supported = {
        "numerical_aperture",
        "track_pitch_nm",
        "temperature_c",
        "relative_humidity",
        "laser_wavelength_nm"
    }
    if sweep_parameter not in supported:
        return {
            "error": f"Unsupported sweep_parameter. Supported: {sorted(list(supported))}"
        }

    values = np.linspace(float(data.start), float(data.end), steps)
    candidates = []
    for value in values:
        frame = dict(base)
        frame[sweep_parameter] = float(value)
        candidates.append(frame)

    batch_metrics = predict_batch_metrics(candidates, modulation=data.modulation)

    timeline = []
    for idx, (value, metrics) in enumerate(zip(values, batch_metrics)):
        timeline.append({
            "t": idx,
            "parameter": sweep_parameter,
            "value": float(value),
            "physics_snr_db": metrics["physics_snr_db"],
            "predicted_snr_db": metrics["predicted_snr_db"],
            "estimated_ber": metrics["estimated_ber"]
        })

    return {
        "simulation_mode": "real_time_dashboard_concept",
        "sweep_parameter": sweep_parameter,
        "frames": timeline
    }


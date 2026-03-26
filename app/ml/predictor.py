# app/ml/predictor.py
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from app.utils.constants import K_BOLTZMANN
from app.ml.model_loader import registry

def calculate_physics_snr(wavelength: float, NA: float, isi: float, crosstalk: float, thermal_factor: float) -> float:
    """Deterministic Physics Baseline SNR."""
    return (85 + 30 * NA - 0.02 * wavelength - 15 * isi - 10 * crosstalk + 5 * thermal_factor)

def snr_db_to_linear(snr_db: float) -> float:
    return 10 ** (snr_db / 10)

def estimate_ber_from_snr(snr_db: float, modulation: str = "OOK-NRZ") -> float:
    """BER approximations for AWGN channels."""
    snr_linear = max(snr_db_to_linear(snr_db), 1e-12)
    mode = modulation.upper().strip()

    if mode in {"BPSK", "QPSK"}:
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))
    else:
        ber = 0.5 * math.erfc(math.sqrt(snr_linear / 2.0))

    return max(min(ber, 0.5), 1e-15)

def safe_feature_pipeline(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Safely builds features and matches the exact trained column order."""
    if not registry.is_loaded:
        raise RuntimeError("ML Models not loaded. Call registry.load_models() during startup.")
        
    temp_k = input_dict.get('temperature_c', 25.0) + 273.15
    ae = input_dict.get('activation_energy_ev', 1.0)
    thermal_factor = math.exp(-ae / (K_BOLTZMANN * temp_k))
    input_dict['thermal_factor'] = thermal_factor

    input_dict['physics_snr_db'] = calculate_physics_snr(
        input_dict.get('laser_wavelength_nm', 405),
        input_dict.get('numerical_aperture', 0.85),
        input_dict.get('isi_factor', 0.0),
        input_dict.get('crosstalk_factor', 0.0),
        thermal_factor
    )

    na = input_dict.get('numerical_aperture', 0.85)
    input_dict['NA_sq'] = na ** 2
    input_dict['wavelength_div_NA'] = input_dict.get('laser_wavelength_nm', 405) / na if na > 0 else 0
    pitch = input_dict.get('track_pitch_nm', 320)
    input_dict['spot_div_pitch'] = input_dict.get('spot_size_nm', 290) / pitch if pitch > 0 else 0
    input_dict['temp_x_humidity'] = input_dict.get('temperature_c', 25) * input_dict.get('relative_humidity', 50)

    material = input_dict.pop("recording_material", "GST_HTL")
    input_dict['recording_material_GST_HTL'] = 1 if material == "GST_HTL" else 0
    input_dict['recording_material_MDISC'] = 1 if material == "MDISC" else 0
    
    input_dict['reflectivity'] = input_dict.get('reflectivity', 0.7)

    df = pd.DataFrame([input_dict])
    
    # 🔴 Safety Layer: strictly align with expected trained columns
    df = df.reindex(columns=registry.feature_columns, fill_value=0)
    return df, input_dict['physics_snr_db']

def predict_snr_ber(input_dict: Dict[str, Any], modulation: str = "OOK-NRZ") -> Dict[str, Any]:
    """Generates the main ML output using global models."""
    try:
        df, physics_snr = safe_feature_pipeline(input_dict)
        
        ml_residual = float(registry.model.predict(df)[0])
        final_snr = float(physics_snr + ml_residual)
        ber = estimate_ber_from_snr(final_snr, modulation)
        
        return {
            "physics_snr_db": physics_snr,
            "ml_residual_db": ml_residual,
            "predicted_snr_db": final_snr,
            "estimated_ber": ber
        }
    except Exception as e:
        raise ValueError(f"Feature pipeline or prediction failed: {str(e)}")

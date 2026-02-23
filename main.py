from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import math

# Load trained residual model and feature list
model = joblib.load("osis_snr_model.pkl")
feature_columns = joblib.load("osis_features.pkl")

app = FastAPI(title="OSIS Hybrid SNR Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# -------------------------
# PHYSICS CONSTANTS
# -------------------------
K_BOLTZMANN = 8.617e-5

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

# -------------------------
# PREDICTION API
# -------------------------
@app.post("/predict_snr")
def predict_snr(data: OSISInput):
    input_dict = data.dict()
    
    # 1. Calculate Derived Physics Terms
    # Thermal Factor
    temp_k = input_dict['temperature_c'] + 273.15
    thermal_factor = math.exp(-input_dict['activation_energy_ev'] / (K_BOLTZMANN * temp_k))
    input_dict['thermal_factor'] = thermal_factor
    
    # Physics Baseline SNR
    physics_snr = calculate_physics_snr(
        input_dict['laser_wavelength_nm'],
        input_dict['numerical_aperture'],
        input_dict['isi_factor'],
        input_dict['crosstalk_factor'],
        thermal_factor
    )
    input_dict['physics_snr_db'] = physics_snr
    
    # 2. Feature Engineering (Must match training exactly)
    input_dict['NA_sq'] = input_dict['numerical_aperture'] ** 2
    input_dict['wavelength_div_NA'] = input_dict['laser_wavelength_nm'] / input_dict['numerical_aperture']
    input_dict['spot_div_pitch'] = input_dict['spot_size_nm'] / input_dict['track_pitch_nm']
    input_dict['temp_x_humidity'] = input_dict['temperature_c'] * input_dict['relative_humidity']
    
    # One-hot encoding
    material = input_dict.pop("recording_material")
    input_dict['recording_material_GST_HTL'] = 1 if material == "GST_HTL" else 0
    input_dict['recording_material_MDISC'] = 1 if material == "MDISC" else 0
    # DYE_LTH is implicit 0
    
    # Create DataFrame and ensure column order
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    # 3. Predict Residual
    ml_residual = model.predict(df)[0]
    
    # 4. Final SNR
    final_snr = physics_snr + ml_residual
    
    return {
        "physics_snr_db": round(physics_snr, 2),
        "ml_residual_db": round(ml_residual, 2),
        "predicted_snr_db": round(final_snr, 2)
    }


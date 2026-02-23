import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import pandas as pd
import joblib
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

print("Current working directory:", os.getcwd())

# -------------------------
# PHYSICS CONSTANTS & HELPER
# -------------------------
K_BOLTZMANN = 8.617e-5

def calculate_physics_snr(wavelength, NA, isi, crosstalk, thermal_factor):
    """
    Deterministic Physics Baseline SNR (Same as training/main)
    """
    return (85 
            + 30 * NA 
            - 0.02 * wavelength 
            - 15 * isi 
            - 10 * crosstalk 
            + 5 * thermal_factor)

try:
    # Load model and features
    model = joblib.load("osis_snr_model.pkl")
    feature_columns = joblib.load("osis_features.pkl")
    print(f"Feature columns loaded: {len(feature_columns)}")

    # Load dataset
    df = pd.read_csv("osis_dataset.csv")

    # =====================================================
    # PREPROCESSING & FEATURE ENGINEERING
    # =====================================================
    
    # 1. Physics Terms (Ensure they exist or recalculate if needed)
    # The dataset should have 'physics_snr_db' from generation, but let's recalculate 
    # to be 100% sure we verify the INFERENCE logic, not just the file values.
    # Actually, main.py recalculates it. Let's do the same here.
    
    # Thermal Factor
    # Temp is in C, converting to K
    df['temp_k'] = df['temperature_c'] + 273.15
    df['thermal_factor'] = np.exp(-df['activation_energy_ev'] / (K_BOLTZMANN * df['temp_k']))
    
    # Physics SNR Baseline
    df['calc_physics_snr'] = (
        85 
        + 30 * df['numerical_aperture'] 
        - 0.02 * df['laser_wavelength_nm'] 
        - 15 * df['isi_factor'] 
        - 10 * df['crosstalk_factor'] 
        + 5 * df['thermal_factor']
    )
    
    # Note: df['physics_snr_db'] exists in CSV, comparing to verify consistency
    # diff = (df['calc_physics_snr'] - df['physics_snr_db']).abs().max()
    # print(f"Max difference between CSV physics_snr and calculated: {diff}")

    # 2. Interaction Features
    df['NA_sq'] = df['numerical_aperture'] ** 2
    df['wavelength_div_NA'] = df['laser_wavelength_nm'] / df['numerical_aperture']
    df['spot_div_pitch'] = df['spot_size_nm'] / df['track_pitch_nm']
    df['temp_x_humidity'] = df['temperature_c'] * df['relative_humidity']

    # 3. One-hot encoding
    df['recording_material_GST_HTL'] = (df['recording_material'] == 'GST_HTL').astype(int)
    df['recording_material_MDISC'] = (df['recording_material'] == 'MDISC').astype(int)
    
    # Ensure 'physics_snr_db' is used as feature if model expects it
    # The training script used 'physics_snr_db' from the CSV. 
    # We should use the calculated one for pure inference simulation, 
    # but strictly the feature name must match.
    # We'll map our calculated one to the feature name expected.
    df['physics_snr_db'] = df['calc_physics_snr']

    # Align columns to feature list
    X = df.reindex(columns=feature_columns, fill_value=0)
    
    # Target (Actual Measured SNR)
    y_true = df['measured_snr_db']

    # =====================================================
    # PREDICTION
    # =====================================================
    
    # Predict Residual
    y_pred_residual = model.predict(X)
    
    # Final Prediction = Physics Baseline + Residual
    y_pred_final = df['calc_physics_snr'] + y_pred_residual

    # =====================================================
    # METRICS
    # =====================================================
    
    mse = mean_squared_error(y_true, y_pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_final)
    r2 = r2_score(y_true, y_pred_final)

    print("-" * 30)
    print("Hybrid Model Evaluation Metrics:")
    print("-" * 30)
    print(f"R² Score: {r2:.5f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} dB")
    print(f"Mean Absolute Error (MAE): {mae:.4f} dB")
    print("-" * 30)
    
    if r2 > 0.99:
        print("✅ Status: EXCELLENT (Target > 0.99 met)")
    else:
        print("⚠️ Status: NEEDS IMPROVEMENT")

except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================================================
# CONFIGURATION
# =====================================================
DATA_FILE = "osis_dataset.csv"
MODEL_FILE = "osis_snr_model.pkl"
FEATURES_FILE = "osis_features.pkl"

def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    print("Engineering features...")
    
    # 1. Base Physics Features (Already in dataset, but ensuring we use them)
    # ['spot_size_nm', 'isi_factor', 'crosstalk_factor', 'thermal_factor']
    
    # 2. Interaction Features
    df['NA_sq'] = df['numerical_aperture'] ** 2
    df['wavelength_div_NA'] = df['laser_wavelength_nm'] / df['numerical_aperture']
    df['spot_div_pitch'] = df['spot_size_nm'] / df['track_pitch_nm']
    df['temp_x_humidity'] = df['temperature_c'] * df['relative_humidity']
    
    # 3. Encoding Material
    df['recording_material_GST_HTL'] = (df['recording_material'] == 'GST_HTL').astype(int)
    df['recording_material_MDISC'] = (df['recording_material'] == 'MDISC').astype(int)
    # DYE_LTH is implicit

    # Feature Vector
    features = [
        # Raw Inputs
        'laser_wavelength_nm', 'numerical_aperture', 'track_pitch_nm',
        'layer_count', 'layer_spacing_nm',
        'temperature_c', 'relative_humidity',
        'prml_enabled', 'ctc_enabled',
        'thermal_conductivity_w_mk', 'activation_energy_ev',
        
        # Physics Features
        'spot_size_nm', 'isi_factor', 'crosstalk_factor', 'thermal_factor',
        'physics_snr_db', # Important: The baseline is a feature!
        
        # Interaction Features
        'NA_sq', 'wavelength_div_NA', 'spot_div_pitch', 'temp_x_humidity',
        
        # Material encoding
        'recording_material_GST_HTL', 'recording_material_MDISC'
    ]
    
    X = df[features]
    
    # =====================================================
    # TARGET DEFINITION (RESIDUAL LEARNING)
    # =====================================================
    # We want to predict the *difference* between measured and physics baseline
    # residual = measured - physics
    y_residual = df['measured_snr_db'] - df['physics_snr_db']
    
    print(f"Training on {len(df)} samples using {len(features)} features.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_residual, test_size=0.2, random_state=42)

    # =====================================================
    # MODEL TRAINING
    # =====================================================
    print("Training Gradient Boosting Regressor (Residual Model)...")
    
    gb_model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    gb_model.fit(X_train, y_train)

    # =====================================================
    # EVALUATION
    # =====================================================
    print("Evaluating model...")
    
    # Predict residual
    y_pred_residual = gb_model.predict(X_test)
    
    # Reconstruct Final SNR for evaluation
    # Final = Physics Baseline + Predicted Residual
    physics_baseline_test = X_test['physics_snr_db']
    y_pred_final = physics_baseline_test + y_pred_residual
    
    # Actual Measured SNR
    # Measured = Physics Baseline + Actual Residual
    y_true_final = physics_baseline_test + y_test

    # Metrics
    r2 = r2_score(y_true_final, y_pred_final)
    mse = mean_squared_error(y_true_final, y_pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_final, y_pred_final)

    print("-" * 30)
    print("Model Evaluation Metrics (Hybrid Physics + ML):")
    print("-" * 30)
    print(f"R² Score: {r2:.5f}")
    print(f"RMSE: {rmse:.4f} dB")
    print(f"MAE:  {mae:.4f} dB")
    print("-" * 30)

    if r2 > 0.95:
        print("[SUCCESS] Target Accuracy (R² > 0.95) Achieved!")
    else:
        print("[WARNING] Target Accuracy Not Met.")

    # =====================================================
    # SAVE ARTIFACTS
    # =====================================================
    print(f"Saving model to {MODEL_FILE}...")
    joblib.dump(gb_model, MODEL_FILE)
    joblib.dump(features, FEATURES_FILE)
    print("Done.")

if __name__ == "__main__":
    train()

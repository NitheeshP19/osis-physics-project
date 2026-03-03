import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import pandas as pd
import numpy as np
import joblib
import optuna
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================================================
# CONFIGURATION
# =====================================================
DATA_FILE = "osis_dataset.csv"
MODEL_FILE = "osis_snr_model.pkl"
MODEL_LOWER_FILE = "osis_snr_model_lower.pkl"
MODEL_UPPER_FILE = "osis_snr_model_upper.pkl"
FEATURES_FILE = "osis_features.pkl"
EXPLAINER_FILE = "osis_explainer.pkl"
SHAP_BACKGROUND = "osis_shap_background.pkl"

def load_data():
    df = pd.read_csv(DATA_FILE)
    
    # 2. Interaction Features
    df['NA_sq'] = df['numerical_aperture'] ** 2
    df['wavelength_div_NA'] = df['laser_wavelength_nm'] / df['numerical_aperture']
    df['spot_div_pitch'] = df['spot_size_nm'] / df['track_pitch_nm']
    df['temp_x_humidity'] = df['temperature_c'] * df['relative_humidity']
    
    # 3. Encoding Material
    df['recording_material_GST_HTL'] = (df['recording_material'] == 'GST_HTL').astype(int)
    df['recording_material_MDISC'] = (df['recording_material'] == 'MDISC').astype(int)

    features = [
        # Raw Inputs
        'laser_wavelength_nm', 'numerical_aperture', 'track_pitch_nm',
        'layer_count', 'layer_spacing_nm',
        'temperature_c', 'relative_humidity',
        'prml_enabled', 'ctc_enabled',
        'thermal_conductivity_w_mk', 'activation_energy_ev',
        
        # Physics Features
        'spot_size_nm', 'isi_factor', 'crosstalk_factor', 'thermal_factor',
        'physics_snr_db', 
        
        # Interaction Features
        'NA_sq', 'wavelength_div_NA', 'spot_div_pitch', 'temp_x_humidity',
        
        # Material encoding
        'recording_material_GST_HTL', 'recording_material_MDISC'
    ]
    
    X = df[features]
    y_residual = df['measured_snr_db'] - df['physics_snr_db']
    
    return X, y_residual, features, df['physics_snr_db']

def train():
    print("Loading dataset...")
    X, y, features, pb_full = load_data()
    print(f"Dataset Size: {len(X)} | Features: {len(features)}")
    
    X_train, X_test, y_train, y_test, pb_train, pb_test = train_test_split(
        X, y, pb_full, test_size=0.2, random_state=42
    )

    print("\n--- PHASE 1: BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA) ---")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 60),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 4),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2)
        }
        
        model = GradientBoostingRegressor(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=2, scoring='r2').mean()
        return score

    study = optuna.create_study(direction="maximize")
    # ONLY 1 TRIAL FOR FAST DEMONSTRATION SPEED
    study.optimize(objective, n_trials=1) 
    best_params = study.best_params
    print("Optimization Complete. Best Params:", best_params)

    print("\n--- PHASE 2: TRAINING HETEROGENEOUS ENSEMBLE ---")
    gb_opt = GradientBoostingRegressor(**best_params, random_state=42)
    rf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
    
    ensemble = StackingRegressor(
        estimators=[('gb', gb_opt), ('rf', rf)],
        final_estimator=GradientBoostingRegressor(n_estimators=20, random_state=42)
    )
    ensemble.fit(X_train, y_train)

    print("\n--- PHASE 3: TRAINING UNCERTAINTY QUANTIFICATION (UQ) MODELS ---")
    uq_lower = GradientBoostingRegressor(**best_params, loss='quantile', alpha=0.05, random_state=42)
    uq_upper = GradientBoostingRegressor(**best_params, loss='quantile', alpha=0.95, random_state=42)
    uq_lower.fit(X_train, y_train)
    uq_upper.fit(X_train, y_train)
    print("Quantile Regressors Trained (5% and 95% Confidence Bounds).")

    print("\n--- PHASE 4: INTEGRATING SHAP / EXPLAINABLE AI (XAI) ---")
    gb_opt.fit(X_train, y_train)
    explainer = shap.TreeExplainer(gb_opt)

    print("\n--- EVALUATION METRICS ---")
    y_pred_residual = ensemble.predict(X_test)
    y_pred_final = pb_test + y_pred_residual
    y_true_final = pb_test + y_test

    r2 = r2_score(y_true_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    mae = mean_absolute_error(y_true_final, y_pred_final)

    print(f"Ensemble R² Score: {r2:.5f}")
    print(f"Ensemble RMSE:     {rmse:.4f} dB")
    print(f"Ensemble MAE:      {mae:.4f} dB")

    if r2 > 0.99:
        print("✅ SUCCESS: Advanced ML Model Exceeded Baseline Target!")

    print("\n--- SAVING ARTIFACTS ---")
    joblib.dump(ensemble, MODEL_FILE)
    joblib.dump(uq_lower, MODEL_LOWER_FILE)
    joblib.dump(uq_upper, MODEL_UPPER_FILE)
    joblib.dump(features, FEATURES_FILE)
    joblib.dump(explainer, EXPLAINER_FILE)
    
    # 10 records for fast load times
    bg_data = X_train.sample(10, random_state=42)
    joblib.dump(bg_data, SHAP_BACKGROUND)
    
    print("✅ All advanced components successfully exported.")

if __name__ == "__main__":
    train()

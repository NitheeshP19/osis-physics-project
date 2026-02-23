import numpy as np
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================
N_SAMPLES = 20000
OUTPUT_FILE = "osis_dataset.csv"
np.random.seed(42)

# =====================================================
# PHYSICS-BASED HELPER FUNCTIONS
# =====================================================

def spot_size_nm(wavelength_nm, NA):
    return (0.61 * wavelength_nm) / NA

def isi_factor(spot_nm, track_pitch_nm):
    return spot_nm / track_pitch_nm

def crosstalk_factor(track_pitch_nm, spot_nm, alpha=0.002):
    # Modified to match spec: exp(-alpha * (track_pitch - spot_size))
    # Using a reasonable alpha value if not specified
    return np.exp(-alpha * (track_pitch_nm - spot_nm))

def thermal_factor_calc(activation_energy, temperature_c, k_boltzmann=8.617e-5):
    temp_k = temperature_c + 273.15
    return np.exp(-activation_energy / (k_boltzmann * temp_k))

def calculate_physics_snr(wavelength, NA, isi, crosstalk, thermal_factor):
    """
    Deterministic Physics Baseline SNR
    Formula provided:
    physics_snr = 85 + 30*NA - 0.02*wavelength - 15*isi - 10*crosstalk + 5*thermal_factor
    """
    return (85 
            + 30 * NA 
            - 0.02 * wavelength 
            - 15 * isi 
            - 10 * crosstalk 
            + 5 * thermal_factor)

# =====================================================
# PARAMETER DEFINITIONS
# =====================================================

wavelengths = [405, 650, 780]
NA_range = {405: (0.80, 0.95), 650: (0.60, 0.70), 780: (0.40, 0.55)}
track_pitch_base = {405: 225, 650: 740, 780: 1600}
materials = ["GST_HTL", "DYE_LTH", "MDISC"]

# =====================================================
# DATA GENERATION
# =====================================================

data = []

for _ in range(N_SAMPLES):
    wl = np.random.choice(wavelengths)
    
    # Randomize NA within realistic range for the wavelength
    na_min, na_max = NA_range[wl]
    NA = np.random.uniform(na_min, na_max)
    
    # Track pitch variation
    base_pitch = track_pitch_base[wl]
    track_pitch = base_pitch * np.random.uniform(0.9, 1.1)
    
    spot = spot_size_nm(wl, NA)
    isi = isi_factor(spot, track_pitch)

    layers = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
    spacing = np.random.uniform(15000, 30000) if layers > 1 else 1e6
    
    # Updated crosstalk calculation based on spec
    # Note: Using alpha=0.002 as a scaling factor for the exponent to make it reasonable
    crosstalk = crosstalk_factor(track_pitch, spot)

    material = np.random.choice(materials, p=[0.5, 0.3, 0.2])

    if material == "GST_HTL":
        thermal_k = np.random.uniform(0.5, 1.5)
        activation = np.random.uniform(1.8, 2.2)
    elif material == "DYE_LTH":
        thermal_k = np.random.uniform(0.1, 0.4)
        activation = np.random.uniform(0.8, 1.2)
    else: # MDISC
        thermal_k = np.random.uniform(1.2, 2.0)
        activation = np.random.uniform(2.0, 2.5)

    temp = np.random.uniform(20, 80)
    humidity = np.random.uniform(10, 90)

    prml = np.random.choice([0, 1], p=[0.5, 0.5])
    ctc = np.random.choice([0, 1], p=[0.5, 0.5])

    # 1. Deterministic Physics Baseline
    thermal_f = thermal_factor_calc(activation, temp)
    physics_snr = calculate_physics_snr(wl, NA, isi, crosstalk, thermal_f)
    
    # Add small noise to physics_snr as requested ("smooth curves")
    physics_snr += np.random.normal(0, 0.2)

    # 2. Simulate "Measured" SNR with Non-linear Interactions
    # These are the "residuals" the ML model will learn
    
    # Interaction: High NA with small track pitch is worse than linear prediction
    density_penalty = 0
    if isi > 0.8:
        density_penalty = 5 * (isi - 0.8)**2
        
    # Interaction: Humidity affects dye more than others
    humidity_penalty = 0
    if material == "DYE_LTH":
        humidity_penalty = 0.05 * (humidity - 40) if humidity > 40 else 0
        
    # Interaction: Multi-layer penalty increases non-linearly
    layer_penalty = 0
    if layers > 1:
        layer_penalty = 2 * (layers - 1)**1.5
        
    # Electronic gains
    prml_gain = 2.5 if prml else 0
    ctc_gain = 1.5 if ctc else 0
    
    # Final Measured SNR
    # measured_snr = physics_snr - penalties + gains + noise
    measured_snr = (physics_snr 
                    - density_penalty 
                    - humidity_penalty 
                    - layer_penalty 
                    + prml_gain 
                    + ctc_gain 
                    + np.random.normal(0, 0.5)) # Small measurement noise

    # Ensure SNR doesn't go below physical floor
    measured_snr = max(measured_snr, 1.0)

    data.append([
        wl, NA, spot, track_pitch, layers, spacing,
        isi, crosstalk, material, thermal_k, activation,
        temp, humidity, prml, ctc, 
        physics_snr, measured_snr, thermal_f
    ])

# =====================================================
# SAVE CSV FILE
# =====================================================

columns = [
    "laser_wavelength_nm", "numerical_aperture", "spot_size_nm",
    "track_pitch_nm", "layer_count", "layer_spacing_nm",
    "isi_factor", "crosstalk_factor", "recording_material",
    "thermal_conductivity_w_mk", "activation_energy_ev",
    "temperature_c", "relative_humidity", "prml_enabled",
    "ctc_enabled", "physics_snr_db", "measured_snr_db", "thermal_factor"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ {OUTPUT_FILE} created successfully with {N_SAMPLES} samples.")
print("Columns:", columns)


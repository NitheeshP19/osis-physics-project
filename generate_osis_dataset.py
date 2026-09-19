"""
Dataset generator for OSIS (Optical Storage Intelligence Simulator).

Generates synthetic measurement datasets by evaluating the rigorous OSIS physics
engine (Transfer Matrix Method + Optical Transfer Function + optoelectronic noise)
across a broad parameter space of disc architectures, then applies physical
channel impairments (crosstalk, inter-symbol interference, environmental humidity,
multi-layer optical cross-coupling) and electronic signal processing gains
(PRML detection, Cross-Talk Cancellation CTC).

The resulting dataset provides honest ground truth for training machine-learning
surrogate models to predict physical residual corrections.
"""

from __future__ import annotations

import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to Python path so osis can be imported directly
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import osis
from osis.configs import DiscConfig, ThinFilmLayer

# =====================================================
# CONFIGURATION
# =====================================================
N_SAMPLES = 5000
OUTPUT_FILE = "osis_dataset.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Boltzmann constant in eV/K for thermal activation calculations
K_BOLTZMANN_EV = 8.617333262e-5

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def thermal_factor_calc(activation_energy_ev: float, temperature_c: float) -> float:
    """Arrhenius thermal stability factor exp(-Ea / (kB * T))."""
    temp_k = temperature_c + 273.15
    return float(np.exp(-activation_energy_ev / (K_BOLTZMANN_EV * temp_k)))


def crosstalk_factor(track_pitch_nm: float, spot_nm: float, alpha: float = 0.002) -> float:
    """Optical crosstalk fraction from adjacent tracks."""
    delta = track_pitch_nm - spot_nm
    return float(np.exp(-alpha * max(delta, 0.0)))


def build_disc_config(
    wl_nm: float,
    na: float,
    material: str,
    temp_c: float,
) -> DiscConfig:
    """Construct a rigorous DiscConfig for a given set of sampled parameters."""
    wl_m = wl_nm * 1e-9
    temp_k = temp_c + 273.15

    if wl_nm < 500:  # Blu-ray regime (~405 nm)
        base = osis.BluRayConfig(
            numerical_aperture=na,
            temperature_k=temp_k,
        )
    elif wl_nm < 700:  # DVD regime (~650 nm)
        base = osis.DVDConfig(
            numerical_aperture=na,
            temperature_k=temp_k,
        )
    else:  # CD regime (~780 nm)
        base = osis.CDConfig(
            numerical_aperture=na,
            temperature_k=temp_k,
        )

    # Adjust active layer properties according to recording material
    if material == "DYE_LTH":
        # Organic dye: lower baseline reflectance, higher thermal sensitivity
        land_stack = [
            ThinFilmLayer("Dye-unwritten", 80e-9, 2.2, 0.05, "Representative azo-dye"),
            ThinFilmLayer("Al", 70e-9, 2.8, 8.45, "Reflector"),
        ]
        mark_stack = [
            ThinFilmLayer("Dye-bleached", 80e-9, 1.8, 0.01, "Bleached dye mark"),
            ThinFilmLayer("Al", 70e-9, 2.8, 8.45, "Reflector"),
        ]
        return DiscConfig(
            name=f"Custom-Dye-{int(wl_nm)}nm",
            wavelength_m=wl_m,
            numerical_aperture=na,
            track_pitch_m=base.track_pitch_m,
            min_mark_m=base.min_mark_m,
            laser_power_w=base.laser_power_w,
            coupling_efficiency=base.coupling_efficiency,
            detector_bandwidth_hz=base.detector_bandwidth_hz,
            rin_per_hz=base.rin_per_hz,
            load_resistance_ohm=base.load_resistance_ohm,
            temperature_k=temp_k,
            quantum_efficiency=base.quantum_efficiency,
            n_incident=1.0,
            n_substrate=1.58 + 0j,
            land_stack=land_stack,
            mark_stack=mark_stack,
        )
    elif material == "MDISC":
        # Inorganic glassy carbon / rock-like inorganic stack with high endurance
        land_stack = [
            ThinFilmLayer("Dielectric-TaOx", 50e-9, 2.1, 0.0, "Oxide protective"),
            ThinFilmLayer("Active-Inorganic", 30e-9, 3.2, 1.8, "Inorganic absorbent"),
            ThinFilmLayer("Dielectric-TaOx", 30e-9, 2.1, 0.0, "Oxide protective"),
            ThinFilmLayer("Reflector-Ti", 60e-9, 2.5, 3.5, "Titanium reflector"),
        ]
        mark_stack = [
            ThinFilmLayer("Dielectric-TaOx", 50e-9, 2.1, 0.0, "Oxide protective"),
            ThinFilmLayer("Active-Inorganic-Pit", 30e-9, 2.2, 0.4, "Decomposed mark"),
            ThinFilmLayer("Dielectric-TaOx", 30e-9, 2.1, 0.0, "Oxide protective"),
            ThinFilmLayer("Reflector-Ti", 60e-9, 2.5, 3.5, "Titanium reflector"),
        ]
        return DiscConfig(
            name=f"Custom-MDISC-{int(wl_nm)}nm",
            wavelength_m=wl_m,
            numerical_aperture=na,
            track_pitch_m=base.track_pitch_m,
            min_mark_m=base.min_mark_m,
            laser_power_w=base.laser_power_w,
            coupling_efficiency=base.coupling_efficiency,
            detector_bandwidth_hz=base.detector_bandwidth_hz,
            rin_per_hz=base.rin_per_hz,
            load_resistance_ohm=base.load_resistance_ohm,
            temperature_k=temp_k,
            quantum_efficiency=base.quantum_efficiency,
            n_incident=1.0,
            n_substrate=1.58 + 0j,
            land_stack=land_stack,
            mark_stack=mark_stack,
        )
    else:
        # Standard Phase-Change GST_HTL
        return base


# =====================================================
# DATA GENERATION LOOP
# =====================================================

wavelengths = [405, 650, 780]
NA_range = {405: (0.75, 0.88), 650: (0.55, 0.65), 780: (0.42, 0.52)}
track_pitch_base = {405: 320, 650: 740, 780: 1600}
materials = ["GST_HTL", "DYE_LTH", "MDISC"]

print(f"Generating {N_SAMPLES} samples using OSIS Transfer Matrix & Optics engine...")

data = []

for idx in range(N_SAMPLES):
    wl = int(np.random.choice(wavelengths))
    na_min, na_max = NA_range[wl]
    na = float(np.random.uniform(na_min, na_max))

    base_pitch = track_pitch_base[wl]
    track_pitch = float(base_pitch * np.random.uniform(0.92, 1.08))

    # Spot radius (Rayleigh: 0.61 * lambda / NA)
    spot = (0.61 * wl) / na
    isi = spot / track_pitch

    layers = int(np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05]))
    spacing = float(np.random.uniform(15000, 30000) if layers > 1 else 1e6)

    crosstalk = crosstalk_factor(track_pitch, spot)
    material = str(np.random.choice(materials, p=[0.5, 0.3, 0.2]))

    if material == "GST_HTL":
        thermal_k = float(np.random.uniform(0.5, 1.5))
        activation = float(np.random.uniform(1.8, 2.2))
    elif material == "DYE_LTH":
        thermal_k = float(np.random.uniform(0.1, 0.4))
        activation = float(np.random.uniform(0.8, 1.2))
    else:  # MDISC
        thermal_k = float(np.random.uniform(1.2, 2.0))
        activation = float(np.random.uniform(2.0, 2.5))

    temp = float(np.random.uniform(20, 70))
    humidity = float(np.random.uniform(10, 90))

    prml = int(np.random.choice([0, 1], p=[0.5, 0.5]))
    ctc = int(np.random.choice([0, 1], p=[0.5, 0.5]))

    # 1. Rigorous Physics Evaluation via OSIS
    cfg = build_disc_config(wl, na, material, temp)
    sim_res = osis.simulate(cfg)
    physics_snr = float(sim_res["cnr_db"])

    # Arrhenius thermal factor
    thermal_f = thermal_factor_calc(activation, temp)

    # 2. Channel Impairments & Processing Gains (Learned Residuals)
    # Density penalty when spot exceeds track pitch significantly
    density_penalty = 0.0
    if isi > 0.8:
        density_penalty = 4.0 * (isi - 0.8) ** 2

    # Humidity penalty (primarily degrades organic dye layers over time)
    humidity_penalty = 0.0
    if material == "DYE_LTH" and humidity > 45:
        humidity_penalty = 0.04 * (humidity - 45)

    # Multi-layer optical crosstalk and spherical aberration penalty
    layer_penalty = 0.0
    if layers > 1:
        layer_penalty = 1.8 * ((layers - 1) ** 1.3)

    # Electronic channel detection gains
    prml_gain = 2.5 if prml else 0.0
    ctc_gain = 1.5 if ctc else 0.0

    # Small measurement noise (AWGN channel fluctuation)
    noise = float(np.random.normal(0, 0.12))

    measured_snr = (
        physics_snr
        - density_penalty
        - humidity_penalty
        - layer_penalty
        + prml_gain
        + ctc_gain
        + noise
    )
    measured_snr = max(measured_snr, 1.0)

    data.append([
        wl, na, spot, track_pitch, layers, spacing,
        isi, crosstalk, material, thermal_k, activation,
        temp, humidity, prml, ctc,
        physics_snr, measured_snr, thermal_f,
    ])

# =====================================================
# SAVE CSV
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

print(f"Dataset successfully created: {OUTPUT_FILE} ({len(df)} rows).")
print(f"Physics SNR Range: {df['physics_snr_db'].min():.2f} to {df['physics_snr_db'].max():.2f} dB")
print(f"Measured SNR Range: {df['measured_snr_db'].min():.2f} to {df['measured_snr_db'].max():.2f} dB")
print(f"Mean Residual: {(df['measured_snr_db'] - df['physics_snr_db']).mean():.2f} dB")

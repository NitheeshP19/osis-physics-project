"""Level 1 Validation: Analytical limiting cases for OSIS physics engine."""

import math
import numpy as np
import sys
from pathlib import Path

# Add src to sys.path
src_dir = str(Path(__file__).resolve().parents[2] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import osis
from osis.configs import ThinFilmLayer
from osis.physics.tmm import compute_stack_reflectance, compute_stack_rt

from osis.physics.optics import spot_radius_m, coherent_cutoff_freq, incoherent_cutoff_freq, mtf_incoherent
from osis.physics.channel import shot_noise_power, thermal_noise_power, readout_signal
from osis.physics.snr import cnr_db, ber_from_cnr, photon_energy_j



def validate_analytical():
    print("=" * 70)
    print("Level 1 Validation: Analytical Limiting Cases")
    print("=" * 70)

    # 1. Fresnel normal incidence reflection at single glass/air interface
    n1, n2 = 1.0, 1.50
    expected_r = ((n1 - n2) / (n1 + n2)) ** 2  # 0.04000
    calc_r = compute_stack_reflectance(500e-9, [], n_incident=n1, n_substrate=complex(n2, 0.0))
    diff_fresnel = abs(calc_r - expected_r)
    print(f"[1] Fresnel Single Interface Limit: expected {expected_r:.6f}, got {calc_r:.6f} (error: {diff_fresnel:.2e}) -> {'PASS' if diff_fresnel < 1e-12 else 'FAIL'}")
    assert diff_fresnel < 1e-12

    # 2. Energy conservation in lossless 3-layer dielectric stack: R + T = 1.0
    layers = [
        ThinFilmLayer("SiO2", 120e-9, 1.45, 0.0),
        ThinFilmLayer("TiO2", 65e-9, 2.30, 0.0),
        ThinFilmLayer("SiO2", 110e-9, 1.45, 0.0),
    ]
    r, t = compute_stack_rt(632.8e-9, layers, n_incident=1.0, n_substrate=complex(1.52, 0.0))
    sum_rt = r + t
    diff_energy = abs(sum_rt - 1.0)
    print(f"[2] Lossless Dielectric Energy Conservation (R+T=1): R={r:.4f}, T={t:.4f}, sum={sum_rt:.12f} -> {'PASS' if diff_energy < 1e-12 else 'FAIL'}")
    assert diff_energy < 1e-12

    # 3. Quarter-wave anti-reflection coating: n_ar = sqrt(n_sub)
    n_sub = 2.25
    n_ar = math.sqrt(n_sub)  # 1.50
    wl = 600e-9
    d_ar = wl / (4.0 * n_ar)  # 100 nm
    ar_layer = [ThinFilmLayer("AR", d_ar, n_ar, 0.0)]
    r_ar = compute_stack_reflectance(wl, ar_layer, n_incident=1.0, n_substrate=complex(n_sub, 0.0))
    print(f"[3] Quarter-Wave AR Zero Reflectance Limit: R = {r_ar:.8f} (expected <= 1e-6) -> {'PASS' if r_ar <= 1e-6 else 'FAIL'}")
    assert r_ar <= 1e-6

    # 4. Rayleigh spot size linear wavelength scaling: r(2*lambda) = 2*r(lambda)
    r_base = spot_radius_m(400e-9, 0.85)
    r_double = spot_radius_m(800e-9, 0.85)
    ratio = r_double / r_base
    print(f"[4] Spot Radius Rayleigh Wavelength Linearity: r(2*lambda)/r(lambda) = {ratio:.6f} (expected 2.000000) -> {'PASS' if abs(ratio - 2.0) < 1e-12 else 'FAIL'}")
    assert abs(ratio - 2.0) < 1e-12

    # 5. MTF boundary conditions: MTF(0) = 1.0, MTF(nu >= nu_c) = 0.0
    nu_c = incoherent_cutoff_freq(500e-9, 0.5)
    mtf_zero = float(mtf_incoherent(np.array([0.0]), 500e-9, 0.5)[0])
    mtf_cutoff = float(mtf_incoherent(np.array([nu_c * 1.05]), 500e-9, 0.5)[0])
    print(f"[5] MTF Boundary Limits: MTF(0)={mtf_zero:.4f}, MTF(1.05*nu_c)={mtf_cutoff:.4f} -> {'PASS' if mtf_zero == 1.0 and mtf_cutoff == 0.0 else 'FAIL'}")
    assert mtf_zero == 1.0 and mtf_cutoff == 0.0


    # 6. BER asymptotic limits: infinite SNR -> 0.0, zero SNR -> 0.5
    ber_low = ber_from_cnr(-100.0)
    ber_high = ber_from_cnr(50.0)
    print(f"[6] BER Detection Asymptotes: BER(-100 dB) = {ber_low:.4f} (expected 0.5), BER(50 dB) = {ber_high:.2e} (expected 0.0) -> {'PASS' if abs(ber_low - 0.5) < 1e-3 and ber_high == 0.0 else 'FAIL'}")
    assert abs(ber_low - 0.5) < 1e-3 and ber_high == 0.0

    print("All Level 1 Analytical Validations PASSED successfully.\n")


if __name__ == "__main__":
    validate_analytical()

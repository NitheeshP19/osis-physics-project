"""Level 2 Validation: Independent Numerical Validation against Alternative Computational Methods.

Compares OSIS algorithms against independent mathematical implementations:
1. Independent Heavens (1955) Recurrence Formula vs. OSIS Abelès Matrix TMM.
2. Numerical 2D Pupil Autocorrelation (via 2D FFT) vs. OSIS Analytical MTF.
"""

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
from osis.physics.tmm import compute_stack_reflectance

from osis.physics.optics import mtf_incoherent, incoherent_cutoff_freq


def independent_heavens_recurrence(wavelength_m: float, layers: list, n_incident: float, n_substrate: complex) -> float:
    """Independent implementation of Heavens (1955) recurrence relations for multilayer films.

    Does not use 2x2 matrix multiplication; uses iterative Fresnel recurrence from the substrate up:
        r_j = (r_{j, j+1} + r_{j+1} * exp(2i delta_j)) / (1 + r_{j, j+1} * r_{j+1} * exp(2i delta_j))
    """
    import cmath
    all_indices = [complex(n_incident, 0.0)] + [complex(l.n, l.k) for l in layers] + [complex(n_substrate.real, n_substrate.imag)]
    thicknesses = [0.0] + [l.thickness_m for l in layers]

    N = len(all_indices) - 1
    # Bottom interface Fresnel reflection
    r_current = (all_indices[-2] - all_indices[-1]) / (all_indices[-2] + all_indices[-1])

    for j in range(N - 2, -1, -1):
        r_ij = (all_indices[j] - all_indices[j + 1]) / (all_indices[j] + all_indices[j + 1])
        delta = (2.0 * math.pi / wavelength_m) * all_indices[j + 1] * thicknesses[j + 1]
        phase = cmath.exp(2.0j * delta)
        r_current = (r_ij + r_current * phase) / (1.0 + r_ij * r_current * phase)

    return float(abs(r_current) ** 2)



def independent_2d_pupil_autocorrelation(spatial_freq: float, wavelength_m: float, na: float, grid_size: int = 512) -> float:
    """Numerical 2D circular pupil autocorrelation via discrete 2D spatial grid integration."""
    nu_c = 2.0 * na / wavelength_m
    if spatial_freq >= nu_c:
        return 0.0
    if spatial_freq <= 0.0:
        return 1.0

    # Dimensionless frequency shift in pupil coordinates [-1, 1]
    # Pupil radius R corresponds to coherent cutoff nu_coh = NA / lambda
    shift_norm = spatial_freq / (2.0 * na / wavelength_m)  # shift normalized from 0 to 1

    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)

    # Pupil 1 centered at origin
    pupil1 = (X**2 + Y**2) <= 1.0
    # Pupil 2 shifted by 2 * shift_norm
    pupil2 = ((X - 2.0 * shift_norm)**2 + Y**2) <= 1.0

    intersection = np.sum(pupil1 & pupil2)
    single_pupil = np.sum(pupil1)

    return float(intersection / single_pupil)


def validate_numerical():
    print("=" * 70)
    print("Level 2 Validation: Independent Numerical Method Cross-Checks")
    print("=" * 70)

    # Test Case 1: Complex 4-layer optical recording stack (GST Phase Change disc)
    layers = [
        ThinFilmLayer("ZnS-SiO2", 130e-9, 2.15, 0.0),
        ThinFilmLayer("Ge2Sb2Te5", 20e-9, 4.35, 2.10),
        ThinFilmLayer("ZnS-SiO2", 40e-9, 2.15, 0.0),
        ThinFilmLayer("Al-Alloy", 100e-9, 1.65, 6.20),
    ]
    wavelength = 650e-9
    n_inc = 1.0
    n_sub = complex(1.58, 0.0)

    tmm_r = compute_stack_reflectance(wavelength, layers, n_incident=n_inc, n_substrate=n_sub)
    heavens_r = independent_heavens_recurrence(wavelength, layers, n_incident=n_inc, n_substrate=n_sub)
    diff_tmm = abs(tmm_r - heavens_r)

    print(f"[1] Abeles TMM vs. Heavens Recurrence (4-Layer Complex Absorbing Stack):")
    print(f"    OSIS TMM R:             {tmm_r:.10f}")
    print(f"    Independent Heavens R:  {heavens_r:.10f}")
    print(f"    Absolute Difference:    {diff_tmm:.2e} -> {'PASS' if diff_tmm < 1e-10 else 'FAIL'}")
    assert diff_tmm < 1e-10


    # Test Case 2: Analytical MTF vs. 2D Numerical Pupil Overlap
    frequencies = [0.2e6, 0.5e6, 1.0e6, 1.5e6, 2.0e6]
    wl = 405e-9
    na = 0.85
    nu_c = 2.0 * na / wl

    print(f"\n[2] OSIS Analytical MTF vs. Independent 2D Numerical Pupil Autocorrelation:")
    print(f"    Cutoff Spatial Frequency: {nu_c * 1e-6:.2f} cycles/um")
    max_mtf_diff = 0.0

    for nu in frequencies:
        mtf_analytical = float(mtf_incoherent(np.array([nu]), wl, na)[0])
        mtf_numerical = independent_2d_pupil_autocorrelation(nu, wl, na, grid_size=1024)
        diff_mtf = abs(mtf_analytical - mtf_numerical)
        max_mtf_diff = max(max_mtf_diff, diff_mtf)
        print(f"    nu = {nu*1e-6:>4.2f} cyc/um | Analytic: {mtf_analytical:.5f} | Numerical: {mtf_numerical:.5f} | Diff: {diff_mtf:.4e}")
        assert diff_mtf < 5e-3  # Discrete pixelation tolerance

    print(f"    Max MTF Discretization Error: {max_mtf_diff:.2e} -> {'PASS' if max_mtf_diff < 5e-3 else 'FAIL'}")
    print("\nAll Level 2 Numerical Cross-Checks PASSED successfully.\n")


if __name__ == "__main__":
    validate_numerical()

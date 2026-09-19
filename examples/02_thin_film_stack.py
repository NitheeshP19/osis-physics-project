"""
Example 02: Custom Multilayer Thin-Film Stack Design using Abelès TMM.

This example demonstrates how to:
1. Define custom thin-film layers with complex refractive indices (n + ik).
2. Model phase-change optical contrast between crystalline (unwritten) and amorphous (written) marks.
3. Compute and display reflection spectra across the visible/near-IR spectrum.
"""

from __future__ import annotations

import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pathlib import Path
import numpy as np

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from osis.configs import ThinFilmLayer
from osis.physics.tmm import compute_stack_reflectance, compute_stack_rt


def main() -> None:
    print("=" * 75)
    print("Example 02: Multilayer Phase-Change Optical Stack Analysis (Abeles TMM)")
    print("=" * 75)

    # Define a 4-layer optical recording stack:
    # Air | ZnS-SiO2 (upper dielectric) | Ge2Sb2Te5 (active phase-change) | ZnS-SiO2 (lower) | Ag (reflector) | Substrate

    upper_dielectric = ThinFilmLayer("ZnS-SiO2", thickness_m=60e-9, n=2.15, k=0.0)
    lower_dielectric = ThinFilmLayer("ZnS-SiO2", thickness_m=20e-9, n=2.15, k=0.0)
    reflector        = ThinFilmLayer("Ag",      thickness_m=100e-9, n=0.065, k=1.88)

    # Crystalline phase (unwritten/land) vs Amorphous phase (written/mark) at 405 nm
    gst_crystalline = ThinFilmLayer("GST-crystalline", thickness_m=12e-9, n=3.60, k=3.70)
    gst_amorphous   = ThinFilmLayer("GST-amorphous",   thickness_m=12e-9, n=4.50, k=1.50)

    land_stack = [upper_dielectric, gst_crystalline, lower_dielectric, reflector]
    mark_stack = [upper_dielectric, gst_amorphous,   lower_dielectric, reflector]

    design_wl = 405e-9

    r_land, t_land = compute_stack_rt(design_wl, land_stack, n_incident=1.0, n_substrate=1.58 + 0j)
    r_mark, t_mark = compute_stack_rt(design_wl, mark_stack, n_incident=1.0, n_substrate=1.58 + 0j)

    print(f"\nEvaluation at design wavelength: {design_wl * 1e9:.1f} nm")
    print(f"  Land State (Crystalline):")
    print(f"    Reflectance R: {r_land * 100:.2f} %")
    print(f"    Transmittance T: {t_land * 100:.2f} %")
    print(f"    Absorption A:  {(1.0 - r_land - t_land) * 100:.2f} %")

    print(f"\n  Mark State (Amorphous):")
    print(f"    Reflectance R: {r_mark * 100:.2f} %")
    print(f"    Transmittance T: {t_mark * 100:.2f} %")
    print(f"    Absorption A:  {(1.0 - r_mark - t_mark) * 100:.2f} %")

    delta_r = abs(r_land - r_mark)
    contrast_ratio = max(r_land, r_mark) / max(min(r_land, r_mark), 1e-6)
    print(f"\n  Readout Contrast Metrics:")
    print(f"    Absolute Contrast (|Delta R|): {delta_r * 100:.2f} %")
    print(f"    Modulation Contrast:          {(delta_r / max(r_land, r_mark)) * 100:.2f} %")
    print(f"    Contrast Ratio:               {contrast_ratio:.2f} : 1")

    # Thickness tolerance sweep on upper dielectric layer
    print("\n--- Upper Dielectric Thickness Sensitivity ---")
    print(f"{'Thickness (nm)':<16} | {'R_land (%)':<12} | {'R_mark (%)':<12} | {'Contrast |Delta R| (%)':<16}")
    print("-" * 62)
    for d_nm in np.linspace(40, 80, 9):
        test_dielectric = ThinFilmLayer("ZnS-SiO2", thickness_m=d_nm * 1e-9, n=2.15, k=0.0)
        test_land = [test_dielectric, gst_crystalline, lower_dielectric, reflector]
        test_mark = [test_dielectric, gst_amorphous, lower_dielectric, reflector]

        rl = compute_stack_reflectance(design_wl, test_land)
        rm = compute_stack_reflectance(design_wl, test_mark)
        dr = abs(rl - rm)
        print(f"{d_nm:<16.1f} | {rl * 100:>10.2f} % | {rm * 100:>10.2f} % | {dr * 100:>14.2f} %")


if __name__ == "__main__":
    main()

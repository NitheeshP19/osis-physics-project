"""
Example 01: Comparing Standard Optical Storage Formats (CD, DVD, Blu-ray).

This example demonstrates how to:
1. Load standard pre-calibrated disc presets.
2. Run end-to-end physics simulations.
3. Inspect and compare key optical characteristics (spot size, reflectance, MTF, CNR).
"""

from __future__ import annotations

import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pathlib import Path

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import osis


def main() -> None:
    print("=" * 70)
    print("Example 01: Standard Optical Storage Format Comparison")
    print("=" * 70)

    presets = [
        ("Compact Disc (CD-RW)", osis.CDConfig()),
        ("Digital Versatile Disc (DVD-RW)", osis.DVDConfig()),
        ("Blu-ray Disc (BD-RE)", osis.BluRayConfig()),
    ]

    for title, config in presets:
        print(f"\n--- {title} ---")
        result = osis.simulate(config)

        print(f"  Laser Wavelength:      {config.wavelength_m * 1e9:.1f} nm")
        print(f"  Numerical Aperture:    {config.numerical_aperture:.2f}")
        print(f"  Rayleigh Spot Radius:  {result['spot_radius_m'] * 1e6:.3f} um ({result['spot_radius_m']*1e9:.1f} nm)")
        print(f"  Minimum Mark Length:   {config.min_mark_m * 1e6:.3f} um")
        print(f"  Land Reflectance:      {result['r_land'] * 100:.2f} %")
        print(f"  Mark Reflectance:      {result['r_mark'] * 100:.2f} %")
        print(f"  Optical Contrast (|Delta R|): {abs(result['r_land'] - result['r_mark']) * 100:.2f} %")
        print(f"  MTF Factor at T_min:   {result['mtf']:.3f}")
        print(f"  Signal Power:          {result['signal_power_w'] * 1e3:.4f} mW")
        print(f"  Total Noise Power:     {result['noise_power_w'] * 1e9:.4f} nW")
        print(f"  Carrier-to-Noise Ratio:{result['cnr_db']:.2f} dB")
        print(f"  Estimated BER:         {result['ber']:.2e}")


if __name__ == "__main__":
    main()

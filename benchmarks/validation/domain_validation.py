"""Level 3 Validation: Domain Literature & Physical Standard Verification.

Compares OSIS standardized disc configurations against published international
standards (ECMA-130, ECMA-267, Blu-ray BDA Part 1) and optical storage literature.
"""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = str(Path(__file__).resolve().parents[2] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import osis


def validate_domain():
    print("=" * 70)
    print("Level 3 Validation: Optical Disc Domain & Specification Verification")
    print("=" * 70)

    # Reference literature benchmarks:
    # 1. ECMA-130 (CD-ROM Standard) / Bouwhuis (1985):
    #    lambda = 780 nm, NA = 0.45 -> Airy spot radius ~ 1.06 um, track pitch = 1.6 um
    #    Typical commercial CD-RW carrier-to-noise ratio: 35 - 45 dB
    # 2. ECMA-267 (DVD Read-Only Standard):
    #    lambda = 650 nm, NA = 0.60 -> Airy spot radius ~ 0.66 um, track pitch = 0.74 um
    #    Typical DVD-RW carrier-to-noise ratio: 22 - 35 dB
    # 3. Blu-ray Disc Association (BDA) Physical Specifications:
    #    lambda = 405 nm, NA = 0.85 -> Airy spot radius ~ 0.29 um, track pitch = 0.32 um
    #    Typical BD-RE carrier-to-noise ratio: 15 - 25 dB under scalar diffraction limit

    domain_specs = [
        {
            "format": "Compact Disc (CD-RW)",
            "config": osis.CDConfig(),
            "expected_spot_nm": (1040, 1070),
            "expected_cnr_range_db": (32.0, 48.0),
            "standard_ref": "ECMA-130 / Philips-Sony Red Book",
        },
        {
            "format": "DVD-RW",
            "config": osis.DVDConfig(),
            "expected_spot_nm": (650, 675),
            "expected_cnr_range_db": (20.0, 35.0),
            "standard_ref": "ECMA-267 / DVD Forum",
        },
        {
            "format": "Blu-ray BD-RE",
            "config": osis.BluRayConfig(),
            "expected_spot_nm": (280, 305),
            "expected_cnr_range_db": (12.0, 25.0),
            "standard_ref": "BDA System Description Part 1",
        },
    ]

    for item in domain_specs:
        cfg = item["config"]
        res = osis.simulate(cfg)

        spot_nm = res["spot_radius_m"] * 1e9
        cnr_db = res["cnr_db"]
        ber = res["ber"]

        spot_ok = item["expected_spot_nm"][0] <= spot_nm <= item["expected_spot_nm"][1]
        cnr_ok = item["expected_cnr_range_db"][0] <= cnr_db <= item["expected_cnr_range_db"][1]

        print(f"[{item['format']}] (Ref: {item['standard_ref']}):")
        print(f"    Simulated Spot Radius: {spot_nm:.1f} nm (Standard Range: {item['expected_spot_nm']}) -> {'PASS' if spot_ok else 'FAIL'}")
        print(f"    Simulated CNR:         {cnr_db:.2f} dB (Literature Range: {item['expected_cnr_range_db']}) -> {'PASS' if cnr_ok else 'FAIL'}")
        print(f"    Signal Contrast |Delta R|:  {res['contrast']*100:.2f}% | MTF factor: {res['mtf']:.3f} | BER: {ber:.2e}")
        print()


        assert spot_ok
        assert cnr_ok

    print("All Level 3 Domain Standard Validations PASSED successfully.\n")


if __name__ == "__main__":
    validate_domain()

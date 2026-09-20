"""Command-line interface for the Optical Storage Intelligence Simulator (OSIS).

Commands:
    osis simulate [--preset {cd,dvd,bluray}]
    osis validate
    osis reproduce
"""

import argparse
import sys
from pathlib import Path

src_dir = str(Path(__file__).resolve().parents[1])
root_dir = str(Path(__file__).resolve().parents[2])
for p in [src_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


import osis
from osis.configs import CDConfig, DVDConfig, BluRayConfig



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="osis",
        description="OSIS: Optical Storage Intelligence Simulator CLI",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"OSIS {osis.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: simulate
    sim_parser = subparsers.add_parser("simulate", help="Run simulation for a preset format")
    sim_parser.add_argument(
        "--preset",
        choices=["cd", "dvd", "bluray"],
        default="bluray",
        help="Disc format preset (default: bluray)",
    )

    # Command: validate
    subparsers.add_parser("validate", help="Run the 3-level physical validation suite")

    # Command: reproduce
    subparsers.add_parser("reproduce", help="Reproduce the full research benchmark study")

    args = parser.parse_args(argv)

    if args.command == "simulate":
        cfg_map = {
            "cd": CDConfig(),
            "dvd": DVDConfig(),
            "bluray": BluRayConfig(),
        }
        cfg = cfg_map[args.preset]
        res = osis.simulate(cfg)
        print("=" * 60)
        print(f"OSIS Simulation: {cfg.name}")
        print("=" * 60)
        print(f"  Wavelength:         {cfg.wavelength_m * 1e9:.1f} nm")
        print(f"  Numerical Aperture: {cfg.numerical_aperture:.2f}")
        print(f"  Spot Radius:        {res['spot_radius_m'] * 1e9:.1f} nm")
        print(f"  Land Reflectance:   {res['r_land'] * 100:.2f} %")
        print(f"  Mark Reflectance:   {res['r_mark'] * 100:.2f} %")
        print(f"  Optical Contrast:   {res['contrast'] * 100:.2f} %")
        print(f"  MTF Factor (T_min): {res['mtf']:.3f}")
        print(f"  CNR:                {res['cnr_db']:.2f} dB")
        print(f"  Estimated BER:      {res['ber']:.2e}")
        print("=" * 60)
        return 0

    elif args.command == "validate":
        from benchmarks.validation.run_all_validations import run_all
        run_all()
        return 0

    elif args.command == "reproduce":
        from osis.research import run_full_study
        run_full_study()
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

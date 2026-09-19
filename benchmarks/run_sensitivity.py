"""
Benchmark analyzing parameter sensitivity across optical and detector variables.

Usage:
    python benchmarks/run_sensitivity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import osis


def benchmark_sensitivity() -> None:
    print("=" * 80)
    print("OSIS Parameter Sensitivity Benchmark (Blu-ray BD-RE Baseline)")
    print("=" * 80)

    cfg = osis.BluRayConfig()
    parameters = [
        "numerical_aperture",
        "laser_power_w",
        "detector_bandwidth_hz",
        "rin_per_hz",
        "load_resistance_ohm",
        "temperature_k",
        "coupling_efficiency",
        "quantum_efficiency",
    ]

    t0 = time.perf_counter()
    sens = osis.one_at_a_time_sensitivity(cfg, parameters=parameters, delta_fraction=0.05)
    dt = time.perf_counter() - t0

    ranked = osis.rank_parameters_by_influence(sens, metric="delta_output")

    print(f"Evaluated {len(parameters)} parameters in {dt * 1e3:.2f} ms.\n")
    header = f"{'Rank':<4} | {'Parameter':<24} | {'Base Value':<14} | {'Gradient (dB/unit)':<20} | {'Elasticity':<12} | {'CNR Swing (dB)':<14}"
    print(header)
    print("-" * len(header))

    for rank, (param, swing) in enumerate(ranked, start=1):
        m = sens[param]
        print(
            f"{rank:<4} | {param:<24} | {m['baseline_value']:<14.4e} | "
            f"{m['gradient']:<20.4e} | {m['elasticity']:>+10.4f}  | {swing:>11.3f} dB"
        )

    print("-" * len(header))
    print("Note: Elasticity = (p0 / y0) * (dy / dp), representing % change in CNR per 1% parameter shift.\n")


if __name__ == "__main__":
    benchmark_sensitivity()

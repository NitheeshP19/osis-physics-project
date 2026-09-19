"""
Benchmark comparing optical characteristics and simulation speed across CD, DVD, and Blu-ray.

Usage:
    python benchmarks/run_disc_comparison.py
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


def benchmark_disc_comparison(iterations: int = 1000) -> None:
    presets = [
        ("CD-RW", osis.CDConfig()),
        ("DVD-RW", osis.DVDConfig()),
        ("Blu-ray BD-RE", osis.BluRayConfig()),
    ]

    print("=" * 80)
    print("OSIS Standard Disc Comparison & Throughput Benchmark")
    print("=" * 80)
    print(f"Iterations per preset: {iterations}\n")

    header = f"{'Standard':<16} | {'Wavelength':<10} | {'NA':<5} | {'Spot (nm)':<10} | {'MTF':<6} | {'CNR (dB)':<8} | {'Latency':<10}"
    print(header)
    print("-" * len(header))

    for name, cfg in presets:
        # Warmup
        _ = osis.simulate(cfg)

        t0 = time.perf_counter()
        for _ in range(iterations):
            res = osis.simulate(cfg)
        dt = time.perf_counter() - t0

        avg_latency_us = (dt / iterations) * 1e6
        wl_nm = cfg.wavelength_m * 1e9
        spot_nm = res["spot_radius_m"] * 1e9

        print(
            f"{name:<16} | {wl_nm:>7.1f} nm | {cfg.numerical_aperture:>5.2f} | "
            f"{spot_nm:>8.1f} nm | {res['mtf']:>6.3f} | {res['cnr_db']:>7.2f}  | "
            f"{avg_latency_us:>7.1f} µs"
        )

    print("-" * len(header))
    print("Benchmark complete. All simulations executed through full TMM + MTF + Noise pipeline.\n")


if __name__ == "__main__":
    benchmark_disc_comparison()

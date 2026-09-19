"""Integration tests for end-to-end OSIS simulation pipelines."""

from __future__ import annotations

import math
import pytest
import osis
from osis.configs import CDConfig, DVDConfig, BluRayConfig


class TestSimulationPipeline:
    """End-to-end integration tests for the OSIS simulation pipeline."""

    def test_standard_disc_presets_e2e(self) -> None:
        """Verify full simulation runs across all standardized disc presets."""
        presets = [
            ("CD", CDConfig(), (30.0, 50.0), (0.9e-6, 1.2e-6)),
            ("DVD", DVDConfig(), (20.0, 40.0), (0.5e-6, 0.8e-6)),
            ("Blu-ray", BluRayConfig(), (10.0, 30.0), (0.2e-6, 0.4e-6)),
        ]

        expected_keys = {
            "cnr_db",
            "ber",
            "spot_radius_m",
            "mtf",
            "r_land",
            "r_mark",
            "signal_power_w",
            "noise_power_w",
            "config",
        }

        spot_sizes = []

        for name, cfg, (cnr_min, cnr_max), (spot_min, spot_max) in presets:
            res = osis.simulate(cfg)

            # 1. Check all keys present
            assert expected_keys.issubset(res.keys())

            # 2. Check value boundaries
            assert cnr_min <= res["cnr_db"] <= cnr_max, (
                f"{name} CNR {res['cnr_db']:.2f} dB outside expected range [{cnr_min}, {cnr_max}]"
            )
            assert spot_min <= res["spot_radius_m"] <= spot_max
            assert 0.0 <= res["ber"] <= 0.5
            assert 0.0 < res["r_land"] < 1.0
            assert 0.0 < res["r_mark"] < 1.0
            assert 0.0 < res["mtf"] <= 1.0
            assert res["signal_power_w"] > 0
            assert res["noise_power_w"] > 0

            spot_sizes.append(res["spot_radius_m"])

        # 3. Check physical progression: CD spot > DVD spot > Blu-ray spot
        assert spot_sizes[0] > spot_sizes[1] > spot_sizes[2]

    def test_mark_length_mtf_impact(self) -> None:
        """Verify longer marks produce higher MTF and stronger readout CNR."""
        cfg = BluRayConfig()

        # Short mark (T3 ≈ 0.149 um) vs long mark (T8 ≈ 0.400 um)
        res_t3 = osis.simulate(cfg, mark_length_m=0.149e-6)
        res_t8 = osis.simulate(cfg, mark_length_m=0.400e-6)

        assert res_t8["mtf"] > res_t3["mtf"]
        assert res_t8["cnr_db"] > res_t3["cnr_db"]

    def test_deterministic_reproducibility(self) -> None:
        """Verify that identical configurations produce bitwise-identical results."""
        cfg = BluRayConfig()
        res1 = osis.simulate(cfg)
        res2 = osis.simulate(cfg)

        assert res1["cnr_db"] == res2["cnr_db"]
        assert res1["ber"] == res2["ber"]
        assert res1["signal_power_w"] == res2["signal_power_w"]
        assert res1["noise_power_w"] == res2["noise_power_w"]

"""Unit tests for parameter sweep and sensitivity analysis modules."""

from __future__ import annotations

import numpy as np
import pytest

from osis.configs import BluRayConfig, CDConfig
from osis.analysis.sweep import parameter_sweep
from osis.analysis.sensitivity import (
    one_at_a_time_sensitivity,
    rank_parameters_by_influence,
)


class TestAnalysisModules:
    """Test suite for parameter sweep and sensitivity tools."""

    def test_parameter_sweep_na(self) -> None:
        """Verify parameter sweep returns correct results across NA values."""
        cfg = BluRayConfig()
        na_vals = [0.75, 0.80, 0.85, 0.90]

        results = parameter_sweep(cfg, "numerical_aperture", na_vals)
        assert len(results) == 4

        # Verify values preserved
        for i, row in enumerate(results):
            assert row["value"] == na_vals[i]
            assert not np.isnan(row["cnr_db"])
            assert row["spot_radius_m"] > 0

        # Monotonicity: higher NA increases CNR through improved MTF
        cnrs = [row["cnr_db"] for row in results]
        assert cnrs[0] < cnrs[-1]

    def test_parameter_sweep_invalid_attribute(self) -> None:
        """Verify AttributeError is raised when sweeping a nonexistent attribute."""
        cfg = CDConfig()
        with pytest.raises(AttributeError):
            parameter_sweep(cfg, "nonexistent_parameter_xyz", [1.0, 2.0])

    def test_one_at_a_time_sensitivity(self) -> None:
        """Verify OAT sensitivity yields finite gradients and valid metrics."""
        cfg = BluRayConfig()
        params = ["numerical_aperture", "laser_power_w"]

        sens = one_at_a_time_sensitivity(cfg, parameters=params, delta_fraction=0.05)

        for p in params:
            assert p in sens
            entry = sens[p]
            assert "baseline_value" in entry
            assert "gradient" in entry
            assert "elasticity" in entry
            assert not np.isnan(entry["gradient"])

        # NA has positive elasticity on CNR
        assert sens["numerical_aperture"]["elasticity"] > 0

    def test_one_at_a_time_invalid_fraction(self) -> None:
        """Verify ValueError is raised for invalid delta fractions."""
        cfg = BluRayConfig()
        with pytest.raises(ValueError):
            one_at_a_time_sensitivity(cfg, delta_fraction=0.0)

        with pytest.raises(ValueError):
            one_at_a_time_sensitivity(cfg, delta_fraction=-0.1)

        with pytest.raises(ValueError):
            one_at_a_time_sensitivity(cfg, delta_fraction=1.5)

    def test_rank_parameters_by_influence(self) -> None:
        """Verify ranking produces descending order of absolute influence."""
        cfg = BluRayConfig()
        sens = one_at_a_time_sensitivity(
            cfg,
            parameters=["numerical_aperture", "laser_power_w", "temperature_k"],
            delta_fraction=0.05,
        )
        ranked = rank_parameters_by_influence(sens, metric="delta_output")

        assert len(ranked) == 3
        # Check sorting: first element has largest absolute swing
        assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]

    def test_analysis_edge_cases_and_error_handling(self) -> None:
        """Verify error handling in sweep and sensitivity analysis."""
        cfg = BluRayConfig()

        # Sweeping an invalid value (e.g., negative wavelength) records error row
        results = parameter_sweep(cfg, "wavelength_m", [-405e-9, 405e-9])
        assert len(results) == 2
        assert "error" in results[0]
        assert np.isnan(results[0]["cnr_db"])
        assert not np.isnan(results[1]["cnr_db"])

        # Invalid parameter in sensitivity raises AttributeError
        with pytest.raises(AttributeError):
            one_at_a_time_sensitivity(cfg, parameters=["invalid_param"])

        # Invalid output key raises KeyError
        with pytest.raises(KeyError):
            one_at_a_time_sensitivity(cfg, output_key="nonexistent_key")

        # Large delta_fraction guards against negative values (val0 * 1e-3)
        sens = one_at_a_time_sensitivity(cfg, parameters=["laser_power_w"], delta_fraction=0.999)
        assert "laser_power_w" in sens

        # Invalid metric in rank_parameters_by_influence raises ValueError
        with pytest.raises(ValueError):
            rank_parameters_by_influence(sens, metric="invalid_metric")


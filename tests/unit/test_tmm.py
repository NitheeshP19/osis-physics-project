"""Unit tests for the Abelès Transfer Matrix Method (TMM) solver."""

from __future__ import annotations

import math
import numpy as np
import pytest

from osis.configs import ThinFilmLayer, CDConfig, DVDConfig, BluRayConfig
from osis.physics.tmm import compute_stack_reflectance, compute_stack_rt


class TestTMMReflectance:
    """Test suite for TMM reflectivity and transmission calculations."""

    def test_fresnel_single_interface_limit(self) -> None:
        """Verify TMM matches the standard normal-incidence Fresnel equation."""
        n0 = 1.0
        n_sub = 1.50
        expected_r = ((n0 - n_sub) / (n0 + n_sub)) ** 2  # ≈ 0.04 (4%)

        # In TMM, an empty or single matching layer should reproduce the boundary
        # A single layer with same index as substrate
        layer = ThinFilmLayer("Glass", 100e-9, n_sub, 0.0)
        r = compute_stack_reflectance(
            wavelength_m=600e-9,
            layers=[layer],
            n_incident=n0,
            n_substrate=n_sub + 0j,
        )
        assert pytest.approx(r, rel=1e-4) == expected_r

    def test_energy_conservation_lossless_stack(self) -> None:
        """Verify R + T = 1.0 for non-absorbing (k=0) multilayer stacks."""
        wavelength = 500e-9
        layers = [
            ThinFilmLayer("TiO2", 55e-9, 2.30, 0.0),
            ThinFilmLayer("SiO2", 85e-9, 1.46, 0.0),
            ThinFilmLayer("TiO2", 55e-9, 2.30, 0.0),
        ]
        r, t = compute_stack_rt(
            wavelength_m=wavelength,
            layers=layers,
            n_incident=1.0,
            n_substrate=1.52 + 0j,
        )
        assert 0.0 <= r <= 1.0
        assert 0.0 <= t <= 1.0
        assert pytest.approx(r + t, abs=1e-5) == 1.0

    def test_quarter_wave_antireflection_coating(self) -> None:
        """Verify that an ideal quarter-wave AR coating yields R ≈ 0 at design wavelength."""
        # For incident n0=1.0 and substrate ns=2.25, ideal layer index n1 = sqrt(n0*ns) = 1.50
        # Layer thickness d = lambda0 / (4 * n1)
        lambda0 = 600e-9
        n1 = 1.50
        d1 = lambda0 / (4.0 * n1)
        ar_layer = ThinFilmLayer("Ideal-AR", d1, n1, 0.0)

        r = compute_stack_reflectance(
            wavelength_m=lambda0,
            layers=[ar_layer],
            n_incident=1.0,
            n_substrate=2.25 + 0j,
        )
        assert r < 1e-4

    def test_metal_reflector_reflectivity(self) -> None:
        """Verify high reflectance for a thick opaque Ag layer."""
        # Ag at 650 nm: n≈0.14, k≈3.98
        ag_layer = ThinFilmLayer("Ag", 100e-9, 0.14, 3.98)
        r = compute_stack_reflectance(
            wavelength_m=650e-9,
            layers=[ag_layer],
            n_incident=1.0,
            n_substrate=1.58 + 0j,
        )
        # Silver at 650nm is a known high-reflector (R > 90%)
        assert r > 0.90

    def test_standard_disc_presets_valid_reflectance(self) -> None:
        """Verify that all standard disc presets yield physically valid reflectances."""
        for cfg in [CDConfig(), DVDConfig(), BluRayConfig()]:
            r_land = compute_stack_reflectance(cfg.wavelength_m, cfg.land_stack)
            r_mark = compute_stack_reflectance(cfg.wavelength_m, cfg.mark_stack)

            assert 0.0 < r_land < 1.0
            assert 0.0 < r_mark < 1.0
            # Ensure nonzero optical contrast between land and mark states
            assert abs(r_land - r_mark) > 0.01

    def test_invalid_arguments_raise_errors(self) -> None:
        """Verify TMM and ThinFilmLayer raise ValueError on unphysical input parameters."""
        valid_layer = ThinFilmLayer("Dielectric", 50e-9, 2.0, 0.0)

        with pytest.raises(ValueError):
            # Negative wavelength
            compute_stack_reflectance(-500e-9, [valid_layer])

        with pytest.raises(ValueError):
            # Zero wavelength
            compute_stack_reflectance(0.0, [valid_layer])

        with pytest.raises(ValueError):
            # Non-positive thickness
            ThinFilmLayer("Bad", -10e-9, 2.0, 0.0)

        with pytest.raises(ValueError):
            # Negative extinction coefficient
            ThinFilmLayer("Bad", 10e-9, 2.0, -0.5)

    def test_empty_layers_returns_substrate_fresnel(self) -> None:
        """Verify empty layer list returns the bare substrate Fresnel reflection."""
        n0 = 1.0
        n_sub = 1.58
        expected = ((n0 - n_sub) / (n0 + n_sub)) ** 2
        r = compute_stack_reflectance(500e-9, [], n_incident=n0, n_substrate=n_sub + 0j)
        assert pytest.approx(r, rel=1e-4) == expected

    def test_spectral_reflectance_and_rt_validation(self) -> None:
        """Verify spectral reflectance array computation and compute_stack_rt validation."""
        from osis.physics.tmm import compute_stack_rt, compute_stack_reflectance_spectrum

        valid_layer = ThinFilmLayer("Dielectric", 50e-9, 2.0, 0.0)

        # Invalid wavelength in compute_stack_rt
        with pytest.raises(ValueError):
            compute_stack_rt(-400e-9, [valid_layer])

        # Array of wavelengths
        wls = np.linspace(400e-9, 700e-9, 5)
        spec = compute_stack_reflectance_spectrum(wls, [valid_layer])
        assert len(spec) == 5
        assert np.all((spec >= 0.0) & (spec <= 1.0))



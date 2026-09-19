"""Unit tests for scalar diffraction optics and MTF modules."""

from __future__ import annotations

import math
import numpy as np
import pytest

from osis.physics.optics import (
    spot_radius_m,
    coherent_cutoff_freq,
    incoherent_cutoff_freq,
    otf_coherent,
    mtf_incoherent,
    airy_intensity,
)


class TestDiffractionOptics:
    """Test suite for diffraction spot sizing, OTF, and MTF."""

    def test_spot_radius_standard_wavelengths(self) -> None:
        """Verify Rayleigh spot radius values for standard disc architectures."""
        # CD: 780 nm, NA = 0.45 -> r ≈ 1057.3 nm
        r_cd = spot_radius_m(780e-9, 0.45)
        assert pytest.approx(r_cd * 1e9, abs=1.0) == 1057.3

        # DVD: 650 nm, NA = 0.60 -> r ≈ 660.8 nm
        r_dvd = spot_radius_m(650e-9, 0.60)
        assert pytest.approx(r_dvd * 1e9, abs=1.0) == 660.8

        # Blu-ray: 405 nm, NA = 0.85 -> r ≈ 290.8 nm
        r_bd = spot_radius_m(405e-9, 0.85)
        assert pytest.approx(r_bd * 1e9, abs=1.0) == 290.8

    def test_spot_radius_monotonicity(self) -> None:
        """Verify spot radius scales inverse with NA and linearly with wavelength."""
        wl = 405e-9
        na_low = 0.70
        na_high = 0.85
        assert spot_radius_m(wl, na_high) < spot_radius_m(wl, na_low)

        na = 0.60
        wl_short = 405e-9
        wl_long = 650e-9
        assert spot_radius_m(wl_short, na) < spot_radius_m(wl_long, na)

    def test_spot_radius_validation_errors(self) -> None:
        """Verify error handling on invalid physical parameters."""
        with pytest.raises(ValueError):
            spot_radius_m(-405e-9, 0.85)

        with pytest.raises(ValueError):
            spot_radius_m(405e-9, 0.0)

        with pytest.raises(ValueError):
            spot_radius_m(405e-9, 1.2)  # NA >= 1 in scalar immersion-free

    def test_cutoff_frequencies(self) -> None:
        """Verify coherent and incoherent optical cutoff frequencies."""
        wl = 500e-9
        na = 0.50

        fc_coh = coherent_cutoff_freq(wl, na)
        assert pytest.approx(fc_coh) == 1e6  # 0.5 / 500e-9 = 1e6 cycles/m

        fc_inc = incoherent_cutoff_freq(wl, na)
        assert pytest.approx(fc_inc) == 2e6  # 2 * 1e6

    def test_mtf_incoherent_properties(self) -> None:
        """Verify MTF boundary conditions, cutoff, and monotonic decay."""
        wl = 405e-9
        na = 0.85
        fc = incoherent_cutoff_freq(wl, na)

        # DC component: MTF(0) == 1.0
        val_0 = mtf_incoherent(np.array([0.0]), wl, na)[0]
        assert pytest.approx(val_0, abs=1e-6) == 1.0

        # Cutoff: MTF(fc) == 0.0
        val_fc = mtf_incoherent(np.array([fc]), wl, na)[0]
        assert pytest.approx(val_fc, abs=1e-6) == 0.0

        # Beyond cutoff: MTF(> fc) == 0.0
        val_beyond = mtf_incoherent(np.array([1.5 * fc]), wl, na)[0]
        assert val_beyond == 0.0

        # Monotonicity test across frequency range
        freqs = np.linspace(0, fc, 25)
        mtf_vals = mtf_incoherent(freqs, wl, na)
        diffs = np.diff(mtf_vals)
        assert np.all(diffs <= 1e-12)

    def test_airy_pattern_intensity(self) -> None:
        """Verify central peak and symmetry of Airy intensity."""
        wl = 405e-9
        na = 0.85
        r_pts = np.linspace(-1e-6, 1e-6, 51)
        intensity = airy_intensity(r_pts, wl, na)

        # Peak at center (r=0) must be 1.0
        center_idx = len(r_pts) // 2
        assert pytest.approx(intensity[center_idx], abs=1e-6) == 1.0

        # Must be bounded in [0, 1]
        assert np.all(intensity >= 0.0)
        assert np.all(intensity <= 1.0)

        # Symmetry: I(r) == I(-r)
        assert np.allclose(intensity, intensity[::-1])

"""Unit tests for readout channel and noise models."""

from __future__ import annotations

import math
import pytest

from osis.physics.channel import (
    readout_signal,
    shot_noise_power,
    rin_noise_power,
    thermal_noise_power,
    total_noise_power,
)


class TestChannelModels:
    """Test suite for readout signal conversion and noise components."""

    def test_readout_signal_zero_contrast(self) -> None:
        """Verify signal power is zero when land and mark reflectances are identical."""
        sig = readout_signal(
            r_land=0.5,
            r_mark=0.5,
            laser_power_w=1e-3,
            coupling_efficiency=0.35,
        )
        assert sig == 0.0

    def test_readout_signal_quadratic_power_scaling(self) -> None:
        """Verify detected electrical signal power scales with the square of optical power."""
        p1 = 1e-3
        p2 = 2e-3
        sig1 = readout_signal(0.25, 0.05, p1, 0.35)
        sig2 = readout_signal(0.25, 0.05, p2, 0.35)
        # Power doubles -> electrical signal power quadruples (factor of 4)
        assert pytest.approx(sig2 / sig1, rel=1e-5) == 4.0

    def test_readout_signal_validation(self) -> None:
        """Verify error checking on unphysical parameters."""
        with pytest.raises(ValueError):
            readout_signal(1.5, 0.2, 1e-3, 0.35)  # R > 1.0

        with pytest.raises(ValueError):
            readout_signal(0.5, 0.2, -1e-3, 0.35)  # negative power

    def test_shot_noise_linear_scaling(self) -> None:
        """Verify shot noise power scales linearly with optical power and bandwidth."""
        n1 = shot_noise_power(r_mean=0.3, laser_power_w=1e-3, coupling_efficiency=0.3, bandwidth_hz=50e6)
        n2 = shot_noise_power(r_mean=0.3, laser_power_w=2e-3, coupling_efficiency=0.3, bandwidth_hz=50e6)
        n_bw = shot_noise_power(r_mean=0.3, laser_power_w=1e-3, coupling_efficiency=0.3, bandwidth_hz=100e6)

        assert pytest.approx(n2 / n1, rel=1e-5) == 2.0
        assert pytest.approx(n_bw / n1, rel=1e-5) == 2.0

    def test_rin_noise_quadratic_power_scaling(self) -> None:
        """Verify laser RIN noise scales quadratically with optical power."""
        n1 = rin_noise_power(r_mean=0.3, laser_power_w=1e-3, coupling_efficiency=0.3, bandwidth_hz=50e6)
        n2 = rin_noise_power(r_mean=0.3, laser_power_w=2e-3, coupling_efficiency=0.3, bandwidth_hz=50e6)
        assert pytest.approx(n2 / n1, rel=1e-5) == 4.0

    def test_thermal_noise_independence_of_laser(self) -> None:
        """Verify Johnson thermal noise is strictly independent of laser power and scales with T."""
        bw = 100e6
        nth_300k = thermal_noise_power(bandwidth_hz=bw, temperature_k=300.0)
        nth_600k = thermal_noise_power(bandwidth_hz=bw, temperature_k=600.0)

        # Doubling absolute temperature doubles thermal noise
        assert pytest.approx(nth_600k / nth_300k, rel=1e-5) == 2.0

    def test_total_noise_summation(self) -> None:
        """Verify total noise equals the sum of its three constituent components."""
        sig = 1e-4
        p_laser = 1e-3
        bw = 50e6
        rin = 1e-14
        r_l = 500.0
        temp = 300.0
        qe = 0.8
        e_ph = 4.9e-19
        c_eff = 0.35
        r_mean = 0.4

        n_shot = shot_noise_power(r_mean, p_laser, c_eff, bw, qe, e_ph, r_l)
        n_rin = rin_noise_power(r_mean, p_laser, c_eff, bw, rin, qe, e_ph, r_l)
        n_th = thermal_noise_power(bw, r_l, temp)

        n_tot = total_noise_power(
            signal_w=sig,
            laser_power_w=p_laser,
            bandwidth_hz=bw,
            rin_per_hz=rin,
            load_resistance_ohm=r_l,
            temperature_k=temp,
            quantum_efficiency=qe,
            photon_energy_j=e_ph,
            coupling_efficiency=c_eff,
            r_mean=r_mean,
        )

        assert pytest.approx(n_tot, rel=1e-7) == (n_shot + n_rin + n_th)

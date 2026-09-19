"""Unit tests for SNR, CNR, and BER calculations."""

from __future__ import annotations

import math
import pytest

from osis.physics.snr import (
    cnr_db,
    ber_from_cnr,
    photon_energy_j,
    q_factor_from_cnr,
)


class TestSNRMetrics:
    """Test suite for CNR, BER, and fundamental photon calculations."""

    def test_cnr_db_exact_ratios(self) -> None:
        """Verify exact decibel calculation for known power ratios."""
        # 100x power ratio = 20 dB
        assert pytest.approx(cnr_db(100.0, 1.0)) == 20.0

        # 10,000x power ratio = 40 dB
        assert pytest.approx(cnr_db(10000.0, 1.0)) == 40.0

        # Zero signal power produces -infinity
        assert math.isinf(cnr_db(0.0, 1.0)) and cnr_db(0.0, 1.0) < 0

    def test_cnr_db_invalid_noise_raises(self) -> None:
        """Verify error handling on non-positive noise power."""
        with pytest.raises(ValueError):
            cnr_db(1.0, 0.0)

        with pytest.raises(ValueError):
            cnr_db(1.0, -1.0)

    def test_ber_monotonic_decay(self) -> None:
        """Verify bit error rate decreases strictly monotonically with increasing CNR."""
        cnr_values = [0.0, 5.0, 10.0, 15.0, 20.0]
        ber_values = [ber_from_cnr(c) for c in cnr_values]

        for i in range(len(ber_values) - 1):
            assert ber_values[i] > ber_values[i + 1]

    def test_ber_asymptotes(self) -> None:
        """Verify BER asymptotes: 0.5 at poor SNR, 0.0 at extremely high SNR."""
        # Infinitely bad SNR (CNR -> -inf) should produce max random chance (0.5)
        assert pytest.approx(ber_from_cnr(-100.0), abs=1e-3) == 0.5

        # Very high CNR (> 35 dB) should produce practically 0.0
        assert ber_from_cnr(40.0) == 0.0

    def test_photon_energy_calculation(self) -> None:
        """Verify Planck-Einstein relation E = hc / lambda."""
        # 405 nm Blu-ray photon: ~4.908e-19 J (~3.06 eV)
        e_405 = photon_energy_j(405e-9)
        assert pytest.approx(e_405 * 1e19, rel=1e-3) == 4.908

        # 780 nm CD photon: ~2.547e-19 J (~1.59 eV)
        e_780 = photon_energy_j(780e-9)
        assert pytest.approx(e_780 * 1e19, rel=1e-3) == 2.547

        with pytest.raises(ValueError):
            photon_energy_j(-500e-9)

    def test_snr_edge_cases_and_q_factor(self) -> None:
        """Verify edge cases for negative signal power, q-factor and invalid modulation."""
        from osis.physics.snr import q_factor_from_cnr_linear

        with pytest.raises(ValueError):
            cnr_db(-1.0, 1.0)

        with pytest.raises(NotImplementedError):
            ber_from_cnr(20.0, modulation="BPSK")

        with pytest.raises(ValueError):
            q_factor_from_cnr_linear(-1.0)

        # Q-factor at 20 dB (CNR_linear = 100) -> sqrt(100) / 2 = 5.0
        q = q_factor_from_cnr(20.0)
        assert pytest.approx(q) == 5.0


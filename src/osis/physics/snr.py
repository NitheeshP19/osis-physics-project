"""
SNR / CNR and Bit Error Rate estimation for optical-storage readout.

Physical model
--------------
Optical-disc readout quality is primarily characterised by the
**Carrier-to-Noise Ratio** (CNR), which is the ratio of the detected
signal power to the total noise power.

This module provides:

1. ``cnr_db`` — Compute CNR in dB from signal and noise powers.
2. ``ber_from_cnr`` — Estimate BER from CNR via the Q-factor (Gaussian noise approximation).
3. ``photon_energy_j`` — Compute photon energy at a given wavelength (used in shot-noise calculation).

BER model
---------
Under a Gaussian noise approximation and on-off-keying (OOK), the
bit error rate is related to the Q-factor by (Proakis & Salehi 2007):

    Q = √(CNR_linear) / 2        (SNR in power, OOK)
    BER = 0.5 · erfc(Q / √2)

This is an approximation that assumes:
- Additive white Gaussian noise (AWGN).
- Equal probability of 0 and 1 bits.
- Optimal decision threshold.

For PRML and more complex channel codes, the effective BER would differ;
this model provides a baseline.  Actual BER in optical-storage channels
depends strongly on ISI, media defects, and equaliser performance, which
are outside the scope of this simplified model.

References
----------
Proakis, J. G. & Salehi, M. (2007). *Digital Communications* (5th ed.),
    §4.2. McGraw-Hill.
Saleh, B. E. A. & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.),
    §18.5. Wiley.
Marchetti, S. (1993). *Optical Data Storage*. SPIE Tutorial Texts.
"""

from __future__ import annotations

import math
from scipy.special import erfc


# Physical constants
_H_PLANCK = 6.62607015e-34   # Planck's constant [J·s]
_C_LIGHT  = 2.99792458e8     # Speed of light in vacuum [m/s]


def photon_energy_j(wavelength_m: float) -> float:
    """Return the energy of a single photon at the given wavelength.

    E = h·c / λ

    Parameters
    ----------
    wavelength_m : float
        Free-space wavelength in metres.  Must be > 0.

    Returns
    -------
    float
        Photon energy in joules.

    Examples
    --------
    >>> photon_energy_j(405e-9)  # Blu-ray laser
    4.904...e-19
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    return _H_PLANCK * _C_LIGHT / wavelength_m


def cnr_db(signal_power_w: float, noise_power_w: float) -> float:
    """Compute Carrier-to-Noise Ratio in dB.

    CNR = 10 · log10(signal_power / noise_power)

    Parameters
    ----------
    signal_power_w : float
        Detected signal power in watts.  Must be ≥ 0.
    noise_power_w : float
        Total noise power in watts.  Must be > 0.

    Returns
    -------
    float
        CNR in dB.  Returns ``-inf`` if ``signal_power_w = 0``.

    Raises
    ------
    ValueError
        If ``noise_power_w`` ≤ 0.
    """
    if noise_power_w <= 0:
        raise ValueError(f"noise_power_w must be > 0, got {noise_power_w}")
    if signal_power_w < 0:
        raise ValueError(f"signal_power_w must be >= 0, got {signal_power_w}")
    if signal_power_w == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_power_w / noise_power_w)


def ber_from_cnr(cnr_db_value: float, modulation: str = "OOK-NRZ") -> float:
    """Estimate Bit Error Rate from CNR using the Q-factor (Gaussian noise) approximation.

    Under AWGN and OOK-NRZ modulation:

        CNR_linear = 10^(CNR_dB / 10)
        Q = √(CNR_linear) / 2
        BER = 0.5 · erfc(Q / √2)

    This is a theoretical lower bound assuming a matched filter, optimal
    threshold, and Gaussian noise.  Real channels with ISI and media defects
    will exhibit higher BER.

    Parameters
    ----------
    cnr_db_value : float
        Carrier-to-Noise Ratio in dB.
    modulation : str, optional
        Modulation scheme.  Currently only ``"OOK-NRZ"`` is implemented.
        Other values raise ``NotImplementedError``.

    Returns
    -------
    float
        Estimated BER in [0, 0.5].

    Raises
    ------
    NotImplementedError
        If ``modulation`` is not ``"OOK-NRZ"``.

    Examples
    --------
    >>> ber_from_cnr(20.0)    # 20 dB CNR should give very low BER
    ...
    >>> ber_from_cnr(0.0)     # 0 dB CNR: BER near 0.5
    ...

    References
    ----------
    Proakis, J. G. & Salehi, M. (2007). *Digital Communications* (5th ed.), §4.2.
    """
    if modulation.upper() != "OOK-NRZ":
        raise NotImplementedError(
            f"Only OOK-NRZ modulation is implemented; got '{modulation}'"
        )
    cnr_linear = 10.0 ** (cnr_db_value / 10.0)
    q = math.sqrt(cnr_linear) / 2.0
    ber = 0.5 * float(erfc(q / math.sqrt(2.0)))
    return float(min(max(ber, 0.0), 0.5))


def q_factor_from_cnr_linear(cnr_linear: float) -> float:
    """Return the Q-factor from linear CNR for OOK-NRZ.

        Q = √(CNR) / 2

    Parameters
    ----------
    cnr_linear : float
        Linear CNR (dimensionless power ratio). Must be ≥ 0.

    Returns
    -------
    float
        Q-factor (dimensionless).
    """
    if cnr_linear < 0:
        raise ValueError(f"cnr_linear must be >= 0, got {cnr_linear}")
    return math.sqrt(cnr_linear) / 2.0


def q_factor_from_cnr(cnr_db_val: float) -> float:
    """Return the Q-factor from CNR in decibels.

    Parameters
    ----------
    cnr_db_val : float
        Carrier-to-Noise Ratio in dB.

    Returns
    -------
    float
        Q-factor (dimensionless).
    """
    cnr_linear = 10.0 ** (cnr_db_val / 10.0)
    return q_factor_from_cnr_linear(cnr_linear)


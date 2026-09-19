"""
Readout signal and physical noise models for optical-disc readout.

Physical model overview
-----------------------
The optical readout process detects the contrast in reflected laser power
between the land (unrecorded/crystalline) and mark (pit/amorphous) states.
The detector converts reflected photons to a photocurrent proportional to
incident power.

Signal model
~~~~~~~~~~~~
A simplified optical contrast signal:

    S = (η_q / E_ph) · coupling · laser_power · (R_land − R_mark)  [A]

where

- η_q  : detector quantum efficiency (photons → electrons).
- E_ph : photon energy at the laser wavelength [J].
- coupling : fractional optical efficiency of the return beam path (0–1).
- laser_power : incident read-laser power at the disc [W].
- R_land, R_mark : reflectances of the land and mark states (from TMM).

The detected photocurrent signal is then converted to signal power via the
transimpedance amplifier and load resistance.

Noise model
~~~~~~~~~~~
Three primary noise mechanisms are included:

1. **Shot noise** (quantum noise on the photocurrent, Poisson statistics):

       σ²_shot = 2 · e · I_dc · B

   where e is the electron charge, I_dc is the mean photocurrent, and B is
   the detector bandwidth.  Reference: Saleh & Teich (2019), §18.5.

2. **Relative Intensity Noise (RIN)** (laser power fluctuations):

       σ²_RIN = RIN · I_dc² · B

   where RIN is the spectral density of fractional intensity noise [1/Hz].
   Reference: Petermann (1988), *Laser Diode Modulation and Noise*, §5.

3. **Johnson–Nyquist thermal noise** (resistor thermal noise):

       σ²_thermal = 4 · k_B · T · B / R_L

   where k_B is Boltzmann's constant, T is temperature [K], and R_L is
   the load resistance [Ω].  Reference: Johnson (1928), *Phys. Rev.* 32, 97.

The total noise power is:

    N_total = σ²_shot + σ²_RIN + σ²_thermal

Assumptions and limitations
-----------------------------
- **1/f (flicker) noise** is not modelled.
- **Inter-symbol interference (ISI)** from the OTF is not modelled here;
  the channel model treats it as a separate impairment.
- **Media noise** (jitter, defects) is not included.
- The signal model treats optical contrast as a simple difference in
  reflectance; it does not convolve the PSF with a specific mark geometry.
- All noise sources are assumed to be additive, Gaussian, and stationary.

References
----------
Saleh, B. E. A. & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.).
    Wiley.
Johnson, J. B. (1928). Thermal agitation of electricity in conductors.
    *Physical Review*, 32, 97–109.
Petermann, K. (1988). *Laser Diode Modulation and Noise*. Kluwer.
"""

from __future__ import annotations

import math

# Physical constants
_K_BOLTZMANN = 1.380649e-23   # Boltzmann constant [J/K]
_E_CHARGE    = 1.602176634e-19  # Electron charge [C]


def readout_signal(
    r_land: float,
    r_mark: float,
    laser_power_w: float,
    coupling_efficiency: float,
    quantum_efficiency: float = 0.80,
    photon_energy_j: float = 4.9e-19,
    load_resistance_ohm: float = 500.0,
) -> float:
    """Compute the detected optical signal power.

    Models the detected signal as proportional to the land–mark reflectance
    contrast, the laser power, and coupling/detector efficiency.

    Signal photocurrent:
        I_signal = (η_q / E_ph) · η_c · P_laser · |R_land − R_mark|   [A]

    Signal power at the transimpedance stage:
        P_signal = I_signal² · R_L                                      [W]

    Parameters
    ----------
    r_land : float
        Land (unrecorded) reflectance, in [0, 1].
    r_mark : float
        Mark (pit/amorphous) reflectance, in [0, 1].
    laser_power_w : float
        Incident read-laser power in watts.
    coupling_efficiency : float
        Fraction of return signal captured by the detector, in [0, 1].
    quantum_efficiency : float, optional
        Detector quantum efficiency (electrons per photon), in [0, 1].
    photon_energy_j : float, optional
        Single-photon energy in joules.
    load_resistance_ohm : float, optional
        Transimpedance load resistance in ohms.

    Returns
    -------
    float
        Signal power in watts.

    Raises
    ------
    ValueError
        If any argument is out of physical range.
    """
    for name, val in [
        ("r_land", r_land), ("r_mark", r_mark),
        ("coupling_efficiency", coupling_efficiency),
        ("quantum_efficiency", quantum_efficiency),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"'{name}' must be in [0, 1], got {val}")
    if laser_power_w <= 0:
        raise ValueError("laser_power_w must be > 0")
    if photon_energy_j <= 0:
        raise ValueError("photon_energy_j must be > 0")

    delta_r = abs(r_land - r_mark)
    responsivity = (quantum_efficiency * _E_CHARGE) / photon_energy_j
    i_signal = responsivity * coupling_efficiency * laser_power_w * delta_r
    p_signal = i_signal**2 * load_resistance_ohm
    return float(p_signal)


def shot_noise_power(
    r_mean: float,
    laser_power_w: float,
    coupling_efficiency: float,
    bandwidth_hz: float,
    quantum_efficiency: float = 0.80,
    photon_energy_j: float = 4.9e-19,
    load_resistance_ohm: float = 500.0,
) -> float:
    """Compute shot-noise power.

    Shot noise arises from the discrete (Poisson) nature of photon detection.

        I_dc   = (η_q · e / E_ph) · η_c · P_laser · R_mean    [A]
        σ²_shot = 2 · e · I_dc · B                             [A²]
        P_shot  = σ²_shot · R_L                                [W]

    Parameters
    ----------
    r_mean : float
        Mean disc reflectance (average of land and mark), in [0, 1].
    laser_power_w : float
        Incident read power in watts.
    coupling_efficiency : float
        Return-path optical efficiency.
    bandwidth_hz : float
        Detector bandwidth in Hz.
    quantum_efficiency : float, optional
    photon_energy_j : float, optional
    load_resistance_ohm : float, optional

    Returns
    -------
    float
        Shot-noise power in watts.

    References
    ----------
    Saleh & Teich (2019), *Fundamentals of Photonics* (3rd ed.), §18.5.
    """
    responsivity = (quantum_efficiency * _E_CHARGE) / photon_energy_j
    i_dc = responsivity * coupling_efficiency * laser_power_w * r_mean
    sigma2_shot = 2.0 * _E_CHARGE * abs(i_dc) * bandwidth_hz
    return float(sigma2_shot * load_resistance_ohm)


def rin_noise_power(
    r_mean: float,
    laser_power_w: float,
    coupling_efficiency: float,
    bandwidth_hz: float,
    rin_per_hz: float = 1e-14,
    quantum_efficiency: float = 0.80,
    photon_energy_j: float = 4.9e-19,
    load_resistance_ohm: float = 500.0,
) -> float:
    """Compute Relative Intensity Noise (RIN) power.

        I_dc    = (η_q · e / E_ph) · η_c · P_laser · R_mean    [A]
        σ²_RIN  = RIN · I_dc² · B                              [A²]
        P_RIN   = σ²_RIN · R_L                                 [W]

    Parameters
    ----------
    r_mean : float
        Mean disc reflectance in [0, 1].
    laser_power_w : float
        Read power in watts.
    coupling_efficiency : float
    bandwidth_hz : float
    rin_per_hz : float, optional
        Laser RIN spectral density in 1/Hz.
    quantum_efficiency : float, optional
    photon_energy_j : float, optional
    load_resistance_ohm : float, optional

    Returns
    -------
    float
        RIN noise power in watts.

    References
    ----------
    Petermann, K. (1988). *Laser Diode Modulation and Noise*. Kluwer, §5.
    """
    responsivity = (quantum_efficiency * _E_CHARGE) / photon_energy_j
    i_dc = responsivity * coupling_efficiency * laser_power_w * r_mean
    sigma2_rin = rin_per_hz * i_dc**2 * bandwidth_hz
    return float(sigma2_rin * load_resistance_ohm)


def thermal_noise_power(
    bandwidth_hz: float,
    load_resistance_ohm: float = 500.0,
    temperature_k: float = 300.0,
) -> float:
    """Compute Johnson–Nyquist thermal noise power.

        σ²_thermal = 4 · k_B · T · B / R_L        [A²]
        P_thermal  = σ²_thermal · R_L = 4 k_B T B [W]

    Note: this simplifies to 4·k_B·T·B independent of R_L when expressed
    as power into R_L (Saleh & Teich, eq. 18.5-7).

    Parameters
    ----------
    bandwidth_hz : float
        Noise bandwidth in Hz.
    load_resistance_ohm : float, optional
        Load resistance in ohms.
    temperature_k : float, optional
        Temperature in kelvin.

    Returns
    -------
    float
        Thermal noise power in watts.

    References
    ----------
    Johnson, J. B. (1928). *Physical Review*, 32, 97–109.
    Nyquist, H. (1928). *Physical Review*, 32, 110–113.
    """
    return float(4.0 * _K_BOLTZMANN * temperature_k * bandwidth_hz)


def total_noise_power(
    signal_w: float,
    laser_power_w: float,
    bandwidth_hz: float,
    rin_per_hz: float = 1e-14,
    load_resistance_ohm: float = 500.0,
    temperature_k: float = 300.0,
    quantum_efficiency: float = 0.80,
    photon_energy_j: float = 4.9e-19,
    coupling_efficiency: float = 0.35,
    r_mean: float = 0.5,
) -> float:
    """Compute total noise power (shot + RIN + thermal).

    Parameters
    ----------
    signal_w : float
        Detected signal power [W].
    laser_power_w : float
        Read laser power [W].
    bandwidth_hz : float
        Detector bandwidth [Hz].
    rin_per_hz : float, optional
        Laser RIN spectral density [1/Hz].
    load_resistance_ohm : float, optional
    temperature_k : float, optional
    quantum_efficiency : float, optional
    photon_energy_j : float, optional
    coupling_efficiency : float, optional
    r_mean : float, optional
        Effective mean disc reflectance. Defaults to 0.5.

    Returns
    -------
    float
        Total noise power in watts.
    """
    n_shot = shot_noise_power(
        r_mean=r_mean,
        laser_power_w=laser_power_w,
        coupling_efficiency=coupling_efficiency,
        bandwidth_hz=bandwidth_hz,
        quantum_efficiency=quantum_efficiency,
        photon_energy_j=photon_energy_j,
        load_resistance_ohm=load_resistance_ohm,
    )
    n_rin = rin_noise_power(
        r_mean=r_mean,
        laser_power_w=laser_power_w,
        coupling_efficiency=coupling_efficiency,
        bandwidth_hz=bandwidth_hz,
        rin_per_hz=rin_per_hz,
        quantum_efficiency=quantum_efficiency,
        photon_energy_j=photon_energy_j,
        load_resistance_ohm=load_resistance_ohm,
    )
    n_thermal = thermal_noise_power(bandwidth_hz, load_resistance_ohm, temperature_k)
    return float(n_shot + n_rin + n_thermal)

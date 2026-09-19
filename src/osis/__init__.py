"""
OSIS — Optical Storage Intelligence Simulator
==============================================

A Python library for simulating optical-storage readout systems.

Core modules
------------
- ``osis.physics.tmm``     — Abeles Transfer Matrix Method for multilayer thin-film optics
- ``osis.physics.optics``  — Scalar diffraction spot model and Optical Transfer Function
- ``osis.physics.channel`` — 1D readout signal and physical noise models
- ``osis.physics.snr``     — Carrier-to-Noise Ratio and Bit Error Rate estimation
- ``osis.configs``         — Literature-sourced disc configuration presets (CD, DVD, Blu-ray)
- ``osis.analysis.sweep``  — Parameter sweep framework
- ``osis.analysis.sensitivity`` — One-at-a-time and finite-difference sensitivity analysis

Quick start
-----------
>>> import osis
>>> config = osis.BluRayConfig()
>>> result = osis.simulate(config)
>>> print(f"CNR = {result['cnr_db']:.1f} dB")

The library operates independently of the optional FastAPI web interface.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("osis")
except PackageNotFoundError:
    __version__ = "unknown"

from osis.configs import (
    DiscConfig,
    CDConfig,
    DVDConfig,
    BluRayConfig,
    PRESETS,
)
from osis.physics.tmm import compute_stack_reflectance
from osis.physics.optics import spot_radius_m, otf_coherent, mtf_incoherent
from osis.physics.channel import readout_signal, total_noise_power
from osis.physics.snr import cnr_db, ber_from_cnr
from osis.analysis.sweep import parameter_sweep
from osis.analysis.sensitivity import (
    one_at_a_time_sensitivity,
    rank_parameters_by_influence,
)


def simulate(config: "DiscConfig", *, mark_length_m: float | None = None) -> dict:
    """Run the full optical-storage readout simulation pipeline for a given configuration.

    This is the primary entry-point for the OSIS library. It chains:
    1. Thin-film stack reflectance via TMM (land and mark optical states).
    2. Scalar diffraction spot size and OTF cutoff.
    3. Readout signal and total noise power.
    4. CNR and estimated BER.

    Parameters
    ----------
    config : DiscConfig
        Disc and optical-head configuration. Use ``CDConfig()``, ``DVDConfig()``,
        or ``BluRayConfig()`` for standard presets.
    mark_length_m : float, optional
        Minimum pit/mark length in metres. Defaults to ``config.min_mark_m``.

    Returns
    -------
    dict
        Keys:
        - ``cnr_db`` (float): Carrier-to-Noise Ratio in dB.
        - ``ber`` (float): Estimated Bit Error Rate.
        - ``spot_radius_m`` (float): Rayleigh spot radius in metres.
        - ``r_land`` (float): Land reflectance (fraction 0–1).
        - ``r_mark`` (float): Mark/pit reflectance (fraction 0–1).
        - ``signal_power_w`` (float): Optical signal power in watts.
        - ``noise_power_w`` (float): Total noise power in watts.
        - ``config`` (DiscConfig): The configuration used.

    Examples
    --------
    >>> result = osis.simulate(osis.BluRayConfig())
    >>> result['cnr_db']  # doctest: +ELLIPSIS
    ...
    """
    import numpy as np
    from osis.physics import snr as snr_mod

    ml = mark_length_m if mark_length_m is not None else config.min_mark_m

    # 1. Thin-film reflectance for land (crystalline) and mark (amorphous) states
    r_land = compute_stack_reflectance(
        wavelength_m=config.wavelength_m,
        layers=config.land_stack,
    )
    r_mark = compute_stack_reflectance(
        wavelength_m=config.wavelength_m,
        layers=config.mark_stack,
    )

    # 2. Spot size and MTF modulation factor at minimum mark spatial frequency
    r_spot = spot_radius_m(config.wavelength_m, config.numerical_aperture)
    f_spatial = 1.0 / (2.0 * ml)
    mtf_val = float(
        mtf_incoherent(
            np.array([f_spatial]),
            config.wavelength_m,
            config.numerical_aperture,
        )[0]
    )

    # 3. Readout signal (modulated by MTF^2 power transfer) and noise
    e_photon = snr_mod.photon_energy_j(config.wavelength_m)
    sig_raw = readout_signal(
        r_land=r_land,
        r_mark=r_mark,
        laser_power_w=config.laser_power_w,
        coupling_efficiency=config.coupling_efficiency,
        quantum_efficiency=config.quantum_efficiency,
        photon_energy_j=e_photon,
        load_resistance_ohm=config.load_resistance_ohm,
    )
    sig = sig_raw * (mtf_val ** 2)

    r_mean = 0.5 * (r_land + r_mark)
    noise = total_noise_power(
        signal_w=sig,
        laser_power_w=config.laser_power_w,
        bandwidth_hz=config.detector_bandwidth_hz,
        rin_per_hz=config.rin_per_hz,
        load_resistance_ohm=config.load_resistance_ohm,
        temperature_k=config.temperature_k,
        quantum_efficiency=config.quantum_efficiency,
        photon_energy_j=e_photon,
        coupling_efficiency=config.coupling_efficiency,
        r_mean=r_mean,
    )

    cnr = cnr_db(sig, noise)
    ber = ber_from_cnr(cnr)

    return {
        "cnr_db": cnr,
        "ber": ber,
        "spot_radius_m": r_spot,
        "mtf": mtf_val,
        "r_land": r_land,
        "r_mark": r_mark,
        "signal_power_w": sig,
        "noise_power_w": noise,
        "config": config,
    }


__all__ = [
    "__version__",
    "simulate",
    "DiscConfig",
    "CDConfig",
    "DVDConfig",
    "BluRayConfig",
    "PRESETS",
    "compute_stack_reflectance",
    "spot_radius_m",
    "otf_coherent",
    "mtf_incoherent",
    "readout_signal",
    "total_noise_power",
    "cnr_db",
    "ber_from_cnr",
    "parameter_sweep",
    "one_at_a_time_sensitivity",
    "rank_parameters_by_influence",
]

# OSIS API Reference

This document provides a concise reference for the public Python API exposed by `osis`.

---

## Top-Level Functions

### `osis.simulate(config: DiscPresetConfig) -> dict[str, Any]`

Runs the complete physics readout channel pipeline for a given disc configuration preset.

**Parameters:**
- `config` (`DiscPresetConfig`): A configuration instance, such as `CDConfig()`, `DVDConfig()`, or `BluRayConfig()`.

**Returns:**
A dictionary containing the simulation diagnostics and output metrics:
- `"r_land"` (`float`): Power reflectance of the unwritten crystalline land state.
- `"r_mark"` (`float`): Power reflectance of the written amorphous mark state.
- `"contrast"` (`float`): Absolute reflectance contrast `|r_land - r_mark|`.
- `"spot_radius_m"` (`float`): Diffraction-limited Airy spot radius in meters ($0.61 \lambda / \mathrm{NA}$).
- `"cutoff_freq_lp_m"` (`float`): Optical cutoff spatial frequency in line pairs per meter ($2\mathrm{NA}/\lambda$).
- `"mtf"` (`float`): Incoherent Modulation Transfer Function factor evaluated at the minimum mark spatial frequency.
- `"signal_current_a"` (`float`): Peak signal photocurrent in Amperes.
- `"signal_power_w"` (`float`): Signal electrical power in Watts ($I_\text{sig}^2 R_L$).
- `"shot_noise_w"` (`float`): Shot noise electrical power in Watts ($2 e I_\text{dc} B R_L$).
- `"rin_noise_w"` (`float`): Laser Relative Intensity Noise power in Watts ($\mathrm{RIN} I_\text{dc}^2 B R_L$).
- `"thermal_noise_w"` (`float`): Johnson–Nyquist thermal noise power in Watts ($4 k_B T B$).
- `"total_noise_w"` (`float`): Total noise electrical power in Watts.
- `"cnr_db"` (`float`): Carrier-to-Noise Ratio in decibels ($10 \log_{10}(P_\text{sig} / P_\text{noise})$).
- `"ber"` (`float`): Estimated Bit Error Rate for an OOK-NRZ channel under additive Gaussian noise.

---

## Configuration Presets

### `osis.CDConfig()`
Compact Disc standard preset (ECMA-130):
- $\lambda = 780\text{ nm}$
- $\mathrm{NA} = 0.45$
- Minimum mark length $L_\text{min} = 833\text{ nm}$

### `osis.DVDConfig()`
DVD standard preset (ECMA-267):
- $\lambda = 650\text{ nm}$
- $\mathrm{NA} = 0.60$
- Minimum mark length $L_\text{min} = 400\text{ nm}$

### `osis.BluRayConfig()`
Blu-ray Disc standard preset (BDA System Description):
- $\lambda = 405\text{ nm}$
- $\mathrm{NA} = 0.85$
- Minimum mark length $L_\text{min} = 149\text{ nm}$

---

## Analysis Modules

### `osis.parameter_sweep(base_config, parameter_name, values) -> list[dict[str, Any]]`
Sweeps a single scalar attribute of the configuration over an iterable of values and returns the evaluated simulation outputs.

### `osis.one_at_a_time_sensitivity(base_config, parameters=None, delta_fraction=0.05) -> dict[str, dict[str, float]]`
Performs a symmetric two-sided perturbation sensitivity analysis for each parameter, returning:
- `"baseline_value"`: Nominal parameter value.
- `"delta_output"`: Output swing in CNR (dB).
- `"gradient"`: Local gradient $\partial y / \partial p$.
- `"elasticity"`: Normalized elasticity $(p_0 / y_0) \cdot (\partial y / \partial p)$.

### `osis.rank_parameters_by_influence(sensitivities, metric="delta_output") -> list[tuple[str, float]]`
Ranks parameters in descending order of sensitivity swing or elasticity.

---

## Core Physics Submodules

- `osis.physics.tmm`: Abelès Transfer Matrix Method implementation (`compute_stack_reflectance`, `compute_stack_rt`).
- `osis.physics.optics`: Scalar diffraction spot sizing (`airy_spot_radius`) and MTF calculation (`incoherent_mtf`).
- `osis.physics.channel`: Signal photocurrent (`readout_signal`) and physical noise components (`shot_noise_variance`, `rin_noise_variance`, `thermal_noise_variance`, `total_noise_power`).
- `osis.physics.snr`: CNR computation (`cnr_db`), theoretical Bit Error Rate (`ber_from_cnr`), and photon energy (`photon_energy`).

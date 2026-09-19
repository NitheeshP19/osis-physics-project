"""
Sensitivity analysis framework for OSIS.

Provides One-At-a-Time (OAT) sensitivity screening, local finite-difference
gradients, and elasticity metrics for evaluating the influence of optical,
detector, and thermal parameters on disc readout performance (CNR, BER).

Theoretical background
----------------------
Local sensitivity of metric :math:`y` with respect to parameter :math:`p`
evaluated at baseline :math:`p_0`:

.. math::
    \\frac{\\partial y}{\\partial p} \\approx \\frac{y(p_0 + \\Delta p) - y(p_0 - \\Delta p)}{2 \\Delta p}

Normalized sensitivity (elasticity / logarithmic sensitivity):

.. math::
    S_p = \\frac{p_0}{y_0} \\frac{\\partial y}{\\partial p}

A normalized sensitivity of :math:`+1.0` means a 1% increase in :math:`p` yields
an approximate 1% increase in :math:`y`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from osis.configs import DiscConfig


DEFAULT_SENSITIVITY_PARAMETERS: list[str] = [
    "laser_power_w",
    "numerical_aperture",
    "detector_bandwidth_hz",
    "rin_per_hz",
    "load_resistance_ohm",
    "temperature_k",
    "coupling_efficiency",
    "quantum_efficiency",
]


def one_at_a_time_sensitivity(
    config: DiscConfig,
    parameters: Sequence[str] | None = None,
    delta_fraction: float = 0.05,
    *,
    output_key: str = "cnr_db",
    extra_simulate_kwargs: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Compute One-At-a-Time (OAT) local sensitivity for scalar parameters.

    Perturbs each parameter by :math:`\\pm \\delta \\times p_0` (where :math:`\\delta` is
    ``delta_fraction``), evaluates the simulation at both points, and calculates
    the finite-difference gradient and normalized elasticity.

    Parameters
    ----------
    config : DiscConfig
        Baseline disc configuration.
    parameters : Sequence[str], optional
        List of parameter attribute names on ``config`` to evaluate. If None,
        defaults to ``DEFAULT_SENSITIVITY_PARAMETERS``.
    delta_fraction : float, default=0.05
        Fractional perturbation step size (e.g. 0.05 corresponds to ±5%).
        Must be strictly positive and typically <= 0.20.
    output_key : str, default="cnr_db"
        Simulation result dictionary key to track (e.g., ``"cnr_db"``, ``"ber"``,
        ``"spot_radius_m"``, ``"signal_power_w"``, ``"noise_power_w"``).
    extra_simulate_kwargs : dict, optional
        Additional kwargs passed directly to :func:`osis.simulate`.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of parameter name to a dictionary containing:
        - ``"baseline_value"``: initial value of parameter :math:`p_0`.
        - ``"baseline_output"``: output metric :math:`y(p_0)`.
        - ``"val_low"``: perturbed value :math:`p_0 (1 - \\delta)`.
        - ``"val_high"``: perturbed value :math:`p_0 (1 + \\delta)`.
        - ``"output_low"``: output at :math:`p_0 (1 - \\delta)`.
        - ``"output_high"``: output at :math:`p_0 (1 + \\delta)`.
        - ``"delta_output"``: swing :math:`y_{high} - y_{low}`.
        - ``"gradient"``: finite-difference derivative :math:`\\partial y / \\partial p`.
        - ``"elasticity"``: normalized elasticity :math:`(p_0 / y_0) (\\partial y / \\partial p)`.

    Raises
    ------
    ValueError
        If ``delta_fraction <= 0`` or ``delta_fraction >= 1``.
    AttributeError
        If any specified parameter is not a field of ``DiscConfig``.

    Examples
    --------
    >>> import osis
    >>> sens = osis.one_at_a_time_sensitivity(osis.BluRayConfig(), ["laser_power_w"])
    >>> "laser_power_w" in sens
    True
    >>> "gradient" in sens["laser_power_w"]
    True
    """
    import osis as _osis

    if delta_fraction <= 0 or delta_fraction >= 1.0:
        raise ValueError(
            f"delta_fraction must be in (0, 1), got {delta_fraction}"
        )

    param_list = list(parameters) if parameters is not None else DEFAULT_SENSITIVITY_PARAMETERS

    for p in param_list:
        if not hasattr(config, p):
            raise AttributeError(f"'{type(config).__name__}' has no attribute '{p}'")

    kwargs = extra_simulate_kwargs or {}

    # Baseline evaluation
    base_res = _osis.simulate(config, **kwargs)
    if output_key not in base_res:
        raise KeyError(
            f"Simulation output missing requested key '{output_key}'. Available keys: {list(base_res.keys())}"
        )
    y0 = float(base_res[output_key])

    results: dict[str, dict[str, float]] = {}

    for param in param_list:
        val0 = float(getattr(config, param))
        step = val0 * delta_fraction

        val_low = val0 - step
        val_high = val0 + step

        # Guard against unphysical negative values if baseline is strictly positive
        if val0 > 0 and val_low <= 0:
            val_low = val0 * 1e-3

        cfg_low = replace(config, **{param: val_low})
        cfg_high = replace(config, **{param: val_high})

        try:
            res_low = _osis.simulate(cfg_low, **kwargs)
            res_high = _osis.simulate(cfg_high, **kwargs)
            y_low = float(res_low[output_key])
            y_high = float(res_high[output_key])
        except Exception:
            results[param] = {
                "baseline_value": val0,
                "baseline_output": y0,
                "val_low": val_low,
                "val_high": val_high,
                "output_low": float("nan"),
                "output_high": float("nan"),
                "delta_output": float("nan"),
                "gradient": float("nan"),
                "elasticity": float("nan"),
            }
            continue

        delta_p = val_high - val_low
        delta_y = y_high - y_low

        gradient = delta_y / delta_p if delta_p != 0 else 0.0

        if y0 != 0 and not np.isnan(y0):
            elasticity = (val0 / y0) * gradient
        else:
            elasticity = float("nan")

        results[param] = {
            "baseline_value": val0,
            "baseline_output": y0,
            "val_low": val_low,
            "val_high": val_high,
            "output_low": y_low,
            "output_high": y_high,
            "delta_output": delta_y,
            "gradient": gradient,
            "elasticity": elasticity,
        }

    return results


def rank_parameters_by_influence(
    sensitivity_results: dict[str, dict[str, float]],
    metric: str = "delta_output",
) -> list[tuple[str, float]]:
    """Rank parameters by absolute sensitivity metric.

    Parameters
    ----------
    sensitivity_results : dict[str, dict[str, float]]
        Output dictionary from :func:`one_at_a_time_sensitivity`.
    metric : str, default="delta_output"
        Metric to rank by: ``"delta_output"``, ``"gradient"``, or ``"elasticity"``.

    Returns
    -------
    list[tuple[str, float]]
        List of (parameter_name, absolute_value) sorted descending by importance.
    """
    valid_metrics = {"delta_output", "gradient", "elasticity"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

    ranked = []
    for param, metrics in sensitivity_results.items():
        val = metrics.get(metric, 0.0)
        mag = abs(val) if not np.isnan(val) else -1.0
        ranked.append((param, mag))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


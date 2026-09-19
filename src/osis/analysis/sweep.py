"""
Parameter sweep framework for OSIS.

Provides :func:`parameter_sweep` for sweeping any scalar field of a
:class:`~osis.configs.DiscConfig` over a range of values and collecting
the simulation output at each point.

Usage
-----
::

    import numpy as np
    import osis

    config = osis.BluRayConfig()
    na_values = np.linspace(0.70, 0.90, 11)
    results = osis.parameter_sweep(config, "numerical_aperture", na_values)

    for row in results:
        print(f"NA={row['value']:.3f}  CNR={row['cnr_db']:.1f} dB")
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from osis.configs import DiscConfig


def parameter_sweep(
    config: DiscConfig,
    parameter: str,
    values: np.ndarray | list[float],
    *,
    extra_simulate_kwargs: dict | None = None,
) -> list[dict[str, Any]]:
    """Sweep a single scalar parameter of a disc configuration.

    For each value in ``values``, creates a modified copy of ``config`` with
    ``parameter`` set to that value, runs the full simulation, and collects
    the results.

    Parameters
    ----------
    config : DiscConfig
        Base disc configuration.
    parameter : str
        Name of the :class:`~osis.configs.DiscConfig` attribute to sweep.
        Must be a scalar (float/int) field.  Nested dataclass fields are
        not supported.
    values : array-like of float
        Values to sweep over.
    extra_simulate_kwargs : dict, optional
        Additional keyword arguments forwarded to :func:`osis.simulate`.

    Returns
    -------
    list of dict
        Each element contains:

        - ``"value"`` (float): the parameter value used.
        - ``"cnr_db"`` (float): CNR in dB.
        - ``"ber"`` (float): estimated BER.
        - ``"r_land"`` (float): land reflectance.
        - ``"r_mark"`` (float): mark reflectance.
        - ``"spot_radius_m"`` (float): spot radius in metres.

    Raises
    ------
    AttributeError
        If ``parameter`` is not a field of ``DiscConfig``.
    ValueError
        If the modified configuration is physically invalid.

    Examples
    --------
    >>> import numpy as np, osis
    >>> results = osis.parameter_sweep(osis.CDConfig(), "numerical_aperture",
    ...                                np.linspace(0.40, 0.55, 4))
    >>> len(results)
    4
    """
    import osis as _osis

    if not hasattr(config, parameter):
        raise AttributeError(
            f"'{type(config).__name__}' has no attribute '{parameter}'"
        )

    kwargs = extra_simulate_kwargs or {}
    rows: list[dict[str, Any]] = []

    for v in values:
        modified = _set_field(config, parameter, float(v))
        try:
            result = _osis.simulate(modified, **kwargs)
        except Exception as exc:
            rows.append(
                {
                    "value": float(v),
                    "cnr_db": float("nan"),
                    "ber": float("nan"),
                    "r_land": float("nan"),
                    "r_mark": float("nan"),
                    "spot_radius_m": float("nan"),
                    "error": str(exc),
                }
            )
            continue

        rows.append(
            {
                "value": float(v),
                "cnr_db": result["cnr_db"],
                "ber": result["ber"],
                "r_land": result["r_land"],
                "r_mark": result["r_mark"],
                "spot_radius_m": result["spot_radius_m"],
            }
        )

    return rows


def _set_field(config: DiscConfig, field: str, value: float) -> DiscConfig:
    """Return a shallow copy of ``config`` with one field replaced."""
    return replace(config, **{field: value})

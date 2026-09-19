"""
Abeles Transfer Matrix Method (TMM) for multilayer thin-film optics.

This module implements the standard 2×2 characteristic matrix (Abeles) method
for computing the normal-incidence amplitude reflectance and power reflectance
of a planar multilayer thin-film stack.  The formulation follows the treatment
in:

    Born, M. & Wolf, E. (2019). *Principles of Optics* (7th ed.), §1.6.
    Cambridge University Press.

    Heavens, O. S. (1955). *Optical Properties of Thin Solid Films*.
    Butterworths Scientific.

    Hecht, E. (2017). *Optics* (5th ed.), §9.7. Pearson.

Notation
--------
- ``n`` : real part of the complex refractive index.
- ``k`` : imaginary part (extinction coefficient). k > 0 means the medium absorbs.
- ``ñ = n + i·k`` : complex refractive index (physics sign convention, e−iωt).
- ``d`` : layer thickness in metres.
- ``λ`` : free-space wavelength in metres.
- ``δ = (2π/λ)·ñ·d`` : complex phase accumulated through one pass of the layer.

Assumptions and limitations
-----------------------------
- **Normal incidence only.** Oblique-incidence generalisation is straightforward
  (replace ``ñ`` by ``ñ·cos(θ)`` for TE polarisation) but is not implemented here
  because optical-disc read-heads operate at normal incidence.
- **Planar, infinite layers** — no lateral structure, scattering, or surface roughness.
- **Incoherent substrate** is not modelled; all layers are treated coherently.
- **Wavelength-independent optical constants.** The caller supplies constants at a
  single wavelength. For broadband simulations call this function at each wavelength.
- Material constants must be supplied by the caller with appropriate provenance.
  This module contains *no built-in optical-constant database*.
"""

from __future__ import annotations

import cmath
import math
from typing import Sequence

import numpy as np


def _layer_matrix(n_complex: complex, thickness_m: float, wavelength_m: float) -> np.ndarray:
    """Return the 2×2 characteristic matrix for a single thin-film layer.

    The matrix is defined (Born & Wolf §1.6.2) as::

        M = [[  cos(δ),       -i·sin(δ)/η ],
             [ -i·η·sin(δ),   cos(δ)      ]]

    where ``δ = (2π/λ)·ñ·d`` is the complex phase and ``η = ñ`` for normal
    incidence (TE/TM degenerate).

    Parameters
    ----------
    n_complex : complex
        Complex refractive index ``ñ = n + ik`` of the layer.
    thickness_m : float
        Physical thickness of the layer in metres.
    wavelength_m : float
        Free-space wavelength in metres.

    Returns
    -------
    numpy.ndarray, shape (2, 2), dtype complex
        Characteristic matrix for this layer.

    Notes
    -----
    For a non-absorbing layer (k = 0), ``δ`` is real and ``M`` is unitary.
    For an absorbing layer (k > 0), ``δ`` is complex and the matrix is not unitary.
    """
    delta = (2 * math.pi / wavelength_m) * n_complex * thickness_m
    cos_d = cmath.cos(delta)
    sin_d = cmath.sin(delta)
    eta = n_complex  # for normal incidence, η = ñ

    M = np.array(
        [
            [cos_d,              -1j * sin_d / eta],
            [-1j * eta * sin_d,  cos_d            ],
        ],
        dtype=complex,
    )
    return M


def compute_stack_reflectance(
    wavelength_m: float,
    layers: Sequence,
    n_incident: float = 1.0,
    n_substrate: complex = 1.58 + 0j,
) -> float:
    """Compute the normal-incidence power reflectance of a multilayer thin-film stack.

    The total characteristic matrix is the ordered product of the individual
    layer matrices (layer 0 is closest to the incident medium)::

        M_total = M_0 · M_1 · ... · M_{N-1}

    The amplitude reflection coefficient is then (Born & Wolf §1.6.4)::

        r = (η_i·(m00 + m01·η_s) − (m10 + m11·η_s))
            / (η_i·(m00 + m01·η_s) + (m10 + m11·η_s))

    and the power reflectance is ``R = |r|²``.

    Parameters
    ----------
    wavelength_m : float
        Free-space wavelength in metres.  Must be > 0.
    layers : sequence of objects with attributes ``n``, ``k``, ``thickness_m``
        Ordered thin-film layers from incident medium (index 0) to substrate.
        Each element must have:

        - ``n`` (float): real part of the refractive index.
        - ``k`` (float): extinction coefficient (imaginary part). k ≥ 0.
        - ``thickness_m`` (float): physical thickness in metres. > 0.

        The incident medium and substrate are *not* included in this list.
    n_incident : float, optional
        Real refractive index of the incident medium (default: 1.0 for air).
    n_substrate : complex, optional
        Complex refractive index of the substrate
        (default: 1.58+0j, representative of optical-disc polycarbonate).

    Returns
    -------
    float
        Power reflectance R in the range [0, 1].

    Raises
    ------
    ValueError
        If ``wavelength_m`` ≤ 0.
    ValueError
        If any layer has non-physical parameters (thickness ≤ 0, k < 0).

    Examples
    --------
    Single-layer Ag film at 405 nm against air/polycarbonate:

    >>> from osis.physics.tmm import compute_stack_reflectance
    >>> from osis.configs import ThinFilmLayer
    >>> layer = ThinFilmLayer("Ag", 100e-9, n=0.065, k=1.88, source="Rakic 1998")
    >>> R = compute_stack_reflectance(405e-9, [layer])
    >>> 0.0 <= R <= 1.0
    True

    Notes
    -----
    If the layer list is empty, the result reduces to the Fresnel formula for a
    single air/substrate interface:
    ``R = |(n_i − n_s)/(n_i + n_s)|²``.

    References
    ----------
    Born, M. & Wolf, E. (2019). *Principles of Optics* (7th ed.), §1.6.
    Heavens, O. S. (1955). *Optical Properties of Thin Solid Films*.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")

    # Validate layers
    for i, layer in enumerate(layers):
        if layer.thickness_m <= 0:
            raise ValueError(f"Layer {i} ('{layer.name}') thickness must be > 0")
        if layer.k < 0:
            raise ValueError(f"Layer {i} ('{layer.name}') k must be >= 0")

    # Start with the identity matrix
    M_total = np.eye(2, dtype=complex)

    for layer in layers:
        n_cplx = complex(layer.n, layer.k)
        M_layer = _layer_matrix(n_cplx, layer.thickness_m, wavelength_m)
        M_total = M_total @ M_layer

    # Admittances at normal incidence: η = ñ
    eta_i = complex(n_incident, 0.0)
    eta_s = complex(n_substrate.real, n_substrate.imag)

    m00, m01 = M_total[0, 0], M_total[0, 1]
    m10, m11 = M_total[1, 0], M_total[1, 1]

    numerator   = eta_i * (m00 + m01 * eta_s) - (m10 + m11 * eta_s)
    denominator = eta_i * (m00 + m01 * eta_s) + (m10 + m11 * eta_s)

    if abs(denominator) < 1e-30:
        # Degenerate case (impedance matched): R = 0
        return 0.0

    r = numerator / denominator
    R = abs(r) ** 2
    # Clamp numerical noise to physical range
    return float(min(max(R, 0.0), 1.0))


def compute_stack_rt(
    wavelength_m: float,
    layers: Sequence,
    n_incident: float = 1.0,
    n_substrate: complex = 1.58 + 0j,
) -> tuple[float, float]:
    """Compute both power reflectance (R) and power transmittance (T).

    Parameters
    ----------
    wavelength_m : float
        Free-space wavelength in metres. Must be > 0.
    layers : sequence
        Ordered thin-film layers from incident medium to substrate.
    n_incident : float, optional
        Real refractive index of the incident medium.
    n_substrate : complex, optional
        Complex refractive index of the substrate.

    Returns
    -------
    tuple[float, float]
        (R, T) where R is power reflectance and T is power transmittance.
        For non-absorbing layers (k = 0), R + T = 1.0.

    References
    ----------
    Born, M. & Wolf, E. (2019). *Principles of Optics* (7th ed.), §1.6.4.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")

    for i, layer in enumerate(layers):
        if layer.thickness_m <= 0:
            raise ValueError(f"Layer {i} ('{layer.name}') thickness must be > 0")
        if layer.k < 0:
            raise ValueError(f"Layer {i} ('{layer.name}') k must be >= 0")

    M_total = np.eye(2, dtype=complex)
    for layer in layers:
        n_cplx = complex(layer.n, layer.k)
        M_layer = _layer_matrix(n_cplx, layer.thickness_m, wavelength_m)
        M_total = M_total @ M_layer

    eta_i = complex(n_incident, 0.0)
    eta_s = complex(n_substrate.real, n_substrate.imag)

    m00, m01 = M_total[0, 0], M_total[0, 1]
    m10, m11 = M_total[1, 0], M_total[1, 1]

    denom = eta_i * (m00 + m01 * eta_s) + (m10 + m11 * eta_s)
    if abs(denom) < 1e-30:
        return 0.0, 0.0

    r = (eta_i * (m00 + m01 * eta_s) - (m10 + m11 * eta_s)) / denom
    t = (2.0 * eta_i) / denom

    R = abs(r) ** 2
    # Transmittance into substrate: T = (Re(eta_s) / Re(eta_i)) * |t|^2
    T = (eta_s.real / eta_i.real) * (abs(t) ** 2) if eta_i.real > 0 else 0.0

    R = float(min(max(R, 0.0), 1.0))
    T = float(min(max(T, 0.0), 1.0))
    return R, T


def compute_stack_reflectance_spectrum(
    wavelengths_m: np.ndarray,
    layers: Sequence,
    n_incident: float = 1.0,
    n_substrate: complex = 1.58 + 0j,
) -> np.ndarray:
    """Compute power reflectance at multiple wavelengths.

    This is a convenience wrapper that calls :func:`compute_stack_reflectance`
    at each wavelength.  Optical constants are assumed constant over the
    supplied wavelength range (single-wavelength approximation per call).
    For broadband accuracy, supply wavelength-dependent optical constants
    externally and call :func:`compute_stack_reflectance` directly.

    Parameters
    ----------
    wavelengths_m : numpy.ndarray, shape (N,)
        Free-space wavelengths in metres.
    layers : sequence
        Same as :func:`compute_stack_reflectance`.
    n_incident : float, optional
        Incident medium refractive index.
    n_substrate : complex, optional
        Substrate complex refractive index.

    Returns
    -------
    numpy.ndarray, shape (N,)
        Power reflectance at each wavelength.
    """
    return np.array(
        [
            compute_stack_reflectance(wl, layers, n_incident, n_substrate)
            for wl in wavelengths_m
        ]
    )

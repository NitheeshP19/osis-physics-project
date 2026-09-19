"""
Scalar diffraction model and Optical Transfer Function for optical-disc readout.

This module implements a scalar diffraction model appropriate for the
paraxial/moderate-NA regime commonly encountered in optical storage analysis.

Physical model
--------------
The read-laser beam is focused by an objective lens of numerical aperture NA.
Under the scalar (paraxial) diffraction theory, the intensity point-spread
function (PSF) of a diffraction-limited lens is the squared modulus of the
Fourier transform of the pupil function.  For a circular, aberration-free
pupil the result is an Airy pattern.

The key scalar results used here are:

**Spot radius (first zero of Airy pattern):**

    r_airy = 0.61 · λ / NA                              (1)

This is the Rayleigh criterion.  The factor 0.61 = 1.22/2 because the
Airy formula is usually stated in terms of the half-angle.  Reference:
Hecht (2017), *Optics* §10.2.

**Optical Transfer Function (OTF) — coherent illumination:**

Optical-disc readout uses coherent laser illumination.  The coherent OTF
(also called the pupil function projected into frequency space) is a
rectangle/pillbox that is unity up to the coherent cut-off and zero beyond:

    H_coh(ν) = 1    if |ν| ≤ ν_c = NA / λ
               0    otherwise                            (2)

Here ν is the spatial frequency in 1/m (cycles per metre).  Reference:
Goodman (2017), *Introduction to Fourier Optics* (4th ed.), §6.2.

**Optical Transfer Function — incoherent illumination:**

For incoherent imaging (included for completeness, not directly applicable
to disc readout), the OTF is the autocorrelation of the pupil:

    H_inc(ν) = (2/π) · [arccos(ν/ν_c) − (ν/ν_c)·√(1−(ν/ν_c)²)]   (3)

for |ν| ≤ ν_c = 2·NA/λ, and 0 otherwise. Reference: Goodman §6.3.

Limitations and scope
---------------------
- **Scalar theory** is accurate for NA ≲ 0.6.  For Blu-ray (NA = 0.85),
  vector diffraction effects cause measurable deviation from the scalar Airy
  pattern, particularly in polarisation-dependent spot distortion and
  sidelobe suppression.  The scalar model is used here as an established
  analytical baseline; users should be aware of its limitations at high NA.
  See: Mansuripur (2002), *The Physical Principles of Magneto-Optical Recording*.

- **Aberration-free** : no defocus, spherical aberration, or coma is modelled.

- **Monochromatic** : the model applies at a single wavelength.

References
----------
Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W. H. Freeman.
Hecht, E. (2017). *Optics* (5th ed.). Pearson.
Mansuripur, M. (2002). *The Physical Principles of Magneto-Optical Recording*.
    Cambridge University Press.
Born, M. & Wolf, E. (2019). *Principles of Optics* (7th ed.). Cambridge University Press.
"""

from __future__ import annotations

import math
import numpy as np


def spot_radius_m(wavelength_m: float, numerical_aperture: float) -> float:
    """Compute the Rayleigh diffraction-limited spot radius.

    Uses the Rayleigh criterion: the radius is the distance from the centre
    of the Airy disc to its first zero:

        r = 0.61 · λ / NA

    Parameters
    ----------
    wavelength_m : float
        Free-space laser wavelength in metres.  Must be > 0.
    numerical_aperture : float
        Objective lens numerical aperture.  Must satisfy 0 < NA < 1.

    Returns
    -------
    float
        Spot radius (first Airy zero) in metres.

    Raises
    ------
    ValueError
        If wavelength_m ≤ 0 or numerical_aperture ≤ 0 or ≥ 1.

    Examples
    --------
    Blu-ray (405 nm, NA = 0.85):

    >>> spot_radius_m(405e-9, 0.85)  # ≈ 291 nm
    2.9076470588235294e-07

    CD (780 nm, NA = 0.45):

    >>> round(spot_radius_m(780e-9, 0.45) * 1e9)
    1057

    References
    ----------
    Rayleigh, Lord (1879). *Phil. Mag.*, 8, 261.
    Hecht, E. (2017). *Optics* (5th ed.), §10.2.5.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    if not (0 < numerical_aperture < 1):
        raise ValueError(
            f"numerical_aperture must be in (0, 1), got {numerical_aperture}"
        )
    return 0.61 * wavelength_m / numerical_aperture


def coherent_cutoff_freq(wavelength_m: float, numerical_aperture: float) -> float:
    """Return the coherent spatial-frequency cut-off: ν_c = NA / λ.

    Parameters
    ----------
    wavelength_m : float
        Free-space wavelength in metres.
    numerical_aperture : float
        Objective lens NA.

    Returns
    -------
    float
        Coherent cut-off frequency in cycles per metre (1/m).

    References
    ----------
    Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.), §6.2.
    """
    if wavelength_m <= 0 or numerical_aperture <= 0:
        raise ValueError("wavelength_m and numerical_aperture must be > 0")
    return numerical_aperture / wavelength_m


def incoherent_cutoff_freq(wavelength_m: float, numerical_aperture: float) -> float:
    """Return the incoherent spatial-frequency cut-off: ν_c = 2·NA / λ.

    Parameters
    ----------
    wavelength_m : float
    numerical_aperture : float

    Returns
    -------
    float
        Incoherent cut-off in 1/m.

    References
    ----------
    Goodman, J. W. (2017). §6.3.
    """
    return 2.0 * coherent_cutoff_freq(wavelength_m, numerical_aperture)


def otf_coherent(
    spatial_freqs: np.ndarray,
    wavelength_m: float,
    numerical_aperture: float,
) -> np.ndarray:
    """Evaluate the coherent OTF (pupil transfer function) at given spatial frequencies.

    Returns 1 for |ν| ≤ ν_c = NA/λ and 0 otherwise (ideal, aberration-free).

    Parameters
    ----------
    spatial_freqs : numpy.ndarray
        Array of spatial frequencies in 1/m.  Can be 1-D or N-D; shape is preserved.
    wavelength_m : float
        Free-space wavelength in metres.
    numerical_aperture : float
        Objective lens NA.

    Returns
    -------
    numpy.ndarray
        OTF values (0 or 1) with the same shape as ``spatial_freqs``.

    Examples
    --------
    >>> import numpy as np
    >>> nu = np.array([0, 1e6, 3e6])
    >>> otf_coherent(nu, 405e-9, 0.85)
    array([1., 1., 0.])
    """
    nu_c = coherent_cutoff_freq(wavelength_m, numerical_aperture)
    freqs = np.asarray(spatial_freqs, dtype=float)
    return np.where(np.abs(freqs) <= nu_c, 1.0, 0.0)


def mtf_incoherent(
    spatial_freqs: np.ndarray,
    wavelength_m: float,
    numerical_aperture: float,
) -> np.ndarray:
    """Evaluate the incoherent MTF (modulation transfer function) at given spatial frequencies.

    The incoherent MTF is the normalised autocorrelation of the pupil function.
    For a circular, aberration-free pupil at normal incidence:

        H(ν) = (2/π) · [arccos(ν/ν_c) − (ν/ν_c)·√(1−(ν/ν_c)²)]

    for 0 ≤ ν ≤ ν_c, where ν_c = 2·NA/λ.  H(0) = 1 by normalisation.

    Parameters
    ----------
    spatial_freqs : numpy.ndarray
        Spatial frequencies in 1/m (positive values).
    wavelength_m : float
    numerical_aperture : float

    Returns
    -------
    numpy.ndarray
        MTF values in [0, 1].

    References
    ----------
    Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.), §6.3.2, eq. 6-41.
    """
    nu_c = incoherent_cutoff_freq(wavelength_m, numerical_aperture)
    freqs = np.asarray(spatial_freqs, dtype=float)
    u = np.abs(freqs) / nu_c
    # Vectorised implementation; arccos is valid only for u ≤ 1
    u_clipped = np.clip(u, 0.0, 1.0)
    mtf = (2.0 / math.pi) * (np.arccos(u_clipped) - u_clipped * np.sqrt(1.0 - u_clipped**2))
    mtf = np.where(np.abs(freqs) > nu_c, 0.0, mtf)
    return np.clip(mtf, 0.0, 1.0)


def airy_intensity(
    radial_m: np.ndarray,
    wavelength_m: float,
    numerical_aperture: float,
) -> np.ndarray:
    """Evaluate the normalised Airy disc intensity pattern.

    The Airy pattern is the far-field diffraction intensity for a circular
    aperture.  It is normalised so that the peak (at r = 0) equals 1.

        I(r) = [2·J₁(x)/x]²     where x = π·r·D_NA / λ, D_NA = 2·NA

    Parameters
    ----------
    radial_m : numpy.ndarray
        Radial distances from the optical axis in metres.
    wavelength_m : float
    numerical_aperture : float

    Returns
    -------
    numpy.ndarray
        Normalised intensity values in [0, 1].

    References
    ----------
    Hecht, E. (2017). *Optics* (5th ed.), eq. 10.56.
    """
    from scipy.special import j1

    r = np.asarray(radial_m, dtype=float)
    x = math.pi * r * 2.0 * numerical_aperture / wavelength_m
    x_safe = np.where(np.abs(x) < 1e-12, 1.0, x)
    val = (2.0 * j1(x_safe) / x_safe) ** 2
    result = np.where(np.abs(x) < 1e-12, 1.0, val)
    return np.clip(result, 0.0, 1.0)

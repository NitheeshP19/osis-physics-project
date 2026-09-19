# Limitations and Scope

This document defines the physical validity domain and engineering limitations of OSIS.

---

## 1. Optical Modeling Scope

- **Scalar (Paraxial) Diffraction:** The scalar Airy spot formula and incoherent MTF are accurate for low to moderate numerical apertures ($\mathrm{NA} \lesssim 0.60$). At Blu-ray $\mathrm{NA} = 0.85$, non-paraxial vector diffraction effects (polarization-dependent spot distortion and apodization) cause systematic deviations on the order of 5–15% in absolute CNR. Vector (Richards–Wolf) diffraction is not modeled.
- **Normal Incidence:** Thin-film reflectance is evaluated assuming planar waves at normal incidence ($\theta = 0^\circ$). Oblique rays from high-NA focusing cones are neglected.
- **Monochromatic Laser:** Chromatic dispersion and finite laser linewidth ($\Delta\lambda \approx 1\text{ nm}$) are neglected; optical constants are evaluated at the single nominal center wavelength.
- **Aberration-Free Pupil:** Assumes an un-aberrated circular pupil. Defocus, spherical aberration, astigmatism, and coma are not included.

---

## 2. Channel & Media Limitations

- **Binary Readout (OOK-NRZ):** Signal detection assumes binary on/off keying. Advanced run-length limited (RLL) modulation schemes (EFM for CD, EFMPlus for DVD, 17PP for Blu-ray) and PRML (Partial Response Maximum Likelihood) viterbi decoders are not implemented.
- **Gaussian Stationary Noise:** Noise sources are assumed to be independent, additive, and stationary Gaussian processes. 1/f (flicker) noise, pattern-dependent jitter, and media defect bursts are not modeled.
- **Single-Layer Discs:** Multi-layer crosstalk and inter-layer attenuation in dual-layer or multi-layer media are only captured via empirical parameters in the ML surrogate, not analytically in the core physics solver.

---

## 3. Practical Usage Guidance

OSIS is designed for:
- Comparative format evaluations (CD vs. DVD vs. Blu-ray).
- Material thickness and index sensitivity studies.
- Educational demonstration of the optical storage readout chain.
- Rapid parameter exploration for machine-learning surrogate models.

OSIS is **not** a tool for manufacturing tolerance certification or hardware ASIC design.

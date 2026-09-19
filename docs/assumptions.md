# OSIS Assumptions and Limitations

This document explicitly states the physical assumptions and known limitations of the OSIS simulation framework, as required by scientific reproducibility standards.

---

## Physical Assumptions

### Optical Model

| Assumption | Scope | Impact if violated |
|:-----------|:------|:-------------------|
| Scalar (paraxial) diffraction theory | Valid for NA ≲ 0.6 | At BD NA=0.85, vector effects cause ~5–15% CNR error (not modelled) |
| Normal incidence, planar waves | All TMM calculations | Oblique incidence would require angle-dependent Fresnel coefficients |
| Single wavelength per simulation call | No chromatic dispersion | Laser linewidth effects (~0.5–2 nm) are neglected |
| Circular pupil, no aberrations | MTF calculation | Real optical heads may have pupil apodization or Zernike aberrations |
| Coherent-to-incoherent MTF transition | Simplified to incoherent MTF | A partially coherent model would be more accurate for DVD/BD |

### Material Model

| Assumption | Detail |
|:-----------|:-------|
| Phase-change constants (n, k) | Representative literature values at specific wavelengths; users are strongly advised to substitute measured values for their specific material system |
| Single-temperature optical constants | No thermal lensing, no write-process temperature dependence |
| Uniform layer thicknesses | No thickness gradients or roughness effects |
| Sharp crystalline–amorphous boundaries | No partial crystallization fringe effects |

### Noise Model

| Assumption | Detail |
|:-----------|:-------|
| Three additive Gaussian noise sources only | 1/f (flicker) noise, inter-symbol interference (ISI), and pattern-dependent jitter are not modelled |
| White-spectrum RIN | Frequency-independent RIN spectral density over bandwidth B |
| Fixed load resistance and temperature | No thermal drift of detector parameters |
| Independent noise sources | Quadrature addition; cross-correlations are neglected |

### Channel Model

| Assumption | Detail |
|:-----------|:-------|
| OOK-NRZ modulation | Binary on/off keying; no EFM/EFMPlus run-length encoding |
| AWGN channel for BER | Gaussian noise assumption; no dispersion, ISI, or non-linear distortion |

---

## Known Limitations

1. **Blu-ray high-NA accuracy.** At NA = 0.85, scalar diffraction introduces systematic errors in spot sizing and MTF. A vector (Richards–Wolf) diffraction model would be required for high-accuracy BD simulation. The current CNR values for BD-RE should be treated as lower bounds under scalar assumptions.

2. **No write-process simulation.** OSIS simulates the readout channel only. Write-process effects (amorphization threshold, crystallization kinetics) are not modelled.

3. **Single-channel, single-layer recording.** Multi-layer (dual/quad-layer) disc architectures with inter-layer crosstalk are not supported.

4. **No servo modelling.** Focus and tracking servo errors, which degrade effective CNR in practice, are not included.

5. **Optical constant single-source.** Default preset material constants were taken from published literature at representative wavelengths. Significant batch-to-batch and temperature variation exists in real phase-change media.

---

## Validation Status

The OSIS physics engine has been validated in the following limited sense:
- TMM reflectance converges to the analytic Fresnel single-interface limit (±10⁻¹² relative tolerance).
- Quarter-wave antireflection coatings suppress reflectance to ≤0.01% as expected.
- Metal (Al) reflectance matches literature value (Al at 780 nm: R ≈ 88–92%).
- CNR vs. NA monotonicity is consistent with established optical storage theory.
- CNR decreases from CD→DVD→BD due to increasing spatial frequency challenge, consistent with Bouwhuis et al. (1985).

**No experimental measurement validation against physical disc samples has been performed.** OSIS is not certified for engineering design; it is intended as a research and educational tool for systematic parameter exploration.

---

## Suggested Extensions for Future Work

- Vector diffraction (Richards–Wolf) for NA > 0.7
- Partial coherence MTF model
- 1/f noise floor
- Multi-layer disc architectures
- Wavelength-dependent material dispersion (Sellmeier models)
- ISI and run-length limited channel coding

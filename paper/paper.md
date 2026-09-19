---
title: 'OSIS: Optical Storage Intelligence Simulator — A Physics-Based Python Framework for Optical Disc Readout Channel Modeling'
tags:
  - Python
  - optical storage
  - thin-film optics
  - transfer matrix method
  - signal processing
  - noise modeling
  - machine learning surrogate
authors:
  - name: Nitheesh P.
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 19 September 2026
bibliography: paper.bib
---

# Summary

**OSIS** (Optical Storage Intelligence Simulator) is an open-source Python library for simulating the complete optical readout channel of phase-change optical storage discs — including CD, DVD, and Blu-ray architectures. The software chains four physically rigorous sequential sub-models: (1) multilayer thin-film reflectance via the Abelès Transfer Matrix Method (TMM) [@Born2019; @Heavens1955]; (2) Rayleigh scalar diffraction spot sizing and an incoherent Modulation Transfer Function (MTF) for spatial frequency resolution evaluation [@Goodman2017; @Hecht2017]; (3) optoelectronic signal conversion with a photodetector responsivity model; and (4) a three-component physical noise model incorporating quantum shot noise, laser Relative Intensity Noise (RIN), and Johnson–Nyquist thermal noise [@SalehTeich2019; @Johnson1928; @Petermann1988]. The library provides standard calibrated presets for CD-RW, DVD-RW, and Blu-ray BD-RE sourced from published specifications [@Ecma130; @Ecma267; @BDA2004] and peer-reviewed optical constant databases [@Yamada1991; @Ohta1998; @Rakic1998; @Palik1985].

OSIS produces as primary outputs the Carrier-to-Noise Ratio (CNR, in dB) and Bit Error Rate (BER), together with intermediate diagnostics (reflectances, spot radii, MTF values, signal and noise powers). A vectorized parameter sweep framework and One-At-a-Time (OAT) sensitivity analysis with finite-difference gradient and elasticity computation complement the core simulation. An optional machine-learning surrogate component — trained on the physics engine outputs — demonstrates residual learning of multi-layer channel impairments (crosstalk, layer attenuation, detector non-linearity) at sub-millisecond inference latency.

# Statement of Need

Optical data storage is experiencing renewed research and commercial interest driven by next-generation archival applications: cold-tier cloud infrastructure, ultra-long-life 5D optical memory [@Bouwhuis1985], and phase-change material research for neuromorphic computing and non-volatile memory. Investigating the fundamental limits of optical readout — how material properties, layer thicknesses, and optical-head parameters govern CNR and BER — requires either access to proprietary closed-source simulation environments (Zemax, VirtualLab, CODE V) or the manual implementation of optics equations within ad-hoc, non-reproducible scripts.

No existing open-source Python library provides an end-to-end, physically grounded, modular optical disc readout simulation framework. OSIS fills this gap by exposing a clean, importable API — `osis.simulate(config)` — that runs the complete TMM-to-BER pipeline transparently, with all physical equations and assumptions documented and cited inline. The library is designed for researchers in optical storage physics, photonics instrumentation, and materials science who need a reproducible, auditable, and extensible computational baseline without dependency on commercial software.

# Software Design and Physical Architecture

OSIS is engineered as a modular, pure-Python library (`src/osis/`) with zero mandatory web dependencies. Immutable dataclass configurations (`DiscConfig`) define disc stacks and optical parameters. The computational engine couples four physical stages (\autoref{fig:pipeline}):


![The OSIS simulation architecture: four-stage physical model chaining thin-film matrix optics, scalar diffraction and optical transfer function, optoelectronic signal conversion, and additive physical noise into Carrier-to-Noise Ratio (CNR) and Bit Error Rate (BER) metrics.\label{fig:pipeline}](figure1_pipeline.png)

**1. Multilayer Thin-Film Reflectance (Abelès TMM).**
The reflectance of the disc recording stack is computed using the standard 2×2 characteristic matrix (Abelès) formalism [@Born2019, §1.6]:

$$M_{\text{total}} = \prod_{i=1}^{N} \begin{bmatrix} \cos\delta_i & -\mathrm{i}\sin\delta_i/\eta_i \\ -\mathrm{i}\eta_i\sin\delta_i & \cos\delta_i \end{bmatrix}$$

where $\delta_i = (2\pi/\lambda)\tilde{n}_i d_i$ is the complex phase accumulated through layer $i$, $\tilde{n}_i = n_i + \mathrm{i}k_i$ is the complex refractive index, and $\eta_i = \tilde{n}_i$ for normal incidence. Power reflectance is $R = |r|^2$ where $r$ is the amplitude reflection coefficient. Separate reflectances $R_\text{land}$ and $R_\text{mark}$ are computed for the crystalline (unwritten) and amorphous (written) states of the phase-change recording layer.

**2. Scalar Diffraction and Optical Transfer Function.**
The Rayleigh diffraction-limited spot radius is:

$$r_\text{spot} = 0.61\,\lambda / \mathrm{NA}$$

The incoherent MTF at the minimum mark spatial frequency $\nu_s = 1/(2L_\text{min})$ is evaluated using the normalised autocorrelation of a circular pupil [@Goodman2017, §6.3]:

$$\mathrm{MTF}(\nu) = \frac{2}{\pi}\left[\arccos\!\left(\frac{\nu}{\nu_c}\right) - \frac{\nu}{\nu_c}\sqrt{1-\!\left(\frac{\nu}{\nu_c}\right)^2}\right]$$

where $\nu_c = 2\,\mathrm{NA}/\lambda$. The MTF factor $M = \mathrm{MTF}(\nu_s)$ attenuates the detected signal power as $P_\text{sig} \propto M^2$.

**3. Optoelectronic Signal Model.**
The detected signal photocurrent is:

$$I_\text{sig} = \mathcal{R}\cdot\eta_c\cdot P_\text{laser}\cdot|R_\text{land} - R_\text{mark}|\cdot M$$

where $\mathcal{R} = \eta_q e / E_\text{ph}$ is the photodetector responsivity (A/W), $\eta_q$ is the quantum efficiency, $e$ is the electron charge, $E_\text{ph} = hc/\lambda$ is the photon energy, and $\eta_c$ is the optical coupling efficiency. Signal electrical power is $P_\text{sig} = I_\text{sig}^2 R_L$.

**4. Physical Noise Channel.**
Total noise power is the quadrature sum of three independent Gaussian components [@SalehTeich2019; @Johnson1928; @Petermann1988]:

$$P_\text{noise} = (2eI_\text{dc}B + \mathrm{RIN}\cdot I_\text{dc}^2 B + 4k_BT B/R_L)\cdot R_L$$

where $B$ is the detector bandwidth, $I_\text{dc}$ is the mean photocurrent, RIN is the laser Relative Intensity Noise spectral density, $k_B$ is Boltzmann's constant, $T$ is temperature, and $R_L$ is the load resistance.

**5. CNR and BER.**
The Carrier-to-Noise Ratio is $\mathrm{CNR} = 10\log_{10}(P_\text{sig}/P_\text{noise})$, and the Bit Error Rate under a Gaussian/AWGN OOK-NRZ channel model is:

$$\mathrm{BER} = \tfrac{1}{2}\,\mathrm{erfc}\!\left(\frac{\sqrt{\mathrm{CNR}_\text{linear}}}{2\sqrt{2}}\right)$$

# State of the Field

OSIS was written from scratch as a research-software contribution, without being a fork of any existing software. Existing open tools such as `tmm` [@SalehTeich2019] cover thin-film reflectance in isolation, but do not couple to readout signal modeling, noise physics, or analysis frameworks. Commercial tools (Zemax, VirtualLab) support vector diffraction and full optical system modeling but are closed-source, require expensive licenses, and do not expose programmatic APIs for parameter sweep automation or ML surrogate integration. No identified open-source Python package provides the complete optical storage readout channel (TMM + MTF + noise → CNR/BER) in a single, documented, tested library.

# Research Impact Statement

The scientific utility and computational reproducibility of OSIS are demonstrated through an automated, end-to-end research benchmark study included in the repository (`research/run_study.py`, executable via `osis reproduce`). The study investigates how wavelength compression, numerical aperture, and minimum mark spatial frequency dictate the trade-offs between optical spot compression, MTF modulation depth, and optoelectronic Carrier-to-Noise Ratio (CNR) across the three generations of optical disc formats:

| Format | $\lambda$ (nm) | NA | Spot Radius (nm) | MTF at $T_\text{min}$ | CNR (dB) |
|:-------|:---------------|:---|:-----------------|:----------------------|:---------|
| CD-RW | 780 | 0.45 | 1057 | 0.369 | 38.98 |
| DVD-RW | 650 | 0.60 | 661 | 0.209 | 26.19 |
| Blu-ray BD-RE | 405 | 0.85 | 291 | 0.105 | 16.79 |

![Quantitative OSIS model outputs across standardized disc configurations: (a) diffraction-limited focal spot intensity profiles (Airy patterns); (b) incoherent Modulation Transfer Function (MTF) versus spatial frequency up to optical cutoff; (c) One-At-a-Time sensitivity analysis of Blu-ray BD-RE baseline showing parameter elasticities (% $\Delta\text{CNR}$ per 1% parameter shift).\label{fig:disc_comparison}](figure2_disc_comparison.png)

The MTF-limited CNR decrease from CD to Blu-ray (\autoref{fig:disc_comparison}a, b) is physically consistent with the increasing spatial frequency challenge at smaller minimum mark lengths. Sensitivity analysis (OAT, ±5% perturbation) identifies numerical aperture as the dominant influence on CNR (elasticity $\approx +3.11$, \autoref{fig:disc_comparison}c), consistent with established optical storage theory [@Bouwhuis1985].

A gradient-boosted machine-learning surrogate trained on 5,000 OSIS physics-engine evaluations achieves RMSE = 0.168 dB and MAE = 0.123 dB on a held-out test set (R² = 0.9999), while reducing per-evaluation latency from ~0.25 ms (full physics) to ~0.05 ms (surrogate inference). The surrogate learns channel impairments not captured by the analytic model — density-dependent crosstalk, multi-layer optical attenuation, and humidity-induced dye degradation — without synthetic or circular training targets.



# Assumptions and Limitations

OSIS explicitly documents its physical scope and limitations in `docs/assumptions.md` and module docstrings. Key limitations include: (1) scalar diffraction theory is used throughout (valid for NA ≲ 0.6; vector corrections are noted but not implemented for Blu-ray NA=0.85); (2) 1/f flicker noise and inter-symbol interference from the channel filter are not modelled; (3) optical constants are supplied at a single wavelength per simulation call (no chromatic dispersion); (4) the phase-change material constants used in presets are representative approximate values and users are advised to substitute measured values for specific material systems.

# AI Usage Disclosure

Generative AI tools (large language models) were used as coding assistants to scaffold initial module structures, generate docstring templates, and check syntax. All physical equations, parameter values, bibliographic citations, and scientific design decisions were independently verified by the author against primary literature. No AI-generated physical reasoning was incorporated without verification.

# Acknowledgements

The author thanks the maintainers of NumPy [@Harris2020] and SciPy [@Virtanen2020] for the scientific Python infrastructure that powers OSIS.

# References

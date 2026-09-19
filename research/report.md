# Research Study Report: Simulation-Based Analysis of Optical Readout Quality Across Representative CD, DVD, and Blu-ray Configurations

**Principal Investigator:** Nitheesh P.  
**Software Version:** OSIS v1.0.0  
**Timestamp:** 2026-09-19 17:37:16 UTC  

---

## 1. Executive Summary & Research Question
This investigation addresses the fundamental physical question:
> **How do optical spot compression, spatial frequency MTF attenuation, and optoelectronic noise scale across three generations of phase-change optical discs (CD-RW, DVD-RW, Blu-ray BD-RE)?**

Using the deterministic, validated physics engine of the **Optical Storage Intelligence Simulator (OSIS)**, we chained Abelès thin-film matrix optics, scalar diffraction, and a three-component noise channel to evaluate readout Carrier-to-Noise Ratio (CNR) and Bit Error Rate (BER).

---

## 2. Quantitative Results Summary

| Format | Wavelength λ (nm) | NA | Spot Radius (nm) | Min Mark L_min (nm) | Contrast |ΔR| | MTF(T_min) | Readout CNR (dB) | Theoretical BER |
|:-------|:------------------|:---|:-----------------|:--------------------|:--------------|:-----------|:-----------------|:----------------|
| **CD-RW** | 780.0 | 0.45 | 1057.33 | 833.0 | 12.2% | 0.3689 | 38.98 dB | 0.00e+00 |
| **DVD-RW** | 650.0 | 0.60 | 660.83 | 400.0 | 7.94% | 0.2093 | 26.19 dB | 1.07e-24 |
| **Blu-ray BD-RE** | 405.0 | 0.85 | 290.65 | 149.0 | 15.06% | 0.1045 | 16.79 dB | 2.74e-04 |

---

## 3. Scientific Insights & Key Findings

1. **Diffraction Compression vs. Spatial Frequency Penalty:**
   - Moving from CD (780 nm, NA 0.45) to Blu-ray (405 nm, NA 0.85) compresses the Airy spot radius by **3.64×** (from 1057.3 nm to 290.6 nm).
   - However, the minimum recorded mark length was scaled down by **5.59×** (from 833 nm to 149 nm).
   - As a direct consequence of operating closer to the optical cutoff frequency ($2\mathrm{NA}/\lambda$), the incoherent MTF modulation drops from **0.369** in CD to **0.105** in Blu-ray, imposing a **10.9 dB** geometric signal attenuation penalty.

2. **Dominant Parameter Sensitivity:**
   - Across all formats, Numerical Aperture ($\mathrm{NA}$) exhibits the highest positive elasticity on CNR (+3.11 in Blu-ray), demonstrating that optical focusing power dominates detection headroom over raw laser power (+0.50 elasticity).
   - Thermal and detector bandwidth noise parameters exhibit negative elasticities (bandwidth elasticity $\\approx -0.26$), confirming that transimpedance filtering must be tightly matched to format data rates.

---

## 4. Reproducibility
This study is 100% reproducible from clean source code by running:
```bash
python research/run_study.py
```
All intermediate datasets, tables, and figures are automatically re-computed and stored in `research/results/`, `research/tables/`, and `research/figures/`.

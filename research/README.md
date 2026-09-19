# OSIS Reproducible Research Study

## Simulation-Based Analysis of Optical Readout Quality Across Representative CD, DVD, and Blu-ray Configurations

This directory contains a complete, automated, and reproducible computational physics study conducted using the **Optical Storage Intelligence Simulator (OSIS)**.

### Research Question
> *How do wavelength scaling, numerical aperture, and minimum mark spatial frequency dictate the trade-offs between optical spot compression, MTF modulation depth, and optoelectronic Carrier-to-Noise Ratio (CNR) across the three generations of optical disc formats?*

---

## Directory Structure

```
research/
├── README.md               # Study overview and execution instructions
├── run_study.py            # Master reproduction script
├── report.md               # Comprehensive scientific report and conclusions
├── configuration/          # Study input configurations (JSON)
│   └── study_config.json
├── results/                # Machine-readable evaluation outputs
│   ├── baseline_summary.json
│   ├── mark_length_sweep.json
│   └── sensitivity_summary.json
├── tables/                 # Formatted publication tables (Markdown)
│   ├── table1_disc_comparison.md
│   └── table2_sensitivity_rankings.md
└── figures/                # Publication-grade figures (300 DPI PNG)
    ├── study_fig1_mark_length_rolloff.png
    └── study_fig2_elasticity_comparison.png
```

---

## How to Reproduce

In a clean environment with `osis` installed:

```bash
python research/run_study.py
```

Execution takes approximately **1.5 seconds** and will re-evaluate all configurations, output JSON datasets, markdown tables, and multi-panel figures.

---

## Key Findings

1. **Spot Compression vs. Spatial Frequency Penalty:** Spot size shrinks by 3.64× (1057 nm to 291 nm) from CD to Blu-ray, but mark length shrinks by 5.59× (833 nm to 149 nm), reducing MTF modulation from 0.369 to 0.105.
2. **Focusing Power Dominance:** Numerical aperture exhibits the highest positive elasticity (+3.11 for Blu-ray), demonstrating that NA governs detection headroom much more strongly than laser power (+0.50).

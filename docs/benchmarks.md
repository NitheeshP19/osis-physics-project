# Benchmarks and Reproducible Research Study

OSIS includes both micro-benchmarks for execution throughput and a macro research study investigating optical storage channel trade-offs across disc generations.

---

## 1. Reproducible Research Study

The repository contains an automated research study located in [`research/`](file:///c:/Users/dell/Documents/physics/research/):

**Title:** *Simulation-Based Analysis of Optical Readout Quality Across Representative CD, DVD, and Blu-ray Configurations*  
**Research Question:** *How do wavelength scaling, numerical aperture, and minimum mark spatial frequency dictate the trade-offs between optical spot compression, MTF modulation depth, and optoelectronic Carrier-to-Noise Ratio (CNR) across the three generations of optical disc formats?*

### Running the Study

```bash
osis reproduce
# or: python research/run_study.py
```

This reproduces:
1. `research/results/baseline_summary.json`: Multi-format CNR and BER outputs.
2. `research/results/mark_length_sweep.json`: Spatial frequency MTF roll-off data.
3. `research/results/sensitivity_summary.json`: Normalized parameter elasticities.
4. `research/tables/table1_disc_comparison.md`: Formatted markdown comparison table.
5. `research/tables/table2_sensitivity_rankings.md`: Formatted elasticity rankings.
6. `research/figures/study_fig1_mark_length_rolloff.png`: 300 DPI publication figure.
7. `research/figures/study_fig2_elasticity_comparison.png`: 300 DPI publication figure.
8. `research/report.md`: Complete scientific report.

---

## 2. Micro-Benchmarks

### Standard Disc Throughput Benchmark

Measures per-simulation execution latency across presets:

```bash
python benchmarks/run_disc_comparison.py
```

Typical latency on standard x86-64 hardware:
- CD-RW: $\approx 230\ \mu\text{s}$
- DVD-RW: $\approx 205\ \mu\text{s}$
- Blu-ray BD-RE: $\approx 135\ \mu\text{s}$

### Parameter Sensitivity Benchmark

Evaluates local gradients and elasticities across 8 optical and optoelectronic parameters:

```bash
python benchmarks/run_sensitivity.py
```

Numerical Aperture ($\mathrm{NA}$) is identified as the dominant parameter governing CNR (elasticity $\approx +3.11$).

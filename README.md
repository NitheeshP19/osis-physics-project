# OSIS — Optical Storage Intelligence Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python ≥ 3.10](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://python.org)
[![Tests](https://github.com/NitheeshP19/osis-physics-project/actions/workflows/ci.yml/badge.svg)](https://github.com/NitheeshP19/osis-physics-project/actions/workflows/ci.yml)

OSIS is an open-source Python library for **physics-based simulation of optical disc readout channels** (CD, DVD, Blu-ray). It chains four physically rigorous sub-models — thin-film reflectance, scalar diffraction, optoelectronic signal conversion, and noise — to produce Carrier-to-Noise Ratio (CNR) and Bit Error Rate (BER) from first principles.

## Installation

```bash
git clone https://github.com/NitheeshP19/osis-physics-project.git
cd osis-physics-project
pip install -e .
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.

## Quick Start

```python
import osis

# Full Blu-ray readout simulation in one call
result = osis.simulate(osis.BluRayConfig())
print(f"CNR: {result['cnr_db']:.2f} dB   BER: {result['ber']:.2e}")
```

**Output:**
```
CNR: 16.79 dB   BER: 9.11e-04
```

See [`docs/quickstart.md`](docs/quickstart.md) for parameter sweeps and sensitivity analysis.

## Physics Model

OSIS implements:

1. **Abelès Transfer Matrix Method (TMM)** — multilayer thin-film reflectance for crystalline (land) and amorphous (mark) phase-change states (Born & Wolf, 2019)
2. **Rayleigh scalar diffraction** — diffraction-limited spot radius `r = 0.61 λ/NA`
3. **Incoherent MTF** — circular pupil autocorrelation for spatial frequency modulation (Goodman, 2017)
4. **Optoelectronic signal model** — photodetector responsivity, signal photocurrent (Saleh & Teich, 2019)
5. **Three-component noise model** — quantum shot noise, laser RIN, Johnson–Nyquist thermal noise (Johnson, 1928; Petermann, 1988)
6. **CNR/BER** — Carrier-to-Noise Ratio (dB) and Bit Error Rate under OOK-NRZ/AWGN (Proakis & Salehi, 2007)

See [`docs/physics.md`](docs/physics.md) for full equation derivation and [`docs/assumptions.md`](docs/assumptions.md) for explicit scope and limitations.

## Simulation Results

| Format | λ (nm) | NA | CNR (dB) | BER |
|:-------|:-------|:---|:---------|:----|
| CD-RW | 780 | 0.45 | 38.98 | 1.44×10⁻⁹ |
| DVD-RW | 650 | 0.60 | 26.19 | 1.65×10⁻⁴ |
| Blu-ray BD-RE | 405 | 0.85 | 16.79 | 9.11×10⁻⁴ |

**Note:** CNR decreases from CD→BD due to increasing spatial frequency challenge at smaller minimum mark lengths (`L_min`). These values are produced by the physics engine under scalar diffraction assumptions; see [limitations](docs/assumptions.md).

## Analysis Framework

```python
import numpy as np, osis

# Parameter sweep: NA effect on BD CNR
cfg = osis.BluRayConfig()
results = osis.parameter_sweep(cfg, "numerical_aperture", np.linspace(0.75, 0.90, 7))

# Sensitivity analysis
sens = osis.one_at_a_time_sensitivity(cfg, delta_fraction=0.05)
ranked = osis.rank_parameters_by_influence(sens)
# → numerical_aperture: elasticity ≈ +1.37 (dominant parameter)
```

## ML Surrogate (Optional)

An optional gradient-boosted ML surrogate is provided for rapid parameter sweeps. It is trained on OSIS physics-engine outputs and learns multi-layer channel impairments (crosstalk, humidity-induced dye degradation, detector non-linearity) not present in the analytic model:

- RMSE: 0.168 dB | MAE: 0.123 dB | R²: 0.9999
- Latency: ~0.05 ms (surrogate) vs ~0.25 ms (full physics)

```python
from ml.predict import predict_cnr
result = predict_cnr({"wavelength_nm": 405, "numerical_aperture": 0.85, ...})
```

Requires `pip install -e ".[ml]"`. See [`ml/README.md`](ml/) for details.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
# → runs unit, API, and end-to-end integration tests
```

Tests cover TMM accuracy (Fresnel limit, energy conservation, quarter-wave AR coating, metal reflectance), optics, channel signal, SNR metrics, analysis modules, and full end-to-end integration.

## Citing OSIS

If you use OSIS in published research, please cite:

```bibtex
@article{osis2026,
  title   = {{OSIS}: Optical Storage Intelligence Simulator --- A Physics-Based Python Framework for Optical Disc Readout Channel Modeling},
  author  = {P., Nitheesh},
  journal = {Journal of Open Source Software},
  year    = {2026},
  note    = {Submitted}
}
```

See [`CITATION.cff`](CITATION.cff) for full citation metadata.

## License

MIT License — see [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests are welcome. Please run `pytest tests/` before submitting. See [open issues](https://github.com/NitheeshP19/osis-physics-project/issues).

## AI Disclosure

Generative AI tools were used as coding assistants to scaffold module structures and docstrings during development. All physical equations, parameter values, citations, and design decisions were independently verified by the author against primary literature.

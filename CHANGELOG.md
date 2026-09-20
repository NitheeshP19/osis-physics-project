# Changelog

All notable changes to OSIS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
OSIS uses [Semantic Versioning](https://semver.org/).

---

## [1.0.1] — 2026-09-19

### Fixed
- Made the `dev` extra self-contained for the full API and physics test suite.
- Rebuilt the bundled surrogate model artefacts and constrained their dependencies to versions supporting Python 3.10–3.13.
- Made the CLI's research-study loader namespaced so it cannot be shadowed by an unrelated installed `research` package.
- **Python 3.10 CI fix:** `osis_explainer.pkl` contains Numba-JIT compiled SHAP internals whose pickle format depends on the Python code-object layout (18 args in 3.11+, 16 in 3.10). The explainer is now loaded in an isolated `try/except` so a version mismatch degrades gracefully (feature-importance explanations become unavailable) rather than crashing app startup. All 45 tests now pass on Python 3.10 and 3.11–3.13.
- Removed accidentally committed `.python310-runtime/` interpreter directory from version control; added to `.gitignore`.
- Fixed incorrect `tmm` package citation in `paper/paper.md` (was citing Saleh & Teich textbook; corrected to Byrnes 2016, arXiv:1603.02720).
- Fixed broken ML surrogate import example in `README.md`.
- Added ORCID `0009-0005-1003-3695` to `paper/paper.md` and `CITATION.cff`.
- Removed stray root-level development test scripts and internal planning documents from version control.

### Changed (paper)
- Corrected JOSS paper section heading from `# Software Design and Physical Architecture` to the required `# Software design`.
- Expanded `# State of the Field` with per-tool analysis for `tmm`, POPPY, and LightPipes; added explicit build-vs-contribute argument; removed unsupported absolute claims.
- Added `@Byrnes2016` (arXiv:1603.02720), `@perrin2012` (POPPY/SPIE), and `@tovar1996` (LightPipes/JOSA) BibTeX entries to `paper/paper.bib`.

---

## [1.0.0] — 2026-09-19

### Added
- `src/osis/` installable Python library, importable independently of the web API.
- `src/osis/physics/tmm.py`: Abeles Transfer Matrix Method for N-layer thin-film stacks with complex refractive indices.
- `src/osis/physics/optics.py`: Scalar diffraction spot model, Optical Transfer Function (OTF/MTF) computation.
- `src/osis/physics/channel.py`: 1D readout signal convolution, shot noise, thermal noise, and RIN noise models.
- `src/osis/physics/snr.py`: Carrier-to-Noise Ratio (CNR) and BER estimation via the Q-factor/erfc formulation.
- `src/osis/configs.py`: Literature-sourced configuration presets for CD, DVD, and Blu-ray disc standards.
- `src/osis/analysis/sweep.py`: Parameter sweep framework over any configuration field.
- `src/osis/analysis/sensitivity.py`: One-at-a-time and finite-difference sensitivity analysis.
- `tests/unit/`: Unit tests for TMM, optics, SNR, and channel modules with analytical validation cases.
- `tests/integration/`: End-to-end pipeline tests for all three disc configurations.
- `benchmarks/`: Reproducible parameter-sweep and sensitivity benchmark scripts.
- `examples/`: Standalone Python scripts demonstrating core workflows.
- `paper/paper.md` and `paper/paper.bib`: JOSS submission manuscript.
- `docs/`: MkDocs documentation with physics model descriptions, assumptions, and limitations.
- `pyproject.toml`: Standard packaging configuration.
- `LICENSE`: MIT license.
- `CITATION.cff`: Machine-readable citation metadata.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`: Community guidelines.

### Changed
- Replaced the heuristic polynomial SNR formula with a physically-grounded TMM + scalar diffraction + noise model pipeline.
- ML surrogate training target is now derived from the rigorous physics engine (not from the training formula itself).
- FastAPI web application updated to delegate all physics computation to `src/osis/` library.

### Removed
- Arbitrary hand-written SNR formula (`85 + 30*NA - 0.02*wavelength + ...`) — replaced with physical model.
- Circular ML training target (where "measured SNR" was derived from the same formula the model trained on).

---

## [0.3.0] — 2026-04-15

### Added
- FastAPI refactoring into modular `app/` package structure.
- Advanced simulation platform endpoint (`/api/v1/simulate_platform`).
- Manufacturing yield Monte Carlo module.

---

## [0.2.0] — 2026-03-03

### Added
- Optuna Bayesian hyperparameter optimization for ML training.
- Stacking ensemble with quantile regression uncertainty bounds.
- SHAP explainability integration.
- Lenis scroll UI for the web frontend.

---

## [0.1.0] — 2026-02-23

### Added
- Initial OSIS project with FastAPI backend, static frontend.
- Synthetic dataset generation and Random Forest/Gradient Boosting SNR model.
- Physics parameter inputs: wavelength, NA, track pitch, ISI, crosstalk, material, temperature, humidity.

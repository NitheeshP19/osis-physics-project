# OSIS Documentation

## Installation

```bash
git clone https://github.com/NitheeshP19/osis-physics-project.git
cd osis-physics-project
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.10
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0

Optional:
- scikit-learn ≥ 1.7, < 1.8 (ML surrogate; required for the bundled model artefacts)
- fastapi, uvicorn (web API)

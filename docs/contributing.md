# Contributing to OSIS

We welcome contributions from the scientific computing, photonics, and optical storage communities.

## Development Setup

```bash
git clone https://github.com/NitheeshP19/osis-physics-project.git
cd osis-physics-project
pip install -e ".[dev]"
```

## Running Tests and Validation

Before submitting any code changes, ensure all tests and validation suites pass:

```bash
# Unit and integration test suite
pytest tests/ -v

# Physical validation suite (Levels 1, 2, and 3)
osis validate
# or: python benchmarks/validation/run_all_validations.py
```

## Guidelines for Physics Contributions

1. **Cite Primary Sources:** Every new equation or constant must cite a peer-reviewed paper, standard, or textbook.
2. **Include Limiting-Case Tests:** Any added physical model must be accompanied by unit tests verifying analytical or limiting-case behavior.
3. **Document Scope & Limitations:** Explicitly document the assumptions and limits of validity in module docstrings and `docs/limitations.md`.

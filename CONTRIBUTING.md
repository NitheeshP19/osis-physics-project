# Contributing to OSIS

Thank you for your interest in contributing to the Optical Storage Intelligence Simulator.

## Ways to Contribute

- **Bug reports** — Open a GitHub Issue describing the bug, steps to reproduce it, and your environment.
- **Feature requests** — Open a GitHub Issue with the label `enhancement`. Describe the scientific motivation.
- **Documentation improvements** — Submit a pull request against the `docs/` directory.
- **Bug fixes and code improvements** — Submit a pull request. All changes require tests.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/NitheeshP19/osis-physics-project.git
cd osis-physics-project

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  (Linux/macOS)

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Run the tests
python -m pytest tests/ -v
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Scientific Contributions

All contributions to the physics modules must:

1. Cite the source equation or algorithm (paper, textbook, or standard).
2. Include a unit test against an independently verifiable result (analytical case, limiting case, or independent implementation).
3. Document assumptions and limitations in the module docstring.
4. Not introduce unverifiable "magic constants" without provenance.

## Pull Request Checklist

- [ ] Tests pass (`python -m pytest tests/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] New physics functions have docstrings with equation references and units.
- [ ] New features have at least one unit test.
- [ ] CHANGELOG.md entry added (Unreleased section).

## Reporting Bugs

Use the GitHub issue tracker. Include:
- OSIS version (`python -c "import osis; print(osis.__version__)"`)
- Python version
- Operating system
- Minimal reproducible example
- Observed vs. expected output

## Support

For questions about usage, open a GitHub Discussion or Issue. For questions about the physics models, cite the relevant section of `docs/physics.md`.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive.

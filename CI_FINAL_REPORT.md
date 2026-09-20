# CI Final Report

## Root Cause

The CI `dev` extra did not contain all dependencies required by the tests it
runs. In particular, API tests required FastAPI's test-client runtime and the
serialized ML artefacts required SHAP and a compatible scikit-learn release.
The CLI also used a generic `research` import that can be shadowed by a third
party package.

The first remote fix also showed that the latest scikit-learn, SHAP, and
Matplotlib releases had dropped Python 3.10. The bundled ML artefacts were
therefore rebuilt from the existing dataset with scikit-learn 1.7.2 and SHAP
0.49.1, the latest compatible release series.

## Environments Tested

| Environment | Result |
|---|---|
| Python 3.10 Ubuntu | Not remotely verified |
| Python 3.11 Ubuntu | Not remotely verified |
| Python 3.12 Ubuntu | Not remotely verified |
| Python 3.13 Ubuntu | Not remotely verified |
| Python 3.10 Windows | Not remotely verified |
| Python 3.11 Windows | Not remotely verified |
| Python 3.12 Windows | Not remotely verified |
| Python 3.13 Windows | PASS locally in a clean virtual environment |

Remote verification is pending because this workspace has no GitHub CLI and
cannot access the GitHub Actions API/logs. No remote result is claimed here.

## Tests

- 45 passed, 0 failed on Python 3.13 / Windows.
- Coverage: 92% for `src/osis` (365/395 statements covered).
- The run used the workflow-equivalent editable install with the `dev` extra.

## Package and CLI

- Built and installed `osis` from the project successfully.
- `import osis` reported version `1.0.0`.
- `osis validate` completed all analytical, numerical, and domain validation levels.
- `osis reproduce` completed and regenerated the research study outputs.
- `paper/generate_figures.py` completed successfully.
- Notebook execution is not part of the CI workflow and was not run because no notebook execution dependency is declared.

## Changes

- `pyproject.toml`: declares all test runtime dependencies and pins Python 3.10–3.13-compatible model dependencies.
- The bundled surrogate model, interval models, feature list, explainer, and SHAP background were regenerated from the existing dataset using scikit-learn 1.7.2 and SHAP 0.49.1.
- `.github/workflows/ci.yml`: invokes pip and pytest via the selected matrix Python interpreter.
- `src/osis/research.py` and `src/osis/cli.py`: use a namespaced, repository-local research loader.
- `tests/unit/test_cli.py`: retains the CLI reproduction assertion against the namespaced loader.
- `CHANGELOG.md`, `README.md`, and `docs/installation.md`: document the corrected dependency contract.

## Remaining Risks

- The eight remote GitHub Actions jobs require a push and remote workflow run before they can be reported as green.
- The serialized surrogate artefacts intentionally require scikit-learn 1.7.x; future model retraining should update this bound and test all supported Python versions.

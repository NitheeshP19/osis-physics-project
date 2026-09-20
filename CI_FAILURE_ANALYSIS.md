# CI Failure Analysis

## Evidence gathered

The workflow is an eight-job matrix: Ubuntu and Windows for Python 3.10, 3.11,
3.12, and 3.13. The GitHub CLI is not installed in this workspace, and the
GitHub Actions log/API endpoints could not be reached from this environment, so
remote job logs were not available for direct inspection.

An isolated Python 3.13 Windows environment reproduced the workflow's
`python -m pip install -e ".[dev]"` installation and then exposed the following
failures in order:

| Environment | Failed step | First meaningful error | Root cause | Fix |
|---|---|---|---|---|
| Clean Windows / Python 3.13 | Test collection | `ModuleNotFoundError: No module named 'fastapi'` | The full suite includes FastAPI integration tests, but `dev` did not declare their runtime dependencies. | Declare the API test dependencies in `dev`. |
| Clean Windows / Python 3.13 | API test client setup | `No module named '_loss'` | The bundled model artefacts were produced by scikit-learn 1.8.0, while `>=1.3` resolved 1.9.1. Pickled scikit-learn artefacts are not minor-version portable. | Constrain scikit-learn to `>=1.8,<1.9`. |
| Clean Windows / Python 3.13 | API application startup | `No module named 'shap'` | The application loads a serialized SHAP explainer during its startup lifecycle, but the test extra did not include SHAP. | Declare SHAP in `dev`. |
| Clean Windows / Python 3.13 | CLI unit test | An installed `research` module shadowed the repository study script. | The CLI imported an overly generic top-level package name. | Load the repository study through `osis.research`, using a private file-module name. |

The first missing FastAPI dependency is sufficient to fail every matrix job at
test collection. The later failures were found only after fixing each preceding
one in a clean environment.

## Follow-up remote matrix finding

The first pushed fix made Python 3.11–3.13 jobs pass but both Python 3.10 jobs
failed during dependency installation. Published package metadata identified the
cause: scikit-learn 1.8, SHAP 0.52, and Matplotlib 3.11 no longer support Python
3.10. The model artefacts were rebuilt from the existing dataset with
scikit-learn 1.7.2 and SHAP 0.49.1, and the dependency ranges now select the
latest releases that support Python 3.10–3.13.

## Scope

No platform-specific path, shell, line-ending, or scientific-calculation defect
was found. The corrective changes are dependency metadata and import isolation.

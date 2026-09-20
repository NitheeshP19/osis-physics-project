"""Load the repository's reproducible study without a top-level name collision."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def run_full_study() -> None:
    """Run the bundled research study from its repository-local source file."""
    study_path = Path(__file__).resolve().parents[2] / "research" / "run_study.py"
    spec = importlib.util.spec_from_file_location("_osis_research_study", study_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load research study from {study_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_full_study()

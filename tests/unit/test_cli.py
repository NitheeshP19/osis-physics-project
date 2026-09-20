"""Unit tests for the OSIS CLI entry point."""

from unittest.mock import patch
import pytest

from osis.cli import main


def test_cli_version(capsys: pytest.CaptureFixture) -> None:
    """Verify osis --version prints version string."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_cli_simulate(capsys: pytest.CaptureFixture) -> None:
    """Verify osis simulate executes preset simulation."""
    code = main(["simulate", "--preset", "bluray"])
    assert code == 0
    captured = capsys.readouterr()
    assert "OSIS Simulation: Blu-ray BD-RE" in captured.out
    assert "CNR:" in captured.out


def test_cli_simulate_cd(capsys: pytest.CaptureFixture) -> None:
    """Verify osis simulate executes CD preset."""
    code = main(["simulate", "--preset", "cd"])
    assert code == 0
    captured = capsys.readouterr()
    assert "OSIS Simulation: CD" in captured.out



def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Verify running osis with no arguments displays help."""
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "OSIS: Optical Storage Intelligence Simulator" in captured.out


def test_cli_validate_mocked() -> None:
    """Verify osis validate invokes validation runner."""
    with patch("benchmarks.validation.run_all_validations.run_all") as mock_run:
        code = main(["validate"])
        assert code == 0
        mock_run.assert_called_once()


def test_cli_reproduce_mocked() -> None:
    """Verify osis reproduce invokes research study runner."""
    with patch("osis.research.run_full_study") as mock_run:
        code = main(["reproduce"])
        assert code == 0
        mock_run.assert_called_once()

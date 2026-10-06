"""Entry-script error handling: a handled failure prints Python's full
error first and ends on one plain line saying what was written and what to
rerun, and a failure the script was not written to expect is not caught."""

from __future__ import annotations

import importlib.util
import sys
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"


def _load_script(name: str) -> ModuleType:
    """Import one entry script as a module so its main() can be called."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _last_line(text: str) -> str:
    return text.strip().splitlines()[-1]


class TestFetchHillstrom:
    def test_failed_download_shows_the_error_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetch = _load_script("run_fetch_hillstrom")
        destination = tmp_path / "hillstrom" / "hillstrom.csv"
        monkeypatch.setattr(fetch, "DEST", destination)

        def refuse_connection(*args: object, **kwargs: object) -> None:
            raise OSError("simulated network failure")

        monkeypatch.setattr(urllib.request, "urlopen", refuse_connection)
        assert fetch.main() == 2
        stderr = capsys.readouterr().err
        assert "Traceback" in stderr
        assert "simulated network failure" in stderr
        assert _last_line(stderr).startswith("error: the download failed, so nothing was written")
        assert list(destination.parent.iterdir()) == []


class TestAnalyzeExperiment:
    def test_missing_data_file_shows_the_error_then_what_to_do(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")
        config = tmp_path / "config.toml"
        config.write_text(
            (FIXTURES / "aggregates_config.toml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_analyze_experiment.py", "--config", str(config), "--out", str(out_dir)],
        )
        assert analyze.main() == 2
        stderr = capsys.readouterr().err
        assert "Traceback" in stderr
        assert "data file not found" in stderr
        assert _last_line(stderr).startswith("error: the analysis stopped before report.md")
        assert not out_dir.exists()

    def test_missing_matplotlib_names_the_install_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")

        def matplotlib_missing(path: Path) -> dict:
            raise ImportError("simulated", name="matplotlib.pyplot")

        monkeypatch.setattr(analyze, "load_config", matplotlib_missing)
        monkeypatch.setattr(
            sys, "argv", ["run_analyze_experiment.py", "--config", str(tmp_path / "unused.toml")]
        )
        assert analyze.main() == 2
        stderr = capsys.readouterr().err
        assert "Traceback" in stderr
        assert 'pip install -e ".[dev]"' in _last_line(stderr)

    def test_other_import_errors_are_not_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        analyze = _load_script("run_analyze_experiment")

        def broken_import(path: Path) -> dict:
            raise ImportError("simulated broken install", name="scipy")

        monkeypatch.setattr(analyze, "load_config", broken_import)
        monkeypatch.setattr(
            sys, "argv", ["run_analyze_experiment.py", "--config", str(tmp_path / "unused.toml")]
        )
        with pytest.raises(ImportError, match="simulated broken install"):
            analyze.main()

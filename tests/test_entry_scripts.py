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
        monkeypatch.setattr(sys, "argv", ["run_fetch_hillstrom.py"])
        with pytest.raises(SystemExit) as stopped:
            fetch.main()
        assert stopped.value.code == 2
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
            (FIXTURES / "experiment_ok.toml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_analyze_experiment.py", "--config", str(config), "--out", str(out_dir)],
        )
        with pytest.raises(SystemExit) as stopped:
            analyze.main()
        assert stopped.value.code == 2
        stderr = capsys.readouterr().err
        assert "Traceback" in stderr
        assert "data file not found" in stderr
        assert _last_line(stderr).startswith("error: the analysis stopped before readout.txt")
        assert "run_fetch_hillstrom" not in _last_line(stderr)
        assert not out_dir.exists()

    def test_missing_fetched_data_file_ends_on_the_fetch_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")
        config_text = (FIXTURES / "experiment_ok.toml").read_text(encoding="utf-8")
        assert "hillstrom" not in str(tmp_path).lower()
        config_text = config_text.replace(
            'path = "user_rows_ok.csv"', 'path = "data/hillstrom/hillstrom.csv"'
        )
        assert "data/hillstrom/hillstrom.csv" in config_text
        config = tmp_path / "config.toml"
        config.write_text(config_text, encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_analyze_experiment.py", "--config", str(config), "--out", str(tmp_path / "out")],
        )
        with pytest.raises(SystemExit) as stopped:
            analyze.main()
        assert stopped.value.code == 2
        last = _last_line(capsys.readouterr().err)
        assert last.startswith("error: the analysis stopped before readout.txt")
        assert last.endswith("run scripts/run_fetch_hillstrom.py first.")

    def test_a_mistyped_config_path_gets_no_fetch_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")
        missing_config = tmp_path / "analysis" / "hillstrom" / "confg.toml"
        monkeypatch.setattr(
            sys, "argv", ["run_analyze_experiment.py", "--config", str(missing_config)]
        )
        with pytest.raises(SystemExit) as stopped:
            analyze.main()
        assert stopped.value.code == 2
        last = _last_line(capsys.readouterr().err)
        assert last.startswith("error: the analysis stopped before readout.txt")
        assert "run_fetch_hillstrom" not in last

    def test_user_rows_fixture_writes_the_readout_and_its_figures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")
        for name in ("experiment_ok.toml", "user_rows_ok.csv"):
            (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_analyze_experiment.py",
                "--config",
                str(tmp_path / "experiment_ok.toml"),
                "--out",
                str(out_dir),
            ],
        )
        analyze.main()
        printed = capsys.readouterr().out
        readout = (out_dir / "readout.txt").read_text(encoding="utf-8")
        assert readout.startswith("fixture-experiment\n")
        assert readout in printed
        assert "Decision for" in readout
        assert sorted(path.name for path in (out_dir / "figures").iterdir()) == ["monitoring.png"]

    def test_md5_mismatch_stops_before_reading_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")
        (tmp_path / "user_rows_ok.csv").write_bytes((FIXTURES / "user_rows_ok.csv").read_bytes())
        config_text = (FIXTURES / "experiment_ok.toml").read_text(encoding="utf-8")
        config_text = config_text.replace(
            'path = "user_rows_ok.csv"', 'path = "user_rows_ok.csv"\nmd5 = "' + "0" * 32 + '"'
        )
        config = tmp_path / "config.toml"
        config.write_text(config_text, encoding="utf-8")
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_analyze_experiment.py", "--config", str(config), "--out", str(out_dir)],
        )
        with pytest.raises(SystemExit) as stopped:
            analyze.main()
        assert stopped.value.code == 2
        stderr = capsys.readouterr().err
        assert "but the config pins" in stderr
        assert not out_dir.exists()

    def test_missing_matplotlib_names_the_install_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        analyze = _load_script("run_analyze_experiment")

        def matplotlib_missing(*args: object) -> list[str]:
            raise ImportError("simulated", name="matplotlib.pyplot")

        monkeypatch.setattr(analyze, "write_experiment_readout", matplotlib_missing)
        monkeypatch.setattr(
            sys, "argv", ["run_analyze_experiment.py", "--config", str(tmp_path / "unused.toml")]
        )
        with pytest.raises(SystemExit) as stopped:
            analyze.main()
        assert stopped.value.code == 2
        stderr = capsys.readouterr().err
        assert "Traceback" in stderr
        assert "pip install -r requirements.txt" in _last_line(stderr)

    def test_other_import_errors_are_not_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        analyze = _load_script("run_analyze_experiment")

        def broken_import(*args: object) -> list[str]:
            raise ImportError("simulated broken install", name="scipy")

        monkeypatch.setattr(analyze, "write_experiment_readout", broken_import)
        monkeypatch.setattr(
            sys, "argv", ["run_analyze_experiment.py", "--config", str(tmp_path / "unused.toml")]
        )
        with pytest.raises(ImportError, match="simulated broken install"):
            analyze.main()


class TestPlanExperiment:
    def test_rate_plan_matches_the_library_function(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from abkit.power_analysis import sample_size_two_proportions

        plan = _load_script("run_plan_experiment")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_plan_experiment.py", "--baseline-rate", "0.05", "--minimum-lift", "0.005"],
        )
        plan.main()
        printed = capsys.readouterr().out
        per_group = sample_size_two_proportions(0.05, 0.055)
        assert f"per group   {per_group:,}   total   {2 * per_group:,}" in printed
        assert "smallest lift worth detecting +0.50 pp (10.0% of baseline)" in printed
        assert printed.strip().splitlines()[-1].startswith("Decision: run until each group has")

    def test_mean_plan_matches_the_library_function(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from abkit.power_analysis import sample_size_two_means

        plan = _load_script("run_plan_experiment")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_plan_experiment.py",
                "--standard-deviation",
                "12",
                "--minimum-difference",
                "0.5",
                "--power",
                "0.9",
            ],
        )
        plan.main()
        printed = capsys.readouterr().out
        per_group = sample_size_two_means(0.5, 12.0, power=0.9)
        assert f"per group   {per_group:,}" in printed
        assert "90% power" in printed

    def test_bad_input_shows_the_error_then_one_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        plan = _load_script("run_plan_experiment")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_plan_experiment.py", "--baseline-rate", "1.5", "--minimum-lift", "0.01"],
        )
        with pytest.raises(SystemExit) as stopped:
            plan.main()
        assert stopped.value.code == 2
        stderr = capsys.readouterr().err
        assert "Traceback" in stderr
        assert _last_line(stderr).startswith("error: nothing was planned")

    def test_mixing_the_two_modes_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        plan = _load_script("run_plan_experiment")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_plan_experiment.py", "--baseline-rate", "0.05", "--standard-deviation", "2"],
        )
        with pytest.raises(SystemExit) as stopped:
            plan.main()
        assert stopped.value.code == 2
        assert "not both" in capsys.readouterr().err

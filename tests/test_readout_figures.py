"""Figure tests: the right files are drawn for what the readout holds, and the
readout lists them."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from abkit.experiment_intake import load_config, load_user_rows
from abkit.readout_analysis import analyze_experiment
from abkit.readout_figures import draw_figures
from abkit.readout_text import readout_lines

FIXTURES = Path(__file__).parent / "fixtures"


def _user_rows_experiment(tmp_path: Path):
    text = (
        '[experiment]\nname = "t"\nalpha = 0.05\n\n'
        '[input]\npath = "user_rows_ok.csv"\n\n'
        '[columns]\ngroup = "group"\nday = "day"\nuser_id = "user_id"\n\n'
        '[groups]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n\n'
        '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\nprimary = true\n\n'
        '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\nnonnegative = true\n\n'
        '[[metrics]]\nname = "converted_again"\ncolumn = "converted"\nkind = "binary"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    return load_user_rows(load_config(config_path), FIXTURES / "user_rows_ok.csv")


def test_only_a_synthetic_run_is_labeled_on_the_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    titles: list[str] = []
    monkeypatch.setattr(Figure, "suptitle", lambda _, title, **__: titles.append(title))
    readout = analyze_experiment(_user_rows_experiment(tmp_path))
    draw_figures(readout, tmp_path / "real")
    assert titles == []
    draw_figures(dataclasses.replace(readout, truth={"seed": 1}), tmp_path / "synthetic")
    assert titles == ["Synthetic check: generated data, not a real experiment"]


def test_both_figures_are_drawn_and_the_readout_lists_them(tmp_path: Path) -> None:
    readout = analyze_experiment(_user_rows_experiment(tmp_path))
    output_dir = tmp_path / "out"
    paths = draw_figures(readout, output_dir / "figures")
    assert [path.name for path in paths] == ["corrections.png", "monitoring.png"]
    assert all(path.stat().st_size > 1000 for path in paths)
    lines = readout_lines(readout, output_dir, paths)
    assert lines[-2:] == ["  figures/corrections.png", "  figures/monitoring.png"]

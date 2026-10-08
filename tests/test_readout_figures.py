"""Figure tests: the right files are drawn for each data shape, and the
readout lists them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from abkit.io import load_aggregates, load_config, load_user_rows
from abkit.readout_analysis import analyze_experiment
from abkit.readout_figures import draw_figures
from abkit.readout_text import readout_lines

FIXTURES = Path(__file__).parent / "fixtures"


def _user_rows_experiment(tmp_path: Path):
    text = (
        '[experiment]\nname = "t"\nalpha = 0.05\n\n'
        '[input]\nshape = "user_rows"\npath = "user_rows_ok.csv"\n\n'
        '[columns]\narm = "group"\nday = "day"\nuser_id = "user_id"\n\n'
        '[arms]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n\n'
        '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\nprimary = true\n\n'
        '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\nnonnegative = true\n\n'
        '[[metrics]]\nname = "converted_again"\ncolumn = "converted"\nkind = "binary"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    return load_user_rows(load_config(config_path), FIXTURES / "user_rows_ok.csv")


def test_user_rows_draw_all_four_figures_and_the_readout_lists_them(tmp_path: Path) -> None:
    experiment = _user_rows_experiment(tmp_path)
    readout = analyze_experiment(experiment, np.random.default_rng(0))
    output_dir = tmp_path / "out"
    paths = draw_figures(readout, experiment, output_dir / "figures")
    assert [path.name for path in paths] == [
        "corrections.png",
        "revenue.png",
        "early_late.png",
        "monitoring.png",
    ]
    assert all(path.stat().st_size > 1000 for path in paths)
    lines = readout_lines(readout, output_dir, paths)
    assert lines[-4:] == [
        "  figures/corrections.png",
        "  figures/revenue.png",
        "  figures/early_late.png",
        "  figures/monitoring.png",
    ]


def test_aggregates_draw_only_the_time_figures(tmp_path: Path) -> None:
    config = load_config(FIXTURES / "aggregates_config.toml")
    experiment = load_aggregates(config, FIXTURES / "aggregates_ok.csv")
    readout = analyze_experiment(experiment, np.random.default_rng(0))
    paths = draw_figures(readout, experiment, tmp_path / "figures")
    assert [path.name for path in paths] == ["early_late.png", "monitoring.png"]

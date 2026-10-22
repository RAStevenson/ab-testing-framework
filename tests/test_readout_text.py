"""Readout text tests: golden files for both fixture shapes, the line rules
that keep prose out, and the decision line's order of precedence."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from abkit.io import load_config, load_user_rows
from abkit.readout_analysis import Limitation, analyze_experiment
from abkit.readout_text import readout_lines, save_readout

FIXTURES = Path(__file__).parent / "fixtures"
MAX_WORDS_PER_SENTENCE = 20


def _user_rows_lines(tmp_path: Path) -> list[str]:
    text = (
        '[experiment]\nname = "fixture experiment"\nalpha = 0.05\n\n'
        '[input]\npath = "user_rows_ok.csv"\n\n'
        '[columns]\narm = "group"\nday = "day"\nuser_id = "user_id"\n\n'
        '[arms]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n\n'
        '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\nprimary = true\n\n'
        '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\nnonnegative = true\n\n'
        '[[metrics]]\nname = "converted_again"\ncolumn = "converted"\nkind = "binary"\n\n'
        "[truth]\nbase_rate = 0.27\nabsolute_lift = 0.13\n"
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    experiment = load_user_rows(load_config(config_path), FIXTURES / "user_rows_ok.csv")
    readout = analyze_experiment(experiment)
    figures = [tmp_path / "out" / "figures" / "monitoring.png"]
    return readout_lines(readout, tmp_path / "out", figures)


def _assert_matches_golden(lines: list[str], name: str) -> None:
    golden = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    assert lines == golden


def test_readout_matches_its_golden_file(tmp_path: Path) -> None:
    _assert_matches_golden(_user_rows_lines(tmp_path), "readout_user_rows.txt")


def test_lines_stay_short_and_prose_free(tmp_path: Path) -> None:
    for line in _user_rows_lines(tmp_path):
        assert len(line) <= 100, line
        for sentence in re.split(r"[.;]\s+", line):
            assert len(sentence.split()) <= MAX_WORDS_PER_SENTENCE, line


def test_user_rows_readout_has_every_block_in_order(tmp_path: Path) -> None:
    lines = _user_rows_lines(tmp_path)
    headings = [
        "Data:",
        "Limitations (*):",
        "Group split:",
        "Sensitivity:",
        "Primary converted:",
        "Other metrics:",
        "Monitoring replay:",
        "Planted versus found",
        "Decision for",
        "Figures:",
    ]
    positions = [
        next(i for i, line in enumerate(lines) if line.startswith(heading)) for heading in headings
    ]
    assert positions == sorted(positions)
    assert lines[-1] == "  figures/monitoring.png"
    assert "  * " in "\n".join(lines)  # the fixture is tiny, so flags are raised


def test_decision_prefers_a_broken_split(tmp_path: Path) -> None:
    experiment = load_user_rows(
        load_config(FIXTURES / "experiment_ok.toml"), FIXTURES / "user_rows_ok.csv"
    )
    readout = analyze_experiment(experiment)
    broken = dataclasses.replace(
        readout, limitations=(Limitation("group_split", "the split is off"), *readout.limitations)
    )
    lines = readout_lines(broken, tmp_path, [])
    assert lines[-1] == "Decision: do not act; the group split is broken"
    assert "  * the split is off" in lines


def test_save_readout_writes_the_same_lines_with_unix_endings(tmp_path: Path) -> None:
    experiment = load_user_rows(
        load_config(FIXTURES / "experiment_ok.toml"), FIXTURES / "user_rows_ok.csv"
    )
    readout = analyze_experiment(experiment)
    lines = save_readout(readout, tmp_path / "out", [])
    raw = (tmp_path / "out" / "readout.txt").read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8") == "\n".join(lines) + "\n"


def test_several_treatments_get_a_decision_and_sensitivity_each(tmp_path: Path) -> None:
    text = (
        '[experiment]\nname = "hillstrom-fixture"\nalpha = 0.05\n\n'
        ""
        '[columns]\narm = "segment"\n\n'
        '[columns.aliases.zip_code]\n"Surburban" = "Suburban"\n\n'
        '[arms]\nlabels = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]\n'
        "intended_split = [0.3334, 0.3333, 0.3333]\n\n"
        '[[metrics]]\nname = "visit"\nkind = "binary"\nprimary = true\n\n'
        '[[metrics]]\nname = "spend"\nkind = "continuous"\nnonnegative = true\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    experiment = load_user_rows(load_config(config_path), FIXTURES / "hillstrom_shaped.csv")
    lines = readout_lines(analyze_experiment(experiment), tmp_path, [])
    decisions = [line for line in lines if line.startswith("Decision for")]
    assert [line.split(":")[0] for line in decisions] == [
        "Decision for Mens E-Mail vs No E-Mail",
        "Decision for Womens E-Mail vs No E-Mail",
    ]
    assert (
        "  Holm-corrected across the 2 treatments; detection judged on the Holm p; "
        "intervals unadjusted"
    ) in lines
    sensitivity = lines.index("Sensitivity: smallest lift this sample could detect, 80% power")
    assert (
        lines[sensitivity + 1]
        == "  each at alpha 0.05/2, the strictest level the Holm correction applies"
    )
    block = lines[sensitivity + 2 : lines.index("", sensitivity)]
    assert [line.split("   ")[0] for line in block if not line.startswith("    ")] == [
        "  Mens E-Mail vs No E-Mail",
        "  Womens E-Mail vs No E-Mail",
    ]

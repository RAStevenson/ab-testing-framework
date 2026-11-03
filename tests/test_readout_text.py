"""Readout text tests: golden files for both fixture shapes, the line rules
that keep prose out, and the decision line's order of precedence."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from abkit.experiment_intake import load_config, load_user_rows
from abkit.readout_analysis import Limitation, analyze_experiment
from abkit.readout_text import readout_lines, save_readout

FIXTURES = Path(__file__).parent / "fixtures"
MAX_WORDS_PER_SENTENCE = 20


def _user_rows_lines(tmp_path: Path, planned_users_per_group: int | None = None) -> list[str]:
    plan = (
        f"planned_users_per_group = {planned_users_per_group}\n"
        if planned_users_per_group is not None
        else ""
    )
    text = (
        f'[experiment]\nname = "fixture experiment"\nalpha = 0.05\n{plan}\n'
        '[input]\npath = "user_rows_ok.csv"\n\n'
        '[columns]\ngroup = "group"\nday = "day"\nuser_id = "user_id"\n\n'
        '[groups]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n\n'
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
        "Primary metric, converted:",
        "Other metrics:",
        "Monitoring replay:",
        "Planted versus found",
        "Synthetic check output for",
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


def test_a_sample_too_small_for_any_lift_says_so_on_the_decision_line(tmp_path: Path) -> None:
    # the README's four-row example: no lift is reachable at 80% power
    (tmp_path / "rows.csv").write_text(
        "variant,converted,spend\ncontrol,0,0\ncontrol,1,12.50\nnew_page,1,49.99\nnew_page,0,0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[experiment]\nname = "tiny"\n\n[input]\npath = "rows.csv"\n\n'
        '[columns]\ngroup = "variant"\n\n[groups]\nlabels = ["control", "new_page"]\n\n'
        '[[metrics]]\nname = "converted"\nkind = "binary"\nprimary = true\n',
        encoding="utf-8",
    )
    readout = analyze_experiment(load_user_rows(load_config(config_path), tmp_path / "rows.csv"))
    lines = readout_lines(readout, tmp_path, [])
    assert lines[-1] == (
        "Decision for new_page vs control: no detectable effect; "
        "no lift was detectable at this size"
    )


def test_save_readout_writes_the_same_lines_with_unix_endings(tmp_path: Path) -> None:
    experiment = load_user_rows(
        load_config(FIXTURES / "experiment_ok.toml"), FIXTURES / "user_rows_ok.csv"
    )
    readout = analyze_experiment(experiment)
    lines = save_readout(readout, tmp_path / "out", [])
    raw = (tmp_path / "out" / "readout.txt").read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8") == "\n".join(lines) + "\n"


def test_a_synthetic_run_prints_check_output_never_decisions(tmp_path: Path) -> None:
    lines = _user_rows_lines(tmp_path)
    assert not any(line.startswith("Decision") for line in lines)
    assert any(line.startswith("Synthetic check output for") for line in lines)


def test_several_treatments_get_a_decision_and_sensitivity_each(tmp_path: Path) -> None:
    text = (
        '[experiment]\nname = "hillstrom-fixture"\nalpha = 0.05\n\n'
        ""
        '[columns]\ngroup = "segment"\n\n'
        '[columns.aliases.zip_code]\n"Surburban" = "Suburban"\n\n'
        '[groups]\nlabels = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]\n'
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


def test_a_run_in_flight_matches_its_golden_file(tmp_path: Path) -> None:
    # 15 users per arm reached against a plan of 40 per group: 38% of the plan
    _assert_matches_golden(
        _user_rows_lines(tmp_path, planned_users_per_group=40), "readout_user_rows_in_flight.txt"
    )


def test_a_run_in_flight_reads_its_decision_from_the_boundary(tmp_path: Path) -> None:
    lines = _user_rows_lines(tmp_path, planned_users_per_group=40)
    assert any(line.startswith("Monitoring: look 5, 38% of the plan;") for line in lines)
    assert (
        "  read at 38% of the planned size; the p above is a peek, judged on the monitoring block"
        in lines
    )
    assert "  plan   40 users per group, 38% reached" in lines
    assert any(line.startswith("  latest look   effect") for line in lines)
    assert any(
        line.startswith(
            "Synthetic check output for treatment vs control: no stop allowed yet at look 5, 38%"
        )
        for line in lines
    )
    assert (
        "* the run is at 38% of its planned 40 users per group; the results below are a peek"
        in ("\n".join(lines).replace("\n      ", " "))
    )


def test_a_crossing_in_flight_allows_a_stop_with_that_looks_interval(tmp_path: Path) -> None:
    rows = "".join(
        f"{arm},{day},{1 if user < share else 0}\n"
        for day in range(5)
        for arm, share in (("control", 10), ("treatment", 60))
        for user in range(100)
    )
    data_path = tmp_path / "rows.csv"
    data_path.write_text("group,day,converted\n" + rows, encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[experiment]\nname = "t"\nplanned_users_per_group = 2000\n\n'
        '[columns]\ngroup = "group"\nday = "day"\n\n'
        '[groups]\nlabels = ["control", "treatment", "other"]\n\n'
        '[[metrics]]\nname = "converted"\nkind = "binary"\nprimary = true\n',
        encoding="utf-8",
    )
    with data_path.open("a", encoding="utf-8") as data_file:
        for day in range(5):
            for user in range(100):
                data_file.write(f"other,{day},{1 if user < 10 else 0}\n")
    readout = analyze_experiment(load_user_rows(load_config(config_path), data_path))
    assert readout.monitoring is not None and not readout.monitoring.finished
    stopped = readout.monitoring.outcome.stopped_at_look
    assert stopped is not None and stopped <= 5
    # decision lines wrap at the readout's width, so read them joined
    text = "\n".join(readout_lines(readout, tmp_path, [])).replace("\n    ", " ")
    decisions = [line for line in text.splitlines() if line.startswith("Decision for")]
    assert decisions[0].startswith(
        f"Decision for treatment vs control: a stop is allowed from look {stopped}; "
        "higher on converted by"
    )
    assert "(adjusted +" in decisions[0]
    assert decisions[1].startswith("Decision for other vs control: none yet; the run is at")
    assert "the monitoring covers the first treatment only" in decisions[1]

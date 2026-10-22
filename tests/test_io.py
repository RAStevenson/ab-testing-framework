"""Intake tests: config shape-checking, the user-rows loader,
capabilities, and every validator rule proven on injected bad data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abkit.io import (
    Experiment,
    capabilities,
    load_config,
    load_user_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _config_text(**overrides: str) -> str:
    """The ok fixture config with named sections replaced verbatim."""
    base = {
        "experiment": '[experiment]\nname = "t"\nalpha = 0.05\n',
        "input": '[input]\npath = "user_rows_ok.csv"\n',
        "columns": '[columns]\narm = "group"\nday = "day"\nuser_id = "user_id"\n',
        "arms": '[arms]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n',
        "metrics": (
            '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\n'
            "primary = true\n"
            '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\n'
            "nonnegative = true\n"
        ),
    }
    base.update(overrides)
    return "\n".join(base.values())


def _write_config(tmp_path: Path, text: str) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _append_clean_filler(data_path: Path, n_filler: int = 300) -> None:
    """Append clean rows so a test's few planted drops stay under the 1% stop."""
    with open(data_path, "a", encoding="utf-8") as data_file:
        for i in range(n_filler):
            group = "control" if i % 2 else "treatment"
            data_file.write(f"f{i},{group},{i % 3},{i % 2},{i % 4}\n")


@pytest.fixture()
def ok_experiment() -> Experiment:
    config = load_config(FIXTURES / "experiment_ok.toml")
    return load_user_rows(config, FIXTURES / "user_rows_ok.csv")


class TestConfig:
    def test_ok_fixture_parses(self) -> None:
        config = load_config(FIXTURES / "experiment_ok.toml")
        assert config["experiment"]["name"] == "fixture-experiment"
        assert [metric["name"] for metric in config["metrics"]] == ["converted", "spend"]

    def test_two_primaries_rejected(self, tmp_path: Path) -> None:
        text = _config_text(
            metrics=(
                '[[metrics]]\nname = "a"\nkind = "binary"\nprimary = true\n'
                '[[metrics]]\nname = "b"\nkind = "binary"\nprimary = true\n'
            )
        )
        with pytest.raises(ValueError, match="exactly one metric"):
            load_config(_write_config(tmp_path, text))

    def test_unknown_metric_kind_rejected(self, tmp_path: Path) -> None:
        text = _config_text(metrics='[[metrics]]\nname = "a"\nkind = "count"\nprimary = true\n')
        with pytest.raises(ValueError, match="kind must be one of"):
            load_config(_write_config(tmp_path, text))

    def test_bad_split_rejected(self, tmp_path: Path) -> None:
        text = _config_text(
            arms='[arms]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.6]\n'
        )
        with pytest.raises(ValueError, match="intended_split"):
            load_config(_write_config(tmp_path, text))

    def test_duplicate_metric_name_rejected(self, tmp_path: Path) -> None:
        text = _config_text(
            metrics=(
                '[[metrics]]\nname = "a"\nkind = "binary"\nprimary = true\n'
                '[[metrics]]\nname = "a"\nkind = "binary"\n'
            )
        )
        with pytest.raises(ValueError, match="duplicate metric name"):
            load_config(_write_config(tmp_path, text))


class TestCleanLoad:
    def test_capabilities(self, ok_experiment: Experiment) -> None:
        assert capabilities(ok_experiment) == frozenset(
            {"time_axis", "multi_metric", "continuous_metric", "intended_split"}
        )

    def test_checkpoints_are_sorted_unique_days(self, ok_experiment: Experiment) -> None:
        assert ok_experiment.checkpoints == (0.0, 1.0, 2.0, 3.0, 4.0)

    def test_no_validation_events_on_clean_data(self, ok_experiment: Experiment) -> None:
        assert ok_experiment.validation == ()

    def test_end_of_run_aggregates_match_hand_computed(self, ok_experiment: Experiment) -> None:
        # treatment/converted: 6 ones, 9 zeros -> mean 0.4, ddof=1 var 3.6/14
        cells_by_key = {
            (cell.arm, cell.metric, cell.checkpoint): cell for cell in ok_experiment.aggregates
        }
        conversions_treatment = cells_by_key[("treatment", "converted", None)]
        assert conversions_treatment.n == 15
        assert conversions_treatment.mean == pytest.approx(0.4)
        assert conversions_treatment.variance == pytest.approx(3.6 / 14)
        # control/spend: values 10,20,30,40 among 15 rows -> mean 100/15
        spend_control_cell = cells_by_key[("control", "spend", None)]
        assert spend_control_cell.mean == pytest.approx(100 / 15)

    def test_cumulative_aggregates_grow_with_checkpoints(self, ok_experiment: Experiment) -> None:
        counts = [
            cell.n
            for cell in ok_experiment.aggregates
            if cell.arm == "control" and cell.metric == "converted" and cell.checkpoint is not None
        ]
        assert counts == [3, 6, 9, 12, 15]

    def test_missing_declared_column_rejected(self, tmp_path: Path) -> None:
        config = load_config(FIXTURES / "experiment_ok.toml")
        bad = tmp_path / "bad.csv"
        bad.write_text("user_id,group,day,converted\nu1,control,0,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing declared columns"):
            load_user_rows(config, bad)


class TestValidatorRules:
    """Each rule proven to FIRE on the dirty fixture, per the plan."""

    @pytest.fixture()
    def dirty(self, tmp_path: Path) -> Experiment:
        text = _config_text(
            columns=(
                '[columns]\narm = "group"\nday = "day"\nuser_id = "user_id"\n'
                '[columns.aliases.group]\n"treatmnet" = "treatment"\n'
            )
        )
        config = load_config(_write_config(tmp_path, text))
        return load_user_rows(config, FIXTURES / "user_rows_dirty.csv")

    def _count(self, experiment: Experiment, rule: str) -> int:
        return sum(event.count for event in experiment.validation if event.rule == rule)

    def test_alias_normalized_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "alias_normalized") == 1
        assert dirty.rows is not None
        # u03 kept as treatment, with the 200 even-numbered filler rows
        assert np.sum(dirty.rows["arm"] == "treatment") == 204

    def test_null_row_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "null_row_dropped") == 1  # u04, empty arm

    def test_negative_value_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "negative_value_dropped") == 1  # u06 spend -5

    def test_uncoercible_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "uncoercible_dropped") == 1  # u07 spend N/A

    def test_surviving_row_count(self, dirty: Experiment) -> None:
        # 412 data rows, 3 dropped (null arm, negative, uncoercible): 0.7%,
        # under the 1% stop
        assert dirty.rows is not None
        assert dirty.rows["arm"].size == 409

    def test_duplicate_user_stops_the_run(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "duplicate.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,control,0,1,5\nu2,treatment,0,0,0\nu1,control,1,0,0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"duplicate_user_error.*'u1'.*rows 2 and 4"):
            load_user_rows(config, data_path)

    def test_drop_share_above_one_percent_stops_the_run(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        clean = [
            f"u{i},{'control' if i % 2 else 'treatment'},{i % 3},{i % 2},{i % 4}" for i in range(98)
        ]
        header = "user_id,group,day,converted,spend\n"
        one_bad = tmp_path / "one_bad.csv"
        one_bad.write_text(
            header + "\n".join([*clean, "x1,control,0,1,N/A", "x2,control,0,1,3"]) + "\n",
            encoding="utf-8",
        )
        experiment = load_user_rows(config, one_bad)
        assert experiment.rows is not None
        assert experiment.rows["arm"].size == 99  # 1 of 100 dropped, under the stop
        two_bad = tmp_path / "two_bad.csv"
        two_bad.write_text(
            header + "\n".join([*clean, "x1,control,0,1,N/A", "x2,control,0,1,-3"]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(
            ValueError, match=r"too_many_dropped_error.*2 of 100.*uncoercible_dropped"
        ):
            load_user_rows(config, two_bad)

    def test_pre_column_round_trips(self, tmp_path: Path) -> None:
        text = _config_text(
            metrics=(
                '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\n'
                "primary = true\n"
                '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\n'
                'nonnegative = true\npre_column = "pre_spend"\n'
            )
        )
        config = load_config(_write_config(tmp_path, text))
        assert config["metrics"][1]["pre_column"] == "pre_spend"
        assert config["metrics"][0]["pre_column"] is None
        data_path = tmp_path / "pre.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend,pre_spend\n"
            "u1,control,0,1,5,2\nu2,control,0,0,0,0\nu3,treatment,1,1,7,3.5\nu4,treatment,1,0,0,1\n",
            encoding="utf-8",
        )
        experiment = load_user_rows(config, data_path)
        assert experiment.pre_period is not None
        assert list(experiment.pre_period) == ["spend"]
        np.testing.assert_array_equal(experiment.pre_period["spend"], [2.0, 0.0, 3.5, 1.0])
        assert experiment.rows is not None
        assert "pre_spend" not in experiment.rows
        assert experiment.validation == ()

    def test_pre_column_follows_the_metric_coercion_rules(self, tmp_path: Path) -> None:
        text = _config_text(
            metrics=(
                '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\n'
                "primary = true\n"
                '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\n'
                'nonnegative = true\npre_column = "pre_spend"\n'
            )
        )
        config = load_config(_write_config(tmp_path, text))
        clean = [
            f"u{i},{'control' if i % 2 else 'treatment'},{i % 3},{i % 2},{i % 4},{i % 5}"
            for i in range(300)
        ]
        data_path = tmp_path / "pre_dirty.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend,pre_spend\n"
            + "\n".join([*clean, "x1,control,0,1,5,-1", "x2,control,0,1,5,"])
            + "\n",
            encoding="utf-8",
        )
        experiment = load_user_rows(config, data_path)
        assert self._count(experiment, "negative_value_dropped") == 1
        assert self._count(experiment, "null_row_dropped") == 1
        assert experiment.pre_period is not None
        assert experiment.pre_period["spend"].size == 300

    def test_rows_use_canonical_keys(self, dirty: Experiment) -> None:
        assert dirty.rows is not None
        assert set(dirty.rows) == {"arm", "day", "user_id", "converted", "spend"}

    def test_unmapped_arm_raises(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))  # no alias table
        with pytest.raises(ValueError, match="unmapped_arm_error"):
            load_user_rows(config, FIXTURES / "user_rows_dirty.csv")

    def test_events_carry_example_row(self, dirty: Experiment) -> None:
        by_rule = {event.rule: event.detail for event in dirty.validation}
        assert by_rule["negative_value_dropped"] == "column 'spend', first at data row 7"


class TestBatteryRegressions:
    """Pins for the 2026-08-12 review findings: each was a demonstrated
    silent-corruption or unhelpful-crash path before its fix."""

    def test_reserved_metric_name_rejected(self, tmp_path: Path) -> None:
        text = _config_text(
            metrics=(
                '[[metrics]]\nname = "day"\ncolumn = "converted"\nkind = "binary"\nprimary = true\n'
            )
        )
        with pytest.raises(ValueError, match="reserved for a structural column"):
            load_config(_write_config(tmp_path, text))

    def test_case_colliding_arm_labels_rejected(self, tmp_path: Path) -> None:
        text = _config_text(arms='[arms]\nlabels = ["control", "Treatment", "treatment"]\n')
        with pytest.raises(ValueError, match="unique ignoring case"):
            load_config(_write_config(tmp_path, text))

    def test_string_primary_rejected(self, tmp_path: Path) -> None:
        text = _config_text(
            metrics=(
                '[[metrics]]\nname = "a"\nkind = "binary"\nprimary = true\n'
                '[[metrics]]\nname = "b"\nkind = "binary"\nprimary = "false"\n'
            )
        )
        with pytest.raises(ValueError, match="must be a boolean"):
            load_config(_write_config(tmp_path, text))

    def test_blank_user_ids_do_not_deduplicate_each_other(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "blank_ids.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            ",control,0,1,5\n,control,0,0,0\n,treatment,1,1,7\n,treatment,1,0,0\n",
            encoding="utf-8",
        )
        experiment = load_user_rows(config, data_path)
        assert experiment.rows is not None
        assert experiment.rows["arm"].size == 4
        assert experiment.validation == ()

    def test_nan_and_inf_metric_values_dropped(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "nonfinite.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,control,0,1,nan\nu2,control,0,0,inf\nu3,treatment,0,1,5\nu4,treatment,0,0,0\n"
            "u5,control,1,0,0\nu6,treatment,1,1,2\n",
            encoding="utf-8",
        )
        _append_clean_filler(data_path)
        experiment = load_user_rows(config, data_path)
        assert (
            sum(
                event.count
                for event in experiment.validation
                if event.rule == "uncoercible_dropped"
            )
            == 2
        )
        assert experiment.rows is not None
        assert np.all(np.isfinite(experiment.rows["spend"]))

    def test_nonbinary_value_in_binary_metric_dropped(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "nonbinary.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,control,0,3.7,0\nu2,control,0,0,0\nu3,treatment,0,1,5\nu4,treatment,1,0,0\n",
            encoding="utf-8",
        )
        _append_clean_filler(data_path)
        experiment = load_user_rows(config, data_path)
        assert (
            sum(event.count for event in experiment.validation if event.rule == "nonbinary_dropped")
            == 1
        )

    def test_garbage_day_dropped_not_crashed(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "bad_day.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,control,N/A,1,0\nu2,control,0,0,0\nu3,treatment,,1,5\nu4,treatment,1,0,0\n",
            encoding="utf-8",
        )
        _append_clean_filler(data_path)
        experiment = load_user_rows(config, data_path)
        assert experiment.rows is not None
        assert experiment.rows["arm"].size == 302  # u2, u4, and the 300 filler rows
        assert (
            sum(
                event.count
                for event in experiment.validation
                if event.rule == "uncoercible_dropped"
            )
            == 2
        )

    def test_bom_header_accepted(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "bom.csv"
        data_path.write_bytes(
            b"\xef\xbb\xbfuser_id,group,day,converted,spend\nu1,control,0,1,0\nu2,treatment,0,0,0\n"
        )
        experiment = load_user_rows(config, data_path)
        assert experiment.rows is not None
        assert experiment.rows["arm"].size == 2

    def test_arm_case_normalization_is_counted(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "cased.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,Control,0,1,0\nu2, treatment,0,0,0\nu3,control,1,0,0\nu4,treatment,1,1,5\n",
            encoding="utf-8",
        )
        experiment = load_user_rows(config, data_path)
        assert (
            sum(
                event.count
                for event in experiment.validation
                if event.rule == "arm_label_normalized"
            )
            == 2
        )
        assert experiment.rows is not None
        assert experiment.rows["arm"].size == 4


def test_alias_table_on_a_numeric_column_stops_the_run(tmp_path: Path) -> None:
    text = (
        '[experiment]\nname = "aliases on spend"\n\n'
        ""
        '[columns]\narm = "group"\n\n'
        '[columns.aliases.spend]\n"" = "0"\n\n'
        '[arms]\nlabels = ["control", "treatment"]\n\n'
        '[[metrics]]\nname = "spend"\nkind = "continuous"\nprimary = true\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="read as numbers"):
        load_config(config_path)


def test_aggregates_shape_is_refused(tmp_path: Path) -> None:
    text = _config_text(input='[input]\nshape = "aggregates"\npath = "x.csv"\n')
    with pytest.raises(ValueError, match="reads one row per user"):
        load_config(_write_config(tmp_path, text))

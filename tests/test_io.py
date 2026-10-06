"""Intake tests: config shape-checking, the user-rows loader, aggregates,
capabilities, and every validator rule proven on injected bad data."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from abkit.io import (
    Experiment,
    capabilities,
    interval_from_cumulative,
    load_aggregates,
    load_config,
    load_user_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _config_text(**overrides: str) -> str:
    """The ok fixture config with named sections replaced verbatim."""
    base = {
        "experiment": '[experiment]\nname = "t"\nalpha = 0.05\n',
        "input": '[input]\nshape = "user_rows"\npath = "user_rows_ok.csv"\n',
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
            {"user_rows", "time_axis", "multi_metric", "continuous_metric", "intended_split"}
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
        assert np.sum(dirty.rows["arm"] == "treatment") == 4  # u03 kept as treatment

    def test_null_row_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "null_row_dropped") == 1  # u04, empty arm

    def test_duplicate_user_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "duplicate_user_dropped") == 1  # second u05

    def test_negative_value_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "negative_value_dropped") == 1  # u06 spend -5

    def test_uncoercible_dropped_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "uncoercible_dropped") == 1  # u07 spend N/A

    def test_surviving_row_count(self, dirty: Experiment) -> None:
        # 13 data rows, 4 dropped (null arm, duplicate, negative, uncoercible)
        assert dirty.rows["arm"].size == 9

    def test_rows_use_canonical_keys(self, dirty: Experiment) -> None:
        assert set(dirty.rows) == {"arm", "day", "user_id", "converted", "spend"}

    def test_unmapped_arm_raises(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))  # no alias table
        with pytest.raises(ValueError, match="unmapped_arm_error"):
            load_user_rows(config, FIXTURES / "user_rows_dirty.csv")

    def test_events_carry_example_row(self, dirty: Experiment) -> None:
        by_rule = {event.rule: event.detail for event in dirty.validation}
        assert by_rule["negative_value_dropped"] == "first at data row 8"


class TestAggregates:
    @pytest.fixture()
    def agg_experiment(self) -> Experiment:
        config = load_config(FIXTURES / "aggregates_config.toml")
        return load_aggregates(config, FIXTURES / "aggregates_ok.csv")

    def test_arms_derived_from_data(self, agg_experiment: Experiment) -> None:
        assert agg_experiment.arms == ("control", "variant_2")

    def test_capabilities(self, agg_experiment: Experiment) -> None:
        # no user rows, so no user_rows and no continuous_metric even though
        # a continuous metric is declared; time axis and multi-metric hold
        assert capabilities(agg_experiment) == frozenset({"time_axis", "multi_metric"})

    def test_checkpoints(self, agg_experiment: Experiment) -> None:
        assert agg_experiment.checkpoints == (1.0, 2.0, 3.0)

    def test_experiment_filter_applied(self, agg_experiment: Experiment) -> None:
        cells = {
            (cell.arm, cell.metric, cell.checkpoint): cell for cell in agg_experiment.aggregates
        }
        assert cells[("control", "metric_1", 1.0)].n == 100  # expA, not expB's 50

    def test_end_of_run_equals_last_checkpoint(self, agg_experiment: Experiment) -> None:
        cells = {
            (cell.arm, cell.metric, cell.checkpoint): cell for cell in agg_experiment.aggregates
        }
        final = cells[("variant_2", "metric_4", None)]
        last = cells[("variant_2", "metric_4", 3.0)]
        assert (final.n, final.mean, final.variance) == (last.n, last.mean, last.variance)

    @pytest.mark.parametrize(
        ("bad_row", "match"),
        [
            ("expA,2,1,4.0,0,0.1,0.09,400,0.12,0.11", "nonpositive count"),
            ("expA,2,1,4.0,400,0.1,-0.09,400,0.12,0.11", "negative variance"),
            ("expA,2,1,4.0,300,0.1,0.09,400,0.12,0.11", "cumulative_regression_error"),
        ],
    )
    def test_hard_errors(self, tmp_path: Path, bad_row: str, match: str) -> None:
        source = (FIXTURES / "aggregates_ok.csv").read_text(encoding="utf-8")
        bad = tmp_path / "bad.csv"
        bad.write_text(source + bad_row + "\n", encoding="utf-8")
        config = load_config(FIXTURES / "aggregates_config.toml")
        with pytest.raises(ValueError, match=match):
            load_aggregates(config, bad)


class TestIntervalFromCumulative:
    def test_recovery_is_exact_under_ddof1(self) -> None:
        random_generator = np.random.default_rng(7)
        observations = random_generator.lognormal(1.0, 0.8, size=200)
        splits = [60, 140, 200]
        cumulative_counts, cumulative_means, cumulative_variances = [], [], []
        for end in splits:
            chunk = observations[:end]
            cumulative_counts.append(chunk.size)
            cumulative_means.append(chunk.mean())
            cumulative_variances.append(chunk.var(ddof=1))
        interval_counts, interval_means, interval_variances = interval_from_cumulative(
            np.array(cumulative_counts), np.array(cumulative_means), np.array(cumulative_variances)
        )
        expected = [observations[:60], observations[60:140], observations[140:200]]
        assert np.allclose(interval_counts, [60, 80, 60])
        assert np.allclose(interval_means, [chunk.mean() for chunk in expected], rtol=1e-10)
        assert np.allclose(interval_variances, [chunk.var(ddof=1) for chunk in expected], rtol=1e-8)

    def test_repeated_checkpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero observations"):
            interval_from_cumulative(np.array([10, 10]), np.array([1.0, 1.0]), np.array([0.5, 0.5]))

    def test_decreasing_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            interval_from_cumulative(np.array([10, 5]), np.array([1.0, 1.0]), np.array([0.5, 0.5]))

    def test_inconsistent_stats_rejected(self) -> None:
        # second cumulative mean jumps while variance shrinks to zero, which
        # no real cumulative series can do; the recovered interval variance
        # goes materially negative and must raise
        with pytest.raises(ValueError, match="inconsistent"):
            interval_from_cumulative(
                np.array([10, 12]), np.array([1.0, 5.0]), np.array([100.0, 0.0])
            )


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
        assert experiment.rows["arm"].size == 4
        assert not any(event.rule == "duplicate_user_dropped" for event in experiment.validation)

    def test_nan_and_inf_metric_values_dropped(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "nonfinite.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,control,0,1,nan\nu2,control,0,0,inf\nu3,treatment,0,1,5\nu4,treatment,0,0,0\n"
            "u5,control,1,0,0\nu6,treatment,1,1,2\n",
            encoding="utf-8",
        )
        experiment = load_user_rows(config, data_path)
        assert (
            sum(
                event.count
                for event in experiment.validation
                if event.rule == "uncoercible_dropped"
            )
            == 2
        )
        assert np.all(np.isfinite(experiment.rows["spend"]))

    def test_nonbinary_value_in_binary_metric_dropped(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _config_text()))
        data_path = tmp_path / "nonbinary.csv"
        data_path.write_text(
            "user_id,group,day,converted,spend\n"
            "u1,control,0,3.7,0\nu2,control,0,0,0\nu3,treatment,0,1,5\nu4,treatment,1,0,0\n",
            encoding="utf-8",
        )
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
        experiment = load_user_rows(config, data_path)
        assert experiment.rows["arm"].size == 2
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
        assert experiment.rows["arm"].size == 2

    def test_aggregates_missing_declared_metric_rejected(self, tmp_path: Path) -> None:
        config = load_config(FIXTURES / "aggregates_config.toml")
        config["metrics"].append(
            {
                "name": "metric_9",
                "column": "9",
                "kind": "binary",
                "primary": False,
                "nonnegative": False,
            }
        )
        with pytest.raises(ValueError, match="has no data for arm"):
            load_aggregates(config, FIXTURES / "aggregates_ok.csv")

    def test_aggregates_misaligned_checkpoints_rejected(self, tmp_path: Path) -> None:
        source = (FIXTURES / "aggregates_ok.csv").read_text(encoding="utf-8")
        # second variant for expA whose metric-1 grid misses checkpoint 1.0
        extra = (
            "expA,3,1,2.0,220,0.11,0.0983,110,0.14,0.1204\n"
            "expA,3,1,3.0,360,0.105,0.0942,180,0.13,0.1131\n"
            "expA,3,4,1.0,100,5.2,3.1,50,5.4,3.3\n"
            "expA,3,4,2.0,220,5.3,3.2,110,5.5,3.4\n"
            "expA,3,4,3.0,360,5.25,3.15,180,5.6,3.5\n"
        )
        bad = tmp_path / "misaligned.csv"
        bad.write_text(source + extra, encoding="utf-8")
        config = load_config(FIXTURES / "aggregates_config.toml")
        with pytest.raises(ValueError, match="checkpoint grids differ across arms"):
            load_aggregates(config, bad)

    def test_aggregates_nan_statistic_rejected(self, tmp_path: Path) -> None:
        source = (FIXTURES / "aggregates_ok.csv").read_text(encoding="utf-8")
        bad = tmp_path / "nanstat.csv"
        bad.write_text(source + "expA,2,1,4.0,400,NaN,0.09,400,0.12,0.11\n", encoding="utf-8")
        config = load_config(FIXTURES / "aggregates_config.toml")
        with pytest.raises(ValueError, match="non-finite statistics at data row"):
            load_aggregates(config, bad)

    def test_aggregates_noninteger_count_rejected(self, tmp_path: Path) -> None:
        source = (FIXTURES / "aggregates_ok.csv").read_text(encoding="utf-8")
        bad = tmp_path / "fraccount.csv"
        bad.write_text(source + "expA,2,1,4.0,400.9,0.1,0.09,400,0.12,0.11\n", encoding="utf-8")
        config = load_config(FIXTURES / "aggregates_config.toml")
        with pytest.raises(ValueError, match="non-integer count"):
            load_aggregates(config, bad)

    def test_aggregates_filter_keys_must_pair(self, tmp_path: Path) -> None:
        text = (FIXTURES / "aggregates_config.toml").read_text(encoding="utf-8")
        text = text.replace('experiment_id = "expA"\n', "")
        with pytest.raises(ValueError, match="must be set together"):
            load_config(_write_config(tmp_path, text))

    def test_aggregates_intended_split_validated(self, tmp_path: Path) -> None:
        text = (FIXTURES / "aggregates_config.toml").read_text(encoding="utf-8")
        text += "\n[arms]\nintended_split = [3, 7]\n"
        with pytest.raises(ValueError, match="positive shares summing to 1"):
            load_config(_write_config(tmp_path, text))

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
        assert experiment.rows["arm"].size == 4


def test_unparseable_aggregate_keeps_the_parse_error(tmp_path: Path) -> None:
    source = (FIXTURES / "aggregates_ok.csv").read_text(encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_text(source + "expA,2,1,4.0,abc,0.1,0.09,400,0.12,0.11\n", encoding="utf-8")
    config = load_config(FIXTURES / "aggregates_config.toml")
    with pytest.raises(ValueError, match="unparseable") as raised:
        load_aggregates(config, bad)
    assert isinstance(raised.value.__cause__, ValueError)


def test_single_observation_interval_is_nan_without_any_warning() -> None:
    # no warning is blocked here: the division is skipped where it has no answer
    observations = np.random.default_rng(5).lognormal(1.0, 0.8, size=40)
    splits = [20, 21, 40]
    counts = np.array([float(end) for end in splits])
    means = np.array([observations[:end].mean() for end in splits])
    variances = np.array([observations[:end].var(ddof=1) for end in splits])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        interval_counts, _, interval_variances = interval_from_cumulative(counts, means, variances)
    assert interval_counts[1] == 1.0
    assert np.isnan(interval_variances[1])
    expected = [observations[:20].var(ddof=1), observations[21:40].var(ddof=1)]
    assert np.allclose(interval_variances[[0, 2]], expected, rtol=1e-8)

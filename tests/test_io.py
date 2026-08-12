"""Intake tests: config shape-checking, the user-rows loader, aggregates,
capabilities, and every validator rule proven on injected bad data."""

from __future__ import annotations

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
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture()
def ok_experiment() -> Experiment:
    cfg = load_config(FIXTURES / "experiment_ok.toml")
    return load_user_rows(cfg, FIXTURES / "user_rows_ok.csv")


class TestConfig:
    def test_ok_fixture_parses(self) -> None:
        cfg = load_config(FIXTURES / "experiment_ok.toml")
        assert cfg["experiment"]["name"] == "fixture-experiment"
        assert [m["name"] for m in cfg["metrics"]] == ["converted", "spend"]

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
        agg = {(a.arm, a.metric, a.checkpoint): a for a in ok_experiment.aggregates}
        conv_t = agg[("treatment", "converted", None)]
        assert conv_t.n == 15
        assert conv_t.mean == pytest.approx(0.4)
        assert conv_t.variance == pytest.approx(3.6 / 14)
        # control/spend: values 10,20,30,40 among 15 rows -> mean 100/15
        spend_c = agg[("control", "spend", None)]
        assert spend_c.mean == pytest.approx(100 / 15)

    def test_cumulative_aggregates_grow_with_checkpoints(self, ok_experiment: Experiment) -> None:
        ns = [
            a.n
            for a in ok_experiment.aggregates
            if a.arm == "control" and a.metric == "converted" and a.checkpoint is not None
        ]
        assert ns == [3, 6, 9, 12, 15]

    def test_missing_declared_column_rejected(self, tmp_path: Path) -> None:
        cfg = load_config(FIXTURES / "experiment_ok.toml")
        bad = tmp_path / "bad.csv"
        bad.write_text("user_id,group,day,converted\nu1,control,0,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing declared columns"):
            load_user_rows(cfg, bad)


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
        cfg = load_config(_write_config(tmp_path, text))
        return load_user_rows(cfg, FIXTURES / "user_rows_dirty.csv")

    def _count(self, exp: Experiment, rule: str) -> int:
        return sum(e.count for e in exp.validation if e.rule == rule)

    def test_alias_normalized_fires(self, dirty: Experiment) -> None:
        assert self._count(dirty, "alias_normalized") == 1
        assert np.sum(dirty.rows["group"] == "treatment") == 4  # u03 kept as treatment

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
        assert dirty.rows["group"].size == 9

    def test_unmapped_arm_raises(self, tmp_path: Path) -> None:
        cfg = load_config(_write_config(tmp_path, _config_text()))  # no alias table
        with pytest.raises(ValueError, match="unmapped_arm_error"):
            load_user_rows(cfg, FIXTURES / "user_rows_dirty.csv")

    def test_events_carry_example_row(self, dirty: Experiment) -> None:
        by_rule = {e.rule: e.detail for e in dirty.validation}
        assert by_rule["negative_value_dropped"] == "first at data row 8"


class TestAggregates:
    @pytest.fixture()
    def agg_experiment(self) -> Experiment:
        cfg = load_config(FIXTURES / "aggregates_config.toml")
        return load_aggregates(cfg, FIXTURES / "aggregates_ok.csv")

    def test_arms_derived_from_data(self, agg_experiment: Experiment) -> None:
        assert agg_experiment.arms == ("control", "variant_2")

    def test_capabilities(self, agg_experiment: Experiment) -> None:
        # no user rows, so no user_rows and no continuous_metric even though
        # a continuous metric is declared; time axis and multi-metric hold
        assert capabilities(agg_experiment) == frozenset({"time_axis", "multi_metric"})

    def test_checkpoints(self, agg_experiment: Experiment) -> None:
        assert agg_experiment.checkpoints == (1.0, 2.0, 3.0)

    def test_experiment_filter_applied(self, agg_experiment: Experiment) -> None:
        cells = {(a.arm, a.metric, a.checkpoint): a for a in agg_experiment.aggregates}
        assert cells[("control", "metric_1", 1.0)].n == 100  # expA, not expB's 50

    def test_end_of_run_equals_last_checkpoint(self, agg_experiment: Experiment) -> None:
        cells = {(a.arm, a.metric, a.checkpoint): a for a in agg_experiment.aggregates}
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
        cfg = load_config(FIXTURES / "aggregates_config.toml")
        with pytest.raises(ValueError, match=match):
            load_aggregates(cfg, bad)


class TestIntervalFromCumulative:
    def test_recovery_is_exact_under_ddof1(self) -> None:
        rng = np.random.default_rng(7)
        observations = rng.lognormal(1.0, 0.8, size=200)
        splits = [60, 140, 200]
        cum_n, cum_mean, cum_var = [], [], []
        for end in splits:
            chunk = observations[:end]
            cum_n.append(chunk.size)
            cum_mean.append(chunk.mean())
            cum_var.append(chunk.var(ddof=1))
        int_n, int_mean, int_var = interval_from_cumulative(
            np.array(cum_n), np.array(cum_mean), np.array(cum_var)
        )
        expected = [observations[:60], observations[60:140], observations[140:200]]
        assert np.allclose(int_n, [60, 80, 60])
        assert np.allclose(int_mean, [c.mean() for c in expected], rtol=1e-10)
        assert np.allclose(int_var, [c.var(ddof=1) for c in expected], rtol=1e-8)

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

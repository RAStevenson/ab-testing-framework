"""Analysis-layer tests: every block the readout reports, computed on the
fixtures, with one test per limitation code on data built to raise it."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from abkit.io import Experiment, load_aggregates, load_config, load_user_rows
from abkit.readout_analysis import (
    HEAVY_TAIL_TOP_SHARE,
    ExperimentReadout,
    _consistent_prefix,
    analyze_experiment,
    planted_spend_difference,
)
from abkit.results import RateComparison

FIXTURES = Path(__file__).parent / "fixtures"
HEADER = "user_id,group,day,converted,spend\n"


def _user_config_text(
    *, extra_metric: bool = False, truth: bool = False, pre_column: bool = False
) -> str:
    """The user-rows fixture config, optionally with a third metric (so the
    corrections family has 2+ members), a [truth] table, or a pre column on spend."""
    text = (
        '[experiment]\nname = "t"\nalpha = 0.05\n\n'
        '[input]\nshape = "user_rows"\npath = "user_rows_ok.csv"\n\n'
        '[columns]\narm = "group"\nday = "day"\nuser_id = "user_id"\n\n'
        '[arms]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n\n'
        '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\nprimary = true\n\n'
        '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\nnonnegative = true\n'
    )
    if pre_column:
        text += 'pre_column = "pre_spend"\n'
    if extra_metric:
        text += '\n[[metrics]]\nname = "converted_again"\ncolumn = "converted"\nkind = "binary"\n'
    if truth:
        text += "\n[truth]\nbase_rate = 0.27\ntrue_lift = 0.13\n"
    return text


def _user_experiment(tmp_path: Path, **options: bool) -> Experiment:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_user_config_text(**options), encoding="utf-8")
    return load_user_rows(load_config(config_path), FIXTURES / "user_rows_ok.csv")


def _readout(experiment: Experiment) -> ExperimentReadout:
    return analyze_experiment(experiment, np.random.default_rng(0))


def _codes(readout: ExperimentReadout) -> list[str]:
    return [limitation.code for limitation in readout.limitations]


def _generated_rows(
    n_per_arm: int,
    *,
    seed: int,
    days: tuple[int, ...] = (0, 1, 2, 3, 4),
    whale: float | None = None,
    treatment_share: float = 0.5,
    pre_column: bool = False,
) -> str:
    """A clean user-rows file big enough that no limitation fires by accident:
    50% conversion, lognormal spend, five days, optional whale and pre column."""
    random_generator = np.random.default_rng(seed)
    n_treatment = round(2 * n_per_arm * treatment_share)
    n_control = 2 * n_per_arm - n_treatment
    lines = [HEADER.rstrip("\n") + (",pre_spend" if pre_column else "")]
    user = 0
    for arm, size in (("control", n_control), ("treatment", n_treatment)):
        for _ in range(size):
            user += 1
            converted = int(random_generator.random() < 0.5)
            spend = float(random_generator.lognormal(2.0, 0.5)) if converted else 0.0
            line = f"u{user},{arm},{days[user % len(days)]},{converted},{spend:.2f}"
            if pre_column:
                line += f",{spend * 0.8 + random_generator.lognormal(1.0, 0.3):.2f}"
            lines.append(line)
    if whale is not None:
        lines[1] = lines[1].replace(lines[1].split(",")[4], f"{whale:.2f}", 1)
    return "\n".join(lines) + "\n"


def _generated_experiment(
    tmp_path: Path,
    n_per_arm: int = 8000,
    *,
    days: tuple[int, ...] = (0, 1, 2, 3, 4),
    whale: float | None = None,
    treatment_share: float = 0.5,
    pre_column: bool = False,
) -> Experiment:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_user_config_text(pre_column=pre_column), encoding="utf-8")
    data_path = tmp_path / "rows.csv"
    data_path.write_text(
        _generated_rows(
            n_per_arm,
            seed=7,
            days=days,
            whale=whale,
            treatment_share=treatment_share,
            pre_column=pre_column,
        ),
        encoding="utf-8",
    )
    return load_user_rows(load_config(config_path), data_path)


class TestUserRowsFixture:
    def test_primary_number_pinned(self, tmp_path: Path) -> None:
        # control 4/15 vs treatment 6/15 -> absolute lift 13.33 pp
        readout = _readout(_user_experiment(tmp_path))
        (treatment_arm, comparison), *_ = readout.primary
        assert treatment_arm == "treatment"
        assert isinstance(comparison, RateComparison)
        assert comparison.absolute_lift == pytest.approx(0.13333, abs=1e-5)
        assert readout.data_shape == "user_rows"
        assert readout.arm_sizes == (("control", 15), ("treatment", 15))
        assert readout.primary_holm is None  # one treatment, no family to correct

    def test_blocks_present_on_the_fixture(self, tmp_path: Path) -> None:
        readout = _readout(_user_experiment(tmp_path, extra_metric=True))
        assert readout.group_split is not None and readout.ratio_stability is None
        assert readout.heavy_tail is not None
        assert readout.novelty is not None and readout.monitoring is not None
        assert len(readout.corrections) == 2
        assert readout.truth is None
        assert readout.excluded_checkpoints == 0
        assert readout.monitoring.n_usable == 5
        assert len(readout.monitoring.intervals.ci_low) == 5

    def test_corrections_family_of_one_is_empty(self, tmp_path: Path) -> None:
        # two metrics, two arms -> one non-primary comparison -> no family
        assert _readout(_user_experiment(tmp_path)).corrections == ()

    def test_truth_carried_only_when_declared(self, tmp_path: Path) -> None:
        assert _readout(_user_experiment(tmp_path, truth=True)).truth == {
            "base_rate": 0.27,
            "true_lift": 0.13,
        }

    def test_tiny_fixture_raises_the_small_sample_flags(self, tmp_path: Path) -> None:
        codes = _codes(_readout(_user_experiment(tmp_path)))
        assert "few_events" in codes
        assert "low_sensitivity" in codes
        assert "no_pre_period" in codes
        assert "group_split" not in codes


class TestLimitationCodes:
    """One fixture per code, built to trigger it and nothing else it can avoid."""

    def test_clean_generated_data_raises_only_the_missing_pre_column(self, tmp_path: Path) -> None:
        readout = _readout(_generated_experiment(tmp_path))
        assert _codes(readout) == ["no_pre_period"]
        assert 0.0 < readout.sensitivity[0][1]["relative_mde"] <= 0.05

    def test_group_split(self, tmp_path: Path) -> None:
        readout = _readout(_generated_experiment(tmp_path, treatment_share=0.4))
        assert "group_split" in _codes(readout)
        assert readout.group_split is not None
        assert readout.group_split[0][1].p_value < 1e-3

    def test_few_events(self, tmp_path: Path) -> None:
        readout = _readout(_user_experiment(tmp_path))
        few = next(
            limitation for limitation in readout.limitations if limitation.code == "few_events"
        )
        assert "control (4 of 15)" in few.detail

    def test_heavy_tail(self, tmp_path: Path) -> None:
        readout = _readout(_generated_experiment(tmp_path, n_per_arm=500, whale=1e6))
        tail = next(
            limitation for limitation in readout.limitations if limitation.code == "heavy_tail"
        )
        assert "% of spend" in tail.detail
        assert readout.heavy_tail is not None
        assert HEAVY_TAIL_TOP_SHARE == 0.25

    def test_low_sensitivity(self, tmp_path: Path) -> None:
        readout = _readout(_generated_experiment(tmp_path, n_per_arm=200))
        low = next(
            limitation for limitation in readout.limitations if limitation.code == "low_sensitivity"
        )
        assert "smallest detectable effect on converted" in low.detail
        assert "for treatment vs control (" in low.detail
        assert readout.sensitivity[0][1]["relative_mde"] > 0.05

    def test_no_pre_period_clears_when_a_pre_column_is_declared(self, tmp_path: Path) -> None:
        readout = _readout(_generated_experiment(tmp_path, pre_column=True))
        assert "no_pre_period" not in _codes(readout)
        assert len(readout.cuped) == 1
        metric, treatment_arm, result = readout.cuped[0]
        assert (metric, treatment_arm) == ("spend", "treatment")
        assert 0.0 < result.variance_reduction < 1.0

    def test_no_pre_period_on_aggregates(self) -> None:
        config = load_config(FIXTURES / "aggregates_config.toml")
        readout = _readout(load_aggregates(config, FIXTURES / "aggregates_ok.csv"))
        no_pre = next(
            limitation for limitation in readout.limitations if limitation.code == "no_pre_period"
        )
        assert "aggregate data" in no_pre.detail
        assert readout.cuped == ()

    def test_missing_days(self, tmp_path: Path) -> None:
        readout = _readout(_generated_experiment(tmp_path, days=(0, 1, 2, 6, 7, 8)))
        gap = next(
            limitation for limitation in readout.limitations if limitation.code == "missing_days"
        )
        assert (
            "1 gap in the checkpoint sequence (the largest 4 days against a typical 1)"
            in gap.detail
        )

    def test_excluded_checkpoints(self) -> None:
        config = load_config(FIXTURES / "aggregates_config.toml")
        experiment = load_aggregates(config, FIXTURES / "aggregates_ok.csv")
        checkpoints = sorted(_present_checkpoints(experiment))
        last, before_last = checkpoints[-1], checkpoints[-2]
        # corrupt the final checkpoint's primary-metric cell in every arm: ten
        # more users and a cumulative mean that drops to 30% of the previous
        # one, the settled-recompute shape, so the implied increment variance
        # goes negative and the prefix stops
        previous = {
            cell.arm: cell
            for cell in experiment.aggregates
            if cell.checkpoint == before_last and cell.metric == "metric_1"
        }
        damaged = dataclasses.replace(
            experiment,
            aggregates=tuple(
                dataclasses.replace(
                    cell, n=previous[cell.arm].n + 10, mean=0.3 * previous[cell.arm].mean
                )
                if cell.checkpoint == last and cell.metric == "metric_1"
                else cell
                for cell in experiment.aggregates
            ),
        )
        readout = _readout(damaged)
        assert readout.excluded_checkpoints == 1
        excluded = next(
            limitation
            for limitation in readout.limitations
            if limitation.code == "excluded_checkpoints"
        )
        assert "final checkpoint is internally inconsistent" in excluded.detail
        assert readout.monitoring is None  # two usable looks are too few to replay


def _present_checkpoints(experiment: Experiment) -> tuple[float, ...]:
    assert experiment.checkpoints is not None
    return experiment.checkpoints


class TestAggregatesFixture:
    @pytest.fixture()
    def readout(self) -> ExperimentReadout:
        config = load_config(FIXTURES / "aggregates_config.toml")
        return _readout(load_aggregates(config, FIXTURES / "aggregates_ok.csv"))

    def test_no_user_level_blocks(self, readout: ExperimentReadout) -> None:
        assert readout.data_shape == "aggregates"
        assert readout.heavy_tail is None
        assert readout.truth is None
        assert readout.cuped == ()

    def test_split_check_takes_the_ratio_stability_form(self, readout: ExperimentReadout) -> None:
        assert readout.group_split is None
        assert readout.ratio_stability is not None
        _, degrees_of_freedom, p_value, n_intervals = readout.ratio_stability
        assert n_intervals == 3 and degrees_of_freedom == 2
        assert 0.0 <= p_value <= 1.0

    def test_monitoring_replays_the_first_contrast(self, readout: ExperimentReadout) -> None:
        assert readout.monitoring is not None
        assert readout.monitoring.treatment_arm == "variant_2"
        assert readout.novelty is not None
        assert readout.novelty.contrasts[0].treatment_arm == "variant_2"


class TestHillstromShapedFixture:
    """The Hillstrom readout's logic on a FABRICATED 30-row fixture with the
    real file's header, so no test touches the network or the cache."""

    def _experiment(self, tmp_path: Path) -> Experiment:
        text = (
            '[experiment]\nname = "hillstrom-fixture"\nalpha = 0.05\n\n'
            '[input]\nshape = "user_rows"\n\n'
            '[columns]\narm = "segment"\n\n'
            '[columns.aliases.zip_code]\n"Surburban" = "Suburban"\n\n'
            '[arms]\nlabels = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]\n'
            "intended_split = [0.3334, 0.3333, 0.3333]\n\n"
            '[[metrics]]\nname = "visit"\nkind = "binary"\nprimary = true\n\n'
            '[[metrics]]\nname = "conversion"\nkind = "binary"\n\n'
            '[[metrics]]\nname = "spend"\nkind = "continuous"\nnonnegative = true\n'
            'pre_column = "history"\n'
        )
        config_path = tmp_path / "config.toml"
        config_path.write_text(text, encoding="utf-8")
        return load_user_rows(load_config(config_path), FIXTURES / "hillstrom_shaped.csv")

    def test_blocks_without_a_time_axis(self, tmp_path: Path) -> None:
        readout = _readout(self._experiment(tmp_path))
        assert readout.monitoring is None and readout.novelty is None
        assert len(readout.corrections) == 4  # 2 non-primary metrics x 2 contrasts
        assert readout.heavy_tail is not None and readout.heavy_tail.metric == "spend"
        assert [arm for arm, _ in readout.primary] == ["Mens E-Mail", "Womens E-Mail"]
        assert [(metric, arm) for metric, arm, _ in readout.cuped] == [
            ("spend", "Mens E-Mail"),
            ("spend", "Womens E-Mail"),
        ]
        assert "no_pre_period" not in _codes(readout)
        # two treatments: one sensitivity per contrast, and Holm across the primary's contrasts
        assert [arm for arm, _ in readout.sensitivity] == ["Mens E-Mail", "Womens E-Mail"]
        assert readout.primary_holm is not None and len(readout.primary_holm) == 2
        raw = [comparison.p_value for _, comparison in readout.primary]
        assert all(
            adjusted >= p_value for adjusted, p_value in zip(readout.primary_holm, raw, strict=True)
        )


class TestConsistentPrefix:
    def test_clean_series_keeps_everything(self) -> None:
        random_generator = np.random.default_rng(3)
        observations = random_generator.lognormal(1.0, 0.6, size=300)
        splits = [100, 200, 300]
        count = np.array([float(end) for end in splits])
        mean = np.array([observations[:end].mean() for end in splits])
        variance = np.array([observations[:end].var(ddof=1) for end in splits])
        assert _consistent_prefix(count, mean, variance) == 3

    def test_corrupt_final_checkpoint_is_cut(self) -> None:
        # final checkpoint's mean drops while n barely grows, the settled-
        # recompute shape found in the real ASOS archive: the implied
        # increment variance goes hugely negative and the prefix stops
        count = np.array([1000.0, 2000.0, 2010.0])
        mean = np.array([0.10, 0.11, 0.05])
        variance = mean * (1 - mean) * count / (count - 1)
        assert _consistent_prefix(count, mean, variance) == 2

    def test_needs_no_warning_block_for_short_increments(self) -> None:
        # one-observation and zero increments are skipped by the division
        # itself, so the check runs clean with every RuntimeWarning an error
        observations = np.random.default_rng(4).lognormal(1.0, 0.6, size=60)
        splits = [30, 31, 31, 60]
        counts = np.array([float(end) for end in splits])
        means = np.array([observations[:end].mean() for end in splits])
        variances = np.array([observations[:end].var(ddof=1) for end in splits])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert _consistent_prefix(counts, means, variances) == len(splits)


class TestOmittedBlocks:
    """A block the data cannot support is omitted, the run carries on, and
    the console says which block and why, with the error in full."""

    def test_missing_series_omits_both_time_blocks_and_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = load_config(FIXTURES / "aggregates_config.toml")
        experiment = load_aggregates(config, FIXTURES / "aggregates_ok.csv")
        treatment_arm = experiment.arms[1]
        # keep the end-of-run cells, drop that arm's per-checkpoint cells
        damaged = dataclasses.replace(
            experiment,
            aggregates=tuple(
                cell
                for cell in experiment.aggregates
                if not (cell.arm == treatment_arm and cell.checkpoint is not None)
            ),
        )
        readout = _readout(damaged)
        assert readout.novelty is None and readout.monitoring is None
        assert readout.primary  # the end-of-run result still reports
        stderr = capsys.readouterr().err
        assert stderr.count("Traceback") == 1
        assert "no per-checkpoint aggregates" in stderr
        assert "omits the block 'Early versus late, and Monitoring replay'" in stderr


def test_planted_spend_difference_matches_the_recipe_by_hand():
    # 0.004 times the mean spender amount: 0.992 * exp(3.4 + 1.3^2 / 2) for the
    # lognormal plus 0.008 * 400 * 1.5 / 0.5 = 1200 for the whales
    recipe = {
        "spend_rate_lift": 0.004,
        "log_mean": 3.4,
        "log_sigma": 1.3,
        "whale_rate": 0.008,
        "whale_shape": 1.5,
        "whale_scale": 400.0,
    }
    expected = 0.004 * (0.992 * np.exp(3.4 + 0.845) + 0.008 * 1200.0)
    assert planted_spend_difference(recipe) == pytest.approx(expected, rel=1e-12)
    assert planted_spend_difference(recipe) == pytest.approx(0.315, abs=5e-4)
    assert planted_spend_difference({"spend_rate_lift": 0.004}) is None


def test_cumulative_increments_flag_on_aggregates_with_a_time_axis():
    config = load_config(FIXTURES / "aggregates_config.toml")
    readout = _readout(load_aggregates(config, FIXTURES / "aggregates_ok.csv"))
    flag = next(
        limitation
        for limitation in readout.limitations
        if limitation.code == "cumulative_increments"
    )
    assert "earlier users' outcomes stay fixed" in flag.detail


def test_many_looks_flag_past_the_pinned_schedule(tmp_path: Path):
    readout = _readout(_generated_experiment(tmp_path, n_per_arm=4000, days=tuple(range(60))))
    assert readout.monitoring is not None and readout.monitoring.n_usable == 60
    flag = next(limitation for limitation in readout.limitations if limitation.code == "many_looks")
    assert "the replay has 60 looks" in flag.detail
    assert "cumulative_increments" not in _codes(readout)  # user rows carry no such risk


def test_heavy_tail_flag_names_zero_inflation_when_most_users_are_zero(tmp_path: Path):
    text = _user_config_text()
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    lines = [HEADER.rstrip("\n")]
    for user in range(4000):
        arm = "control" if user % 2 else "treatment"
        spend = 50.0 + user % 7 if user % 200 == 0 else 0.0
        lines.append(f"u{user},{arm},{user % 5},{user % 3 == 0:d},{spend}")
    data_path = tmp_path / "rows.csv"
    data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    readout = _readout(load_user_rows(load_config(config_path), data_path))
    flag = next(limitation for limitation in readout.limitations if limitation.code == "heavy_tail")
    assert "fewer than 1% have any spend at all" in flag.detail


def test_heavy_tail_flag_states_both_facts_when_spenders_exceed_one_percent(tmp_path: Path):
    readout = _readout(_generated_experiment(tmp_path, n_per_arm=500, whale=1e6))
    flag = next(limitation for limitation in readout.limitations if limitation.code == "heavy_tail")
    # half the generated users convert and only converters spend, so the
    # spenders are far above 1% and the flag must not claim a cause
    assert "because" not in flag.detail


def test_sensitivity_with_several_treatments_is_solved_at_alpha_over_their_number(
    tmp_path: Path,
):
    text = (
        '[experiment]\nname = "hillstrom-fixture"\nalpha = 0.05\n\n'
        '[input]\nshape = "user_rows"\n\n'
        '[columns]\narm = "segment"\n\n'
        '[columns.aliases.zip_code]\n"Surburban" = "Suburban"\n\n'
        '[arms]\nlabels = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]\n\n'
        '[[metrics]]\nname = "visit"\nkind = "binary"\nprimary = true\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    experiment = load_user_rows(load_config(config_path), FIXTURES / "hillstrom_shaped.csv")
    readout = _readout(experiment)
    assert [values["alpha"] for _, values in readout.sensitivity] == [0.025, 0.025]

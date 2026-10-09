"""The readout's analysis layer: every number the readout reports, computed once, no text.

``analyze_experiment`` turns an ``abkit.io.Experiment`` into an
``ExperimentReadout``, a frozen record of results in the order the
readout prints them: the data and its validation events, the limitations
the checks raised, the group-split check, the sensitivity of the sample,
the primary metric per contrast with its CUPED adjustment where a
pre-experiment column exists, the corrected secondary comparisons, the
revenue cross-check, the early-versus-late comparison, the
daily-monitoring replay with adjusted intervals, and the planted truth
for synthetic runs. ``readout_text`` turns the record into lines and
``readout_figures`` draws from it; neither recomputes anything.

A block the data cannot support is ``None``. The two time-based blocks
omit themselves on a data error: the error prints in full, then one line
names the block, and the run carries on, because a monitoring replay the
archive cannot support is not a reason to withhold the primary result.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from abkit.cuped_adjustment import cuped_mean_diff
from abkit.frequentist import (
    benjamini_hochberg,
    holm,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_stats,
)
from abkit.io import ArmAggregate, Experiment, MetricSpec, ValidationEvent, capabilities
from abkit.power import mde_two_proportions, power_two_proportions
from abkit.resampling import bootstrap_mean_diff, permutation_test_mean_diff, trimmed_mean_diff
from abkit.results import (
    BootstrapResult,
    CupedResult,
    MeanComparison,
    PermutationResult,
    RateComparison,
    RepeatedIntervals,
    SequentialDesign,
    SequentialOutcome,
    SrmResult,
)
from abkit.sequential import (
    design,
    monitor,
    repeated_intervals,
    z_from_cumulative_counts,
    z_from_cumulative_stats,
)

# the flag thresholds, accepted 2026-10-06
SRM_ALARM_P = 1e-3  # the conventional alarm threshold for a split check
MIN_EXPECTED_EVENTS = 10  # successes and failures per arm for the normal approximation
HEAVY_TAIL_TOP_SHARE = 0.25  # top 1% of users carrying this share of a metric's total
MODEST_RELATIVE_LIFT = 0.05  # a 5% relative lift; a smallest detectable effect above it is flagged
TARGET_POWER = 0.8
N_RESAMPLES = 10_000
MIN_LOOKS = 3  # fewer looks than this and neither time-based block can say anything
# a gap in the day sequence this many times the typical spacing counts as missing days
MISSING_DAY_GAP_FACTOR = 1.5


@dataclass(frozen=True)
class Limitation:
    """One flag the checks raised: a short code and the detail behind it."""

    code: str
    detail: str


@dataclass(frozen=True)
class CorrectedComparison:
    """One member of the secondary comparison family with its two adjusted p-values."""

    metric: str
    treatment_arm: str
    comparison: RateComparison | MeanComparison
    holm_p: float
    bh_q: float


@dataclass(frozen=True)
class HeavyTailCheck:
    """The revenue cross-check on the first contrast: four methods on one skewed metric."""

    metric: str
    treatment_arm: str
    welch: MeanComparison
    bootstrap: BootstrapResult
    permutation: PermutationResult
    trimmed_difference: float
    zero_share: float
    trim_degenerate: bool


@dataclass(frozen=True)
class NoveltyContrast:
    """Early-window and late-window effects for one contrast, and the test of their gap."""

    treatment_arm: str
    early: float
    late: float
    decay_z: float
    decay_p: float


@dataclass(frozen=True)
class NoveltyCheck:
    """The early-versus-late comparison over the usable checkpoints."""

    metric: str
    window_size: int
    n_usable: int
    n_checkpoints: int
    contrasts: tuple[NoveltyContrast, ...]


@dataclass(frozen=True)
class MonitoringReplay:
    """The daily-monitoring replay of the first contrast with boundaries and adjusted intervals.

    ``z_naive`` is the fixed dashboard threshold and ``naive_look`` the
    first look it would have fired at, for the comparison the readout
    draws.
    """

    metric: str
    treatment_arm: str
    sequential_design: SequentialDesign
    outcome: SequentialOutcome
    intervals: RepeatedIntervals
    z_naive: float
    naive_look: int | None
    n_usable: int
    n_checkpoints: int


@dataclass(frozen=True)
class ExperimentReadout:
    """Every result the readout reports, in its order, with no text.

    ``sensitivity`` holds the smallest detectable effect of the first
    contrast at the realized sizes and the inputs behind it. ``primary``,
    ``cuped``, and ``corrections`` are per contrast. ``excluded_checkpoints``
    counts the trailing checkpoints the time-based blocks could not use.
    """

    experiment_name: str
    data_shape: str
    primary_metric: str
    alpha: float
    arm_sizes: tuple[tuple[str, int], ...]
    validation: tuple[ValidationEvent, ...]
    limitations: tuple[Limitation, ...]
    group_split: tuple[tuple[str, SrmResult], ...] | None
    ratio_stability: tuple[float, int, float, int] | None
    sensitivity: dict[str, float]
    primary: tuple[tuple[str, RateComparison | MeanComparison], ...]
    cuped: tuple[tuple[str, str, CupedResult], ...]
    corrections: tuple[CorrectedComparison, ...]
    heavy_tail: HeavyTailCheck | None
    novelty: NoveltyCheck | None
    monitoring: MonitoringReplay | None
    excluded_checkpoints: int
    truth: dict | None


_Field = TypeVar("_Field")


def _present(value: _Field | None, what: str) -> _Field:
    """Return an optional Experiment field, raising if it is missing. Callers
    use this only where capabilities() has already promised the field."""
    if value is None:
        raise ValueError(f"this block needs the experiment's {what}, which it does not have")
    return value


def _report_skipped_block(title: str) -> None:
    """Print the error being handled, then one line naming the block the
    readout omits because of it. Call only inside an except block."""
    traceback.print_exc()
    print(
        f"note: the readout omits the block '{title}' because of the error above.", file=sys.stderr
    )


def _primary(experiment: Experiment) -> MetricSpec:
    """Return the experiment's primary metric."""
    return next(metric for metric in experiment.metrics if metric.primary)


def _contrasts(experiment: Experiment) -> list[tuple[str, str]]:
    """Return each (control, treatment) arm pair, control first."""
    return [(experiment.arms[0], arm) for arm in experiment.arms[1:]]


def _agg(experiment: Experiment, arm: str, metric: str, checkpoint: float | None) -> ArmAggregate:
    """Return the aggregate cell for this arm, metric, and checkpoint, raising if there is none."""
    for cell in experiment.aggregates:
        if cell.arm == arm and cell.metric == metric and cell.checkpoint == checkpoint:
            return cell
    raise ValueError(f"no aggregate for ({arm}, {metric}, {checkpoint})")


def _series(experiment: Experiment, arm: str, metric: str) -> tuple[NDArray, NDArray, NDArray]:
    """Cumulative (n, mean, variance) arrays for one arm and metric, in
    checkpoint order. Raises if the experiment has no time axis."""
    cells = sorted(
        (
            cell
            for cell in experiment.aggregates
            if cell.arm == arm and cell.metric == metric and cell.checkpoint is not None
        ),
        key=lambda cell: _present(cell.checkpoint, "checkpoint"),
    )
    if not cells:
        raise ValueError(f"no per-checkpoint aggregates for ({arm}, {metric})")
    return (
        np.array([cell.n for cell in cells], dtype=np.float64),
        np.array([cell.mean for cell in cells], dtype=np.float64),
        np.array([cell.variance for cell in cells], dtype=np.float64),
    )


def _window(
    count: NDArray, mean: NDArray, variance: NDArray, start: int, end: int
) -> tuple[int, float, float]:
    """(n, mean, ddof=1 variance) over cumulative checkpoints start..end-1,
    via the same sum-of-squares identity as interval_from_cumulative."""
    sums = count * mean
    squares = (count - 1.0) * variance + count * mean**2
    count_before = 0.0 if start == 0 else count[start - 1]
    sums_before = 0.0 if start == 0 else sums[start - 1]
    squares_before = 0.0 if start == 0 else squares[start - 1]
    window_count = count[end - 1] - count_before
    if window_count < 2:
        raise ValueError(f"window {start}:{end} has fewer than 2 observations")
    window_mean = (sums[end - 1] - sums_before) / window_count
    window_variance = max(
        ((squares[end - 1] - squares_before) - window_count * window_mean**2)
        / (window_count - 1.0),
        0.0,
    )
    if not (np.isfinite(window_mean) and np.isfinite(window_variance)):
        raise ValueError(f"window {start}:{end} has non-finite statistics")
    return int(window_count), float(window_mean), float(window_variance)


def _successes(cell: ArmAggregate) -> int:
    """Return the cell's success count, recovered from its mean and size."""
    return round(cell.mean * cell.n)


def _ratio_stability(
    control_counts: NDArray, treatment_counts: NDArray
) -> tuple[float, int, float, int]:
    """Chi-square homogeneity of the per-interval assignment split.

    Takes the two arms' CUMULATIVE count series and tests whether the
    share of new users landing in each arm is constant across intervals,
    which is the strongest sample-ratio check available when the
    intended split is not on record. Returns (statistic, df, p,
    n_intervals); p is NaN when fewer than two usable intervals exist.
    """
    control_increments = np.diff(np.asarray(control_counts, dtype=np.float64), prepend=0.0)
    treatment_increments = np.diff(np.asarray(treatment_counts, dtype=np.float64), prepend=0.0)
    keep = (control_increments > 0) | (treatment_increments > 0)
    control_increments, treatment_increments = control_increments[keep], treatment_increments[keep]
    if control_increments.sum() == 0.0 or treatment_increments.sum() == 0.0:
        # one arm never gained a user; a split test has nothing to test
        return float("nan"), 0, float("nan"), int(control_increments.size)
    share = treatment_increments.sum() / (control_increments.sum() + treatment_increments.sum())
    expected_treatment = (control_increments + treatment_increments) * share
    expected_control = (control_increments + treatment_increments) * (1.0 - share)
    statistic = float(
        np.sum(
            (treatment_increments - expected_treatment) ** 2 / expected_treatment
            + (control_increments - expected_control) ** 2 / expected_control
        )
    )
    degrees_of_freedom = int(control_increments.size - 1)
    p_value = (
        float(stats.chi2.sf(statistic, df=degrees_of_freedom))
        if degrees_of_freedom > 0
        else float("nan")
    )
    return statistic, degrees_of_freedom, p_value, int(control_increments.size)


def _consistent_prefix(count: NDArray, mean: NDArray, variance: NDArray) -> int:
    """Number of leading checkpoints whose cumulative statistics are
    internally consistent.

    For any real accumulation of observations, the variance recovered for
    the increment between consecutive checkpoints (via the sum-of-squares
    identity) is non-negative. A materially negative recovery means the
    triplet (count, mean, variance) at some checkpoint is not describing
    a superset of the previous checkpoint's sample - a settled recompute,
    a correction, or a join error. Real archives contain such
    checkpoints; the time-based blocks trim to this prefix and say so.
    """
    sums = count * mean
    squares = (count - 1.0) * variance + count * mean**2
    interval_counts = np.diff(count, prepend=0.0)
    interval_sums = np.diff(sums, prepend=0.0)
    interval_squares = np.diff(squares, prepend=0.0)
    interval_means = np.divide(
        interval_sums,
        interval_counts,
        out=np.full_like(interval_counts, np.nan),
        where=interval_counts != 0,
    )
    interval_variances = np.divide(
        interval_squares - interval_counts * interval_means**2,
        interval_counts - 1.0,
        out=np.full_like(interval_counts, np.nan),
        where=interval_counts != 1.0,
    )
    bad = np.nonzero(
        np.isfinite(interval_variances) & (interval_variances < -1e-6 * np.maximum(variance, 1.0))
    )[0]
    return int(bad[0]) if bad.size else int(count.size)


def _cell_test(
    experiment: Experiment, treatment_arm: str, metric: MetricSpec
) -> RateComparison | MeanComparison:
    """The appropriate two-sample test for one (contrast, metric) cell,
    chosen by metric kind and data shape."""
    control = _agg(experiment, experiment.arms[0], metric.name, None)
    treatment = _agg(experiment, treatment_arm, metric.name, None)
    if metric.kind == "binary":
        return two_proportion_ztest(
            _successes(control), control.n, _successes(treatment), treatment.n
        )
    if experiment.rows is not None:
        control_mask = experiment.rows["arm"] == experiment.arms[0]
        treatment_mask = experiment.rows["arm"] == treatment_arm
        return welch_ttest(
            experiment.rows[metric.name][control_mask], experiment.rows[metric.name][treatment_mask]
        )
    return welch_ttest_from_stats(
        control.n, control.mean, control.variance, treatment.n, treatment.mean, treatment.variance
    )


def _arm_values(experiment: Experiment, arm: str, metric: str) -> NDArray[np.float64]:
    """Return one arm's user-level values of a metric, raising without user rows."""
    rows = _present(experiment.rows, "user-level rows")
    return np.asarray(rows[metric][rows["arm"] == arm], dtype=np.float64)


def _group_split(
    experiment: Experiment, metric: MetricSpec, series_usable: bool
) -> tuple[
    tuple[tuple[str, SrmResult], ...] | None,
    tuple[float, int, float, int] | None,
    Limitation | None,
]:
    """The group-split check: against the intended split when it is declared, as ratio stability
    over time when it is not but every arm has a usable series, and nothing otherwise."""
    data_capabilities = capabilities(experiment)
    if "intended_split" in data_capabilities:
        intended_split = _present(experiment.intended_split, "intended split")
        checks: list[tuple[str, SrmResult]] = []
        failing: list[str] = []
        for control_arm, treatment_arm in _contrasts(experiment):
            control_index = experiment.arms.index(control_arm)
            treatment_index = experiment.arms.index(treatment_arm)
            share = intended_split[treatment_index] / (
                intended_split[control_index] + intended_split[treatment_index]
            )
            srm = srm_check(
                _agg(experiment, control_arm, metric.name, None).n,
                _agg(experiment, treatment_arm, metric.name, None).n,
                expected_treatment_share=share,
            )
            checks.append((treatment_arm, srm))
            if srm.p_value < SRM_ALARM_P:
                failing.append(f"{treatment_arm} (p = {srm.p_value:.1e})")
        limitation = (
            Limitation(
                "group_split",
                "group sizes are off from the intended split for "
                + ", ".join(failing)
                + "; users went missing or were assigned unevenly, so no result below can be "
                "trusted",
            )
            if failing
            else None
        )
        return tuple(checks), None, limitation
    if "time_axis" in data_capabilities and series_usable:
        control_arm, treatment_arm = _contrasts(experiment)[0]
        control_counts = _series(experiment, control_arm, metric.name)[0]
        treatment_counts = _series(experiment, treatment_arm, metric.name)[0]
        if np.any(np.diff(control_counts) < 0) or np.any(np.diff(treatment_counts) < 0):
            # a decreasing cumulative count would feed negative interval
            # counts into the chi-square; the intake stops on this for
            # aggregate input, so it is unreachable from a loaded file
            return None, None, None
        stability = _ratio_stability(control_counts, treatment_counts)
        p_value = stability[2]
        limitation = (
            Limitation(
                "group_split",
                f"the share of new users landing in {treatment_arm} shifts across the "
                f"{stability[3]} checkpoints (p = {p_value:.1e}); a split that drifts "
                "mid-experiment means assignment broke, and no result below can be trusted",
            )
            if np.isfinite(p_value) and p_value < SRM_ALARM_P
            else None
        )
        return None, stability, limitation
    return None, None, None


def _sensitivity(
    experiment: Experiment, metric: MetricSpec
) -> tuple[dict[str, float], Limitation | None]:
    """The smallest effect the first contrast could detect at its realized sizes with the target
    power. Binary metrics solve the pooled z-test's power; continuous ones use the normal
    approximation with each arm's realized variance (accurate to the same order as the
    planning formula, which ignores the wrong-direction tail)."""
    control = _agg(experiment, experiment.arms[0], metric.name, None)
    treatment = _agg(experiment, experiment.arms[1], metric.name, None)
    values: dict[str, float] = {
        "n_control": float(control.n),
        "n_treatment": float(treatment.n),
        "control_mean": float(control.mean),
        "alpha": float(experiment.alpha),
        "power": TARGET_POWER,
    }
    if metric.kind == "binary":
        reachable = (
            0.0 < control.mean < 1.0
            and control.n >= 2
            and treatment.n >= 2
            and power_two_proportions(
                control.mean,
                control.mean + (1.0 - control.mean - 1e-9),
                control.n,
                alpha=experiment.alpha,
                n_treatment=treatment.n,
            )
            >= TARGET_POWER
        )
        if not reachable:
            values["mde"] = float("nan")
            values["relative_mde"] = float("nan")
            return values, Limitation(
                "low_sensitivity",
                f"the sample ({control.n:,} and {treatment.n:,} users) cannot reach "
                f"{TARGET_POWER:.0%} power for any lift from a control rate of {control.mean:.1%}",
            )
        mde = mde_two_proportions(
            control.mean,
            control.n,
            alpha=experiment.alpha,
            power=TARGET_POWER,
            n_treatment=treatment.n,
        )
    else:
        if (
            control.n < 2
            or treatment.n < 2
            or not np.isfinite(control.variance)
            or not np.isfinite(treatment.variance)
        ):
            values["mde"] = float("nan")
            values["relative_mde"] = float("nan")
            return values, None
        standard_error = float(
            np.sqrt(control.variance / control.n + treatment.variance / treatment.n)
        )
        mde = (
            stats.norm.isf(experiment.alpha / 2.0) + stats.norm.isf(1.0 - TARGET_POWER)
        ) * standard_error
    values["mde"] = float(mde)
    values["relative_mde"] = float(mde / control.mean) if control.mean != 0.0 else float("nan")
    limitation = None
    if np.isfinite(values["relative_mde"]) and values["relative_mde"] > MODEST_RELATIVE_LIFT:
        limitation = Limitation(
            "low_sensitivity",
            f"the smallest detectable effect on {metric.name} is {values['relative_mde']:.1%} of "
            f"the control level, above a modest {MODEST_RELATIVE_LIFT:.0%} lift; a 'not detected' "
            "result here rules out little",
        )
    return values, limitation


def _few_events(experiment: Experiment, metric: MetricSpec) -> Limitation | None:
    """Flag a binary primary whose arms have too few successes or failures for the z-test."""
    if metric.kind != "binary":
        return None
    thin: list[str] = []
    for arm in experiment.arms:
        cell = _agg(experiment, arm, metric.name, None)
        successes = _successes(cell)
        if min(successes, cell.n - successes) < MIN_EXPECTED_EVENTS:
            thin.append(f"{arm} ({successes:,} of {cell.n:,})")
    if not thin:
        return None
    return Limitation(
        "few_events",
        f"fewer than {MIN_EXPECTED_EVENTS} successes or failures on {metric.name}: "
        + ", ".join(thin)
        + "; the z-test's normal approximation is unreliable here",
    )


def _primary_comparisons(
    experiment: Experiment, metric: MetricSpec
) -> tuple[tuple[str, RateComparison | MeanComparison], ...]:
    """The primary metric's test, one per contrast."""
    return tuple(
        (treatment_arm, _cell_test(experiment, treatment_arm, metric))
        for _, treatment_arm in _contrasts(experiment)
    )


def _cuped(
    experiment: Experiment,
) -> tuple[tuple[tuple[str, str, CupedResult], ...], Limitation | None]:
    """CUPED for every metric that declared a pre-experiment column, per contrast."""
    if experiment.rows is None:
        return (), Limitation(
            "no_pre_period",
            "aggregate data carries no per-user pre-experiment values, so CUPED cannot run",
        )
    if not experiment.pre_period:
        return (), Limitation(
            "no_pre_period",
            "no metric declares a pre-experiment column (pre_column in the config); CUPED is "
            "skipped",
        )
    pre_period = experiment.pre_period
    results: list[tuple[str, str, CupedResult]] = []
    rows = experiment.rows
    for metric in experiment.metrics:
        if metric.name not in pre_period:
            continue
        for control_arm, treatment_arm in _contrasts(experiment):
            control_mask = rows["arm"] == control_arm
            treatment_mask = rows["arm"] == treatment_arm
            results.append(
                (
                    metric.name,
                    treatment_arm,
                    cuped_mean_diff(
                        rows[metric.name][control_mask],
                        rows[metric.name][treatment_mask],
                        pre_period[metric.name][control_mask],
                        pre_period[metric.name][treatment_mask],
                    ),
                )
            )
    return tuple(results), None


def _corrections(experiment: Experiment) -> tuple[CorrectedComparison, ...]:
    """Holm and Benjamini-Hochberg over the non-primary family (or the primary metric's contrasts
    when there is no other metric); empty when the family has fewer than two members."""
    metric_primary = _primary(experiment)
    family: list[tuple[MetricSpec, str]] = [
        (metric, treatment_arm)
        for metric in experiment.metrics
        if not metric.primary
        for _, treatment_arm in _contrasts(experiment)
    ]
    if not family:
        family = [(metric_primary, treatment_arm) for _, treatment_arm in _contrasts(experiment)]
    if len(family) < 2:
        return ()
    comparisons = [_cell_test(experiment, arm, metric) for metric, arm in family]
    naive = np.array([comparison.p_value for comparison in comparisons])
    holm_p_values = holm(naive)
    bh_q_values = benjamini_hochberg(naive)
    return tuple(
        CorrectedComparison(metric.name, arm, comparison, float(holm_p), float(bh_q))
        for (metric, arm), comparison, holm_p, bh_q in zip(
            family, comparisons, holm_p_values, bh_q_values, strict=True
        )
    )


def _heavy_tail_metric(experiment: Experiment) -> MetricSpec | None:
    """The continuous metric the cross-check runs on: the primary when it is continuous, else the
    first continuous metric, else None."""
    primary = _primary(experiment)
    if primary.kind == "continuous":
        return primary
    return next((metric for metric in experiment.metrics if metric.kind == "continuous"), None)


def _heavy_tail(
    experiment: Experiment, random_generator: np.random.Generator
) -> tuple[HeavyTailCheck | None, Limitation | None]:
    """The revenue cross-check on the first contrast of the heavy-tail metric, and the flag when
    the top 1% of users carry more than HEAVY_TAIL_TOP_SHARE of its total."""
    metric = _heavy_tail_metric(experiment)
    if metric is None or experiment.rows is None:
        return None, None
    control_arm, treatment_arm = _contrasts(experiment)[0]
    control_values = _arm_values(experiment, control_arm, metric.name)
    treatment_values = _arm_values(experiment, treatment_arm, metric.name)
    if (
        control_values.size < 2
        or treatment_values.size < 2
        or (control_values.var(ddof=1) == 0.0 and treatment_values.var(ddof=1) == 0.0)
    ):
        return None, None
    pooled = np.concatenate([control_values, treatment_values])
    limitation = None
    total = float(np.abs(pooled).sum())
    if total > 0.0:
        top = max(1, int(np.ceil(0.01 * pooled.size)))
        top_share = float(np.sort(np.abs(pooled))[-top:].sum() / total)
        if top_share > HEAVY_TAIL_TOP_SHARE:
            limitation = Limitation(
                "heavy_tail",
                f"the top 1% of users carry {top_share:.0%} of {metric.name}; the average rests "
                "on a few large values, so the cross-check below compares three methods",
            )
    welch = welch_ttest(control_values, treatment_values)
    bootstrap = bootstrap_mean_diff(
        control_values, treatment_values, random_generator=random_generator, n_resamples=N_RESAMPLES
    )
    permutation = permutation_test_mean_diff(
        control_values,
        treatment_values,
        random_generator=random_generator,
        n_permutations=N_RESAMPLES,
    )
    trimmed = trimmed_mean_diff(control_values, treatment_values, trim=0.01)
    zero_share = float(np.mean(pooled == 0.0))
    # a 1% trim on a mostly-zero metric removes the nonzero values themselves,
    # so the trimmed row stops being a tail check; the text says so when it happens
    trim_degenerate = 2 * int(0.01 * control_values.size) >= int(
        (control_values > 0).sum()
    ) or 2 * int(0.01 * treatment_values.size) >= int((treatment_values > 0).sum())
    return (
        HeavyTailCheck(
            metric.name,
            treatment_arm,
            welch,
            bootstrap,
            permutation,
            trimmed,
            zero_share,
            trim_degenerate,
        ),
        limitation,
    )


def _usable_checkpoints(experiment: Experiment, metric: MetricSpec) -> tuple[int, int] | None:
    """(usable, total) checkpoints of the primary metric across every arm, or None without a time
    axis. Prints and returns None when an arm has no per-checkpoint cells."""
    if "time_axis" not in capabilities(experiment):
        return None
    try:
        all_series = [_series(experiment, arm, metric.name) for arm in experiment.arms]
    except ValueError:
        _report_skipped_block("Early versus late, and Daily monitoring")
        return None
    n_checkpoints = len(_present(experiment.checkpoints, "checkpoints"))
    n_usable = min(_consistent_prefix(*series) for series in all_series)
    return n_usable, n_checkpoints


def _novelty(
    experiment: Experiment, metric: MetricSpec, usable: tuple[int, int]
) -> NoveltyCheck | None:
    """Early-versus-late effect comparison over the usable checkpoints, per contrast."""
    n_usable, n_checkpoints = usable
    if n_usable < MIN_LOOKS:
        return None
    window_size = max(1, n_usable // 3)
    control_series = _series(experiment, experiment.arms[0], metric.name)
    contrasts: list[NoveltyContrast] = []
    for _, treatment_arm in _contrasts(experiment):
        treatment_series = _series(experiment, treatment_arm, metric.name)
        try:
            early_control = _window(*control_series, 0, window_size)
            late_control = _window(*control_series, n_usable - window_size, n_usable)
            early_treatment = _window(*treatment_series, 0, window_size)
            late_treatment = _window(*treatment_series, n_usable - window_size, n_usable)
        except ValueError:
            _report_skipped_block("Early versus late")
            return None
        early = early_treatment[1] - early_control[1]
        late = late_treatment[1] - late_control[1]
        early_standard_error = np.sqrt(
            early_control[2] / early_control[0] + early_treatment[2] / early_treatment[0]
        )
        late_standard_error = np.sqrt(
            late_control[2] / late_control[0] + late_treatment[2] / late_treatment[0]
        )
        if early_standard_error == 0.0 or late_standard_error == 0.0:
            return None
        decay_z = float((early - late) / np.hypot(early_standard_error, late_standard_error))
        decay_p = float(2.0 * stats.norm.sf(abs(decay_z)))
        contrasts.append(
            NoveltyContrast(treatment_arm, float(early), float(late), decay_z, decay_p)
        )
    return NoveltyCheck(metric.name, window_size, n_usable, n_checkpoints, tuple(contrasts))


def _monitoring(
    experiment: Experiment, metric: MetricSpec, usable: tuple[int, int]
) -> MonitoringReplay | None:
    """Replay the first contrast as daily looks under O'Brien-Fleming spending at the sizes
    reached, with adjusted intervals, against the naive fixed threshold."""
    n_usable, n_checkpoints = usable
    if n_usable < MIN_LOOKS:
        return None
    control_arm, treatment_arm = _contrasts(experiment)[0]
    control_counts, control_means, control_variances = (
        values[:n_usable] for values in _series(experiment, control_arm, metric.name)
    )
    treatment_counts, treatment_means, treatment_variances = (
        values[:n_usable] for values in _series(experiment, treatment_arm, metric.name)
    )
    # statistical information for a two-sample z is the inverse summed
    # variance scale 1/(1/n_c + 1/n_t), which equals a share of the total
    # count only when the arm ratio is constant across looks
    information = 1.0 / (1.0 / control_counts + 1.0 / treatment_counts)
    if information[-1] <= 0 or np.any(np.diff(information) <= 0):
        return None
    fractions = information / information[-1]
    if metric.kind == "binary":
        control_conversions = np.round(control_means * control_counts)
        treatment_conversions = np.round(treatment_means * treatment_counts)
        if np.any(np.diff(control_conversions) < 0) or np.any(np.diff(treatment_conversions) < 0):
            # published means at limited precision can reconstruct a count
            # series that dips by one; the stats-based z needs no
            # reconstruction and its divergence from the pooled form is
            # documented on z_from_cumulative_stats
            z_path = z_from_cumulative_stats(
                control_counts,
                control_means,
                control_variances,
                treatment_counts,
                treatment_means,
                treatment_variances,
            )
        else:
            z_path = z_from_cumulative_counts(
                control_conversions, control_counts, treatment_conversions, treatment_counts
            )
    else:
        z_path = z_from_cumulative_stats(
            control_counts,
            control_means,
            control_variances,
            treatment_counts,
            treatment_means,
            treatment_variances,
        )
    sequential_design = design(
        fractions.tolist(), alpha=experiment.alpha, spending="obrien-fleming"
    )
    outcome = monitor(z_path.tolist(), sequential_design)
    # the adjusted intervals use each look's unpooled standard error, the
    # one that describes the difference that is there
    standard_errors = np.sqrt(
        control_variances / control_counts + treatment_variances / treatment_counts
    )
    intervals = repeated_intervals(
        treatment_means - control_means, standard_errors, sequential_design
    )
    # the naive rule is the conventional dashboard threshold, quoted and
    # applied at two decimals as practitioners actually use it
    z_naive = float(np.round(stats.norm.isf(experiment.alpha / 2.0), 2))
    naive_crossings = np.nonzero(np.abs(z_path) >= z_naive)[0]
    naive_look = int(naive_crossings[0]) + 1 if naive_crossings.size else None
    return MonitoringReplay(
        metric.name,
        treatment_arm,
        sequential_design,
        outcome,
        intervals,
        z_naive,
        naive_look,
        n_usable,
        n_checkpoints,
    )


def _missing_days(experiment: Experiment) -> Limitation | None:
    """Flag gaps in the checkpoint sequence wider than the typical spacing allows."""
    checkpoints = experiment.checkpoints
    if checkpoints is None or len(checkpoints) < MIN_LOOKS:
        return None
    spacing = np.diff(np.asarray(checkpoints, dtype=np.float64))
    typical = float(np.median(spacing))
    if typical <= 0.0:
        return None
    gaps = spacing[spacing > MISSING_DAY_GAP_FACTOR * typical]
    if gaps.size == 0:
        return None
    return Limitation(
        "missing_days",
        f"{gaps.size} gap{'s' if gaps.size != 1 else ''} in the checkpoint sequence (the "
        f"largest {gaps.max():g} day{'s' if gaps.max() != 1 else ''} against a typical "
        f"{typical:g}); the time-based blocks treat the checkpoints present as the whole run",
    )


def analyze_experiment(
    experiment: Experiment, random_generator: np.random.Generator
) -> ExperimentReadout:
    """Compute every block of the readout for one experiment.

    ``random_generator`` drives the bootstrap and permutation resamples of
    the revenue cross-check; everything else is deterministic.
    """
    metric = _primary(experiment)
    limitations: list[Limitation] = []
    usable = _usable_checkpoints(experiment, metric)
    group_split, ratio_stability, split_limitation = _group_split(
        experiment, metric, usable is not None
    )
    if split_limitation is not None:
        limitations.append(split_limitation)
    few_events = _few_events(experiment, metric)
    if few_events is not None:
        limitations.append(few_events)
    heavy_tail, tail_limitation = _heavy_tail(experiment, random_generator)
    if tail_limitation is not None:
        limitations.append(tail_limitation)
    sensitivity, sensitivity_limitation = _sensitivity(experiment, metric)
    if sensitivity_limitation is not None:
        limitations.append(sensitivity_limitation)
    cuped, cuped_limitation = _cuped(experiment)
    if cuped_limitation is not None:
        limitations.append(cuped_limitation)
    missing_days = _missing_days(experiment)
    if missing_days is not None:
        limitations.append(missing_days)
    excluded = 0
    novelty = None
    monitoring = None
    if usable is not None:
        excluded = usable[1] - usable[0]
        if excluded > 0:
            limitations.append(
                Limitation(
                    "excluded_checkpoints",
                    f"the final {excluded} checkpoint{'s are' if excluded != 1 else ' is'} "
                    "internally inconsistent and left out of the time-based blocks; an "
                    "increment's recovered variance is negative, which no real accumulation "
                    "of users produces; the end-of-run results still use the final totals",
                )
            )
        novelty = _novelty(experiment, metric, usable)
        monitoring = _monitoring(experiment, metric, usable)
    return ExperimentReadout(
        experiment_name=experiment.name,
        data_shape="user_rows" if experiment.rows is not None else "aggregates",
        primary_metric=metric.name,
        alpha=experiment.alpha,
        arm_sizes=tuple(
            (arm, _agg(experiment, arm, metric.name, None).n) for arm in experiment.arms
        ),
        validation=experiment.validation,
        limitations=tuple(limitations),
        group_split=group_split,
        ratio_stability=ratio_stability,
        sensitivity=sensitivity,
        primary=_primary_comparisons(experiment, metric),
        cuped=cuped,
        corrections=_corrections(experiment),
        heavy_tail=heavy_tail,
        novelty=novelty,
        monitoring=monitoring,
        excluded_checkpoints=excluded,
        truth=experiment.truth,
    )


def planted_end_of_run_lift(truth: dict) -> float | None:
    """The lift on the primary metric a generator recipe implies at the end of the run: the
    persistent ``absolute_lift`` plus the novelty boost averaged over the run's days, when the
    recipe carries those keys; None when it does not."""
    if "absolute_lift" not in truth:
        return None
    lift = float(truth["absolute_lift"])
    if all(key in truth for key in ("novelty_boost", "novelty_half_life_days", "n_days")):
        days = np.arange(int(truth["n_days"]))
        decay = 0.5 ** (days / float(truth["novelty_half_life_days"]))
        lift += float(truth["novelty_boost"]) * float(decay.mean())
    return lift

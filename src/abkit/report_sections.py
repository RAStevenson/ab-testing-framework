"""Presence-gated section builders: from an Experiment to SectionSpecs.

Each ``sec_*`` function runs one analysis over an ``abkit.io.Experiment``
and returns a ``SectionSpec`` for the report engine, or ``None`` when the
analysis is impossible even though the capability gate passed (an
all-constant metric, say). ``default_sections`` assembles the standard
report in order. Prose here is the neutral template voice shared by every
data source; a dataset-specific builder such as the checkout one may
append its own commentary paragraphs to any section body before rendering.

Figures are drawn at section-build time into the given directory.
matplotlib is imported lazily inside the figure helpers, so the library's
runtime dependencies stay numpy + scipy for everything except report
building, matching the existing split where matplotlib is a dev
dependency of the report scripts.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from abkit.frequentist import (
    benjamini_hochberg,
    holm,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_stats,
)
from abkit.io import (
    ArmAggregate,
    Experiment,
    MetricSpec,
    _metric_specs,
    capabilities,
)
from abkit.power import mde_two_proportions
from abkit.report import FigureSpec, SectionSpec
from abkit.resampling import bootstrap_mean_diff, permutation_test_mean_diff, trimmed_mean_diff
from abkit.results import BootstrapResult, MeanComparison, PermutationResult, RateComparison

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from abkit.sequential import (
    design,
    monitor,
    z_from_cumulative_counts,
    z_from_cumulative_stats,
)

COLOR_CONTROL = "#4269d0"
COLOR_TREATMENT = "#d4615c"
COLOR_REFERENCE = "#8a8f98"
COLOR_INK = "#31343a"
_EXTRA_ARM_COLORS = ("#7fc97f", "#beaed4", "#fdc086")


def pct(fraction: float, digits: int = 2) -> str:
    return f"{100 * fraction:.{digits}f}%"


def pp(difference: float, digits: int = 2) -> str:
    return f"{100 * difference:.{digits}f} pp"


def pp_signed(difference: float, digits: int = 2) -> str:
    """Signed percentage-point difference, so effect columns read one way."""
    return f"{100 * difference:+.{digits}f} pp"


def fmt_p(p_value: float) -> str:
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}" if p_value < 0.001 else f"{p_value:.3f}"


def fmt_perm_p(permutation: PermutationResult) -> str:
    """Permutation p-value as report text, flagged when it sits at the estimator's floor.

    The add-one estimator's smallest possible value is 1 / (n_permutations + 1),
    reached when no relabeling is as extreme as the observed difference. That
    value is still a valid, conservative p-value, but printed bare it reads as
    a measurement rather than as the resolution limit of the relabeling count.
    """
    if permutation.p_value <= 1.0 / (permutation.n_permutations + 1):
        return (
            f"p = {fmt_p(permutation.p_value)}, the smallest value "
            f"{permutation.n_permutations:,} relabelings can give"
        )
    return f"p = {fmt_p(permutation.p_value)}"


def _style_axis(ax: Axes, *, grid_axis: Literal["both", "x", "y"] = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def _arm_color(index: int) -> str:
    if index == 0:
        return COLOR_CONTROL
    if index == 1:
        return COLOR_TREATMENT
    return _EXTRA_ARM_COLORS[(index - 2) % len(_EXTRA_ARM_COLORS)]


def _report_skipped_section(title: str) -> None:
    """Print the error being handled, then one line naming the section the
    report omits because of it. Call only inside an except block."""
    traceback.print_exc()
    print(
        f"note: the report omits the section '{title}' because of the error above.", file=sys.stderr
    )


_Field = TypeVar("_Field")


def _present(value: _Field | None, what: str) -> _Field:
    """Return an optional Experiment field, raising if it is missing. Sections
    call this only where capabilities() has already promised the field."""
    if value is None:
        raise ValueError(f"this section needs the experiment's {what}, which it does not have")
    return value


def _primary(experiment: Experiment) -> MetricSpec:
    return next(metric for metric in experiment.metrics if metric.primary)


def _contrasts(experiment: Experiment) -> list[tuple[str, str]]:
    return [(experiment.arms[0], arm) for arm in experiment.arms[1:]]


def _agg(experiment: Experiment, arm: str, metric: str, checkpoint: float | None) -> ArmAggregate:
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
    checkpoints; sections trim to this prefix and say so in prose.
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


def _effect_and_p(comparison: RateComparison | MeanComparison) -> tuple[str, float]:
    if isinstance(comparison, RateComparison):
        return pp_signed(comparison.absolute_lift), comparison.p_value
    return f"{comparison.difference:+.3f}", comparison.p_value


def _contrast_label(experiment: Experiment, treatment_arm: str) -> str:
    return f"{treatment_arm} vs {experiment.arms[0]}"


def sec_validation(experiment: Experiment) -> SectionSpec:
    """Always renders: what the intake did to the data before any statistic."""
    sizes = ", ".join(
        f"{arm} {_agg(experiment, arm, _primary(experiment).name, None).n:,}"
        for arm in experiment.arms
    )
    if experiment.validation:
        rows = "\n".join(
            f"| {event.rule} | {event.count:,} | {event.detail} |"
            for event in experiment.validation
        )
        body = (
            f"Arm sizes after validation: {sizes} (denominators are the primary "
            "metric's). Every change the intake made or refused is listed here, "
            "because silently cleaned data is how an analysis stops being about "
            "the experiment that actually ran.\n\n"
            "| Rule | Rows | Where |\n|---|---|---|\n" + rows
        )
    else:
        body = (
            f"Arm sizes: {sizes} (denominators are the primary metric's). "
            "The intake altered nothing and dropped nothing. The data passed "
            "every validation rule as delivered."
        )
    return SectionSpec(key="validation", title="Data validation", body=body)


def sec_design_power(experiment: Experiment) -> SectionSpec | None:
    """Sensitivity at the realized sample size, for a binary primary metric."""
    metric = _primary(experiment)
    if metric.kind != "binary":
        return None
    control = _agg(experiment, experiment.arms[0], metric.name, None)
    treatment = _agg(experiment, experiment.arms[1], metric.name, None)
    if not 0.0 < control.mean < 1.0:
        return None
    mde = mde_two_proportions(
        control.mean, control.n, alpha=experiment.alpha, n_treatment=treatment.n
    )
    body = (
        f"At the realized sample sizes of the first contrast ({control.n:,} "
        f"control and {treatment.n:,} treatment users) and the control rate of "
        f"{pct(control.mean, 1)}, the smallest absolute lift that contrast could "
        f"detect with 80% power at alpha {experiment.alpha} is {100 * mde:.2f} "
        "percentage points (pp). This is the minimum detectable effect (MDE). "
        "Effects smaller than that could be real and still "
        "produce an "
        "unremarkable p-value here. That is a property of the sample size, not "
        "evidence of absence."
    )
    return SectionSpec(key="design_power", title="Sensitivity at this sample size", body=body)


def sec_srm(experiment: Experiment) -> SectionSpec:
    """Assignment validation: intended-split chi-square when the split is
    declared, ratio-stability over time when it is not but a time axis
    exists, and a stated non-test otherwise."""
    metric = _primary(experiment)
    data_capabilities = capabilities(experiment)
    if "intended_split" in data_capabilities:
        lines = ["| Contrast | Control n | Treatment n | Chi-square | p |", "|---|---|---|---|---|"]
        worst_p_value = 1.0
        intended_split = _present(experiment.intended_split, "intended split")
        for control_arm, treatment_arm in _contrasts(experiment):
            control_index = experiment.arms.index(control_arm)
            treatment_index = experiment.arms.index(treatment_arm)
            share = intended_split[treatment_index] / (
                intended_split[control_index] + intended_split[treatment_index]
            )
            n_control = _agg(experiment, control_arm, metric.name, None).n
            n_treatment = _agg(experiment, treatment_arm, metric.name, None).n
            srm = srm_check(n_control, n_treatment, expected_treatment_share=share)
            worst_p_value = min(worst_p_value, srm.p_value)
            lines.append(
                f"| {_contrast_label(experiment, treatment_arm)} | {n_control:,} | {n_treatment:,} "
                f"| {srm.statistic:.2f} | {fmt_p(srm.p_value)} |"
            )
        srm_conclusion = (
            "No contrast fails the check (conventional alarm threshold p < 1e-3), "
            "so the realized split is consistent with the declared one."
            if worst_p_value >= 1e-3
            else "At least one contrast FAILS the check: users went missing or were "
            "assigned asymmetrically, and no downstream statistic can repair that. "
            "The results below describe the data as collected, with that caveat."
        )
        body = (
            "A sample-ratio mismatch (SRM) check asks whether the number of users "
            "who actually landed in each arm matches the intended split. A failure "
            "means the assignment or logging lost users asymmetrically, which "
            "biases every metric in ways no downstream statistic can fix.\n\n"
            + "\n".join(lines)
            + "\n\n"
            + srm_conclusion
        )
    elif "time_axis" in data_capabilities:
        lines = ["| Contrast | Checkpoints | Chi-square | df | p |", "|---|---|---|---|---|"]
        worst_p_value = 1.0
        for control_arm, treatment_arm in _contrasts(experiment):
            control_counts = _series(experiment, control_arm, metric.name)[0]
            treatment_counts = _series(experiment, treatment_arm, metric.name)[0]
            if np.any(np.diff(control_counts) < 0) or np.any(np.diff(treatment_counts) < 0):
                # a decreasing cumulative count would feed negative
                # interval counts into the chi-square; say so instead
                lines.append(
                    f"| {_contrast_label(experiment, treatment_arm)} | - | not testable "
                    "(cumulative counts decrease) | - | - |"
                )
                continue
            statistic, degrees_of_freedom, p_value, n_intervals = _ratio_stability(
                control_counts, treatment_counts
            )
            if not np.isfinite(p_value):
                lines.append(
                    f"| {_contrast_label(experiment, treatment_arm)} | {n_intervals} | "
                    "not testable (degenerate split) | - | - |"
                )
                continue
            worst_p_value = min(worst_p_value, p_value)
            lines.append(
                f"| {_contrast_label(experiment, treatment_arm)} | {n_intervals} | {statistic:.2f} "
                f"| {degrees_of_freedom} | {fmt_p(p_value)} |"
            )
        srm_conclusion = (
            "The arm ratio is stable across checkpoints in every contrast "
            "(alarm threshold p < 1e-3), which is what can still be checked "
            "without a declared split."
            if worst_p_value >= 1e-3
            else "The arm ratio SHIFTS over time in at least one contrast (p < 1e-3). "
            "A ratio that drifts mid-experiment is sample-ratio-mismatch evidence "
            "regardless of what split was intended, and the checkpoint where it "
            "shifts shows when assignment broke."
        )
        body = (
            "The intended allocation for this experiment is not on record, so a "
            "direct sample-ratio mismatch (SRM) test against it is impossible. "
            "What can be tested without it is RATIO STABILITY. Whichever split was "
            "intended, it was presumably constant, so the share of new users "
            "landing in each arm should not change across checkpoints. The test "
            "below is a chi-square for homogeneity of the per-checkpoint "
            "assignment counts (denominators from the primary metric).\n\n"
            + "\n".join(lines)
            + "\n\n"
            + srm_conclusion
        )
    else:
        sizes = ", ".join(
            f"{arm} {_agg(experiment, arm, metric.name, None).n:,}" for arm in experiment.arms
        )
        body = (
            f"Observed arm sizes: {sizes}. The intended allocation is not on "
            "record and the data has no time axis, so neither a direct "
            "sample-ratio mismatch test nor a ratio-stability test is possible. "
            "The sizes are reported for the reader's own judgment, and that "
            "absence of a check is itself worth knowing."
        )
    return SectionSpec(key="srm", title="Assignment validation", body=body)


def sec_primary(experiment: Experiment) -> SectionSpec:
    """The primary metric, per contrast."""
    metric = _primary(experiment)
    lines = ["| Contrast | Effect | 95% CI | p |", "|---|---|---|---|"]
    for _, treatment_arm in _contrasts(experiment):
        comparison = _cell_test(experiment, treatment_arm, metric)
        if isinstance(comparison, RateComparison):
            effect = pp_signed(comparison.absolute_lift)
            interval_text = f"{pp_signed(comparison.ci_low)} to {pp_signed(comparison.ci_high)}"
        else:
            effect = f"{comparison.difference:+.3f}"
            interval_text = f"{comparison.ci_low:+.3f} to {comparison.ci_high:+.3f}"
        lines.append(
            f"| {_contrast_label(experiment, treatment_arm)} | {effect} | {interval_text} "
            f"| {fmt_p(comparison.p_value)} |"
        )
    body = (
        f"The primary metric is **{metric.name}** ({metric.kind}). Each treatment "
        "arm is compared against control. The test is a pooled two-proportion "
        "z-test for binary metrics and a Welch t-test for continuous ones. Read "
        "each row as an interval first. The 95% CI is the range of effects the "
        "data is consistent with, and p is how often an effect at least this large would "
        "appear if the arms were identical. A narrow interval clear of zero means "
        "the run measured the effect. A wide interval spanning zero means the run "
        "did not pin it down, which differs from showing no effect exists.\n\n" + "\n".join(lines)
    )
    return SectionSpec(key="primary", title="Primary metric", body=body)


def sec_novelty(experiment: Experiment, figure_dir: Path) -> SectionSpec | None:
    """Early-versus-late effect comparison over the time axis."""
    metric = _primary(experiment)
    control_arm = experiment.arms[0]
    try:
        control_series = _series(experiment, control_arm, metric.name)
        all_series = [control_series] + [
            _series(experiment, arm, metric.name) for _, arm in _contrasts(experiment)
        ]
    except ValueError:
        _report_skipped_section("Effect stability over time")
        return None
    n_checkpoints = len(_present(experiment.checkpoints, "checkpoints"))
    n_usable = min(_consistent_prefix(*series) for series in all_series)
    if n_usable < 3:
        return None
    trim_note = (
        ""
        if n_usable == n_checkpoints
        else "\n\n"
        + (
            "The final checkpoint is EXCLUDED from this section"
            if n_checkpoints - n_usable == 1
            else f"The final {n_checkpoints - n_usable} checkpoints are EXCLUDED from this section"
        )
        + ". Each increment's variance can be recovered from consecutive "
        "cumulative counts, means, and variances. For the first excluded "
        "increment that recovery is negative, which no real accumulation of "
        "observations can produce. "
        "The statistics published there therefore do not describe a superset "
        "of the previous checkpoint's sample. A recompute after late-arriving "
        "data or a correction published in place would produce this. The "
        "exclusion is stated here rather than applied silently. The "
        "end-of-run results in the other sections still use the final "
        "checkpoint's published totals."
    )
    window_size = max(1, n_usable // 3)

    lines = ["| Contrast | Early effect | Late effect | Decay z | p |", "|---|---|---|---|---|"]
    for _, treatment_arm in _contrasts(experiment):
        treatment_series = _series(experiment, treatment_arm, metric.name)
        try:
            early_control = _window(*control_series, 0, window_size)
            late_control = _window(*control_series, n_usable - window_size, n_usable)
            early_treatment = _window(*treatment_series, 0, window_size)
            late_treatment = _window(*treatment_series, n_usable - window_size, n_usable)
        except ValueError:
            _report_skipped_section("Effect stability over time")
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
        decay_z_score = float((early - late) / np.hypot(early_standard_error, late_standard_error))
        decay_p_value = float(2.0 * stats.norm.sf(abs(decay_z_score)))
        early_text = pp_signed(early) if metric.kind == "binary" else f"{early:+.3f}"
        late_text = pp_signed(late) if metric.kind == "binary" else f"{late:+.3f}"
        lines.append(
            f"| {_contrast_label(experiment, treatment_arm)} | {early_text} | {late_text} "
            f"| {decay_z_score:.2f} | {fmt_p(decay_p_value)} |"
        )

    _fig_novelty(experiment, metric, figure_dir, n_usable)
    body = (
        "A treatment effect that fades within the run is the signature of a "
        "novelty effect, where users react to the change itself and the "
        "reaction decays. The comparison below tests the first "
        + (f"{window_size} checkpoints" if window_size != 1 else "checkpoint")
        + f" against the last {window_size} directly, because two "
        "separately significant windows are not by themselves evidence that the "
        "windows differ. Interval values are recovered from the cumulative "
        "series by differencing. {{fig:novelty}} shows the per-interval means.\n\n"
        + "\n".join(lines)
        + "\n\nA significant z is evidence that the early and late windows "
        "differ, so the end-of-run cumulative number blends two regimes and "
        "can sit far from the late-window effect that best describes where "
        "the experiment ended up. The z is early minus late. A positive z "
        "means the effect moved down over the run, and a negative z means it "
        "moved up. Whether that is novelty fading, a harm emerging, or a "
        "primacy pattern depends on the effect signs in the table. Either way "
        "the late window is the number to quote."
        + (
            ""
            if experiment.rows is not None
            else "\n\nOne assumption matters for aggregate archives. "
            "Differencing cumulative statistics treats each increment as new "
            "users only. An archive that recomputes its metrics over all "
            "enrolled users at each checkpoint blurs the increments, and this "
            "test is approximate there."
        )
        + trim_note
    )
    return SectionSpec(
        key="novelty",
        title="Effect stability over time",
        body=body,
        requires=frozenset({"time_axis"}),
        figures=(FigureSpec("novelty", "novelty.png", f"Per-interval {metric.name} by arm"),),
    )


def _fig_novelty(
    experiment: Experiment, metric: MetricSpec, figure_dir: Path, n_usable: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, arm in enumerate(experiment.arms):
        count, mean, variance = _series(experiment, arm, metric.name)
        count, mean, variance = count[:n_usable], mean[:n_usable], variance[:n_usable]
        # computed directly rather than via interval_from_cumulative so a
        # checkpoint where this arm gained no users plots as a gap instead
        # of raising; means are exact, and the figure needs no variances
        interval_counts = np.diff(count, prepend=0.0)
        interval_sums = np.diff(count * mean, prepend=0.0)
        interval_means = np.where(
            interval_counts > 0, interval_sums / np.maximum(interval_counts, 1.0), np.nan
        )
        checkpoints = np.asarray(
            _present(experiment.checkpoints, "checkpoints")[:n_usable], dtype=np.float64
        )
        ax.plot(
            checkpoints,
            interval_means,
            color=_arm_color(i),
            linewidth=2,
            marker="o",
            markersize=4,
            label=arm,
        )
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(f"Interval mean of {metric.name}")
    ax.set_title("Per-interval means by arm", loc="left", color=COLOR_INK)
    ax.legend(frameon=False, fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "novelty.png", dpi=150)
    plt.close(fig)


def sec_sequential(experiment: Experiment, figure_dir: Path) -> SectionSpec | None:
    """Replay of the run as sequential looks under an alpha-spending design,
    against the naive fixed-threshold rule. First contrast only: a
    monitoring design guards one comparison."""
    metric = _primary(experiment)
    control_arm, treatment_arm = _contrasts(experiment)[0]
    try:
        control_counts, control_means, control_variances = _series(
            experiment, control_arm, metric.name
        )
        treatment_counts, treatment_means, treatment_variances = _series(
            experiment, treatment_arm, metric.name
        )
    except ValueError:
        _report_skipped_section("What sequential monitoring would have decided")
        return None
    n_checkpoints = int(control_counts.size)
    n_usable = min(
        _consistent_prefix(control_counts, control_means, control_variances),
        _consistent_prefix(treatment_counts, treatment_means, treatment_variances),
    )
    if n_usable < 3:
        return None
    control_counts, control_means, control_variances = (
        control_counts[:n_usable],
        control_means[:n_usable],
        control_variances[:n_usable],
    )
    treatment_counts, treatment_means, treatment_variances = (
        treatment_counts[:n_usable],
        treatment_means[:n_usable],
        treatment_variances[:n_usable],
    )
    trim_note = (
        ""
        if n_usable == n_checkpoints
        else "\n\nAs in the effect-stability analysis, "
        + (
            "the final checkpoint is excluded"
            if n_checkpoints - n_usable == 1
            else f"the final {n_checkpoints - n_usable} checkpoints are excluded"
        )
        + " because the cumulative statistics there are internally "
        f"inconsistent. The replay covers the {n_usable} consistent looks."
    )
    # statistical information for a two-sample z is the inverse summed
    # variance scale 1/(1/control_counts + 1/treatment_counts), which equals a share of the total
    # count only when the arm ratio is constant across looks
    information = 1.0 / (1.0 / control_counts + 1.0 / treatment_counts)
    if information[-1] <= 0 or np.any(np.diff(information) <= 0):
        return None
    fractions = information / information[-1]
    if metric.kind == "binary":
        control_conversions = np.round(control_means * control_counts)
        treatment_conversions = np.round(treatment_means * treatment_counts)
        if np.any(np.diff(control_conversions) < 0) or np.any(np.diff(treatment_conversions) < 0):
            # published means at limited precision can reconstruct a
            # count series that dips by one; the stats-based z needs no
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
    sequential_section = design(
        fractions.tolist(), alpha=experiment.alpha, spending="obrien-fleming"
    )
    outcome = monitor(z_path.tolist(), sequential_section)
    # the naive rule is the conventional dashboard threshold, quoted and
    # applied at two decimals as practitioners actually use it
    z_naive = float(np.round(stats.norm.isf(experiment.alpha / 2.0), 2))
    naive_crossings = np.nonzero(np.abs(z_path) >= z_naive)[0]
    naive_look = int(naive_crossings[0]) + 1 if naive_crossings.size else None

    _fig_sequential(z_path, np.array(sequential_section.boundaries), z_naive, figure_dir)

    if outcome.stopped_at_look is not None:
        spend_sentence = (
            f"The spending design first crosses at look {outcome.stopped_at_look} "
            f"of {len(fractions)}, at which point stopping is justified with the "
            "full run's false-positive guarantee intact."
        )
    else:
        spend_sentence = (
            f"The spending design never crosses in {len(fractions)} looks, so "
            "under monitoring priced for peeking this run does not support "
            "stopping early in either direction."
        )
    if naive_look is not None:
        naive_sentence = f"The naive rule would have called the result at look {naive_look}" + (
            ", earlier than the spending design and on evidence the design "
            "does not consider sufficient."
            if outcome.stopped_at_look is None or naive_look < outcome.stopped_at_look
            else "."
        )
    else:
        naive_sentence = "The naive rule never fires either. The two agree on this run."

    body = (
        "Anyone watching a dashboard peeks. Checking the result at every "
        f"checkpoint against a fixed |z| > {z_naive:.2f} threshold (the naive rule) "
        "inflates the false-positive rate well beyond the nominal alpha, "
        "because every look is another chance to be fooled. An alpha-spending "
        "design prices the peeking by allocating the total alpha across the "
        "realized look schedule and computing a boundary for each look "
        "(O'Brien-Fleming spending, stingy early and generous late). "
        "{{fig:sequential}} shows the observed |z| path against both rules "
        f"for the first contrast ({_contrast_label(experiment, treatment_arm)}). A "
        "monitoring design guards one pre-declared comparison, and this "
        "replay takes the first.\n\n"
        + spend_sentence
        + " "
        + naive_sentence
        + "\n\nEven a correctly stopped estimate is biased away from zero at the "
        "stopping look. Monitoring is a tool for catching harmful changes early, "
        "not for launching apparent improvements early."
        + (
            ""
            if len(fractions) <= 48
            else f"\n\nOne numerical caveat applies at this look count. The boundaries "
            "come from a recursive numerical integration whose grid convergence "
            "is pinned by test on a 14-look schedule, so treat the "
            f"{len(fractions)}-look boundary values as approximate."
        )
        + trim_note
    )
    return SectionSpec(
        key="sequential",
        title="What sequential monitoring would have decided",
        body=body,
        requires=frozenset({"time_axis"}),
        figures=(
            FigureSpec(
                "sequential",
                "sequential.png",
                "Observed |z| against spending boundaries and the naive threshold",
            ),
        ),
    )


def _fig_sequential(z_path: NDArray, boundaries: NDArray, z_naive: float, figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    looks = np.arange(1, z_path.size + 1)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(
        looks,
        np.abs(z_path),
        color=COLOR_CONTROL,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Observed |z|",
    )
    finite = np.isfinite(boundaries)
    ax.plot(
        looks[finite],
        boundaries[finite],
        color=COLOR_INK,
        linewidth=2,
        drawstyle="steps-mid",
        label="O'Brien-Fleming boundary",
    )
    ax.axhline(
        z_naive,
        color=COLOR_REFERENCE,
        linewidth=1.5,
        linestyle="--",
        label=f"Naive fixed threshold ({z_naive:.2f})",
    )
    ax.set_xlabel("Look")
    ax.set_ylabel("|z|")
    top_values = [float(np.abs(z_path).max()), z_naive]
    if finite.any():
        top_values.append(float(boundaries[finite].max()))
    ax.set_ylim(top=1.2 * max(top_values))
    ax.set_title("Sequential monitoring replay", loc="left", color=COLOR_INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "sequential.png", dpi=150)
    plt.close(fig)


def sec_corrections(experiment: Experiment, figure_dir: Path) -> SectionSpec | None:
    """Multiple-comparison corrections over the non-primary family."""
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
        return None
    comparisons = [_cell_test(experiment, arm, metric) for metric, arm in family]
    naive = np.array([comparison.p_value for comparison in comparisons])
    holm_p_values = holm(naive)
    bh_q_values = benjamini_hochberg(naive)

    lines = [
        "| Metric | Contrast | Effect | Unadjusted p | Holm p | BH q |",
        "|---|---|---|---|---|---|",
    ]
    for (metric, arm), comparison, holm_p, bh_q in zip(
        family, comparisons, holm_p_values, bh_q_values, strict=True
    ):
        effect, p_value = _effect_and_p(comparison)
        lines.append(
            f"| {metric.name} | {_contrast_label(experiment, arm)} | {effect} | {fmt_p(p_value)} "
            f"| {fmt_p(float(holm_p))} | {fmt_p(float(bh_q))} |"
        )
    n_naive = int(np.sum(naive < experiment.alpha))
    n_holm = int(np.sum(holm_p_values < experiment.alpha))
    family_error = 1.0 - (1.0 - experiment.alpha) ** len(family)
    body = (
        (
            f"This experiment has {len(family)} comparisons beyond the primary metric. "
            if any(not metric.primary for metric, _ in family)
            else f"With one metric and several arms, the family is the primary "
            f"metric's {len(family)} contrasts. "
        )
        + "Testing each at alpha "
        f"{experiment.alpha} without adjustment gives roughly a "
        f"{family_error:.0%} chance of at least one false alarm even if nothing "
        "moved (treating the comparisons as independent, which shared arms "
        "make approximate). The family is therefore adjusted two ways: Holm, which controls the "
        "chance of ANY false claim (the right standard for a ship decision), "
        "and Benjamini-Hochberg, which controls the false-discovery RATE (the "
        "right standard for a screen that feeds follow-ups).\n\n"
        + "\n".join(lines)
        + (
            f"\n\nUnadjusted, {n_naive} comparison"
            + ("" if n_naive == 1 else "s")
            + (" clears" if n_naive == 1 else " clear")
            + f" alpha. After Holm, {n_holm}"
            + (" survives" if n_holm == 1 else " survive")
        )
        + ". {{fig:corrections}} shows each comparison's "
        "p-value before and after adjustment."
    )
    _fig_corrections(
        [f"{metric.name}, {arm}" for metric, arm in family],
        naive,
        holm_p_values,
        experiment.alpha,
        figure_dir,
    )
    return SectionSpec(
        key="corrections",
        title="Multiple comparisons",
        body=body,
        requires=frozenset({"multi_metric"}),
        figures=(
            FigureSpec(
                "corrections", "corrections.png", "Unadjusted versus Holm-adjusted p-values"
            ),
        ),
    )


def _fig_corrections(
    labels: list[str], naive: NDArray, adjusted: NDArray, alpha: float, figure_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.8))
    positions = np.arange(len(labels))
    ax.scatter(naive, positions, s=55, color=COLOR_CONTROL, label="Unadjusted p", zorder=3)
    ax.scatter(
        adjusted, positions, s=55, color=COLOR_TREATMENT, marker="s", label="Holm p", zorder=3
    )
    for position in positions:
        ax.plot(
            [naive[position], adjusted[position]],
            [position, position],
            color=COLOR_REFERENCE,
            linewidth=1,
            zorder=2,
        )
    ax.axvline(alpha, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xscale("log")
    ax.set_xlabel("p-value (log scale)")
    ax.set_title("Adjustment across the comparison family", loc="left", color=COLOR_INK)
    ax.legend(frameon=False, fontsize=9)
    ax.invert_yaxis()
    _style_axis(ax, grid_axis="x")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "corrections.png", dpi=150)
    plt.close(fig)


def sec_heavy_tail(
    experiment: Experiment, random_generator: np.random.Generator, figure_dir: Path
) -> SectionSpec | None:
    """Resampling-based analysis of a heavy-tailed continuous metric."""
    metric = (
        _primary(experiment)
        if _primary(experiment).kind == "continuous"
        else next((metric for metric in experiment.metrics if metric.kind == "continuous"), None)
    )
    if metric is None:
        return None
    control_arm, treatment_arm = _contrasts(experiment)[0]
    rows = _present(experiment.rows, "user-level rows")
    control_values = rows[metric.name][rows["arm"] == control_arm]
    treatment_values = rows[metric.name][rows["arm"] == treatment_arm]
    if (
        control_values.size < 2
        or treatment_values.size < 2
        or (control_values.var(ddof=1) == 0.0 and treatment_values.var(ddof=1) == 0.0)
    ):
        return None
    welch = welch_ttest(control_values, treatment_values)
    bootstrap = bootstrap_mean_diff(
        control_values, treatment_values, random_generator=random_generator, n_resamples=10_000
    )
    permutation = permutation_test_mean_diff(
        control_values, treatment_values, random_generator=random_generator, n_permutations=10_000
    )
    trimmed = trimmed_mean_diff(control_values, treatment_values, trim=0.01)
    zero_share = float(np.mean(np.concatenate([control_values, treatment_values]) == 0.0))
    # a 1% trim on a mostly-zero metric removes the nonzero values themselves,
    # so the trimmed row stops being a tail check; say so when it happens
    trim_degenerate = 2 * int(0.01 * control_values.size) >= int(
        (control_values > 0).sum()
    ) or 2 * int(0.01 * treatment_values.size) >= int((treatment_values > 0).sum())

    _fig_heavy_tail(
        control_values, treatment_values, welch, bootstrap, trimmed, metric.name, figure_dir
    )
    boot_excludes_zero = (bootstrap.ci_low > 0.0) == (bootstrap.ci_high > 0.0)
    welch_significant = welch.p_value < experiment.alpha
    agreement = (
        "The three same-estimand methods agree here, which is what a well-behaved tail looks like."
        if welch_significant == (permutation.p_value < experiment.alpha)
        and boot_excludes_zero == welch_significant
        else "The methods DISAGREE at the alpha line. That disagreement is the "
        "finding. It says the estimate rests on the tail, and the right move "
        "is to report it rather than quote whichever method crossed the line."
    )
    trim_note = (
        f" The trimmed row is not a tail check on this data. With {pct(zero_share, 0)} "
        "of users at zero, cutting 1% from each end of each arm removes most or all "
        "of the users with a nonzero value along with the extreme ones, so the "
        "trimmed difference sits near zero whatever the true effect."
        if trim_degenerate
        else ""
    )
    body = (
        f"**{metric.name}** is the kind of metric where a few extreme users can "
        "dominate the average"
        + (
            f" ({pct(zero_share, 0)} of users in the first contrast sit at zero)"
            if zero_share > 0.5
            else ""
        )
        + f". The first contrast ({_contrast_label(experiment, treatment_arm)}) is "
        "analyzed four ways, each stressing the tail differently: a Welch t-test (relies "
        "on the normal approximation of the mean), a percentile bootstrap "
        "(resamples the observed tail), a permutation test (no normal "
        "approximation, with a p-value computed from 10,000 random "
        "relabelings under the null that the arms' distributions are "
        "identical), and a trimmed-mean difference. The trimmed-mean difference "
        "is a diagnostic for how much the estimate depends on the tail. It "
        "deliberately estimates a "
        "different quantity (a different estimand), so its number is not comparable to the other "
        "three.\n\n"
        f"| Method | Estimate | Interval / p |\n|---|---|---|\n"
        f"| Welch t-test | {welch.difference:+.3f} | 95% CI {welch.ci_low:+.3f} "
        f"to {welch.ci_high:+.3f}, p = {fmt_p(welch.p_value)} |\n"
        f"| Percentile bootstrap | {bootstrap.estimate:+.3f} | 95% CI {bootstrap.ci_low:+.3f} "
        f"to {bootstrap.ci_high:+.3f} |\n"
        f"| Permutation test | {permutation.observed:+.3f} | {fmt_perm_p(permutation)} |\n"
        f"| Trimmed mean diff (1%) | {trimmed:+.3f} | different estimand |\n\n"
        f"{{{{fig:heavy_tail}}}} shows the distributions and the interval "
        "comparison. " + agreement + trim_note
    )
    return SectionSpec(
        key="heavy_tail",
        title=f"Heavy-tailed metric: {metric.name}",
        body=body,
        requires=frozenset({"user_rows", "continuous_metric"}),
        figures=(
            FigureSpec("heavy_tail", "heavy_tail.png", "Distribution and interval comparison"),
        ),
    )


def _fig_heavy_tail(
    control_values: NDArray,
    treatment_values: NDArray,
    welch: MeanComparison,
    bootstrap: BootstrapResult,
    trimmed: float,
    metric_name: str,
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (histogram_ax, interval_ax) = plt.subplots(1, 2, figsize=(10, 4.2))
    control_spenders, treatment_spenders = (
        control_values[control_values > 0],
        treatment_values[treatment_values > 0],
    )
    if control_spenders.size and treatment_spenders.size:
        top = float(max(control_spenders.max(), treatment_spenders.max()))
        bins = np.geomspace(
            max(min(control_spenders.min(), treatment_spenders.min()), 1e-9), top, 40
        )
        histogram_ax.hist(
            control_spenders,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=COLOR_CONTROL,
            label="control",
        )
        histogram_ax.hist(
            treatment_spenders,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=COLOR_TREATMENT,
            label="treatment",
        )
        histogram_ax.set_xscale("log")
        histogram_ax.set_yscale("log")
    histogram_ax.set_xlabel(f"{metric_name} per user, positive values (log scale)")
    histogram_ax.set_ylabel("Users (log scale)")
    histogram_ax.set_title(
        "Positive-value distribution by arm", loc="left", color=COLOR_INK, fontsize=10
    )
    histogram_ax.legend(frameon=False, fontsize=9)
    _style_axis(histogram_ax)

    interval_ax.axvline(0, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    intervals = [
        ("Welch 95% CI", welch.difference, welch.ci_low, welch.ci_high),
        ("Bootstrap 95% CI", bootstrap.estimate, bootstrap.ci_low, bootstrap.ci_high),
    ]
    for position, (_, center, ci_low, ci_high) in enumerate(intervals):
        interval_ax.plot([ci_low, ci_high], [position, position], color=COLOR_CONTROL, linewidth=2)
        interval_ax.plot(center, position, "o", color=COLOR_INK, markersize=7)
    interval_ax.plot(trimmed, len(intervals), "D", color=COLOR_TREATMENT, markersize=7)
    labels = [label for label, *_ in intervals] + ["Trimmed diff (1%)\ndifferent estimand"]
    interval_ax.set_yticks(np.arange(len(labels)), labels)
    interval_ax.set_xlabel(f"{metric_name} difference (treatment - control)")
    interval_ax.set_title("Interval comparison", loc="left", color=COLOR_INK, fontsize=10)
    interval_ax.invert_yaxis()
    _style_axis(interval_ax, grid_axis="x")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "heavy_tail.png", dpi=150)
    plt.close(fig)


def sec_truth(experiment: Experiment) -> SectionSpec:
    """For synthetic data only: the parameters the generator actually used."""
    rows = "\n".join(
        f"| {name} | {value} |"
        for name, value in sorted(_present(experiment.truth, "truth table").items())
    )
    body = (
        "This data is SYNTHETIC, and that is disclosed here rather than in a "
        "footnote. The generator's parameters below are the true values the "
        "analysis above can be checked against. No real-data section of any "
        "report produced by this pipeline carries this section, which is "
        "itself the signal of which results came from synthetic "
        "data.\n\n| Parameter | Value |\n|---|---|\n" + rows
    )
    return SectionSpec(
        key="truth",
        title="What the generator actually did",
        body=body,
        requires=frozenset({"truth"}),
    )


def default_sections(
    experiment: Experiment, random_generator: np.random.Generator, figure_dir: Path
) -> list[SectionSpec]:
    """The standard report, in order, with impossible sections omitted."""
    data_capabilities = capabilities(experiment)
    candidates = [
        sec_validation(experiment),
        sec_design_power(experiment),
        sec_srm(experiment),
        sec_primary(experiment),
        sec_novelty(experiment, figure_dir) if "time_axis" in data_capabilities else None,
        sec_sequential(experiment, figure_dir) if "time_axis" in data_capabilities else None,
        sec_corrections(experiment, figure_dir) if "multi_metric" in data_capabilities else None,
        sec_heavy_tail(experiment, random_generator, figure_dir)
        if {"user_rows", "continuous_metric"} <= data_capabilities
        else None,
        sec_truth(experiment) if "truth" in data_capabilities else None,
    ]
    return [section for section in candidates if section is not None]


def sweep_aggregate_archive(config: dict, data_path: Path) -> SectionSpec:
    """One section summarizing an ENTIRE aggregate archive, every
    experiment, alongside the close read the rest of the report gives one.

    Two questions per experiment: is the assignment ratio stable across
    checkpoints (the no-declared-split SRM form, via _ratio_stability on
    the primary metric's counts), and does anything survive a Holm
    correction across the experiment's full end-of-run comparison family.
    Anything the sweep cannot analyze is counted and said out loud,
    because a silently truncated sweep reads as full coverage.
    """
    import csv

    aggregate_columns = config["aggregates"]
    metrics = _metric_specs(config)
    metric_by_column = {metric.column: metric for metric in metrics}
    primary_column = next(metric.column for metric in metrics if metric.primary)
    alpha = config["experiment"]["alpha"]

    experiments: dict[str, list[dict]] = {}
    with open(data_path, newline="", encoding="utf-8") as data_file:
        for row in csv.DictReader(data_file):
            experiments.setdefault(
                row[aggregate_columns.get("experiment_id_column", "experiment_id")], []
            ).append(row)

    unstable: list[str] = []
    stability_skipped = 0
    naive_any = holm_any = 0
    cells_skipped = 0
    analyzed = 0
    skip_decreasing = "a cumulative count decreases between checkpoints"
    skip_few = "fewer than two usable checkpoints exist"
    skip_degenerate = "the split is degenerate"
    skip_reasons = {skip_decreasing: 0, skip_few: 0, skip_degenerate: 0}
    for experiment_id, rows in sorted(experiments.items()):
        # cumulative primary-metric counts per variant, checkpoint-ordered
        variants = sorted({row[aggregate_columns["variant"]] for row in rows})
        worst_p_value = float("nan")
        reasons_hit: set[str] = set()
        for variant in variants:
            series = sorted(
                (
                    row
                    for row in rows
                    if row[aggregate_columns["variant"]] == variant
                    and row[aggregate_columns["metric"]] == primary_column
                ),
                key=lambda row: float(row[aggregate_columns["checkpoint"]]),
            )
            if len(series) < 2:
                reasons_hit.add(skip_few)
                continue
            control_counts = np.array(
                [float(row[aggregate_columns["count_control"]]) for row in series]
            )
            treatment_counts = np.array(
                [float(row[aggregate_columns["count_treatment"]]) for row in series]
            )
            if np.any(np.diff(control_counts) < 0) or np.any(np.diff(treatment_counts) < 0):
                reasons_hit.add(skip_decreasing)
                continue
            _, degrees_of_freedom, p_value, _ = _ratio_stability(control_counts, treatment_counts)
            if degrees_of_freedom == 0 or np.isnan(p_value):
                reasons_hit.add(skip_degenerate)
                continue
            if np.isnan(worst_p_value) or p_value < worst_p_value:
                worst_p_value = p_value
        if np.isnan(worst_p_value):
            stability_skipped += 1
            for reason in reasons_hit:
                skip_reasons[reason] += 1
        else:
            analyzed += 1
            if worst_p_value < 1e-3:
                unstable.append(experiment_id)

        # end-of-run multiplicity across the full (metric, variant) family
        p_values: list[float] = []
        for variant in variants:
            for metric_column, metric in metric_by_column.items():
                series = sorted(
                    (
                        row
                        for row in rows
                        if row[aggregate_columns["variant"]] == variant
                        and row[aggregate_columns["metric"]] == metric_column
                    ),
                    key=lambda row: float(row[aggregate_columns["checkpoint"]]),
                )
                if not series:
                    cells_skipped += 1
                    continue
                last = series[-1]
                try:
                    n_control = int(float(last[aggregate_columns["count_control"]]))
                    n_treatment = int(float(last[aggregate_columns["count_treatment"]]))
                    mean_control = float(last[aggregate_columns["mean_control"]])
                    mean_treatment = float(last[aggregate_columns["mean_treatment"]])
                    variance_control = float(last[aggregate_columns["variance_control"]])
                    variance_treatment = float(last[aggregate_columns["variance_treatment"]])
                except ValueError:
                    cells_skipped += 1
                    continue
                if (
                    n_control < 2
                    or n_treatment < 2
                    or not np.isfinite(mean_control)
                    or not np.isfinite(mean_treatment)
                ):
                    cells_skipped += 1
                    continue
                if metric.kind == "binary":
                    # a proportion test needs valid rates, not variances
                    if not (0.0 <= mean_control <= 1.0 and 0.0 <= mean_treatment <= 1.0):
                        cells_skipped += 1
                        continue
                    comparison = two_proportion_ztest(
                        round(mean_control * n_control),
                        n_control,
                        round(mean_treatment * n_treatment),
                        n_treatment,
                    )
                    p_values.append(comparison.p_value)
                else:
                    if (
                        not np.isfinite(variance_control)
                        or not np.isfinite(variance_treatment)
                        or variance_control < 0.0
                        or variance_treatment < 0.0
                        or variance_control / n_control + variance_treatment / n_treatment == 0.0
                    ):
                        cells_skipped += 1
                        continue
                    p_values.append(
                        welch_ttest_from_stats(
                            n_control,
                            mean_control,
                            variance_control,
                            n_treatment,
                            mean_treatment,
                            variance_treatment,
                        ).p_value
                    )
        if p_values:
            naive_any += int(any(p_value < alpha for p_value in p_values))
            holm_any += int(any(float(holm_p) < alpha for holm_p in holm(np.array(p_values))))

    unstable_list = ", ".join(unstable) if unstable else "none"
    # the skip sentence names only the causes that applied in this archive
    cause_parts = [f"{count} because {cause}" for cause, count in skip_reasons.items() if count]
    if len(cause_parts) == 1:
        causes = "because " + next(cause for cause, count in skip_reasons.items() if count)
    elif len(cause_parts) == 2:
        causes = " and ".join(cause_parts)
    else:
        causes = ", ".join(cause_parts[:-1]) + ", and " + cause_parts[-1]
    skipped_sentence = (
        f"{stability_skipped} could not be tested ({causes}) and are counted rather than hidden. "
        if stability_skipped
        else ""
    )
    body = (
        f"The sections above read one experiment closely. This one asks "
        f"two mechanical questions of every experiment in the archive "
        f"({len(experiments)} total). The first question is assignment "
        "stability. With no intended splits on record, the checkable "
        "property is that each experiment's arm ratio holds constant "
        "across its checkpoints (primary-metric denominators, worst "
        f"contrast per experiment). {analyzed} experiments were testable. "
        + skipped_sentence
        + "Experiments flagged at the conventional p < 1e-3 alarm: "
        f"{unstable_list}.\n\n"
        f"The second question is multiplicity at scale. Each experiment's "
        "full end-of-run comparison family (every metric on every variant) "
        f"was tested unadjusted and after a per-experiment Holm correction. "
        f"{naive_any} of {len(experiments)} experiments show at least one "
        f"nominally significant comparison before correction. {holm_any} "
        "still do after it. The gap between those two counts is the multiple-"
        "comparisons trap measured on real experiments rather than argued "
        "from theory."
        + (
            f" ({cells_skipped} comparison cells were skipped because their "
            "published statistics were missing or unusable. The skips are "
            "counted here.)"
            if cells_skipped
            else ""
        )
    )
    return SectionSpec(
        key="archive_sweep",
        title="The archive at large: stability and multiplicity",
        body=body,
    )

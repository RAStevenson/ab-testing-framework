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
from typing import TYPE_CHECKING

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


def fmt_perm_p(perm: PermutationResult) -> str:
    """Permutation p-value as report text, flagged when it sits at the estimator's floor.

    The add-one estimator's smallest possible value is 1 / (n_permutations + 1),
    reached when no relabeling is as extreme as the observed difference. That
    value is still a valid, conservative p-value, but printed bare it reads as
    a measurement rather than as the resolution limit of the relabeling count.
    """
    if perm.p_value <= 1.0 / (perm.n_permutations + 1):
        return (
            f"p = {fmt_p(perm.p_value)}, the smallest value "
            f"{perm.n_permutations:,} relabelings can give"
        )
    return f"p = {fmt_p(perm.p_value)}"


def _style_axis(ax: Axes, *, grid_axis: str = "y") -> None:
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


def _primary(exp: Experiment) -> MetricSpec:
    return next(metric for metric in exp.metrics if metric.primary)


def _contrasts(exp: Experiment) -> list[tuple[str, str]]:
    return [(exp.arms[0], arm) for arm in exp.arms[1:]]


def _agg(exp: Experiment, arm: str, metric: str, checkpoint: float | None) -> ArmAggregate:
    for cell in exp.aggregates:
        if cell.arm == arm and cell.metric == metric and cell.checkpoint == checkpoint:
            return cell
    raise ValueError(f"no aggregate for ({arm}, {metric}, {checkpoint})")


def _series(exp: Experiment, arm: str, metric: str) -> tuple[NDArray, NDArray, NDArray]:
    """Cumulative (n, mean, variance) arrays for one arm and metric, in
    checkpoint order. Raises if the experiment has no time axis."""
    cells = sorted(
        (
            cell
            for cell in exp.aggregates
            if cell.arm == arm and cell.metric == metric and cell.checkpoint is not None
        ),
        key=lambda cell: cell.checkpoint,
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
    lo_n = 0.0 if start == 0 else count[start - 1]
    lo_s = 0.0 if start == 0 else sums[start - 1]
    lo_q = 0.0 if start == 0 else squares[start - 1]
    w_n = count[end - 1] - lo_n
    if w_n < 2:
        raise ValueError(f"window {start}:{end} has fewer than 2 observations")
    w_mean = (sums[end - 1] - lo_s) / w_n
    w_var = max(((squares[end - 1] - lo_q) - w_n * w_mean**2) / (w_n - 1.0), 0.0)
    if not (np.isfinite(w_mean) and np.isfinite(w_var)):
        raise ValueError(f"window {start}:{end} has non-finite statistics")
    return int(w_n), float(w_mean), float(w_var)


def _successes(agg: ArmAggregate) -> int:
    return int(round(agg.mean * agg.n))


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
    expected_t = (control_increments + treatment_increments) * share
    expected_c = (control_increments + treatment_increments) * (1.0 - share)
    statistic = float(
        np.sum(
            (treatment_increments - expected_t) ** 2 / expected_t
            + (control_increments - expected_c) ** 2 / expected_c
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
    int_n = np.diff(count, prepend=0.0)
    int_s = np.diff(sums, prepend=0.0)
    int_q = np.diff(squares, prepend=0.0)
    int_mean = np.divide(int_s, int_n, out=np.full_like(int_n, np.nan), where=int_n != 0)
    int_var = np.divide(
        int_q - int_n * int_mean**2,
        int_n - 1.0,
        out=np.full_like(int_n, np.nan),
        where=int_n != 1.0,
    )
    bad = np.nonzero(np.isfinite(int_var) & (int_var < -1e-6 * np.maximum(variance, 1.0)))[0]
    return int(bad[0]) if bad.size else int(count.size)


def _cell_test(
    exp: Experiment, treatment_arm: str, metric: MetricSpec
) -> RateComparison | MeanComparison:
    """The appropriate two-sample test for one (contrast, metric) cell,
    chosen by metric kind and data shape."""
    control = _agg(exp, exp.arms[0], metric.name, None)
    treatment = _agg(exp, treatment_arm, metric.name, None)
    if metric.kind == "binary":
        return two_proportion_ztest(
            _successes(control), control.n, _successes(treatment), treatment.n
        )
    if exp.rows is not None:
        mask_c = exp.rows["arm"] == exp.arms[0]
        mask_t = exp.rows["arm"] == treatment_arm
        return welch_ttest(exp.rows[metric.name][mask_c], exp.rows[metric.name][mask_t])
    return welch_ttest_from_stats(
        control.n, control.mean, control.variance, treatment.n, treatment.mean, treatment.variance
    )


def _effect_and_p(result: RateComparison | MeanComparison) -> tuple[str, float]:
    if isinstance(result, RateComparison):
        return pp_signed(result.absolute_lift), result.p_value
    return f"{result.difference:+.3f}", result.p_value


def _contrast_label(exp: Experiment, treatment_arm: str) -> str:
    return f"{treatment_arm} vs {exp.arms[0]}"


def sec_validation(exp: Experiment) -> SectionSpec:
    """Always renders: what the intake did to the data before any statistic."""
    sizes = ", ".join(f"{arm} {_agg(exp, arm, _primary(exp).name, None).n:,}" for arm in exp.arms)
    if exp.validation:
        rows = "\n".join(
            f"| {event.rule} | {event.count:,} | {event.detail} |" for event in exp.validation
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


def sec_design_power(exp: Experiment) -> SectionSpec | None:
    """Sensitivity at the realized sample size, for a binary primary metric."""
    metric = _primary(exp)
    if metric.kind != "binary":
        return None
    control = _agg(exp, exp.arms[0], metric.name, None)
    treatment = _agg(exp, exp.arms[1], metric.name, None)
    if not 0.0 < control.mean < 1.0:
        return None
    mde = mde_two_proportions(control.mean, control.n, alpha=exp.alpha, n_treatment=treatment.n)
    body = (
        f"At the realized sample sizes of the first contrast ({control.n:,} "
        f"control and {treatment.n:,} treatment users) and the control rate of "
        f"{pct(control.mean, 1)}, the smallest absolute lift that contrast could "
        f"detect with 80% power at alpha {exp.alpha} is {100 * mde:.2f} "
        "percentage points (pp). This is the minimum detectable effect (MDE). "
        "Effects smaller than that could be real and still "
        "produce an "
        "unremarkable p-value here. That is a property of the sample size, not "
        "evidence of absence."
    )
    return SectionSpec(key="design_power", title="Sensitivity at this sample size", body=body)


def sec_srm(exp: Experiment) -> SectionSpec:
    """Assignment validation: intended-split chi-square when the split is
    declared, ratio-stability over time when it is not but a time axis
    exists, and a stated non-test otherwise."""
    metric = _primary(exp)
    caps = capabilities(exp)
    if "intended_split" in caps:
        lines = ["| Contrast | Control n | Treatment n | Chi-square | p |", "|---|---|---|---|---|"]
        worst_p = 1.0
        for control_arm, treatment_arm in _contrasts(exp):
            i_c = exp.arms.index(control_arm)
            i_t = exp.arms.index(treatment_arm)
            share = exp.intended_split[i_t] / (exp.intended_split[i_c] + exp.intended_split[i_t])
            n_c = _agg(exp, control_arm, metric.name, None).n
            n_t = _agg(exp, treatment_arm, metric.name, None).n
            srm = srm_check(n_c, n_t, expected_treatment_share=share)
            worst_p = min(worst_p, srm.p_value)
            lines.append(
                f"| {_contrast_label(exp, treatment_arm)} | {n_c:,} | {n_t:,} "
                f"| {srm.statistic:.2f} | {fmt_p(srm.p_value)} |"
            )
        srm_conclusion = (
            "No contrast fails the check (conventional alarm threshold p < 1e-3), "
            "so the realized split is consistent with the declared one."
            if worst_p >= 1e-3
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
    elif "time_axis" in caps:
        lines = ["| Contrast | Checkpoints | Chi-square | df | p |", "|---|---|---|---|---|"]
        worst_p = 1.0
        for control_arm, treatment_arm in _contrasts(exp):
            control_counts = _series(exp, control_arm, metric.name)[0]
            treatment_counts = _series(exp, treatment_arm, metric.name)[0]
            if np.any(np.diff(control_counts) < 0) or np.any(np.diff(treatment_counts) < 0):
                # a decreasing cumulative count would feed negative
                # interval counts into the chi-square; say so instead
                lines.append(
                    f"| {_contrast_label(exp, treatment_arm)} | - | not testable "
                    "(cumulative counts decrease) | - | - |"
                )
                continue
            statistic, degrees_of_freedom, p_value, n_intervals = _ratio_stability(
                control_counts, treatment_counts
            )
            if not np.isfinite(p_value):
                lines.append(
                    f"| {_contrast_label(exp, treatment_arm)} | {n_intervals} | "
                    "not testable (degenerate split) | - | - |"
                )
                continue
            worst_p = min(worst_p, p_value)
            lines.append(
                f"| {_contrast_label(exp, treatment_arm)} | {n_intervals} | {statistic:.2f} "
                f"| {degrees_of_freedom} | {fmt_p(p_value)} |"
            )
        srm_conclusion = (
            "The arm ratio is stable across checkpoints in every contrast "
            "(alarm threshold p < 1e-3), which is what can still be checked "
            "without a declared split."
            if worst_p >= 1e-3
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
        sizes = ", ".join(f"{arm} {_agg(exp, arm, metric.name, None).n:,}" for arm in exp.arms)
        body = (
            f"Observed arm sizes: {sizes}. The intended allocation is not on "
            "record and the data has no time axis, so neither a direct "
            "sample-ratio mismatch test nor a ratio-stability test is possible. "
            "The sizes are reported for the reader's own judgment, and that "
            "absence of a check is itself worth knowing."
        )
    return SectionSpec(key="srm", title="Assignment validation", body=body)


def sec_primary(exp: Experiment) -> SectionSpec:
    """The primary metric, per contrast."""
    metric = _primary(exp)
    lines = ["| Contrast | Effect | 95% CI | p |", "|---|---|---|---|"]
    for _, treatment_arm in _contrasts(exp):
        comparison = _cell_test(exp, treatment_arm, metric)
        if isinstance(comparison, RateComparison):
            effect = pp_signed(comparison.absolute_lift)
            interval_text = f"{pp_signed(comparison.ci_low)} to {pp_signed(comparison.ci_high)}"
        else:
            effect = f"{comparison.difference:+.3f}"
            interval_text = f"{comparison.ci_low:+.3f} to {comparison.ci_high:+.3f}"
        lines.append(
            f"| {_contrast_label(exp, treatment_arm)} | {effect} | {interval_text} "
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


def sec_novelty(exp: Experiment, figdir: Path) -> SectionSpec | None:
    """Early-versus-late effect comparison over the time axis."""
    metric = _primary(exp)
    control_arm = exp.arms[0]
    try:
        series_c = _series(exp, control_arm, metric.name)
        all_series = [series_c] + [_series(exp, arm, metric.name) for _, arm in _contrasts(exp)]
    except ValueError:
        _report_skipped_section("Effect stability over time")
        return None
    k_full = len(exp.checkpoints)
    n_usable = min(_consistent_prefix(*ser) for ser in all_series)
    if n_usable < 3:
        return None
    trim_note = (
        ""
        if n_usable == k_full
        else "\n\n"
        + (
            "The final checkpoint is EXCLUDED from this section"
            if k_full - n_usable == 1
            else f"The final {k_full - n_usable} checkpoints are EXCLUDED from this section"
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
    for _, treatment_arm in _contrasts(exp):
        series_t = _series(exp, treatment_arm, metric.name)
        try:
            e_c = _window(*series_c, 0, window_size)
            l_c = _window(*series_c, n_usable - window_size, n_usable)
            e_t = _window(*series_t, 0, window_size)
            l_t = _window(*series_t, n_usable - window_size, n_usable)
        except ValueError:
            _report_skipped_section("Effect stability over time")
            return None
        early = e_t[1] - e_c[1]
        late = l_t[1] - l_c[1]
        se_early = np.sqrt(e_c[2] / e_c[0] + e_t[2] / e_t[0])
        se_late = np.sqrt(l_c[2] / l_c[0] + l_t[2] / l_t[0])
        if se_early == 0.0 or se_late == 0.0:
            return None
        decay_z = float((early - late) / np.hypot(se_early, se_late))
        decay_p = float(2.0 * stats.norm.sf(abs(decay_z)))
        early_s = pp_signed(early) if metric.kind == "binary" else f"{early:+.3f}"
        late_s = pp_signed(late) if metric.kind == "binary" else f"{late:+.3f}"
        lines.append(
            f"| {_contrast_label(exp, treatment_arm)} | {early_s} | {late_s} "
            f"| {decay_z:.2f} | {fmt_p(decay_p)} |"
        )

    _fig_novelty(exp, metric, figdir, n_usable)
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
            if exp.rows is not None
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


def _fig_novelty(exp: Experiment, metric: MetricSpec, figdir: Path, k_use: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, arm in enumerate(exp.arms):
        count, mean, var = _series(exp, arm, metric.name)
        count, mean, var = count[:k_use], mean[:k_use], var[:k_use]
        # computed directly rather than via interval_from_cumulative so a
        # checkpoint where this arm gained no users plots as a gap instead
        # of raising; means are exact, and the figure needs no variances
        int_n = np.diff(count, prepend=0.0)
        int_sums = np.diff(count * mean, prepend=0.0)
        int_mean = np.where(int_n > 0, int_sums / np.maximum(int_n, 1.0), np.nan)
        checkpoints = np.asarray(exp.checkpoints[:k_use], dtype=np.float64)
        ax.plot(
            checkpoints,
            int_mean,
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
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "novelty.png", dpi=150)
    plt.close(fig)


def sec_sequential(exp: Experiment, figdir: Path) -> SectionSpec | None:
    """Replay of the run as sequential looks under an alpha-spending design,
    against the naive fixed-threshold rule. First contrast only: a
    monitoring design guards one comparison."""
    metric = _primary(exp)
    control_arm, treatment_arm = _contrasts(exp)[0]
    try:
        control_counts, mean_c, var_c = _series(exp, control_arm, metric.name)
        treatment_counts, mean_t, var_t = _series(exp, treatment_arm, metric.name)
    except ValueError:
        _report_skipped_section("What sequential monitoring would have decided")
        return None
    k_full = int(control_counts.size)
    k_use = min(
        _consistent_prefix(control_counts, mean_c, var_c),
        _consistent_prefix(treatment_counts, mean_t, var_t),
    )
    if k_use < 3:
        return None
    control_counts, mean_c, var_c = control_counts[:k_use], mean_c[:k_use], var_c[:k_use]
    treatment_counts, mean_t, var_t = treatment_counts[:k_use], mean_t[:k_use], var_t[:k_use]
    trim_note = (
        ""
        if k_use == k_full
        else "\n\nAs in the effect-stability analysis, "
        + (
            "the final checkpoint is excluded"
            if k_full - k_use == 1
            else f"the final {k_full - k_use} checkpoints are excluded"
        )
        + " because the cumulative statistics there are internally "
        f"inconsistent. The replay covers the {k_use} consistent looks."
    )
    # statistical information for a two-sample z is the inverse summed
    # variance scale 1/(1/control_counts + 1/treatment_counts), which equals a share of the total
    # count only when the arm ratio is constant across looks
    info = 1.0 / (1.0 / control_counts + 1.0 / treatment_counts)
    if info[-1] <= 0 or np.any(np.diff(info) <= 0):
        return None
    fractions = info / info[-1]
    if metric.kind == "binary":
        control_conversions = np.round(mean_c * control_counts)
        treatment_conversions = np.round(mean_t * treatment_counts)
        if np.any(np.diff(control_conversions) < 0) or np.any(np.diff(treatment_conversions) < 0):
            # published means at limited precision can reconstruct a
            # count series that dips by one; the stats-based z needs no
            # reconstruction and its divergence from the pooled form is
            # documented on z_from_cumulative_stats
            z_path = z_from_cumulative_stats(
                control_counts, mean_c, var_c, treatment_counts, mean_t, var_t
            )
        else:
            z_path = z_from_cumulative_counts(
                control_conversions, control_counts, treatment_conversions, treatment_counts
            )
    else:
        z_path = z_from_cumulative_stats(
            control_counts, mean_c, var_c, treatment_counts, mean_t, var_t
        )
    seq = design(fractions.tolist(), alpha=exp.alpha, spending="obrien-fleming")
    outcome = monitor(z_path.tolist(), seq)
    # the naive rule is the conventional dashboard threshold, quoted and
    # applied at two decimals as practitioners actually use it
    z_naive = float(np.round(stats.norm.isf(exp.alpha / 2.0), 2))
    naive_crossings = np.nonzero(np.abs(z_path) >= z_naive)[0]
    naive_look = int(naive_crossings[0]) + 1 if naive_crossings.size else None

    _fig_sequential(z_path, np.array(seq.boundaries), z_naive, figdir)

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
        f"for the first contrast ({_contrast_label(exp, treatment_arm)}). A "
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


def _fig_sequential(z_path: NDArray, boundaries: NDArray, z_naive: float, figdir: Path) -> None:
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
    top_vals = [float(np.abs(z_path).max()), z_naive]
    if finite.any():
        top_vals.append(float(boundaries[finite].max()))
    ax.set_ylim(top=1.2 * max(top_vals))
    ax.set_title("Sequential monitoring replay", loc="left", color=COLOR_INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "sequential.png", dpi=150)
    plt.close(fig)


def sec_corrections(exp: Experiment, figdir: Path) -> SectionSpec | None:
    """Multiple-comparison corrections over the non-primary family."""
    metric_primary = _primary(exp)
    family: list[tuple[MetricSpec, str]] = [
        (metric, treatment_arm)
        for metric in exp.metrics
        if not metric.primary
        for _, treatment_arm in _contrasts(exp)
    ]
    if not family:
        family = [(metric_primary, treatment_arm) for _, treatment_arm in _contrasts(exp)]
    if len(family) < 2:
        return None
    results = [_cell_test(exp, arm, metric) for metric, arm in family]
    naive = np.array([comparison.p_value for comparison in results])
    holm_ps = holm(naive)
    bh_ps = benjamini_hochberg(naive)

    lines = [
        "| Metric | Contrast | Effect | Unadjusted p | Holm p | BH q |",
        "|---|---|---|---|---|---|",
    ]
    for (metric, arm), comparison, holm_p, bh_q in zip(
        family, results, holm_ps, bh_ps, strict=True
    ):
        effect, p_value = _effect_and_p(comparison)
        lines.append(
            f"| {metric.name} | {_contrast_label(exp, arm)} | {effect} | {fmt_p(p_value)} "
            f"| {fmt_p(float(holm_p))} | {fmt_p(float(bh_q))} |"
        )
    n_naive = int(np.sum(naive < exp.alpha))
    n_holm = int(np.sum(holm_ps < exp.alpha))
    family_error = 1.0 - (1.0 - exp.alpha) ** len(family)
    body = (
        (
            f"This experiment has {len(family)} comparisons beyond the primary metric. "
            if any(not metric.primary for metric, _ in family)
            else f"With one metric and several arms, the family is the primary "
            f"metric's {len(family)} contrasts. "
        )
        + "Testing each at alpha "
        f"{exp.alpha} without adjustment gives roughly a "
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
        [f"{metric.name}, {arm}" for metric, arm in family], naive, holm_ps, exp.alpha, figdir
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
    labels: list[str], naive: NDArray, adjusted: NDArray, alpha: float, figdir: Path
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
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "corrections.png", dpi=150)
    plt.close(fig)


def sec_heavy_tail(exp: Experiment, rng: np.random.Generator, figdir: Path) -> SectionSpec | None:
    """Resampling-based analysis of a heavy-tailed continuous metric."""
    metric = (
        _primary(exp)
        if _primary(exp).kind == "continuous"
        else next((metric for metric in exp.metrics if metric.kind == "continuous"), None)
    )
    if metric is None:
        return None
    control_arm, treatment_arm = _contrasts(exp)[0]
    control_values = exp.rows[metric.name][exp.rows["arm"] == control_arm]
    treatment_values = exp.rows[metric.name][exp.rows["arm"] == treatment_arm]
    if (
        control_values.size < 2
        or treatment_values.size < 2
        or (control_values.var(ddof=1) == 0.0 and treatment_values.var(ddof=1) == 0.0)
    ):
        return None
    welch = welch_ttest(control_values, treatment_values)
    boot = bootstrap_mean_diff(control_values, treatment_values, rng=rng, n_resamples=10_000)
    perm = permutation_test_mean_diff(
        control_values, treatment_values, rng=rng, n_permutations=10_000
    )
    trimmed = trimmed_mean_diff(control_values, treatment_values, trim=0.01)
    zero_share = float(np.mean(np.concatenate([control_values, treatment_values]) == 0.0))
    # a 1% trim on a mostly-zero metric removes the nonzero values themselves,
    # so the trimmed row stops being a tail check; say so when it happens
    trim_degenerate = 2 * int(0.01 * control_values.size) >= int(
        (control_values > 0).sum()
    ) or 2 * int(0.01 * treatment_values.size) >= int((treatment_values > 0).sum())

    _fig_heavy_tail(control_values, treatment_values, welch, boot, trimmed, metric.name, figdir)
    boot_excludes_zero = (boot.ci_low > 0.0) == (boot.ci_high > 0.0)
    welch_significant = welch.p_value < exp.alpha
    agreement = (
        "The three same-estimand methods agree here, which is what a well-behaved tail looks like."
        if welch_significant == (perm.p_value < exp.alpha)
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
        + f". The first contrast ({_contrast_label(exp, treatment_arm)}) is "
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
        f"| Percentile bootstrap | {boot.estimate:+.3f} | 95% CI {boot.ci_low:+.3f} "
        f"to {boot.ci_high:+.3f} |\n"
        f"| Permutation test | {perm.observed:+.3f} | {fmt_perm_p(perm)} |\n"
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
    boot: BootstrapResult,
    trimmed: float,
    metric_name: str,
    figdir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    pos_x, pos_y = control_values[control_values > 0], treatment_values[treatment_values > 0]
    if pos_x.size and pos_y.size:
        top = float(max(pos_x.max(), pos_y.max()))
        bins = np.geomspace(max(min(pos_x.min(), pos_y.min()), 1e-9), top, 40)
        ax1.hist(
            pos_x, bins=bins, histtype="step", linewidth=2, color=COLOR_CONTROL, label="control"
        )
        ax1.hist(
            pos_y, bins=bins, histtype="step", linewidth=2, color=COLOR_TREATMENT, label="treatment"
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.set_xlabel(f"{metric_name} per user, positive values (log scale)")
    ax1.set_ylabel("Users (log scale)")
    ax1.set_title("Positive-value distribution by arm", loc="left", color=COLOR_INK, fontsize=10)
    ax1.legend(frameon=False, fontsize=9)
    _style_axis(ax1)

    ax2.axvline(0, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    intervals = [
        ("Welch 95% CI", welch.difference, welch.ci_low, welch.ci_high),
        ("Bootstrap 95% CI", boot.estimate, boot.ci_low, boot.ci_high),
    ]
    for position, (_, center, ci_low, ci_high) in enumerate(intervals):
        ax2.plot([ci_low, ci_high], [position, position], color=COLOR_CONTROL, linewidth=2)
        ax2.plot(center, position, "o", color=COLOR_INK, markersize=7)
    ax2.plot(trimmed, len(intervals), "D", color=COLOR_TREATMENT, markersize=7)
    labels = [label for label, *_ in intervals] + ["Trimmed diff (1%)\ndifferent estimand"]
    ax2.set_yticks(np.arange(len(labels)), labels)
    ax2.set_xlabel(f"{metric_name} difference (treatment - control)")
    ax2.set_title("Interval comparison", loc="left", color=COLOR_INK, fontsize=10)
    ax2.invert_yaxis()
    _style_axis(ax2, grid_axis="x")
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "heavy_tail.png", dpi=150)
    plt.close(fig)


def sec_truth(exp: Experiment) -> SectionSpec:
    """For synthetic data only: the parameters the generator actually used."""
    rows = "\n".join(f"| {name} | {value} |" for name, value in sorted(exp.truth.items()))
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


def default_sections(exp: Experiment, rng: np.random.Generator, figdir: Path) -> list[SectionSpec]:
    """The standard report, in order, with impossible sections omitted."""
    caps = capabilities(exp)
    candidates = [
        sec_validation(exp),
        sec_design_power(exp),
        sec_srm(exp),
        sec_primary(exp),
        sec_novelty(exp, figdir) if "time_axis" in caps else None,
        sec_sequential(exp, figdir) if "time_axis" in caps else None,
        sec_corrections(exp, figdir) if "multi_metric" in caps else None,
        sec_heavy_tail(exp, rng, figdir) if {"user_rows", "continuous_metric"} <= caps else None,
        sec_truth(exp) if "truth" in caps else None,
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

    agg = config["aggregates"]
    metrics = _metric_specs(config)
    metric_by_column = {metric.column: metric for metric in metrics}
    primary_col = next(metric.column for metric in metrics if metric.primary)
    alpha = config["experiment"]["alpha"]

    experiments: dict[str, list[dict]] = {}
    with open(data_path, newline="", encoding="utf-8") as data_file:
        for row in csv.DictReader(data_file):
            experiments.setdefault(
                row[agg.get("experiment_id_column", "experiment_id")], []
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
    for exp_id, rows in sorted(experiments.items()):
        # cumulative primary-metric counts per variant, checkpoint-ordered
        variants = sorted({row[agg["variant"]] for row in rows})
        worst_p = float("nan")
        reasons_hit: set[str] = set()
        for variant in variants:
            series = sorted(
                (
                    row
                    for row in rows
                    if row[agg["variant"]] == variant and row[agg["metric"]] == primary_col
                ),
                key=lambda row: float(row[agg["checkpoint"]]),
            )
            if len(series) < 2:
                reasons_hit.add(skip_few)
                continue
            control_counts = np.array([float(row[agg["count_control"]]) for row in series])
            treatment_counts = np.array([float(row[agg["count_treatment"]]) for row in series])
            if np.any(np.diff(control_counts) < 0) or np.any(np.diff(treatment_counts) < 0):
                reasons_hit.add(skip_decreasing)
                continue
            _, degrees_of_freedom, p_value, _ = _ratio_stability(control_counts, treatment_counts)
            if degrees_of_freedom == 0 or np.isnan(p_value):
                reasons_hit.add(skip_degenerate)
                continue
            if np.isnan(worst_p) or p_value < worst_p:
                worst_p = p_value
        if np.isnan(worst_p):
            stability_skipped += 1
            for reason in reasons_hit:
                skip_reasons[reason] += 1
        else:
            analyzed += 1
            if worst_p < 1e-3:
                unstable.append(exp_id)

        # end-of-run multiplicity across the full (metric, variant) family
        p_values: list[float] = []
        for variant in variants:
            for metric_col, metric in metric_by_column.items():
                series = sorted(
                    (
                        row
                        for row in rows
                        if row[agg["variant"]] == variant and row[agg["metric"]] == metric_col
                    ),
                    key=lambda row: float(row[agg["checkpoint"]]),
                )
                if not series:
                    cells_skipped += 1
                    continue
                last = series[-1]
                try:
                    n_c = int(float(last[agg["count_control"]]))
                    n_t = int(float(last[agg["count_treatment"]]))
                    mean_c = float(last[agg["mean_control"]])
                    mean_t = float(last[agg["mean_treatment"]])
                    var_c = float(last[agg["variance_control"]])
                    var_t = float(last[agg["variance_treatment"]])
                except ValueError:
                    cells_skipped += 1
                    continue
                if n_c < 2 or n_t < 2 or not np.isfinite(mean_c) or not np.isfinite(mean_t):
                    cells_skipped += 1
                    continue
                if metric.kind == "binary":
                    # a proportion test needs valid rates, not variances
                    if not (0.0 <= mean_c <= 1.0 and 0.0 <= mean_t <= 1.0):
                        cells_skipped += 1
                        continue
                    result = two_proportion_ztest(
                        int(round(mean_c * n_c)), n_c, int(round(mean_t * n_t)), n_t
                    )
                    p_values.append(result.p_value)
                else:
                    if (
                        not np.isfinite(var_c)
                        or not np.isfinite(var_t)
                        or var_c < 0.0
                        or var_t < 0.0
                        or var_c / n_c + var_t / n_t == 0.0
                    ):
                        cells_skipped += 1
                        continue
                    p_values.append(
                        welch_ttest_from_stats(n_c, mean_c, var_c, n_t, mean_t, var_t).p_value
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

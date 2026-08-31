"""Presence-gated section builders: from an Experiment to SectionSpecs.

Each ``sec_*`` function runs one analysis over an ``abkit.io.Experiment``
and returns a ``SectionSpec`` for the report engine, or ``None`` when the
analysis is impossible even though the capability gate passed (an
all-constant metric, say). ``default_sections`` assembles the standard
report in order. Prose here is the neutral template voice shared by every
data source; a flagship builder may append its own commentary paragraphs
to any section body before rendering.

Figures are drawn at section-build time into the given directory.
matplotlib is imported lazily inside the figure helpers, so the library's
runtime dependencies stay numpy + scipy for everything except report
building, matching the existing split where matplotlib is a dev
dependency of the report scripts.
"""

from __future__ import annotations

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
from abkit.results import BootstrapResult, MeanComparison, RateComparison

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


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f} pp"


def pp_signed(x: float, digits: int = 2) -> str:
    """Signed percentage-point difference, so effect columns read one way."""
    return f"{100 * x:+.{digits}f} pp"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


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


def _primary(exp: Experiment) -> MetricSpec:
    return next(m for m in exp.metrics if m.primary)


def _contrasts(exp: Experiment) -> list[tuple[str, str]]:
    return [(exp.arms[0], arm) for arm in exp.arms[1:]]


def _agg(exp: Experiment, arm: str, metric: str, checkpoint: float | None) -> ArmAggregate:
    for a in exp.aggregates:
        if a.arm == arm and a.metric == metric and a.checkpoint == checkpoint:
            return a
    raise ValueError(f"no aggregate for ({arm}, {metric}, {checkpoint})")


def _series(exp: Experiment, arm: str, metric: str) -> tuple[NDArray, NDArray, NDArray]:
    """Cumulative (n, mean, variance) arrays for one arm and metric, in
    checkpoint order. Raises if the experiment has no time axis."""
    cells = sorted(
        (
            a
            for a in exp.aggregates
            if a.arm == arm and a.metric == metric and a.checkpoint is not None
        ),
        key=lambda a: a.checkpoint,
    )
    if not cells:
        raise ValueError(f"no per-checkpoint aggregates for ({arm}, {metric})")
    return (
        np.array([c.n for c in cells], dtype=np.float64),
        np.array([c.mean for c in cells], dtype=np.float64),
        np.array([c.variance for c in cells], dtype=np.float64),
    )


def _window(
    n: NDArray, mean: NDArray, variance: NDArray, start: int, end: int
) -> tuple[int, float, float]:
    """(n, mean, ddof=1 variance) over cumulative checkpoints start..end-1,
    via the same sum-of-squares identity as interval_from_cumulative."""
    sums = n * mean
    squares = (n - 1.0) * variance + n * mean**2
    lo_n = 0.0 if start == 0 else n[start - 1]
    lo_s = 0.0 if start == 0 else sums[start - 1]
    lo_q = 0.0 if start == 0 else squares[start - 1]
    w_n = n[end - 1] - lo_n
    if w_n < 2:
        raise ValueError(f"window {start}:{end} has fewer than 2 observations")
    w_mean = (sums[end - 1] - lo_s) / w_n
    w_var = max(((squares[end - 1] - lo_q) - w_n * w_mean**2) / (w_n - 1.0), 0.0)
    if not (np.isfinite(w_mean) and np.isfinite(w_var)):
        raise ValueError(f"window {start}:{end} has non-finite statistics")
    return int(w_n), float(w_mean), float(w_var)


def _successes(agg: ArmAggregate) -> int:
    return int(round(agg.mean * agg.n))


def _ratio_stability(nc: NDArray, nt: NDArray) -> tuple[float, int, float, int]:
    """Chi-square homogeneity of the per-interval assignment split.

    Takes the two arms' CUMULATIVE count series and tests whether the
    share of new users landing in each arm is constant across intervals,
    which is the strongest sample-ratio check available when the
    intended split is not on record. Returns (statistic, df, p,
    n_intervals); p is NaN when fewer than two usable intervals exist.
    """
    ic = np.diff(np.asarray(nc, dtype=np.float64), prepend=0.0)
    it = np.diff(np.asarray(nt, dtype=np.float64), prepend=0.0)
    keep = (ic > 0) | (it > 0)
    ic, it = ic[keep], it[keep]
    if ic.sum() == 0.0 or it.sum() == 0.0:
        # one arm never gained a user; a split test has nothing to test
        return float("nan"), 0, float("nan"), int(ic.size)
    share = it.sum() / (ic.sum() + it.sum())
    expected_t = (ic + it) * share
    expected_c = (ic + it) * (1.0 - share)
    statistic = float(
        np.sum((it - expected_t) ** 2 / expected_t + (ic - expected_c) ** 2 / expected_c)
    )
    df = int(ic.size - 1)
    p = float(stats.chi2.sf(statistic, df=df)) if df > 0 else float("nan")
    return statistic, df, p, int(ic.size)


def _consistent_prefix(n: NDArray, mean: NDArray, variance: NDArray) -> int:
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
    sums = n * mean
    squares = (n - 1.0) * variance + n * mean**2
    int_n = np.diff(n, prepend=0.0)
    int_s = np.diff(sums, prepend=0.0)
    int_q = np.diff(squares, prepend=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        int_mean = int_s / int_n
        int_var = (int_q - int_n * int_mean**2) / (int_n - 1.0)
    bad = np.nonzero(np.isfinite(int_var) & (int_var < -1e-6 * np.maximum(variance, 1.0)))[0]
    return int(bad[0]) if bad.size else int(n.size)


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
        rows = "\n".join(f"| {e.rule} | {e.count:,} | {e.detail} |" for e in exp.validation)
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
            "The intake altered nothing and dropped nothing; the data passed "
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
    n_per_arm = min(control.n, treatment.n)
    mde = mde_two_proportions(control.mean, n_per_arm, alpha=exp.alpha)
    body = (
        f"At the realized sample size ({n_per_arm:,} in the smaller arm of the "
        f"first contrast) and the control rate of {pct(control.mean, 1)}, the "
        f"smallest absolute lift this experiment could detect with 80% power at "
        f"alpha {exp.alpha} is {pp(mde, 2)} (the minimum detectable effect, or "
        "MDE). Effects smaller than that could be real and still produce an "
        "unremarkable p-value here; that is a property of the sample size, not "
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
            r = srm_check(n_c, n_t, expected_treatment_share=share)
            worst_p = min(worst_p, r.p_value)
            lines.append(
                f"| {_contrast_label(exp, treatment_arm)} | {n_c:,} | {n_t:,} "
                f"| {r.statistic:.2f} | {fmt_p(r.p_value)} |"
            )
        verdict = (
            "No contrast fails the check (conventional alarm threshold p < 1e-3), "
            "so the realized split is consistent with the declared one."
            if worst_p >= 1e-3
            else "At least one contrast FAILS the check: users went missing or were "
            "assigned asymmetrically, and no downstream statistic can repair that. "
            "The results below describe the data as collected, under that cloud."
        )
        body = (
            "A sample-ratio mismatch check (SRM) asks whether the number of users "
            "who actually landed in each arm matches the intended split. A failure "
            "means the assignment or logging lost users asymmetrically, which "
            "biases every metric in ways no downstream statistic can fix.\n\n"
            + "\n".join(lines)
            + "\n\n"
            + verdict
        )
    elif "time_axis" in caps:
        lines = ["| Contrast | Checkpoints | Chi-square | df | p |", "|---|---|---|---|---|"]
        worst_p = 1.0
        for control_arm, treatment_arm in _contrasts(exp):
            nc = _series(exp, control_arm, metric.name)[0]
            nt = _series(exp, treatment_arm, metric.name)[0]
            if np.any(np.diff(nc) < 0) or np.any(np.diff(nt) < 0):
                # a decreasing cumulative count would feed negative
                # interval counts into the chi-square; say so instead
                lines.append(
                    f"| {_contrast_label(exp, treatment_arm)} | - | not testable "
                    "(cumulative counts decrease) | - | - |"
                )
                continue
            statistic, df, p, n_intervals = _ratio_stability(nc, nt)
            if not np.isfinite(p):
                lines.append(
                    f"| {_contrast_label(exp, treatment_arm)} | {n_intervals} | "
                    "not testable (degenerate split) | - | - |"
                )
                continue
            worst_p = min(worst_p, p)
            lines.append(
                f"| {_contrast_label(exp, treatment_arm)} | {n_intervals} | {statistic:.2f} "
                f"| {df} | {fmt_p(p)} |"
            )
        verdict = (
            "The arm ratio is stable across checkpoints in every contrast "
            "(alarm threshold p < 1e-3), which is what can still be checked "
            "without a declared split."
            if worst_p >= 1e-3
            else "The arm ratio SHIFTS over time in at least one contrast (p < 1e-3). "
            "A ratio that drifts mid-experiment is sample-ratio-mismatch evidence "
            "regardless of what split was intended, and it dates the breakage."
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
            + verdict
        )
    else:
        sizes = ", ".join(f"{arm} {_agg(exp, arm, metric.name, None).n:,}" for arm in exp.arms)
        body = (
            f"Observed arm sizes: {sizes}. The intended allocation is not on "
            "record and the data has no time axis, so neither a direct "
            "sample-ratio mismatch test nor a ratio-stability test is possible. "
            "The sizes are reported for the reader's own judgement, and that "
            "absence of a check is itself worth knowing."
        )
    return SectionSpec(key="srm", title="Assignment validation", body=body)


def sec_primary(exp: Experiment) -> SectionSpec:
    """The primary metric, per contrast."""
    metric = _primary(exp)
    lines = ["| Contrast | Effect | 95% CI | p |", "|---|---|---|---|"]
    for _, treatment_arm in _contrasts(exp):
        r = _cell_test(exp, treatment_arm, metric)
        if isinstance(r, RateComparison):
            effect = pp_signed(r.absolute_lift)
            ci = f"{pp_signed(r.ci_low)} to {pp_signed(r.ci_high)}"
        else:
            effect = f"{r.difference:+.3f}"
            ci = f"{r.ci_low:+.3f} to {r.ci_high:+.3f}"
        lines.append(
            f"| {_contrast_label(exp, treatment_arm)} | {effect} | {ci} | {fmt_p(r.p_value)} |"
        )
    body = (
        f"The primary metric is **{metric.name}** ({metric.kind}). Each treatment "
        "arm is compared against control; the test is a pooled two-proportion "
        "z-test for binary metrics and a Welch t-test for continuous ones.\n\n" + "\n".join(lines)
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
        return None
    k_full = len(exp.checkpoints)
    k = min(_consistent_prefix(*ser) for ser in all_series)
    if k < 3:
        return None
    trim_note = (
        ""
        if k == k_full
        else "\n\n"
        + (
            "The final checkpoint is EXCLUDED from this section"
            if k_full - k == 1
            else f"The final {k_full - k} checkpoints are EXCLUDED from this section"
        )
        + ". Each increment's variance can be recovered from consecutive "
        "cumulative counts, means, and variances. For the first excluded "
        "increment that recovery is negative, which no real accumulation of "
        "observations can produce. "
        "The statistics published there therefore do not describe a superset "
        "of the previous checkpoint; a settled recompute or late correction "
        "is the usual cause. The exclusion is stated here rather than "
        "applied silently."
    )
    w = max(1, k // 3)

    lines = ["| Contrast | Early effect | Late effect | Decay z | p |", "|---|---|---|---|---|"]
    for _, treatment_arm in _contrasts(exp):
        series_t = _series(exp, treatment_arm, metric.name)
        try:
            e_c = _window(*series_c, 0, w)
            l_c = _window(*series_c, k - w, k)
            e_t = _window(*series_t, 0, w)
            l_t = _window(*series_t, k - w, k)
        except ValueError:
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

    _fig_novelty(exp, metric, figdir, k)
    body = (
        "A treatment effect that fades within the run is the signature of a "
        "novelty effect, where users react to the change itself and the "
        "reaction decays. The comparison below tests the first "
        + (f"{w} checkpoints" if w != 1 else "checkpoint")
        + f" against the last {w} directly, because two "
        "separately-significant windows are not by themselves evidence that the "
        "windows differ. Interval values are recovered from the cumulative "
        "series by differencing; {{fig:novelty}} shows the per-interval means.\n\n"
        + "\n".join(lines)
        + "\n\nA significant z means the early and late windows genuinely "
        "differ, so the end-of-run cumulative number blends two regimes and "
        "can sit far from the late-window effect that best describes where "
        "the experiment ended up. The z is early minus late. A positive z "
        "means the effect moved down over the run, and a negative z means it "
        "moved up. Whether that is novelty fading, a harm emerging, or a "
        "primacy pattern depends on the effect signs in the table. Either way "
        "the late window is the honest number to quote."
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
        n, mean, var = _series(exp, arm, metric.name)
        n, mean, var = n[:k_use], mean[:k_use], var[:k_use]
        # computed directly rather than via interval_from_cumulative so a
        # checkpoint where this arm gained no users plots as a gap instead
        # of raising; means are exact, and the figure needs no variances
        int_n = np.diff(n, prepend=0.0)
        int_sums = np.diff(n * mean, prepend=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
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
        nc, mean_c, var_c = _series(exp, control_arm, metric.name)
        nt, mean_t, var_t = _series(exp, treatment_arm, metric.name)
    except ValueError:
        return None
    k_full = int(nc.size)
    k_use = min(_consistent_prefix(nc, mean_c, var_c), _consistent_prefix(nt, mean_t, var_t))
    if k_use < 3:
        return None
    nc, mean_c, var_c = nc[:k_use], mean_c[:k_use], var_c[:k_use]
    nt, mean_t, var_t = nt[:k_use], mean_t[:k_use], var_t[:k_use]
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
        f"inconsistent; the replay covers the {k_use} consistent looks."
    )
    # statistical information for a two-sample z is the inverse summed
    # variance scale 1/(1/nc + 1/nt), which equals a share of the total
    # count only when the arm ratio is constant across looks
    info = 1.0 / (1.0 / nc + 1.0 / nt)
    if info[-1] <= 0 or np.any(np.diff(info) <= 0):
        return None
    fractions = info / info[-1]
    if metric.kind == "binary":
        xc = np.round(mean_c * nc)
        xt = np.round(mean_t * nt)
        if np.any(np.diff(xc) < 0) or np.any(np.diff(xt) < 0):
            # published means at limited precision can reconstruct a
            # count series that dips by one; the stats-based z needs no
            # reconstruction and its divergence from the pooled form is
            # documented on z_from_cumulative_stats
            z_path = z_from_cumulative_stats(nc, mean_c, var_c, nt, mean_t, var_t)
        else:
            z_path = z_from_cumulative_counts(xc, nc, xt, nt)
    else:
        z_path = z_from_cumulative_stats(nc, mean_c, var_c, nt, mean_t, var_t)
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
        naive_sentence = "The naive rule never fires either; the two agree on this run."

    body = (
        "Anyone watching a dashboard peeks. Checking the result at every "
        f"checkpoint against a fixed |z| > {z_naive:.2f} threshold (the naive rule) "
        "inflates the false-positive rate well beyond the nominal alpha, "
        "because every look is another chance to be fooled. An alpha-spending "
        "design prices the peeking by allocating the total alpha across the "
        "realized look schedule and computing a boundary for each look "
        "(O'Brien-Fleming spending, stingy early and generous late). "
        "{{fig:sequential}} shows the observed |z| path against both rules "
        f"for the first contrast ({_contrast_label(exp, treatment_arm)}), which "
        "is the comparison a monitoring design would have guarded.\n\n"
        + spend_sentence
        + " "
        + naive_sentence
        + "\n\nEven a correctly-stopped estimate is biased away from zero at the "
        "stopping look; monitoring is a tool for catching regressions early, "
        "not for shipping winners early."
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
        (m, t) for m in exp.metrics if not m.primary for _, t in _contrasts(exp)
    ]
    if not family:
        family = [(metric_primary, t) for _, t in _contrasts(exp)]
    if len(family) < 2:
        return None
    results = [_cell_test(exp, arm, m) for m, arm in family]
    naive = np.array([r.p_value for r in results])
    holm_ps = holm(naive)
    bh_ps = benjamini_hochberg(naive)

    lines = [
        "| Metric | Contrast | Effect | Unadjusted p | Holm p | BH q |",
        "|---|---|---|---|---|---|",
    ]
    for (m, arm), r, h, b in zip(family, results, holm_ps, bh_ps, strict=True):
        effect, p = _effect_and_p(r)
        lines.append(
            f"| {m.name} | {_contrast_label(exp, arm)} | {effect} | {fmt_p(p)} "
            f"| {fmt_p(float(h))} | {fmt_p(float(b))} |"
        )
    n_naive = int(np.sum(naive < exp.alpha))
    n_holm = int(np.sum(holm_ps < exp.alpha))
    family_error = 1.0 - (1.0 - exp.alpha) ** len(family)
    body = (
        (
            f"This experiment carries {len(family)} comparisons beyond the primary metric. "
            if any(not m.primary for m, _ in family)
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
            + f" alpha; after Holm, {n_holm}"
            + (" survives" if n_holm == 1 else " survive")
        )
        + ". {{fig:corrections}} shows each comparison's "
        "p-value before and after adjustment."
    )
    _fig_corrections([f"{m.name}, {arm}" for m, arm in family], naive, holm_ps, exp.alpha, figdir)
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
    y = np.arange(len(labels))
    ax.scatter(naive, y, s=55, color=COLOR_CONTROL, label="Unadjusted p", zorder=3)
    ax.scatter(adjusted, y, s=55, color=COLOR_TREATMENT, marker="s", label="Holm p", zorder=3)
    for yi in y:
        ax.plot([naive[yi], adjusted[yi]], [yi, yi], color=COLOR_REFERENCE, linewidth=1, zorder=2)
    ax.axvline(alpha, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    ax.set_yticks(y, labels)
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
        else next((m for m in exp.metrics if m.kind == "continuous"), None)
    )
    if metric is None:
        return None
    control_arm, treatment_arm = _contrasts(exp)[0]
    x = exp.rows[metric.name][exp.rows["arm"] == control_arm]
    y = exp.rows[metric.name][exp.rows["arm"] == treatment_arm]
    if x.size < 2 or y.size < 2 or (x.var(ddof=1) == 0.0 and y.var(ddof=1) == 0.0):
        return None
    welch = welch_ttest(x, y)
    boot = bootstrap_mean_diff(x, y, rng=rng, n_resamples=10_000)
    perm = permutation_test_mean_diff(x, y, rng=rng, n_permutations=10_000)
    trimmed = trimmed_mean_diff(x, y, trim=0.01)
    zero_share = float(np.mean(np.concatenate([x, y]) == 0.0))

    _fig_heavy_tail(x, y, welch, boot, trimmed, metric.name, figdir)
    boot_excludes_zero = (boot.ci_low > 0.0) == (boot.ci_high > 0.0)
    welch_significant = welch.p_value < exp.alpha
    agreement = (
        "The three same-estimand methods agree here, which is what a well-behaved tail looks like."
        if welch_significant == (perm.p_value < exp.alpha)
        and boot_excludes_zero == welch_significant
        else "The methods DISAGREE at the alpha line. That disagreement is the "
        "finding. It says the estimate rests on the tail, and the honest move "
        "is to report it rather than quote whichever method crossed the line."
    )
    body = (
        f"**{metric.name}** is the kind of metric where a few extreme users can "
        "carry the average"
        + (f" ({pct(zero_share, 0)} of users sit at zero here)" if zero_share > 0.5 else "")
        + ". It gets four looks that stress the tail differently: a Welch "
        "t-test (relies on the normal approximation of the mean), a percentile "
        "bootstrap (resamples the observed tail), a permutation test (exact "
        "under the null that the arms' distributions are identical), and a "
        "trimmed-mean difference (a diagnostic for tail dependence, "
        "deliberately a different estimand).\n\n"
        f"| Method | Estimate | Interval / p |\n|---|---|---|\n"
        f"| Welch t-test | {welch.difference:+.3f} | 95% CI {welch.ci_low:+.3f} "
        f"to {welch.ci_high:+.3f}, p = {fmt_p(welch.p_value)} |\n"
        f"| Percentile bootstrap | {boot.estimate:+.3f} | 95% CI {boot.ci_low:+.3f} "
        f"to {boot.ci_high:+.3f} |\n"
        f"| Permutation test | {perm.observed:+.3f} | p = {fmt_p(perm.p_value)} |\n"
        f"| Trimmed mean diff (1%) | {trimmed:+.3f} | different estimand |\n\n"
        f"{{{{fig:heavy_tail}}}} shows the distributions and the interval "
        f"comparison for the first contrast ({_contrast_label(exp, treatment_arm)}). " + agreement
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
    x: NDArray,
    y: NDArray,
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
    pos_x, pos_y = x[x > 0], y[y > 0]
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
    for yi, (_, center, lo, hi) in enumerate(intervals):
        ax2.plot([lo, hi], [yi, yi], color=COLOR_CONTROL, linewidth=2)
        ax2.plot(center, yi, "o", color=COLOR_INK, markersize=7)
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
    rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(exp.truth.items()))
    body = (
        "This data is SYNTHETIC, and that is disclosed here rather than in a "
        "footnote. The generator's parameters below are the exam key the "
        "analysis above can be scored against. No real-data section of any "
        "report produced by this pipeline carries this section, which is "
        "itself the honest signal of which results came from manufactured "
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
    return [s for s in candidates if s is not None]


def sweep_aggregate_archive(config: dict, data_path: Path) -> SectionSpec:
    """One section summarizing an ENTIRE aggregate archive, every
    experiment, alongside the deep dive the rest of the report gives one.

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
    metric_by_column = {m.column: m for m in metrics}
    primary_col = next(m.column for m in metrics if m.primary)
    alpha = config["experiment"]["alpha"]

    experiments: dict[str, list[dict]] = {}
    with open(data_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            experiments.setdefault(
                row[agg.get("experiment_id_column", "experiment_id")], []
            ).append(row)

    unstable: list[str] = []
    stability_skipped = 0
    naive_any = holm_any = 0
    cells_skipped = 0
    analyzed = 0
    for exp_id, rows in sorted(experiments.items()):
        # cumulative primary-metric counts per variant, checkpoint-ordered
        variants = sorted({r[agg["variant"]] for r in rows})
        worst_p = float("nan")
        for variant in variants:
            series = sorted(
                (
                    r
                    for r in rows
                    if r[agg["variant"]] == variant and r[agg["metric"]] == primary_col
                ),
                key=lambda r: float(r[agg["checkpoint"]]),
            )
            if len(series) < 2:
                continue
            nc = np.array([float(r[agg["count_control"]]) for r in series])
            nt = np.array([float(r[agg["count_treatment"]]) for r in series])
            if np.any(np.diff(nc) < 0) or np.any(np.diff(nt) < 0):
                continue
            _, df, p, _ = _ratio_stability(nc, nt)
            if df > 0 and (np.isnan(worst_p) or p < worst_p):
                worst_p = p
        if np.isnan(worst_p):
            stability_skipped += 1
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
                        r
                        for r in rows
                        if r[agg["variant"]] == variant and r[agg["metric"]] == metric_col
                    ),
                    key=lambda r: float(r[agg["checkpoint"]]),
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
            naive_any += int(any(p < alpha for p in p_values))
            holm_any += int(any(float(h) < alpha for h in holm(np.array(p_values))))

    unstable_list = ", ".join(unstable) if unstable else "none"
    body = (
        f"The sections above read one experiment closely; this one asks "
        f"two mechanical questions of every experiment in the archive "
        f"({len(experiments)} total). The first question is assignment "
        "stability. With no intended splits on record, the checkable "
        "property is that each experiment's arm ratio holds constant "
        "across its checkpoints (primary-metric denominators, worst "
        f"contrast per experiment). {analyzed} experiments were testable; "
        f"{stability_skipped} could not be tested (too few usable "
        "checkpoints or a degenerate split) and are counted rather than "
        f"hidden. Experiments flagged at the conventional p < 1e-3 alarm: "
        f"{unstable_list}.\n\n"
        f"The second question is multiplicity at scale. Each experiment's "
        "full end-of-run comparison family (every metric on every variant) "
        f"was tested unadjusted and after a per-experiment Holm correction. "
        f"{naive_any} of {len(experiments)} experiments show at least one "
        f"nominally significant comparison before correction; {holm_any} "
        "still do after it. The gap between those two counts is the multiple-"
        "comparisons trap measured on real experiments rather than argued "
        "from theory."
        + (
            f" ({cells_skipped} comparison cells were skipped because their "
            "published statistics were missing or unusable; the skips are "
            "counted here rather than hidden.)"
            if cells_skipped
            else ""
        )
    )
    return SectionSpec(
        key="archive_sweep",
        title="The archive at large: stability and multiplicity",
        body=body,
    )

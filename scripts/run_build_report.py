"""Regenerate analysis/report.md and analysis/figures/ from scratch.

Fully deterministic: every random draw flows from the fixed seeds below,
so re-running the script reproduces the committed report byte for byte
(figures may differ across matplotlib versions).

Run with: python scripts/run_build_report.py (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats

from abkit.frequentist import benjamini_hochberg, holm, srm_check, two_proportion_ztest, welch_ttest
from abkit.generators import ExperimentData, generate_binary, generate_continuous
from abkit.power import power_two_means, sample_size_two_proportions
from abkit.resampling import bootstrap_mean_diff, permutation_test_mean_diff, trimmed_mean_diff
from abkit.results import BootstrapResult, MeanComparison, RateComparison
from abkit.sequential import design, monitor, z_from_cumulative_counts

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
FIGURES = ANALYSIS / "figures"

# experiment configuration (the "truth" the analysis tries to recover)
N_USERS = 120_000
N_DAYS = 14
BASE_RATE = 0.06
TRUE_LIFT = 0.004
NOVELTY_BOOST = 0.008
NOVELTY_HALF_LIFE = 3.0

SPEND_RATE = 0.06
SPEND_RATE_LIFT = 0.004
LOG_MEAN = 3.4
LOG_SIGMA = 1.3
WHALE_RATE = 0.008
WHALE_SHAPE = 1.5
WHALE_SCALE = 400.0

SEED_ABORTED_RUN = 55
SEED_PRIMARY = 38
SEED_SECONDARY_BASE = 124_000
SEED_REVENUE = 221
SEED_BOOTSTRAP = 1
SEED_PERMUTATION = 2
SEED_NULL_SIM = 97

SECONDARY_METRICS = [
    "Add-to-wishlist rate",
    "Search usage",
    "Account creation",
    "Support contact rate",
    "Coupon usage",
    "Newsletter signup",
]

# blue/red rather than the matplotlib default green/orange pair, which is the
# hardest combination for red-green colour blindness to separate
COLOR_CONTROL = "#4269d0"
COLOR_TREATMENT = "#d4615c"
COLOR_REFERENCE = "#8a8f98"
COLOR_INK = "#31343a"


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f} pp"


_SMALL_NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def spell(n: int) -> str:
    """Small counts spelled out, so interpolated prose still reads as prose."""
    return _SMALL_NUMBERS[n] if n < len(_SMALL_NUMBERS) else f"{n:,}"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def style_axis(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def rate_test(data: ExperimentData) -> RateComparison:
    return two_proportion_ztest(
        int(data.control.sum()), data.n_control, int(data.treatment.sum()), data.n_treatment
    )


def lift_se(r: RateComparison) -> float:
    """Unpooled standard error of the absolute lift in a rate comparison."""
    return float(
        np.sqrt(
            r.rate_control * (1.0 - r.rate_control) / r.n_control
            + r.rate_treatment * (1.0 - r.rate_treatment) / r.n_treatment
        )
    )


def fig_daily_rates(data: ExperimentData) -> None:
    days = np.arange(N_DAYS)
    control_rates, treatment_rates, lifts, errs = [], [], [], []
    for d in days:
        sub = data.subset(data.day == d)
        control_rates.append(sub.control.mean() * 100)
        treatment_rates.append(sub.treatment.mean() * 100)
        r = rate_test(sub)
        lifts.append(r.absolute_lift * 100)
        errs.append((r.ci_high - r.ci_low) / 2 * 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    ax1.plot(days, control_rates, color=COLOR_CONTROL, linewidth=2, label="Control")
    ax1.plot(days, treatment_rates, color=COLOR_TREATMENT, linewidth=2, label="Treatment")
    ax1.set_ylabel("Conversion rate (%)")
    ax1.set_title("Daily conversion rate by arm", loc="left", color=COLOR_INK)
    ax1.legend(frameon=False, loc="upper right")
    style_axis(ax1)

    ax2.errorbar(
        days,
        lifts,
        yerr=errs,
        fmt="o",
        color=COLOR_INK,
        markersize=5,
        elinewidth=1,
        capsize=2,
        label="Observed daily lift (95% CI)",
    )
    truth = (TRUE_LIFT + NOVELTY_BOOST * 0.5 ** (days / NOVELTY_HALF_LIFE)) * 100
    ax2.plot(days, truth, color=COLOR_TREATMENT, linewidth=2, label="True effect (with novelty)")
    ax2.axhline(
        TRUE_LIFT * 100,
        color=COLOR_REFERENCE,
        linewidth=1.5,
        linestyle="--",
        label="True persistent effect",
    )
    ax2.axhline(0, color=COLOR_REFERENCE, linewidth=0.8)
    ax2.set_xlabel("Day of experiment")
    ax2.set_ylabel("Absolute lift (pp)")
    ax2.set_title("Daily lift versus the effect that generated it", loc="left", color=COLOR_INK)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    style_axis(ax2)
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_rates.png", dpi=150)
    plt.close(fig)


def fig_sequential(
    days: NDArray[np.int64],
    z_path: NDArray[np.float64],
    boundaries: NDArray[np.float64],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(
        days + 1,
        np.abs(z_path),
        color=COLOR_CONTROL,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Observed |z|",
    )
    ax.plot(
        days + 1,
        boundaries,
        color=COLOR_INK,
        linewidth=2,
        drawstyle="steps-mid",
        label="O'Brien-Fleming boundary",
    )
    ax.axhline(
        1.96,
        color=COLOR_REFERENCE,
        linewidth=1.5,
        linestyle="--",
        label="Naive fixed threshold (1.96)",
    )
    ax.set_xlabel("Daily look")
    ax.set_ylabel("|z|")
    ax.set_xticks(days + 1)
    # headroom so the legend clears the rising z path; early boundaries can
    # be infinite (no alpha spent yet) and are not drawn
    finite = boundaries[np.isfinite(boundaries)]
    ax.set_ylim(top=1.2 * max(np.abs(z_path).max(), finite.max()))
    # an infinite boundary is a real design property, not missing data: label
    # it so the gap at the left of the step line reads as "cannot stop here"
    for look in np.nonzero(~np.isfinite(boundaries))[0] + 1:
        ax.annotate(
            "no alpha spent:\nstopping impossible",
            (look, ax.get_ylim()[1]),
            xytext=(4, -6),
            textcoords="offset points",
            color=COLOR_INK,
            fontsize=8,
            va="top",
        )
    ax.set_title(
        "Daily monitoring: observed z against spending boundaries", loc="left", color=COLOR_INK
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "sequential.png", dpi=150)
    plt.close(fig)


def fig_secondary(naive: NDArray[np.float64], adjusted: NDArray[np.float64]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.0))
    y = np.arange(len(SECONDARY_METRICS))
    ax.scatter(naive, y, s=55, color=COLOR_CONTROL, label="Unadjusted p", zorder=3)
    ax.scatter(
        adjusted, y, s=55, color=COLOR_TREATMENT, marker="s", label="Holm-adjusted p", zorder=3
    )
    for yi in y:
        ax.plot([naive[yi], adjusted[yi]], [yi, yi], color=COLOR_REFERENCE, linewidth=1, zorder=2)
    ax.axvline(0.05, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    ax.text(
        0.05,
        0.97,
        "  alpha = 0.05",
        transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
        fontsize=9,
        color=COLOR_INK,
        va="top",
    )
    ax.set_yticks(y, SECONDARY_METRICS)
    ax.set_xscale("log")
    ax.set_xlabel("p-value (log scale)")
    ax.set_title(
        "Secondary metrics: adjustment removes the lone 'significant' result",
        loc="left",
        color=COLOR_INK,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.invert_yaxis()
    style_axis(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "secondary_pvalues.png", dpi=150)
    plt.close(fig)


def fig_revenue(
    data: ExperimentData,
    welch: MeanComparison,
    boot: BootstrapResult,
    trimmed: float,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    spender_c = data.control[data.control > 0]
    spender_t = data.treatment[data.treatment > 0]
    spenders = np.concatenate([spender_c, spender_t])
    # the log bins start at 1, which drops the handful of sub-dollar spends;
    # they are irrelevant to a figure about the upper tail
    bins = np.geomspace(1, spenders.max(), 60)
    ax1.hist(
        spender_c, bins=bins, histtype="step", linewidth=2, color=COLOR_CONTROL, label="Control"
    )
    ax1.hist(
        spender_t, bins=bins, histtype="step", linewidth=2, color=COLOR_TREATMENT, label="Treatment"
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Spend per spender (log scale)")
    ax1.set_ylabel("Users (log scale)")
    ax1.set_title(
        f"Spend distribution: the tail runs to {spenders.max() / np.median(spenders):,.0f}x "
        "the median spend",
        loc="left",
        color=COLOR_INK,
        fontsize=10,
    )
    ax1.legend(frameon=False, fontsize=9)
    style_axis(ax1)

    # Welch and the bootstrap estimate the same quantity and are drawn on the
    # same footing. The trimmed difference is a diagnostic, not a rival
    # estimate of the mean effect: trimming 1% of each tail of a 94%-zero
    # distribution removes zeros from the bottom and real spenders from the
    # top, so it targets a different population quantity entirely.
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
    ax2.set_xlabel("Revenue per user difference (treatment - control)")
    ax2.set_title(
        "Same data, two verdicts at the 0.05 line", loc="left", color=COLOR_INK, fontsize=10
    )
    ax2.invert_yaxis()
    style_axis(ax2, grid_axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "revenue.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    md: list[str] = []

    # ------------------------------------------------------------- design
    n_required = sample_size_two_proportions(BASE_RATE, BASE_RATE + TRUE_LIFT)
    md.append(f"""# Checkout redesign: end-to-end experiment analysis

This report walks a synthetic checkout-redesign experiment through the same
pipeline I would use on real data: power the design, validate the assignment,
read the primary metric, check for novelty, replay the monitoring decisions,
correct the secondary metric screen, and work one genuinely ambiguous metric
to a recommendation.

Because the data are synthetic, the true effects are known, and that is the
point. At the end I compare what the analysis concluded against what the
generator actually did. Everything here is produced by `scripts/run_build_report.py`
with fixed seeds; the library functions it calls live in `src/abkit/`.

## 1. Design and power

The primary metric is checkout conversion, baseline {pct(BASE_RATE, 1)}. The
smallest effect worth shipping was set at +{pp(TRUE_LIFT, 1)} absolute. For a
two-sided test at alpha 0.05 and 80% power, that needs
**{n_required:,} users per arm** (`sample_size_two_proportions`), so the
experiment was sized at {N_USERS:,} users over {N_DAYS} days with a 50/50
split, giving roughly {N_USERS // 2:,} per arm - some margin over the minimum.
""")

    # ------------------------------------------- aborted first run: SRM
    rng_bad = np.random.default_rng(SEED_ABORTED_RUN)
    bad = generate_binary(
        N_USERS,
        BASE_RATE,
        TRUE_LIFT,
        rng=rng_bad,
        novelty_boost=NOVELTY_BOOST,
        novelty_half_life_days=NOVELTY_HALF_LIFE,
        treatment_loss_rate=0.03,
    )
    bad_srm = srm_check(bad.n_control, bad.n_treatment)
    bad_result = rate_test(bad)

    rng = np.random.default_rng(SEED_PRIMARY)
    data = generate_binary(
        N_USERS,
        BASE_RATE,
        TRUE_LIFT,
        rng=rng,
        novelty_boost=NOVELTY_BOOST,
        novelty_half_life_days=NOVELTY_HALF_LIFE,
    )
    clean_srm = srm_check(data.n_control, data.n_treatment)

    md.append(f"""## 2. Assignment validation, and the run we threw away

The first launch of this experiment had a client bug. The treatment redirect
dropped the tracking beacon for a slice of users who bounced without
converting. Nothing about the metric looks wrong on the surface - the
treatment "won" with an observed lift of {pp(bad_result.absolute_lift)} - but
the sample-ratio check fails loudly:

| Run | Control n | Treatment n | SRM chi-square | SRM p-value |
|---|---|---|---|---|
| First (bug) | {bad_srm.n_control:,} | {bad_srm.n_treatment:,} | {bad_srm.statistic:.1f} | {fmt_p(bad_srm.p_value)} |
| Rerun (fixed) | {clean_srm.n_control:,} | {clean_srm.n_treatment:,} | {clean_srm.statistic:.2f} | {fmt_p(clean_srm.p_value)} |

A 50/50 split does not produce a {bad_srm.n_treatment / (bad_srm.n_control + bad_srm.n_treatment):.2%}
treatment share at this sample size by chance. Because the missing users were
disproportionately non-converters, the observed treatment rate was biased
upward, and no amount of downstream statistics repairs that. The first run's
data were discarded, the bug was fixed, and everything below analyzes the
rerun, which passes the check.
""")

    # ------------------------------------------------------ primary metric
    primary = rate_test(data)
    md.append(f"""## 3. Primary metric

| | Control | Treatment |
|---|---|---|
| Users | {primary.n_control:,} | {primary.n_treatment:,} |
| Conversions | {primary.conversions_control:,} | {primary.conversions_treatment:,} |
| Rate | {pct(primary.rate_control)} | {pct(primary.rate_treatment)} |

Absolute lift {pp(primary.absolute_lift)} (95% CI {pp(primary.ci_low)} to
{pp(primary.ci_high)}), relative lift {pct(primary.relative_lift, 1)},
z = {primary.statistic:.2f}, p = {fmt_p(primary.p_value)}. The headline result
is comfortably significant. But the headline is not the number I would ship a
forecast on, for the reason in the next section.
""")

    # ------------------------------------------------------- novelty check
    early = data.subset(data.day < 3)
    late = data.subset(data.day >= 10)
    early_r = rate_test(early)
    late_r = rate_test(late)
    # two non-overlapping CIs are not a test of the difference between them,
    # so test it directly; the windows are disjoint, hence independent
    decay_diff = early_r.absolute_lift - late_r.absolute_lift
    decay_se = float(np.hypot(lift_se(early_r), lift_se(late_r)))
    decay_z = decay_diff / decay_se
    decay_p = float(2.0 * stats.norm.sf(abs(decay_z)))
    fig_daily_rates(data)
    md.append(f"""## 4. Novelty check

The daily lift drifts downward over the experiment, which is the classic
signature of a novelty effect. Existing users react to the change itself, and
that reaction fades.

![Daily conversion rates and daily lift](figures/daily_rates.png)

Splitting the window confirms it:

| Window | Lift | 95% CI | p-value |
|---|---|---|---|
| Days 0-2 | {pp(early_r.absolute_lift)} | {pp(early_r.ci_low)} to {pp(early_r.ci_high)} | {fmt_p(early_r.p_value)} |
| Days 10-13 | {pp(late_r.absolute_lift)} | {pp(late_r.ci_low)} to {pp(late_r.ci_high)} | {fmt_p(late_r.p_value)} |

Two non-overlapping intervals are not by themselves evidence that the windows
differ, so I tested the decay directly rather than eyeballing the table. The
early-minus-late difference is {pp(decay_diff)} (SE {pp(decay_se)}),
z = {decay_z:.2f}, p = {fmt_p(decay_p)}. The fade is real, not an artifact of
reading two intervals side by side.

The early window carries most of the headline effect. The late window - the
best available proxy for the persistent effect - shows
{pp(late_r.absolute_lift)} with a confidence interval wide enough to include
both zero and the +{pp(TRUE_LIFT, 1)} the experiment was powered for. So the
honest summary is that the treatment very likely helps, that the headline
{pp(primary.absolute_lift)} overstates the long-run effect, and that the
{N_DAYS}-day window cannot pin the persistent lift precisely. I would report
the late-window interval, not the headline, and keep a small holdback running
to measure the effect after habituation.
""")

    # -------------------------------------------------- sequential replay
    look_days = np.arange(N_DAYS)
    nc = np.array([np.sum((data.assignment == 0) & (data.day <= d)) for d in look_days])
    nt = np.array([np.sum((data.assignment == 1) & (data.day <= d)) for d in look_days])
    xc = np.array(
        [np.sum(data.outcome[(data.assignment == 0) & (data.day <= d)]) for d in look_days]
    )
    xt = np.array(
        [np.sum(data.outcome[(data.assignment == 1) & (data.day <= d)]) for d in look_days]
    )
    fractions = (nc + nt) / (nc[-1] + nt[-1])
    z_path = z_from_cumulative_counts(xc, nc, xt, nt)
    seq = design(fractions.tolist(), alpha=0.05, spending="obrien-fleming")
    outcome = monitor(z_path.tolist(), seq)
    naive_crossings = np.nonzero(np.abs(z_path) >= 1.96)[0]
    # the prose below narrates a specific replay; fail loudly if a code change
    # alters the story instead of silently describing the wrong one. A bare
    # assert would vanish under python -O, which is exactly when it matters.
    if naive_crossings.size == 0 or outcome.stopped_at_look is None:
        raise RuntimeError(
            "sequential replay no longer matches the narrative in section 5: "
            f"naive crossings={naive_crossings.tolist()}, "
            f"spending-design stop={outcome.stopped_at_look}"
        )
    first_naive = int(naive_crossings[0]) + 1
    naive_lift = (
        xt[first_naive - 1] / nt[first_naive - 1] - xc[first_naive - 1] / nc[first_naive - 1]
    )
    fig_sequential(look_days, z_path, np.array(seq.boundaries))

    # type I error of both rules under the null, on the realized look schedule
    rng_null = np.random.default_rng(SEED_NULL_SIM)
    n_sims = 100_000
    increments = rng_null.normal(size=(n_sims, N_DAYS)) * np.sqrt(np.diff(fractions, prepend=0.0))
    z_null = np.cumsum(increments, axis=1) / np.sqrt(fractions)
    naive_alpha = float(np.any(np.abs(z_null) >= 1.96, axis=1).mean())
    spent_alpha = float(np.any(np.abs(z_null) >= np.array(seq.boundaries), axis=1).mean())
    n_unspendable = int(np.sum(~np.isfinite(np.array(seq.boundaries))))
    unspendable_phrase = (
        "the first look has" if n_unspendable == 1 else f"the first {n_unspendable} looks have"
    )
    max_fraction_drift = float(np.max(np.abs(fractions - (look_days + 1) / N_DAYS)))

    md.append(f"""## 5. What daily monitoring would have decided

Everyone peeks. The question is whether the peeking is accounted for. I
replayed the experiment as {N_DAYS} daily looks two ways: a naive rule that
calls the result at the first |z| > 1.96, and an O'Brien-Fleming alpha-spending
design built on the realized information fractions (`abkit.sequential`).

Using realized rather than planned fractions is what the spending-function
approach is for, and it does not cost anything here. The look *schedule* was
fixed in advance at one look per day, and the realized fractions differ from
an even {N_DAYS}-way split by at most {max_fraction_drift:.4f}. What would void the
guarantee is choosing the looks after seeing the data, which is a different
thing entirely.

![Sequential monitoring](figures/sequential.png)

The naive rule would have stopped at **look {first_naive}**, when the novelty
effect was near its peak, locking in a cumulative lift of
{pp(naive_lift)} - about {naive_lift / TRUE_LIFT:.0f}x the persistent effect.
The spending design first crosses at **look {outcome.stopped_at_look}**,
spending almost no alpha in the noisy early days. In fact it spends so little
that {unspendable_phrase} no finite boundary at all. Under O'Brien-Fleming
there is no z-statistic on day 1 large enough to stop the experiment, which is
why the boundary line in the figure starts at look {n_unspendable + 1}.
Under a true null on this exact look schedule, simulation ({n_sims:,} runs)
puts the naive rule's false-positive rate at {pct(naive_alpha, 1)} against the
nominal 5%; the spending design holds {pct(spent_alpha, 1)}.

Early stopping is a tool for catching regressions quickly, not for shipping
winners early. Even the correctly-stopped estimate is biased upward by the
novelty effect at that point in the run.
""")

    # ----------------------------------------- secondary metrics screen
    naive_ps = []
    lifts = []
    for i in range(len(SECONDARY_METRICS)):
        rng_i = np.random.default_rng(SEED_SECONDARY_BASE + i)
        metric_data = generate_binary(N_USERS, 0.20, 0.0, rng=rng_i)
        r = rate_test(metric_data)
        naive_ps.append(r.p_value)
        lifts.append(r.absolute_lift)
    naive_arr = np.array(naive_ps)
    holm_ps = holm(naive_arr)
    bh_ps = benjamini_hochberg(naive_arr)
    fig_secondary(naive_arr, holm_ps)

    # the smallest p-value, whichever direction it points; calling it a "win"
    # without checking the sign is how a regression gets shipped as a feature
    flagged = int(np.argmin(naive_arr))
    flagged_verdict = "win" if lifts[flagged] > 0 else "regression"
    family_error = 1.0 - (1.0 - 0.05) ** len(SECONDARY_METRICS)
    exactly_one = len(SECONDARY_METRICS) * 0.05 * (1.0 - 0.05) ** (len(SECONDARY_METRICS) - 1)
    rows = "\n".join(
        f"| {name} | {pp(lifts[i])} | {fmt_p(naive_arr[i])} | {fmt_p(holm_ps[i])} "
        f"| {fmt_p(bh_ps[i])} |"
        for i, name in enumerate(SECONDARY_METRICS)
    )
    md.append(f"""## 6. Secondary metrics and multiple comparisons

{spell(len(SECONDARY_METRICS)).capitalize()} secondary metrics were tracked. None of them was actually moved by
the treatment (the generator applied zero lift to all {spell(len(SECONDARY_METRICS))}), which makes
this a live-fire test of the correction machinery.

| Metric | Observed lift | Unadjusted p | Holm p | BH q |
|---|---|---|---|---|
{rows}

Read naively, **{SECONDARY_METRICS[flagged]}** clears the line at
p = {fmt_p(naive_arr[flagged])} - and note the direction. The observed lift is
{pp(lifts[flagged])}, so the naive reading is a {flagged_verdict}, not the win a
scan of the p-value column alone would suggest. That is the trap twice over.
With {spell(len(SECONDARY_METRICS))} true nulls the chance of at least one unadjusted p below 0.05 is
{family_error:.0%}, so a flagged metric is the expected outcome rather than a
discovery. Both the Holm adjustment (family-wise control, the right tool for a
ship decision) and Benjamini-Hochberg (FDR control, the right tool for a
screening pass) push every metric back above the line.

The seed here was chosen to land on a run that exhibits the false positive.
Across fresh draws only about {exactly_one:.0%} of runs produce exactly one nominally
significant metric, and most produce none. That is a deliberate choice to show
the machinery working, not a typical draw, and it seemed worth saying out loud
in a document whose whole argument is about not reading luck as signal.

![Secondary metric p-values](figures/secondary_pvalues.png)
""")

    # ------------------------------------------------ revenue: ambiguous
    rng_rev = np.random.default_rng(SEED_REVENUE)
    revenue = generate_continuous(
        N_USERS,
        rng=rng_rev,
        spend_rate=SPEND_RATE,
        spend_rate_lift=SPEND_RATE_LIFT,
        log_mean=LOG_MEAN,
        log_sigma=LOG_SIGMA,
        whale_rate=WHALE_RATE,
        whale_shape=WHALE_SHAPE,
        whale_scale=WHALE_SCALE,
    )
    welch = welch_ttest(revenue.control, revenue.treatment)
    boot = bootstrap_mean_diff(
        revenue.control,
        revenue.treatment,
        rng=np.random.default_rng(SEED_BOOTSTRAP),
        n_resamples=10_000,
    )
    perm = permutation_test_mean_diff(
        revenue.control,
        revenue.treatment,
        rng=np.random.default_rng(SEED_PERMUTATION),
        n_permutations=10_000,
    )
    trimmed = trimmed_mean_diff(revenue.control, revenue.treatment, trim=0.01)

    mean_spend = (1 - WHALE_RATE) * np.exp(LOG_MEAN + LOG_SIGMA**2 / 2) + (
        WHALE_RATE * WHALE_SCALE * WHALE_SHAPE / (WHALE_SHAPE - 1)
    )
    true_rev_diff = SPEND_RATE_LIFT * mean_spend
    sample_sd = np.concatenate([revenue.control, revenue.treatment]).std(ddof=1)
    # what the 1% trim actually removes, given a mostly-zero distribution
    zero_share = float(np.mean(revenue.outcome == 0.0))
    trim_zeros = int(0.01 * revenue.control.size)
    trim_spender_share = trim_zeros / int((revenue.control > 0).sum())
    control_raw_mean = float(revenue.control.mean())
    control_trim_mean = float(stats.trim_mean(revenue.control, 0.01))
    apparent_power = power_two_means(
        true_rev_diff, float(sample_sd), min(welch.n_control, welch.n_treatment)
    )
    fig_revenue(revenue, welch, boot, trimmed)

    md.append(f"""## 7. Revenue per user: the ambiguous one

Revenue per user is the metric leadership actually asks about, and it is the
one this experiment cannot answer cleanly. The distribution is zero-inflated
({pct(zero_share, 0)} of users spend nothing) with a Pareto tail. The generator
gives roughly 1 in {1 / WHALE_RATE:.0f} spenders a "whale" draw whose distribution has
infinite variance. This is not an exotic choice; real revenue data looks like
this.

![Revenue distributions and interval comparison](figures/revenue.png)

| Method | Estimate | Interval / p |
|---|---|---|
| Welch t-test | {welch.difference:+.2f} per user | 95% CI {welch.ci_low:+.2f} to {welch.ci_high:+.2f}, p = {fmt_p(welch.p_value)} |
| Percentile bootstrap | {boot.estimate:+.2f} per user | 95% CI {boot.ci_low:+.2f} to {boot.ci_high:+.2f} |
| Permutation test | {perm.observed:+.2f} per user | p = {fmt_p(perm.p_value)} |
| Trimmed mean diff (1%) | {trimmed:+.2f} per user | different estimand, see below |

The first three estimate the same quantity and disagree about it. The Welch
test says not significant (p = {fmt_p(welch.p_value)}); the permutation test says
significant (p = {fmt_p(perm.p_value)}); the bootstrap interval
({boot.ci_low:+.2f} to {boot.ci_high:+.2f}) excludes zero, but its lower bound is a whisker
above it.

The Welch/permutation split has a specific cause. The permutation test is
exact under the sharp null that the two arms
have *identical distributions*, and the treatment here does change the
distribution - it raises the probability of spending at all. So the permutation
test can correctly reject while the *mean* difference, which is what Welch is
estimating, stays badly determined by a few large draws. The two tests ask
two different questions, and only one of them is the one leadership asked.

The trimmed difference is a diagnostic, not a fourth estimate of the same
thing, and the table above would be misleading if it were read as one. With
{pct(zero_share, 0)} of users at zero, trimming 1% from each tail of each arm cuts
{trim_zeros:,} zeros off the bottom and about {pct(trim_spender_share, 0)} of the *spenders* off
the top. It therefore targets a different population quantity. The control
arm's own mean falls from {control_raw_mean:.2f} to {control_trim_mean:.2f} with no treatment
effect involved at all. The useful signal is that {trimmed:+.2f} and
{welch.difference:+.2f} are far apart, which says the raw difference rests on the
part of the distribution the trim removes - but any heavy right tail produces
that gap, so it points at fragility rather than proving whales specifically.

There is no analysis trick that resolves this, because the ambiguity is real.
The true effect the generator planted is {true_rev_diff:+.2f} per user, and
with this tail the experiment only had roughly
{pct(apparent_power, 0)} power to detect it (using the realized sample
standard deviation of {sample_sd:.0f}, itself an unstable estimate under an
infinite-variance tail). The generator draws whale amounts from the same distribution in both
arms, so every bit of the whale imbalance between arms is sampling noise *by
construction*. A method that reads that imbalance as effect is reading noise,
and we know it is noise because we built the data.

My read is that the revenue evidence is directionally positive and consistent
with the conversion result, that the point estimate rests on draws carrying no
signal, and that I would not quote a revenue impact number from this
experiment. If a
revenue claim matters, the follow-up should use a metric that a single whale
cannot move (capped revenue or spender conversion), declared in advance and
sized for it.

What I would not do is run all four and report the one that crossed 0.05. They
were all planned; the disagreement itself is the finding, and it says
"underpowered for this metric," not "significant if you pick the right test."
""")

    # ------------------------------------------------------- ground truth
    md.append(f"""## 8. What the generator actually did

| Quantity | Truth | Headline estimate | Best estimate |
|---|---|---|---|
| Persistent conversion lift | {pp(TRUE_LIFT, 1)} | {pp(primary.absolute_lift)} | {pp(late_r.absolute_lift)} (days 10-13, CI {pp(late_r.ci_low)} to {pp(late_r.ci_high)}) |
| Day-0 novelty boost | {pp(NOVELTY_BOOST, 1)} | not separately reported | decay visible in the daily-lift panel in section 4 |
| Secondary metric lifts | 0 on all six | one naive false positive | none significant after Holm |
| Revenue per user diff | {true_rev_diff:+.2f} | {welch.difference:+.2f} | interval spanning zero; underpowered |

The pipeline got the qualitative story right everywhere. It caught the broken
first run, flagged the novelty inflation instead of shipping the headline
number, and refused the spurious secondary win. On the underpowered revenue
metric, it declined to quote an impact number at all. The place it cannot
help is where the data simply do not contain the answer - the persistent
lift and the revenue effect both need more time or a better metric, and the
report says so rather than manufacturing certainty.

## Recommendation

Ship the redesign on the strength of the conversion result, quote the
late-window interval rather than the headline, keep a 5% holdback for four
more weeks to measure the post-novelty effect, and define a capped revenue
metric before the follow-up starts.
""")

    (ANALYSIS / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {ANALYSIS / 'report.md'} and {len(list(FIGURES.glob('*.png')))} figures")


if __name__ == "__main__":
    main()

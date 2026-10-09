"""The readout as printable lines: labeled blocks, short guides, no prose.

Every number comes from an ``ExperimentReadout``; nothing is recomputed
here. Each block is a heading carrying a few-word guide to reading it,
then indented labeled lines, then a result phrase chosen from a fixed
table. A marker with a one-line legend stands in for a sentence. The
readout closes with a decision line and the figures' paths, and is saved
as readout.txt beside them.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from abkit.readout_analysis import ExperimentReadout, planted_end_of_run_lift
from abkit.results import CupedResult, MeanComparison, PermutationResult, RateComparison

LINE_WIDTH = 100
_INDENT = "  "

_PRIMARY_RESULTS = {
    "higher": "detected, treatment higher",
    "lower": "detected, treatment lower",
    "not_detected": "not detected",
}
_DECISIONS = {
    "split_broken": "Decision: do not act; the group split is broken",
    "higher": "Decision for {arm} vs {control}: higher on {metric} by {effect} ({low} to {high})",
    "lower": "Decision for {arm} vs {control}: lower on {metric} by {effect} ({low} to {high})",
    "not_detected": (
        "Decision for {arm} vs {control}: no detectable effect; that contrast could have seen "
        "{mde} or more"
    ),
}
_SPLIT_RESULTS = {
    "ok": "consistent with the plan",
    "fail": "FAILS; users went missing or were assigned unevenly",
}
_STABILITY_RESULTS = {"ok": "stable across checkpoints", "fail": "SHIFTS; assignment broke mid-run"}
_TAIL_RESULTS = {
    "agree": "methods agree",
    "disagree": "methods DISAGREE at alpha; the estimate rests on the tail",
}
_NOVELTY_RESULTS = {
    "fell": "effect fell over the run",
    "rose": "effect rose over the run",
    "steady": "no change detected",
}


def pct(fraction: float, digits: int = 2) -> str:
    """Format a fraction as a percentage."""
    return f"{100 * fraction:.{digits}f}%"


def pp(difference: float, digits: int = 2) -> str:
    """Format a difference in proportions as percentage points."""
    return f"{100 * difference:.{digits}f} pp"


def pp_signed(difference: float, digits: int = 2) -> str:
    """Signed percentage-point difference, so effect columns read one way."""
    return f"{100 * difference:+.{digits}f} pp"


def fmt_p(p_value: float) -> str:
    """Format a p-value with a precision that fits its size."""
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}" if p_value < 0.001 else f"{p_value:.3f}"


def fmt_perm_p(permutation: PermutationResult) -> str:
    """The permutation p-value, naming the floor when it sits there.

    scipy's two-sided p-value cannot go below 2 / (n_permutations + 1), reached
    when no relabeling is as extreme as the observed difference; printed bare
    that reads as a measurement rather than as the resolution limit.
    """
    text = f"p {fmt_p(permutation.p_value)}"
    if permutation.p_value <= 2.0 / (permutation.n_permutations + 1):
        text += f", the floor for {permutation.n_permutations:,} relabelings"
    return text


def _effect(comparison: RateComparison | MeanComparison) -> str:
    """A comparison's effect in its own unit: percentage points for rates, plain for means."""
    if isinstance(comparison, RateComparison):
        return pp_signed(comparison.absolute_lift)
    return f"{comparison.difference:+.3f}"


def _interval(comparison: RateComparison | MeanComparison) -> str:
    """A comparison's interval in its own unit."""
    if isinstance(comparison, RateComparison):
        return f"{100 * comparison.ci_low:+.2f} to {pp_signed(comparison.ci_high)}"
    return f"{comparison.ci_low:+.3f} to {comparison.ci_high:+.3f}"


def _primary_outcome(comparison: RateComparison | MeanComparison, alpha: float) -> str:
    """Which phrase table entry a primary comparison lands on."""
    if comparison.p_value >= alpha:
        return "not_detected"
    effect = (
        comparison.absolute_lift
        if isinstance(comparison, RateComparison)
        else comparison.difference
    )
    return "higher" if effect > 0 else "lower"


def _wrapped(text: str, indent: str = _INDENT) -> list[str]:
    """Wrap one labeled line to the readout width, continuation lines indented further."""
    return textwrap.wrap(
        text,
        width=LINE_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + "    ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _binary_metrics(readout: ExperimentReadout) -> set[str]:
    """The metrics whose comparisons are rate comparisons, so CUPED lines format in points."""
    names: set[str] = set()
    if readout.primary and isinstance(readout.primary[0][1], RateComparison):
        names.add(readout.primary_metric)
    names.update(
        corrected.metric
        for corrected in readout.corrections
        if isinstance(corrected.comparison, RateComparison)
    )
    return names


def _cuped_line(treatment_arm: str, control_arm: str, result: CupedResult, binary: bool) -> str:
    """One CUPED contrast: adjusted effect, interval, p, and the share of noise removed."""
    if binary:
        effect = pp_signed(result.difference)
        interval = f"{100 * result.ci_low:+.2f} to {pp_signed(result.ci_high)}"
    else:
        effect = f"{result.difference:+.3f}"
        interval = f"{result.ci_low:+.3f} to {result.ci_high:+.3f}"
    return (
        f"{treatment_arm} vs {control_arm}   {effect}   {interval}   p {fmt_p(result.p_value)}"
        f"   noise removed {result.variance_reduction:.0%}"
    )


def _data_block(readout: ExperimentReadout) -> list[str]:
    """Block 1: source, data shape, arm sizes, and what the intake did."""
    shape = "user rows" if readout.data_shape == "user_rows" else "per-group totals"
    lines = [
        f"{readout.experiment_name}: {shape}",
        "Data: what the intake did before any statistic",
        _INDENT + "users   " + "   ".join(f"{arm} {size:,}" for arm, size in readout.arm_sizes),
    ]
    if readout.validation:
        for event in readout.validation:
            lines += _wrapped(f"{event.rule}   {event.count:,} rows   {event.detail}")
    else:
        lines.append(_INDENT + "repairs and drops   none; the file passed every rule as delivered")
    return lines


def _limitations_block(readout: ExperimentReadout) -> list[str]:
    """Block 2: the flags, one line each, or none."""
    lines = ["", "Limitations (*): read every result below with these in view"]
    if not readout.limitations:
        lines.append(_INDENT + "none raised")
    for limitation in readout.limitations:
        lines += _wrapped(f"* {limitation.detail}")
    return lines


def _group_split_block(readout: ExperimentReadout) -> list[str]:
    """Block 3: the split check in whichever form the data allowed."""
    control_arm = readout.arm_sizes[0][0]
    sizes = dict(readout.arm_sizes)
    lines = ["", "Group split: does assignment match the plan?"]
    if readout.group_split is not None:
        worst = 1.0
        for treatment_arm, srm in readout.group_split:
            worst = min(worst, srm.p_value)
            lines.append(
                _INDENT + f"{treatment_arm} vs {control_arm}   {sizes[treatment_arm]:,} vs "
                f"{sizes[control_arm]:,}   chi-square {srm.statistic:.2f}   p {fmt_p(srm.p_value)}"
            )
        lines.append(_INDENT + "result   " + _SPLIT_RESULTS["ok" if worst >= 1e-3 else "fail"])
    elif readout.ratio_stability is not None:
        statistic, degrees_of_freedom, p_value, n_intervals = readout.ratio_stability
        treatment_arm = readout.arm_sizes[1][0]
        lines.append(
            _INDENT + "intended split not on record; testing whether the share of new users "
            "per checkpoint is constant"
        )
        if np.isfinite(p_value):
            lines.append(
                _INDENT + f"{treatment_arm} vs {control_arm}   {n_intervals} checkpoints   "
                f"chi-square {statistic:.2f}   df {degrees_of_freedom}   p {fmt_p(p_value)}"
            )
            lines.append(
                _INDENT + "result   " + _STABILITY_RESULTS["ok" if p_value >= 1e-3 else "fail"]
            )
        else:
            lines.append(_INDENT + "result   not testable; one arm never gained a user")
    else:
        lines.append(
            _INDENT + "no intended split on record and no day column, so no check is possible"
        )
        lines.append(_INDENT + "result   sizes above are for the reader's own judgment")
    return lines


def _sensitivity_block(readout: ExperimentReadout) -> list[str]:
    """Block 4: the smallest effect the first contrast could detect."""
    values = readout.sensitivity
    binary = readout.primary_metric in _binary_metrics(readout)
    lines = ["", "Sensitivity: smallest lift this sample could detect, 80% power"]
    control = pct(values["control_mean"], 1) if binary else f"{values['control_mean']:.3f}"
    mde = values["mde"]
    if not np.isfinite(mde):
        detectable = "none reachable"
    else:
        detectable = pp(mde) if binary else f"{mde:.3f}"
        if np.isfinite(values["relative_mde"]):
            detectable += f" ({pct(values['relative_mde'], 1)} of control)"
    treatment_arm, control_arm = readout.primary[0][0], readout.arm_sizes[0][0]
    lines += _wrapped(
        f"{treatment_arm} vs {control_arm}   {values['n_treatment']:,.0f} vs "
        f"{values['n_control']:,.0f} users   control {control}   detectable {detectable}"
    )
    return lines


def _primary_block(readout: ExperimentReadout) -> list[str]:
    """Block 5: the primary metric per contrast, interval first, then CUPED where it ran."""
    control_arm = readout.arm_sizes[0][0]
    lines = ["", f"Primary {readout.primary_metric}: effect, 95% interval, p; interval first"]
    for treatment_arm, comparison in readout.primary:
        outcome = _primary_outcome(comparison, readout.alpha)
        lines += _wrapped(
            f"{treatment_arm} vs {control_arm}   {_effect(comparison)}   "
            f"{_interval(comparison)}   p {fmt_p(comparison.p_value)}   "
            + _PRIMARY_RESULTS[outcome]
        )
    binary_metrics = _binary_metrics(readout)
    for metric in dict.fromkeys(metric for metric, _, _ in readout.cuped):
        lines += [
            "",
            f"CUPED on {metric}: adjusted by its pre-experiment column; "
            "narrower interval = less noise",
        ]
        for cuped_metric, treatment_arm, result in readout.cuped:
            if cuped_metric == metric:
                lines += _wrapped(
                    _cuped_line(treatment_arm, control_arm, result, metric in binary_metrics)
                )
    return lines


def _corrections_block(readout: ExperimentReadout) -> list[str]:
    """Block 6: the secondary family under Holm and Benjamini-Hochberg."""
    if not readout.corrections:
        return []
    control_arm = readout.arm_sizes[0][0]
    family = readout.corrections
    lines = [
        "",
        f"Other metrics: corrected for {len(family)} comparisons "
        "(Holm, and BH = Benjamini-Hochberg)",
    ]
    for corrected in family:
        lines += _wrapped(
            f"{corrected.metric}   {corrected.treatment_arm} vs {control_arm}   "
            f"{_effect(corrected.comparison)}   p {fmt_p(corrected.comparison.p_value)}   "
            f"Holm {fmt_p(corrected.holm_p)}   BH {fmt_p(corrected.bh_q)}"
        )
    n_naive = sum(corrected.comparison.p_value < readout.alpha for corrected in family)
    n_holm = sum(corrected.holm_p < readout.alpha for corrected in family)
    lines.append(
        _INDENT + f"result   unadjusted {n_naive} of {len(family)} below {readout.alpha:g}; "
        f"after Holm {n_holm} survive"
    )
    return lines


def _heavy_tail_block(readout: ExperimentReadout) -> list[str]:
    """Block 7: the revenue cross-check, four rows and a result."""
    check = readout.heavy_tail
    if check is None:
        return []
    control_arm = readout.arm_sizes[0][0]
    welch, bootstrap, permutation = check.welch, check.bootstrap, check.permutation
    boot_excludes_zero = (bootstrap.ci_low > 0.0) == (bootstrap.ci_high > 0.0)
    welch_significant = welch.p_value < readout.alpha
    agree = (
        welch_significant == (permutation.p_value < readout.alpha)
        and boot_excludes_zero == welch_significant
    )
    lines = [
        "",
        f"Cross-check on {check.metric}: three methods, same question "
        f"({check.treatment_arm} vs {control_arm})",
        _INDENT + f"Welch         {welch.difference:+.3f}   {welch.ci_low:+.3f} to "
        f"{welch.ci_high:+.3f}   p {fmt_p(welch.p_value)}",
        _INDENT + f"bootstrap     {bootstrap.estimate:+.3f}   {bootstrap.ci_low:+.3f} to "
        f"{bootstrap.ci_high:+.3f}   BCa, {bootstrap.n_resamples:,} resamples",
        _INDENT + f"permutation   {permutation.observed:+.3f}   {fmt_perm_p(permutation)}",
        _INDENT
        + f"trimmed 1%    {check.trimmed_difference:+.3f}   a different quantity, not comparable",
        _INDENT + "result   " + _TAIL_RESULTS["agree" if agree else "disagree"],
    ]
    if check.trim_degenerate:
        lines.append(
            _INDENT + f"trimmed row is not a tail check here: {pct(check.zero_share, 0)} of users "
            "sit at zero"
        )
    return lines


def _novelty_block(readout: ExperimentReadout) -> list[str]:
    """Block 8a: early against late, per contrast."""
    check = readout.novelty
    if check is None:
        return []
    control_arm = readout.arm_sizes[0][0]
    binary = check.metric in _binary_metrics(readout)
    lines = [
        "",
        "Early versus late: did the effect fade?",
        _INDENT
        + (
            f"first {check.window_size} checkpoints against the last {check.window_size}"
            if check.window_size != 1
            else "first checkpoint against the last"
        )
        + f", of {check.n_usable} usable",
    ]
    for contrast in check.contrasts:
        early = pp_signed(contrast.early) if binary else f"{contrast.early:+.3f}"
        late = pp_signed(contrast.late) if binary else f"{contrast.late:+.3f}"
        if contrast.decay_p >= readout.alpha:
            outcome = "steady"
        else:
            outcome = "fell" if contrast.decay_z > 0 else "rose"
        lines += _wrapped(
            f"{contrast.treatment_arm} vs {control_arm}   early {early}   late {late}   "
            f"z {contrast.decay_z:.2f}   p {fmt_p(contrast.decay_p)}   " + _NOVELTY_RESULTS[outcome]
        )
    return lines


def _monitoring_block(readout: ExperimentReadout) -> list[str]:
    """Block 8b: the daily-monitoring replay with the final look's adjusted interval."""
    replay = readout.monitoring
    if replay is None:
        return []
    control_arm = readout.arm_sizes[0][0]
    binary = replay.metric in _binary_metrics(readout)
    n_looks = replay.n_usable
    stopped = replay.outcome.stopped_at_look
    design_line = (
        f"crosses at look {stopped} of {n_looks}; stopping there keeps the full run's guarantee"
        if stopped is not None
        else f"never crosses in {n_looks} looks; no early stop in either direction"
    )
    if replay.naive_look is None:
        naive_line = "never fires"
    elif stopped is None or replay.naive_look < stopped:
        naive_line = f"fires at look {replay.naive_look}, on evidence the design does not accept"
    else:
        naive_line = f"fires at look {replay.naive_look}"
    last = n_looks - 1
    estimate = replay.intervals.estimates[last]
    low, high = replay.intervals.ci_low[last], replay.intervals.ci_high[last]
    if binary:
        final = f"effect {pp_signed(estimate)}   adjusted 95% {100 * low:+.2f} to {pp_signed(high)}"
    else:
        final = f"effect {estimate:+.3f}   adjusted 95% {low:+.3f} to {high:+.3f}"
    return [
        "",
        f"Daily monitoring: {n_looks} looks; the {readout.alpha:.0%} budget is spread across "
        "them; crossing a boundary = safe to stop",
        _INDENT + f"{replay.treatment_arm} vs {control_arm} on {replay.metric}, "
        "O'Brien-Fleming spending at the sizes reached",
        _INDENT + f"spending design   {design_line}",
        _INDENT + f"naive {replay.z_naive:.2f} rule   {naive_line}",
        _INDENT + f"final look   {final}   boundary |z| {replay.intervals.boundaries[last]:.2f}",
    ]


def _truth_block(readout: ExperimentReadout) -> list[str]:
    """Block 9, synthetic runs only: what the generator planted beside what the readout found."""
    if readout.truth is None:
        return []
    lines = ["", "Planted versus found (synthetic only)"]
    found: dict[str, str] = {}
    if readout.primary:
        _, comparison = readout.primary[0]
        if isinstance(comparison, RateComparison):
            found["base_rate"] = f"found {comparison.rate_control:.4f} (control rate)"
            expected = planted_end_of_run_lift(readout.truth)
            if expected is not None:
                inside = comparison.ci_low <= expected <= comparison.ci_high
                novelty_note = ", novelty averaged in" if "novelty_boost" in readout.truth else ""
                found["absolute_lift"] = (
                    f"expected {pp_signed(expected)} at the end of the run{novelty_note}   "
                    f"found {_effect(comparison)} ({_interval(comparison)})   "
                    + ("inside" if inside else "OUTSIDE")
                )
    if readout.novelty is not None and readout.novelty.contrasts:
        first = readout.novelty.contrasts[0]
        found["novelty_boost"] = (
            f"found early {pp_signed(first.early)} against late {pp_signed(first.late)}, "
            + _NOVELTY_RESULTS[
                "steady"
                if first.decay_p >= readout.alpha
                else ("fell" if first.decay_z > 0 else "rose")
            ]
        )
    for metric, _, result in readout.cuped:
        found["pre_correlation"] = (
            f"found CUPED on {metric} removed {result.variance_reduction:.0%} of the noise "
            "(the planted value links hidden draws, not dollar amounts, and the whale tail has "
            "no pre-period counterpart)"
        )
    width = max(len(name) for name in readout.truth)
    for name, value in sorted(readout.truth.items()):
        line = f"{name:<{width}}   planted {_planted(value)}"
        if name in found:
            line += f"   {found[name]}"
        lines += _wrapped(line)
    return lines


def _planted(value: object) -> str:
    """A planted value as the readout prints numbers: thousands separators on whole numbers."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _decision_line(readout: ExperimentReadout) -> str:
    """The closing line: a broken split first, then the primary result of the first contrast,
    named, since a readout with several treatments decides one contrast at a time."""
    if any(limitation.code == "group_split" for limitation in readout.limitations):
        return _DECISIONS["split_broken"]
    treatment_arm, comparison = readout.primary[0]
    control_arm = readout.arm_sizes[0][0]
    outcome = _primary_outcome(comparison, readout.alpha)
    if outcome == "not_detected":
        mde = readout.sensitivity["mde"]
        if not np.isfinite(mde):
            mde_text = "no lift at all with this sample"
        elif isinstance(comparison, RateComparison):
            mde_text = pp(mde)
        else:
            mde_text = f"{mde:.3f}"
        return _DECISIONS[outcome].format(arm=treatment_arm, control=control_arm, mde=mde_text)
    if isinstance(comparison, RateComparison):
        low, high = f"{100 * comparison.ci_low:+.2f}", pp_signed(comparison.ci_high)
    else:
        low, high = f"{comparison.ci_low:+.3f}", f"{comparison.ci_high:+.3f}"
    return _DECISIONS[outcome].format(
        arm=treatment_arm,
        control=control_arm,
        metric=readout.primary_metric,
        effect=_effect(comparison),
        low=low,
        high=high,
    )


def readout_lines(
    readout: ExperimentReadout, output_dir: Path, figure_paths: list[Path]
) -> list[str]:
    """The whole readout as lines, ending with the decision and the figures' paths relative to
    where readout.txt is written."""
    lines = (
        _data_block(readout)
        + _limitations_block(readout)
        + _group_split_block(readout)
        + _sensitivity_block(readout)
        + _primary_block(readout)
        + _corrections_block(readout)
        + _heavy_tail_block(readout)
        + _novelty_block(readout)
        + _monitoring_block(readout)
        + _truth_block(readout)
        + [""]
        + _wrapped(_decision_line(readout), indent="")
    )
    if figure_paths:
        lines.append("Figures:")
        for path in figure_paths:
            shown = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path
            lines.append(_INDENT + shown.as_posix())
    return lines


def save_readout(
    readout: ExperimentReadout, output_dir: Path, figure_paths: list[Path]
) -> list[str]:
    """Write readout.txt into the output directory and return its lines for printing."""
    lines = readout_lines(readout, output_dir, figure_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    # fixed line endings, so a committed readout reads the same from any platform
    (output_dir / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return lines

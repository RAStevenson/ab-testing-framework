"""The readout as printable lines: labeled blocks, short guides, no prose.

Every number comes from an ``ExperimentReadout``; nothing is recomputed
here. Each block is a heading carrying a few-word guide to reading it,
then indented labeled lines, then a result phrase chosen from a fixed
table. A marker with a one-line legend stands in for a sentence. The
readout closes with a decision line per treatment and the figures' paths, and is saved
as readout.txt beside them.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from abkit.readout_analysis import (
    ExperimentReadout,
    MonitoringResult,
    planted_end_of_run_lift,
    planted_spend_difference,
)
from abkit.result_types import CupedResult, MeanComparison, RateComparison

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
    "stop_allowed": (
        "Decision for {arm} vs {control}: a stop is allowed from look {look}; {direction} on "
        "{metric} by {effect} (adjusted {low} to {high})"
    ),
    "no_stop_yet": (
        "Decision for {arm} vs {control}: no stop allowed yet at look {look}, {share} of the "
        "planned size; keep running"
    ),
    "unmonitored": (
        "Decision for {arm} vs {control}: none yet; the run is at {share} of its planned size "
        "and the monitoring covers the first treatment only"
    ),
}
_SPLIT_RESULTS = {
    "ok": "consistent with the plan",
    "fail": "FAILS; users went missing or were assigned unevenly",
}


def format_percent(fraction: float, digits: int = 2) -> str:
    """Format a fraction as a percentage."""
    return f"{100 * fraction:.{digits}f}%"


def format_points(difference: float, digits: int = 2) -> str:
    """Format a difference in proportions as percentage points."""
    return f"{100 * difference:.{digits}f} pp"


def format_points_signed(difference: float, digits: int = 2) -> str:
    """Signed percentage-point difference, so effect columns read one way."""
    return f"{100 * difference:+.{digits}f} pp"


def format_p_value(p_value: float) -> str:
    """Format a p-value with a precision that fits its size."""
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}" if p_value < 0.001 else f"{p_value:.3f}"


def _effect(comparison: RateComparison | MeanComparison) -> str:
    """A comparison's effect in its own unit: percentage points for rates, plain for means."""
    if isinstance(comparison, RateComparison):
        return format_points_signed(comparison.absolute_lift)
    return f"{comparison.difference:+.3f}"


def _interval(comparison: RateComparison | MeanComparison) -> str:
    """A comparison's interval in its own unit."""
    if isinstance(comparison, RateComparison):
        return f"{100 * comparison.ci_low:+.2f} to {format_points_signed(comparison.ci_high)}"
    return f"{comparison.ci_low:+.3f} to {comparison.ci_high:+.3f}"


def _primary_outcome(
    comparison: RateComparison | MeanComparison, alpha: float, p_value: float
) -> str:
    """Which phrase table entry a primary comparison lands on, read at the given p-value (the
    Holm-adjusted one when the primary has two or more contrasts)."""
    if p_value >= alpha:
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
        effect = format_points_signed(result.difference)
        interval = f"{100 * result.ci_low:+.2f} to {format_points_signed(result.ci_high)}"
    else:
        effect = f"{result.difference:+.3f}"
        interval = f"{result.ci_low:+.3f} to {result.ci_high:+.3f}"
    return (
        f"{treatment_arm} vs {control_arm}   {effect}   {interval}   "
        f"p {format_p_value(result.p_value)}   noise removed {result.variance_reduction:.0%}"
    )


def _data_block(readout: ExperimentReadout) -> list[str]:
    """Block 1: the experiment's name, each group's size and share, and what the intake did."""
    total_users = sum(size for _, size in readout.arm_sizes)
    lines = [
        readout.experiment_name,
        "Data: what the intake did before any statistic",
    ]
    lines += _wrapped(
        "users   "
        + "   ".join(
            f"{arm} {size:,} ({size / total_users:.1%})" for arm, size in readout.arm_sizes
        )
    )
    if readout.validation:
        for event in readout.validation:
            label = event.rule.replace("_", " ")
            lines += _wrapped(f"{label}   {event.count:,} rows   {event.detail}")
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
    """Block 3: the split check against the intended split, when one is on record."""
    control_arm = readout.arm_sizes[0][0]
    sizes = dict(readout.arm_sizes)
    lines = ["", "Group split: does assignment match the plan?"]
    if readout.group_split is not None:
        worst = 1.0
        for treatment_arm, srm in readout.group_split:
            worst = min(worst, srm.p_value)
            lines.append(
                _INDENT + f"{treatment_arm} vs {control_arm}   {sizes[treatment_arm]:,} vs "
                f"{sizes[control_arm]:,}   chi-square {srm.statistic:.2f}   "
                f"p {format_p_value(srm.p_value)}"
            )
        lines.append(_INDENT + "result   " + _SPLIT_RESULTS["ok" if worst >= 1e-3 else "fail"])
    else:
        lines.append(_INDENT + "no intended split on record (arms.intended_split), so no check")
        lines.append(_INDENT + "result   the Data block's shares are for the reader's own judgment")
    return lines


def _sensitivity_block(readout: ExperimentReadout) -> list[str]:
    """Block 4: the smallest effect each contrast could detect."""
    binary = readout.primary_metric in _binary_metrics(readout)
    control_arm = readout.arm_sizes[0][0]
    lines = ["", "Sensitivity: smallest lift this sample could detect, 80% power"]
    if len(readout.sensitivity) >= 2:
        lines.append(
            _INDENT + f"each at alpha {readout.alpha:g}/{len(readout.sensitivity)}, the strictest "
            "level the Holm correction applies"
        )
    for treatment_arm, values in readout.sensitivity:
        control = (
            format_percent(values["control_mean"], 1) if binary else f"{values['control_mean']:.3f}"
        )
        mde = values["mde"]
        if not np.isfinite(mde):
            detectable = "none reachable"
        else:
            detectable = format_points(mde) if binary else f"{mde:.3f}"
            if np.isfinite(values["relative_mde"]):
                detectable += f" ({format_percent(values['relative_mde'], 1)} of control)"
        lines += _wrapped(
            f"{treatment_arm} vs {control_arm}   {values['n_treatment']:,.0f} vs "
            f"{values['n_control']:,.0f} users   control {control}   detectable {detectable}"
        )
    return lines


def _decision_p(readout: ExperimentReadout, index: int) -> float:
    """The p-value a contrast's result is read at: Holm-adjusted when the primary has two or more
    contrasts, its own otherwise."""
    if readout.primary_holm is not None:
        return readout.primary_holm[index]
    return readout.primary[index][1].p_value


def _primary_block(readout: ExperimentReadout) -> list[str]:
    """Block 5: the primary metric per contrast, interval first, then CUPED where it ran."""
    control_arm = readout.arm_sizes[0][0]
    lines = [
        "",
        f"Primary metric, {readout.primary_metric}: effect, 95% interval, p; interval first",
    ]
    for index, (treatment_arm, comparison) in enumerate(readout.primary):
        outcome = _primary_outcome(comparison, readout.alpha, _decision_p(readout, index))
        holm_text = (
            f"   Holm {format_p_value(readout.primary_holm[index])}"
            if readout.primary_holm is not None
            else ""
        )
        lines += _wrapped(
            f"{treatment_arm} vs {control_arm}   {_effect(comparison)}   "
            f"{_interval(comparison)}   p {format_p_value(comparison.p_value)}{holm_text}   "
            + _PRIMARY_RESULTS[outcome]
        )
    if readout.primary_holm is not None:
        lines.append(
            _INDENT + f"Holm-corrected across the {len(readout.primary)} treatments; detection "
            "judged on the Holm p; intervals unadjusted"
        )
    monitoring = readout.monitoring
    if monitoring is not None and not monitoring.finished:
        lines.append(
            _INDENT
            + f"read at {_plan_share(monitoring)} of the planned size; the p above is a peek, "
            "judged on the monitoring block"
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
            f"{_effect(corrected.comparison)}   p {format_p_value(corrected.comparison.p_value)}   "
            f"Holm {format_p_value(corrected.holm_p)}   BH {format_p_value(corrected.bh_q)}"
        )
    n_naive = sum(corrected.comparison.p_value < readout.alpha for corrected in family)
    n_holm = sum(corrected.holm_p < readout.alpha for corrected in family)
    lines.append(
        _INDENT + f"result   unadjusted {n_naive} of {len(family)} below {readout.alpha:g}; "
        f"after Holm {n_holm} survive"
    )
    return lines


def _plan_share(monitoring: MonitoringResult) -> str:
    """The last judged look's share of the planned information, as a percentage."""
    return f"{monitoring.sequential_design.information_fractions[-1]:.0%}"


def _monitoring_block(readout: ExperimentReadout) -> list[str]:
    """Block 7: the monitoring of the first contrast, a replay of a finished run or a read of
    one in flight, with the last judged look's adjusted interval."""
    monitoring = readout.monitoring
    if monitoring is None:
        return []
    control_arm = readout.arm_sizes[0][0]
    binary = monitoring.metric in _binary_metrics(readout)
    n_looks = monitoring.n_judged
    stopped = monitoring.outcome.stopped_at_look
    planned = monitoring.planned_users_per_group
    budget = f"{readout.alpha:.0%} budget spread across"
    if not monitoring.finished:
        heading = (
            f"Monitoring: look {n_looks}, {_plan_share(monitoring)} of the plan; "
            f"{budget} the looks; a crossing = a stop allowed"
        )
        design_line = (
            f"crosses at look {stopped}; a stop is allowed from there and keeps the "
            f"{readout.alpha:.0%} false-positive rate"
            if stopped is not None
            else f"no crossing through look {n_looks}; no stop allowed yet"
        )
    else:
        heading = (
            f"Monitoring replay: {n_looks} looks; {budget} them; "
            "a crossing = a stop allowed at that look"
        )
        design_line = (
            f"crosses at look {stopped} of {n_looks}; a stop there keeps the "
            f"{readout.alpha:.0%} false-positive rate"
            if stopped is not None
            else f"never crosses in {n_looks} looks; no early stop in either direction"
        )
    if planned is None:
        plan_line = "plan   none in the config; the last look stands in for it"
    elif monitoring.finished:
        plan_line = f"plan   {planned:,} users per group, reached at look {n_looks}"
    else:
        plan_line = f"plan   {planned:,} users per group, {_plan_share(monitoring)} reached"
    if monitoring.naive_look is None:
        naive_line = "never fires"
    elif stopped is None or monitoring.naive_look < stopped:
        naive_line = (
            f"fires at look {monitoring.naive_look}, on evidence the design does not accept"
        )
    else:
        naive_line = f"fires at look {monitoring.naive_look}"
    last = n_looks - 1
    level = f"{1.0 - monitoring.intervals.alpha:.0%}"
    estimate = monitoring.intervals.estimates[last]
    low, high = monitoring.intervals.ci_low[last], monitoring.intervals.ci_high[last]
    if binary:
        final = (
            f"effect {format_points_signed(estimate)}   "
            f"adjusted {level} {100 * low:+.2f} to {format_points_signed(high)}"
        )
    else:
        final = f"effect {estimate:+.3f}   adjusted {level} {low:+.3f} to {high:+.3f}"
    last_label = "final look" if monitoring.finished else "latest look"
    return [
        "",
        heading,
        *_wrapped(
            f"{monitoring.treatment_arm} vs {control_arm} on {monitoring.metric}, "
            "Lan-DeMets O'Brien-Fleming-style spending "
            + ("at the sizes reached" if planned is None else "against the plan")
        ),
        _INDENT + plan_line,
        *_first_treatment_note(readout),
        _INDENT + f"spending design   {design_line}",
        _INDENT + f"naive {monitoring.z_naive:.2f} rule   {naive_line}",
        _INDENT
        + f"{last_label}   {final}   boundary |z| {monitoring.intervals.boundaries[last]:.2f}",
    ]


def _first_treatment_note(readout: ExperimentReadout) -> list[str]:
    """One line saying why a single-contrast block uses the first treatment, when there are
    several to choose from."""
    if len(readout.primary) < 2:
        return []
    return [_INDENT + "chosen   first treatment in the arm order, not by its result"]


def _truth_block(readout: ExperimentReadout) -> list[str]:
    """Block 8, synthetic runs only: what the generator planted beside what the readout found."""
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
                    f"expected {format_points_signed(expected)} "
                    f"at the end of the run{novelty_note}   "
                    f"found {_effect(comparison)} ({_interval(comparison)})   "
                    + ("inside" if inside else "OUTSIDE")
                )
    expected_spend = planted_spend_difference(readout.truth)
    spend_results = [
        (metric, result) for metric, _, result in readout.cuped if metric != readout.primary_metric
    ]
    if expected_spend is not None and spend_results:
        # the CUPED estimate is the same difference in means as the unadjusted one, with less
        # noise, so it is the comparison the planted value is checked against
        metric, result = spend_results[0]
        inside = result.ci_low <= expected_spend <= result.ci_high
        detected = result.p_value < readout.alpha
        found["spend_rate_lift"] = (
            f"expected {expected_spend:+.3f} per user on {metric}   found after CUPED "
            f"{result.difference:+.3f} ({result.ci_low:+.3f} to {result.ci_high:+.3f})   "
            + ("inside" if inside else "OUTSIDE")
            + (
                ", detected"
                if detected
                else f", too small to detect at this sample (p {format_p_value(result.p_value)})"
            )
        )
    width = max(len(name) for name in readout.truth)
    for name, value in sorted(readout.truth.items()):
        line = f"{name:<{width}}   planted {_planted(value)}"
        if name == "absolute_lift" and isinstance(value, float):
            # The other found values are in pp, so the planted lift gets its pp form beside it.
            line += f" ({format_points_signed(value)})"
        if name in found:
            line += f"   {found[name]}"
        lines += _wrapped(line)
    return lines


def _planted(value: object) -> str:
    """A planted value as the readout prints numbers: thousands separators on whole numbers."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _decision_lines(readout: ExperimentReadout) -> list[str]:
    """The closing lines: a broken split stops everything; otherwise one decision per contrast,
    read at the Holm-adjusted p when there are several treatments. A run with planted values
    prints these as synthetic check output, since no one decides anything on made-up data."""
    if any(limitation.code == "group_split" for limitation in readout.limitations):
        lines = [_DECISIONS["split_broken"]]
    elif readout.monitoring is not None and not readout.monitoring.finished:
        lines = [
            _in_flight_decision_line(readout, readout.monitoring, index)
            for index in range(len(readout.primary))
        ]
    else:
        lines = [_decision_line(readout, index) for index in range(len(readout.primary))]
    if readout.truth is None:
        return lines
    return [line.replace("Decision", "Synthetic check output", 1) for line in lines]


def _in_flight_decision_line(
    readout: ExperimentReadout, monitoring: MonitoringResult, index: int
) -> str:
    """One contrast's decision while the run is short of its plan: for the monitored contrast,
    whether the boundary allows a stop yet, with the stop look's adjusted interval; none for the
    others."""
    treatment_arm, _ = readout.primary[index]
    control_arm = readout.arm_sizes[0][0]
    share = _plan_share(monitoring)
    if treatment_arm != monitoring.treatment_arm:
        return _DECISIONS["unmonitored"].format(arm=treatment_arm, control=control_arm, share=share)
    stopped = monitoring.outcome.stopped_at_look
    if stopped is None:
        return _DECISIONS["no_stop_yet"].format(
            arm=treatment_arm, control=control_arm, look=monitoring.n_judged, share=share
        )
    look = stopped - 1
    estimate = monitoring.intervals.estimates[look]
    low, high = monitoring.intervals.ci_low[look], monitoring.intervals.ci_high[look]
    if monitoring.metric in _binary_metrics(readout):
        effect, low_text, high_text = (
            format_points_signed(estimate),
            f"{100 * low:+.2f}",
            format_points_signed(high),
        )
    else:
        effect, low_text, high_text = f"{estimate:+.3f}", f"{low:+.3f}", f"{high:+.3f}"
    return _DECISIONS["stop_allowed"].format(
        arm=treatment_arm,
        control=control_arm,
        look=stopped,
        direction="higher" if estimate > 0 else "lower",
        metric=monitoring.metric,
        effect=effect,
        low=low_text,
        high=high_text,
    )


def _decision_line(readout: ExperimentReadout, index: int) -> str:
    """One contrast's decision, named, with its interval or its smallest detectable effect."""
    treatment_arm, comparison = readout.primary[index]
    control_arm = readout.arm_sizes[0][0]
    outcome = _primary_outcome(comparison, readout.alpha, _decision_p(readout, index))
    if outcome == "not_detected":
        mde = readout.sensitivity[index][1]["mde"]
        if not np.isfinite(mde):
            mde_text = "no lift at all with this sample"
        elif isinstance(comparison, RateComparison):
            mde_text = format_points(mde)
        else:
            mde_text = f"{mde:.3f}"
        return _DECISIONS[outcome].format(arm=treatment_arm, control=control_arm, mde=mde_text)
    if isinstance(comparison, RateComparison):
        low, high = f"{100 * comparison.ci_low:+.2f}", format_points_signed(comparison.ci_high)
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
        + _monitoring_block(readout)
        + _truth_block(readout)
        + [""]
        + [
            wrapped_line
            for line in _decision_lines(readout)
            for wrapped_line in _wrapped(line, indent="")
        ]
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

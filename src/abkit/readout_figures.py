"""The readout's figures, drawn from an ExperimentReadout and listed by path in the readout.

``draw_figures`` draws only the figures the readout has data for: the
corrections and the monitoring with its adjusted intervals. matplotlib is
imported inside the drawing functions, so the library's runtime
dependencies stay numpy, scipy, and statsmodels for everything except
drawing, where it is a dev dependency of the scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from abkit.readout_analysis import ExperimentReadout, MonitoringResult
from abkit.result_types import RateComparison

if TYPE_CHECKING:
    from matplotlib.axes import Axes

COLOR_CONTROL = "#4269d0"
COLOR_TREATMENT = "#d4615c"
COLOR_REFERENCE = "#8a8f98"
COLOR_INK = "#31343a"


def _style_axis(ax: Axes, *, grid_axis: Literal["both", "x", "y"] = "y") -> None:
    """Apply the plain axis style: no top or right spine, a light grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def _monitoring_figure(
    monitoring: MonitoringResult, binary: bool, synthetic: bool, path: Path
) -> None:
    """Left: the observed |z| path against the spending boundaries and the naive threshold.
    Right: the effect at each look with its adjusted interval band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_path = np.abs(np.asarray(monitoring.outcome.z_values))
    boundaries = np.asarray(monitoring.sequential_design.boundaries)
    looks = np.arange(1, z_path.size + 1)
    scale = 100.0 if binary else 1.0
    unit = "pp" if binary else "units"
    estimates = scale * np.asarray(monitoring.intervals.estimates)
    low = scale * np.asarray(monitoring.intervals.ci_low)
    high = scale * np.asarray(monitoring.intervals.ci_high)

    fig, (z_ax, effect_ax) = plt.subplots(1, 2, figsize=(11, 4.4))
    z_ax.plot(
        looks,
        z_path,
        color=COLOR_CONTROL,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Observed |z|",
    )
    finite = np.isfinite(boundaries)
    z_ax.plot(
        looks[finite],
        boundaries[finite],
        color=COLOR_INK,
        linewidth=2,
        drawstyle="steps-mid",
        label="Spending boundary",
    )
    z_ax.axhline(
        monitoring.z_naive,
        color=COLOR_REFERENCE,
        linewidth=1.5,
        linestyle="--",
        label=f"Naive threshold ({monitoring.z_naive:.2f})",
    )
    top_values = [float(z_path.max()), monitoring.z_naive]
    if finite.any():
        top_values.append(float(boundaries[finite].max()))
    z_ax.set_ylim(top=1.2 * max(top_values))
    z_ax.set_xlabel("Look")
    z_ax.set_ylabel("|z|")
    z_title = (
        "Boundaries at the sizes reached"
        if monitoring.planned_users_per_group is None
        else "Boundaries against the planned size"
    )
    z_ax.set_title(z_title, loc="left", color=COLOR_INK, fontsize=10)
    z_ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(z_ax)

    bounded = np.isfinite(low) & np.isfinite(high)
    effect_ax.fill_between(
        looks[bounded],
        low[bounded],
        high[bounded],
        color=COLOR_CONTROL,
        alpha=0.18,
        linewidth=0,
        label=f"Adjusted {1 - monitoring.intervals.alpha:.0%} interval",
    )
    effect_ax.plot(looks, estimates, color=COLOR_CONTROL, linewidth=2, label="Effect at each look")
    effect_ax.axhline(0.0, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    effect_ax.set_xlabel("Look")
    effect_ax.set_ylabel(f"{monitoring.treatment_arm} minus control ({unit})")
    effect_ax.set_title(
        "Adjusted intervals hold at every look at once", loc="left", color=COLOR_INK, fontsize=10
    )
    effect_ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(effect_ax)
    if synthetic:
        fig.suptitle("Synthetic check: generated data, not a real experiment", color=COLOR_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _corrections_figure(readout: ExperimentReadout, path: Path) -> None:
    """Each comparison's unadjusted and Holm-adjusted p-value on a log scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{corrected.metric}, {corrected.treatment_arm}" for corrected in readout.corrections]
    naive = np.array([corrected.comparison.p_value for corrected in readout.corrections])
    adjusted = np.array([corrected.holm_p for corrected in readout.corrections])
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
    ax.axvline(readout.alpha, color=COLOR_REFERENCE, linewidth=1.5, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xscale("log")
    ax.set_xlabel("p-value (log scale)")
    ax.set_title("Adjustment across the comparison family", loc="left", color=COLOR_INK)
    ax.legend(frameon=False, fontsize=9)
    ax.invert_yaxis()
    _style_axis(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_figures(readout: ExperimentReadout, figure_dir: Path) -> list[Path]:
    """Draw every figure the readout has data for into figure_dir and return their paths, in the
    readout's block order."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if readout.corrections:
        path = figure_dir / "corrections.png"
        _corrections_figure(readout, path)
        paths.append(path)
    if readout.monitoring is not None:
        binary = bool(readout.primary) and isinstance(readout.primary[0][1], RateComparison)
        path = figure_dir / "monitoring.png"
        _monitoring_figure(readout.monitoring, binary, readout.truth is not None, path)
        paths.append(path)
    return paths

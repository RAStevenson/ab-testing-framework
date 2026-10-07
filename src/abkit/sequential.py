"""Group-sequential monitoring with alpha spending.

Peeking at an experiment daily with a fixed critical value inflates the
false-positive rate severalfold. The fix is to pre-commit to a spending
function that allocates the total alpha across looks, and to compute
critical values that account for the correlation between looks.

Boundaries are computed with the standard recursive numerical-integration
construction (Armitage-McPherson-Rowe): carry the sub-density of the
score process over paths that have not yet crossed, and at each look
solve for the critical value that spends exactly the increment the
spending function allows.

Accuracy is limited by the integration grid, which converges as O(h^2).
With the default ``grid_points`` the boundaries are stable to about 1e-4
on the 14-look schedule that ``test_boundaries_converge_on_grid`` pins
by refining the grid, and the two- and three-look designs are checked
against exact multivariate-normal rectangle probabilities from scipy,
which share no code with this recursion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from abkit.results import SequentialDesign, SequentialOutcome

# below this spending increment the boundary exceeds 6.4 sigma; treat the
# look as unspendable rather than chase integration noise
_MIN_INCREMENT = 1e-10


def obrien_fleming_spending(information_fraction: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Lan-DeMets O'Brien-Fleming-like spending: stingy early, most alpha late."""
    fractions = np.asarray(information_fraction, dtype=np.float64)
    z_critical = stats.norm.isf(alpha / 2.0)
    return np.asarray(2.0 * stats.norm.sf(z_critical / np.sqrt(fractions)))


def pocock_spending(information_fraction: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Lan-DeMets Pocock-like spending: spends alpha nearly linearly."""
    fractions = np.asarray(information_fraction, dtype=np.float64)
    return np.asarray(alpha * np.log1p((np.e - 1.0) * fractions))


_SPENDING = {
    "obrien-fleming": obrien_fleming_spending,
    "pocock": pocock_spending,
}


def design(
    information_fractions: Sequence[float],
    *,
    alpha: float = 0.05,
    spending: str = "obrien-fleming",
    grid_points: int = 1024,
) -> SequentialDesign:
    """Two-sided group-sequential design for the given look schedule.

    ``information_fractions`` are the planned looks as fractions of the
    final sample size, strictly increasing in (0, 1]. The design must be
    fixed before the data are seen; choosing looks after peeking voids
    the guarantee.
    """
    fractions = np.asarray(information_fractions, dtype=np.float64)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("information_fractions must be a non-empty 1-d sequence")
    # every comparison below is False against NaN, so an unchecked NaN would
    # survive validation and resurface as an opaque failure inside the solver
    if not np.all(np.isfinite(fractions)):
        raise ValueError(f"information_fractions must all be finite, got {fractions.tolist()}")
    if fractions[0] <= 0.0 or fractions[-1] > 1.0 or np.any(np.diff(fractions) <= 0.0):
        raise ValueError(
            f"information_fractions must be strictly increasing in (0, 1], got {fractions.tolist()}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if spending not in _SPENDING:
        raise ValueError(f"unknown spending function {spending!r}, options: {sorted(_SPENDING)}")
    if not 128 <= grid_points <= 8192:
        # the propagation kernel is a dense grid_points x grid_points array,
        # so the upper bound turns a 2 GB allocation into a clear error
        raise ValueError(f"grid_points must be in [128, 8192], got {grid_points}")

    cumulative = _SPENDING[spending](fractions, alpha)
    increments = np.diff(cumulative, prepend=0.0)

    boundaries: list[float] = []
    # grid density of the un-crossed score process S_k = Z_k * sqrt(t_k)
    grid: NDArray[np.float64] | None = None
    density: NDArray[np.float64] | None = None

    for look in range(fractions.size):
        if look == 0:
            boundary = _first_boundary(increments[0])
            increment_standard_deviation = float(np.sqrt(fractions[0]))
            # an infinite boundary (nothing spent yet) still needs a finite
            # grid; 8 sigma keeps truncation error below integration error
            bound = (
                boundary * increment_standard_deviation
                if np.isfinite(boundary)
                else 8.0 * increment_standard_deviation
            )
            grid = np.linspace(-bound, bound, grid_points)
            density = stats.norm.pdf(grid, scale=increment_standard_deviation)
        else:
            increment_standard_deviation = float(np.sqrt(fractions[look] - fractions[look - 1]))
            assert grid is not None and density is not None
            boundary = _next_boundary(
                grid,
                density,
                increment_standard_deviation,
                float(np.sqrt(fractions[look])),
                increments[look],
            )
            grid, density = _propagate(
                grid,
                density,
                increment_standard_deviation,
                boundary * float(np.sqrt(fractions[look])),
                grid_points,
            )
        boundaries.append(boundary)

    return SequentialDesign(
        information_fractions=tuple(float(value) for value in fractions),
        boundaries=tuple(boundaries),
        cumulative_alpha=tuple(float(value) for value in cumulative),
        alpha=alpha,
        spending=spending,
    )


def _first_boundary(increment: float) -> float:
    """Return the first look's boundary for the alpha spent there, infinite when almost nothing is
    spent."""
    if increment < _MIN_INCREMENT:
        return float("inf")
    return float(stats.norm.isf(increment / 2.0))


def _trapezoid_weights(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return trapezoid-rule integration weights for an evenly spaced grid."""
    spacing = grid[1] - grid[0]
    weights = np.full(grid.size, spacing)
    weights[0] = weights[-1] = spacing / 2.0
    return weights


def _next_boundary(
    grid: NDArray[np.float64],
    density: NDArray[np.float64],
    increment_standard_deviation: float,
    scale: float,
    increment: float,
) -> float:
    """Solve for c such that P(cross at this look, not before) = increment."""
    if increment < _MIN_INCREMENT:
        return float("inf")
    weights = _trapezoid_weights(grid) * density

    def crossing_probability(boundary: float) -> float:
        """Probability that the score crosses this boundary at the look, given the carried density;
        brentq matches it to the alpha increment."""
        bound = boundary * scale
        tails = stats.norm.cdf((-bound - grid) / increment_standard_deviation) + stats.norm.sf(
            (bound - grid) / increment_standard_deviation
        )
        return float(np.sum(weights * tails))

    # crossing_probability is monotone decreasing in the boundary; at a
    # boundary of 0 it equals the remaining un-crossed mass (> increment for
    # any valid spending), and at 12 it is numerically zero, so the bracket
    # always holds
    # brentq returns a float unless full_output=True; scipy leaves it untyped,
    # so the checker infers the tuple form too
    root = cast(
        float,
        optimize.brentq(
            lambda boundary: crossing_probability(boundary) - increment, 0.0, 12.0, xtol=1e-10
        ),
    )
    return float(root)


def _propagate(
    grid: NDArray[np.float64],
    density: NDArray[np.float64],
    increment_standard_deviation: float,
    new_bound: float,
    grid_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convolve the un-crossed density with the look's increment and truncate."""
    if not np.isfinite(new_bound):
        # nothing was spent at this look; keep support wide enough that the
        # truncation error stays below the integration error
        new_bound = float(np.abs(grid).max() + 8.0 * increment_standard_deviation)
    new_grid = np.linspace(-new_bound, new_bound, grid_points)
    weights = _trapezoid_weights(grid) * density
    kernel = (
        stats.norm.pdf((new_grid[:, None] - grid[None, :]) / increment_standard_deviation)
        / increment_standard_deviation
    )
    return new_grid, kernel @ weights


def monitor(z_values: Sequence[float], sequential_design: SequentialDesign) -> SequentialOutcome:
    """Compare observed z-statistics to the design's boundaries, in order.

    ``z_values`` may be shorter than the design (experiment still in
    flight) but not longer. Scanning stops at the first crossing: once a
    boundary is hit the experiment should stop, and later looks have no
    interpretation under the design.
    """
    z_scores = tuple(float(z_score) for z_score in z_values)
    if len(z_scores) > len(sequential_design.boundaries):
        raise ValueError(
            f"got {len(z_scores)} looks but the design has only {len(sequential_design.boundaries)}"
        )
    # a NaN z compares False against every boundary, so without this it would
    # read as "this look did not cross" rather than "this look is unusable"
    if not all(np.isfinite(z_score) for z_score in z_scores):
        raise ValueError(f"z_values must all be finite, got {z_scores}")
    boundaries = sequential_design.boundaries[: len(z_scores)]
    stopped = None
    for i, (look_z, bound) in enumerate(zip(z_scores, boundaries, strict=True)):
        if abs(look_z) >= bound:
            stopped = i + 1
            break
    return SequentialOutcome(stopped_at_look=stopped, z_values=z_scores, boundaries=boundaries)


def z_from_cumulative_counts(
    conversions_control: ArrayLike,
    n_control: ArrayLike,
    conversions_treatment: ArrayLike,
    n_treatment: ArrayLike,
) -> NDArray[np.float64]:
    """Pooled two-proportion z-statistic at each look from cumulative counts.

    Inputs are per-look cumulative totals (look k includes all users
    through look k), which is how monitoring data arrives.
    """
    control_conversions = np.asarray(conversions_control, dtype=np.float64)
    control_counts = np.asarray(n_control, dtype=np.float64)
    treatment_conversions = np.asarray(conversions_treatment, dtype=np.float64)
    treatment_counts = np.asarray(n_treatment, dtype=np.float64)
    if (
        not (
            control_conversions.shape
            == control_counts.shape
            == treatment_conversions.shape
            == treatment_counts.shape
        )
        or control_conversions.ndim != 1
    ):
        raise ValueError("all four inputs must be 1-d arrays of the same length")
    if np.any(control_counts <= 0.0) or np.any(treatment_counts <= 0.0):
        raise ValueError("arm sizes must be positive at every look")
    if np.any(control_conversions < 0.0) or np.any(treatment_conversions < 0.0):
        raise ValueError("conversion counts must be non-negative")
    if np.any(control_conversions > control_counts) or np.any(
        treatment_conversions > treatment_counts
    ):
        raise ValueError("conversions cannot exceed the arm size at any look")
    # conversions matter here as much as arm sizes: a cumulative series that
    # goes backwards is a join or transcription error, and left unchecked it
    # produces a large spurious z that stops the experiment
    if (
        np.any(np.diff(control_counts) < 0.0)
        or np.any(np.diff(treatment_counts) < 0.0)
        or np.any(np.diff(control_conversions) < 0.0)
        or np.any(np.diff(treatment_conversions) < 0.0)
    ):
        raise ValueError("counts must be cumulative (non-decreasing)")

    pooled = (control_conversions + treatment_conversions) / (control_counts + treatment_counts)
    standard_error = np.sqrt(
        pooled * (1.0 - pooled) * (1.0 / control_counts + 1.0 / treatment_counts)
    )
    if not np.all(np.isfinite(standard_error)) or np.any(standard_error == 0.0):
        raise ValueError("a look has all-identical outcomes; z is undefined there")
    return np.asarray(
        (treatment_conversions / treatment_counts - control_conversions / control_counts)
        / standard_error
    )


def z_from_cumulative_stats(
    n_control: ArrayLike,
    mean_control: ArrayLike,
    variance_control: ArrayLike,
    n_treatment: ArrayLike,
    mean_treatment: ArrayLike,
    variance_treatment: ArrayLike,
) -> NDArray[np.float64]:
    """Two-sample z-statistic at each look from cumulative summary stats.

    The continuous-metric analogue of ``z_from_cumulative_counts`` for
    aggregate-only monitoring data (the ASOS shape): per-look cumulative
    count, mean and ddof=1 variance for each arm. The standard error is
    the UNPOOLED two-sample form. On binary data it therefore differs
    from the pooled statistic ``z_from_cumulative_counts`` computes: the
    gap grows with the observed rate difference and shrinks with n, and
    on simulated 6%-rate monitoring data it measures up to ~8% at early
    few-hundred-user looks, tightening to ~1% by the final look (the
    test pins both bounds). The two are interchangeable only where the
    test says they are; for a from-the-start monitoring design on counts,
    prefer ``z_from_cumulative_counts``.
    """
    control_counts = np.asarray(n_control, dtype=np.float64)
    control_means = np.asarray(mean_control, dtype=np.float64)
    control_variances = np.asarray(variance_control, dtype=np.float64)
    treatment_counts = np.asarray(n_treatment, dtype=np.float64)
    treatment_means = np.asarray(mean_treatment, dtype=np.float64)
    treatment_variances = np.asarray(variance_treatment, dtype=np.float64)
    shapes = {
        values.shape
        for values in (
            control_counts,
            control_means,
            control_variances,
            treatment_counts,
            treatment_means,
            treatment_variances,
        )
    }
    if len(shapes) != 1 or control_counts.ndim != 1:
        raise ValueError("all six inputs must be 1-d arrays of the same length")
    if np.any(control_counts < 2.0) or np.any(treatment_counts < 2.0):
        raise ValueError("arm sizes must be at least 2 at every look")
    if np.any(np.diff(control_counts) < 0.0) or np.any(np.diff(treatment_counts) < 0.0):
        raise ValueError("counts must be cumulative (non-decreasing)")
    for name, values in (
        ("means", np.concatenate([control_means, treatment_means])),
        ("variances", np.concatenate([control_variances, treatment_variances])),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must all be finite")
    if np.any(control_variances < 0.0) or np.any(treatment_variances < 0.0):
        raise ValueError("variances must be non-negative")

    standard_error = np.sqrt(
        control_variances / control_counts + treatment_variances / treatment_counts
    )
    if np.any(standard_error == 0.0):
        raise ValueError("a look has zero variance in both arms; z is undefined there")
    return np.asarray((treatment_means - control_means) / standard_error)

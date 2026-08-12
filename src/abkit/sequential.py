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
for schedules up to a few dozen looks; ``test_boundaries_converge_on_grid``
pins that directly by refining the grid, and the two- and three-look
designs are checked against exact multivariate-normal rectangle
probabilities from scipy, which share no code with this recursion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from abkit.results import SequentialDesign, SequentialOutcome

# below this spending increment the boundary exceeds 6.4 sigma; treat the
# look as unspendable rather than chase integration noise
_MIN_INCREMENT = 1e-10


def obrien_fleming_spending(t: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Lan-DeMets O'Brien-Fleming-like spending: stingy early, most alpha late."""
    t_arr = np.asarray(t, dtype=np.float64)
    z = stats.norm.isf(alpha / 2.0)
    return np.asarray(2.0 * stats.norm.sf(z / np.sqrt(t_arr)))


def pocock_spending(t: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Lan-DeMets Pocock-like spending: spends alpha nearly linearly."""
    t_arr = np.asarray(t, dtype=np.float64)
    return np.asarray(alpha * np.log1p((np.e - 1.0) * t_arr))


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
    t = np.asarray(information_fractions, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("information_fractions must be a non-empty 1-d sequence")
    # every comparison below is False against NaN, so an unchecked NaN would
    # survive validation and resurface as an opaque failure inside the solver
    if not np.all(np.isfinite(t)):
        raise ValueError(f"information_fractions must all be finite, got {t.tolist()}")
    if t[0] <= 0.0 or t[-1] > 1.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError(
            f"information_fractions must be strictly increasing in (0, 1], got {t.tolist()}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if spending not in _SPENDING:
        raise ValueError(f"unknown spending function {spending!r}, options: {sorted(_SPENDING)}")
    if not 128 <= grid_points <= 8192:
        # the propagation kernel is a dense grid_points x grid_points array,
        # so the upper bound turns a 2 GB allocation into a clear error
        raise ValueError(f"grid_points must be in [128, 8192], got {grid_points}")

    cumulative = _SPENDING[spending](t, alpha)
    increments = np.diff(cumulative, prepend=0.0)

    boundaries: list[float] = []
    # grid density of the un-crossed score process S_k = Z_k * sqrt(t_k)
    grid: NDArray[np.float64] | None = None
    density: NDArray[np.float64] | None = None

    for k in range(t.size):
        if k == 0:
            c = _first_boundary(increments[0])
            sd = float(np.sqrt(t[0]))
            # an infinite boundary (nothing spent yet) still needs a finite
            # grid; 8 sigma keeps truncation error below integration error
            bound = c * sd if np.isfinite(c) else 8.0 * sd
            grid = np.linspace(-bound, bound, grid_points)
            density = stats.norm.pdf(grid, scale=sd)
        else:
            sd = float(np.sqrt(t[k] - t[k - 1]))
            assert grid is not None and density is not None
            c = _next_boundary(grid, density, sd, float(np.sqrt(t[k])), increments[k])
            grid, density = _propagate(grid, density, sd, c * float(np.sqrt(t[k])), grid_points)
        boundaries.append(c)

    return SequentialDesign(
        information_fractions=tuple(float(v) for v in t),
        boundaries=tuple(boundaries),
        cumulative_alpha=tuple(float(v) for v in cumulative),
        alpha=alpha,
        spending=spending,
    )


def _first_boundary(increment: float) -> float:
    if increment < _MIN_INCREMENT:
        return float("inf")
    return float(stats.norm.isf(increment / 2.0))


def _trapezoid_weights(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    dx = grid[1] - grid[0]
    w = np.full(grid.size, dx)
    w[0] = w[-1] = dx / 2.0
    return w


def _next_boundary(
    grid: NDArray[np.float64],
    density: NDArray[np.float64],
    sd: float,
    scale: float,
    increment: float,
) -> float:
    """Solve for c such that P(cross at this look, not before) = increment."""
    if increment < _MIN_INCREMENT:
        return float("inf")
    weights = _trapezoid_weights(grid) * density

    def crossing_probability(c: float) -> float:
        bound = c * scale
        tails = stats.norm.cdf((-bound - grid) / sd) + stats.norm.sf((bound - grid) / sd)
        return float(np.sum(weights * tails))

    # crossing_probability is monotone decreasing in c; at c=0 it equals the
    # remaining un-crossed mass (> increment for any valid spending), and at
    # c=12 it is numerically zero, so the bracket always holds
    return float(
        optimize.brentq(lambda c: crossing_probability(c) - increment, 0.0, 12.0, xtol=1e-10)
    )


def _propagate(
    grid: NDArray[np.float64],
    density: NDArray[np.float64],
    sd: float,
    new_bound: float,
    grid_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convolve the un-crossed density with the look's increment and truncate."""
    if not np.isfinite(new_bound):
        # nothing was spent at this look; keep support wide enough that the
        # truncation error stays below the integration error
        new_bound = float(np.abs(grid).max() + 8.0 * sd)
    new_grid = np.linspace(-new_bound, new_bound, grid_points)
    weights = _trapezoid_weights(grid) * density
    kernel = stats.norm.pdf((new_grid[:, None] - grid[None, :]) / sd) / sd
    return new_grid, kernel @ weights


def monitor(z_values: Sequence[float], seq_design: SequentialDesign) -> SequentialOutcome:
    """Compare observed z-statistics to the design's boundaries, in order.

    ``z_values`` may be shorter than the design (experiment still in
    flight) but not longer. Scanning stops at the first crossing: once a
    boundary is hit the experiment should stop, and later looks have no
    interpretation under the design.
    """
    z = tuple(float(v) for v in z_values)
    if len(z) > len(seq_design.boundaries):
        raise ValueError(f"got {len(z)} looks but the design has only {len(seq_design.boundaries)}")
    # a NaN z compares False against every boundary, so without this it would
    # read as "this look did not cross" rather than "this look is unusable"
    if not all(np.isfinite(v) for v in z):
        raise ValueError(f"z_values must all be finite, got {z}")
    boundaries = seq_design.boundaries[: len(z)]
    stopped = None
    for i, (z_k, bound) in enumerate(zip(z, boundaries, strict=True)):
        if abs(z_k) >= bound:
            stopped = i + 1
            break
    return SequentialOutcome(stopped_at_look=stopped, z_values=z, boundaries=boundaries)


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
    xc = np.asarray(conversions_control, dtype=np.float64)
    nc = np.asarray(n_control, dtype=np.float64)
    xt = np.asarray(conversions_treatment, dtype=np.float64)
    nt = np.asarray(n_treatment, dtype=np.float64)
    if not (xc.shape == nc.shape == xt.shape == nt.shape) or xc.ndim != 1:
        raise ValueError("all four inputs must be 1-d arrays of the same length")
    if np.any(nc <= 0.0) or np.any(nt <= 0.0):
        raise ValueError("arm sizes must be positive at every look")
    if np.any(xc < 0.0) or np.any(xt < 0.0):
        raise ValueError("conversion counts must be non-negative")
    if np.any(xc > nc) or np.any(xt > nt):
        raise ValueError("conversions cannot exceed the arm size at any look")
    # conversions matter here as much as arm sizes: a cumulative series that
    # goes backwards is a join or transcription error, and left unchecked it
    # produces a large spurious z that stops the experiment
    if (
        np.any(np.diff(nc) < 0.0)
        or np.any(np.diff(nt) < 0.0)
        or np.any(np.diff(xc) < 0.0)
        or np.any(np.diff(xt) < 0.0)
    ):
        raise ValueError("counts must be cumulative (non-decreasing)")

    pooled = (xc + xt) / (nc + nt)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / nc + 1.0 / nt))
    if not np.all(np.isfinite(se)) or np.any(se == 0.0):
        raise ValueError("a look has all-identical outcomes; z is undefined there")
    return np.asarray((xt / nt - xc / nc) / se)


def z_from_cumulative_stats(
    n_control: ArrayLike,
    mean_control: ArrayLike,
    var_control: ArrayLike,
    n_treatment: ArrayLike,
    mean_treatment: ArrayLike,
    var_treatment: ArrayLike,
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
    nc = np.asarray(n_control, dtype=np.float64)
    mc = np.asarray(mean_control, dtype=np.float64)
    vc = np.asarray(var_control, dtype=np.float64)
    nt = np.asarray(n_treatment, dtype=np.float64)
    mt = np.asarray(mean_treatment, dtype=np.float64)
    vt = np.asarray(var_treatment, dtype=np.float64)
    shapes = {arr.shape for arr in (nc, mc, vc, nt, mt, vt)}
    if len(shapes) != 1 or nc.ndim != 1:
        raise ValueError("all six inputs must be 1-d arrays of the same length")
    if np.any(nc < 2.0) or np.any(nt < 2.0):
        raise ValueError("arm sizes must be at least 2 at every look")
    if np.any(np.diff(nc) < 0.0) or np.any(np.diff(nt) < 0.0):
        raise ValueError("counts must be cumulative (non-decreasing)")
    for name, arr in (("means", np.concatenate([mc, mt])), ("variances", np.concatenate([vc, vt]))):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must all be finite")
    if np.any(vc < 0.0) or np.any(vt < 0.0):
        raise ValueError("variances must be non-negative")

    se = np.sqrt(vc / nc + vt / nt)
    if np.any(se == 0.0):
        raise ValueError("a look has zero variance in both arms; z is undefined there")
    return np.asarray((mt - mc) / se)

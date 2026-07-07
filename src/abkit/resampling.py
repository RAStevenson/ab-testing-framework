"""Bootstrap and permutation inference for two-sample comparisons.

These are the workhorses for heavy-tailed metrics (revenue per user)
where the normal-theory tests in ``frequentist`` rely on a CLT that has
not kicked in at realistic sample sizes.

Both routines vectorize over resamples in fixed-size chunks so peak
memory stays near ``_CHUNK * n * 8`` bytes regardless of the resample
count.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from abkit.results import BootstrapResult, PermutationResult

_CHUNK = 256


def _two_samples(control: ArrayLike, treatment: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(control, dtype=np.float64)
    y = np.asarray(treatment, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        raise ValueError(f"need at least 2 observations per arm, got {x.size} and {y.size}")
    return x, y


def bootstrap_mean_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap interval for the difference in means.

    Arms are resampled independently with replacement. The percentile
    interval is adequate here because the bootstrap distribution of a
    mean difference is close to symmetric at experiment-scale n; a BCa
    interval would tighten small-sample skew cases and is on the TODO
    list.
    """
    x, y = _two_samples(control, treatment)
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be at least 100, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    diffs = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        size = min(_CHUNK, n_resamples - done)
        idx_x = rng.integers(0, x.size, size=(size, x.size))
        idx_y = rng.integers(0, y.size, size=(size, y.size))
        diffs[done : done + size] = y[idx_y].mean(axis=1) - x[idx_x].mean(axis=1)
        done += size

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [tail, 1.0 - tail])
    return BootstrapResult(
        estimate=float(y.mean() - x.mean()),
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def permutation_test_mean_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    rng: np.random.Generator,
    n_permutations: int = 10_000,
) -> PermutationResult:
    """Two-sided permutation test for a difference in means.

    Exact conditional on the pooled outcomes: no distributional
    assumptions, so it stays valid for heavy-tailed and zero-inflated
    metrics. Also correct for binary outcomes, where the mean difference
    is the rate difference. Uses the add-one p-value
    ``(1 + hits) / (n_permutations + 1)``, which is conservative and
    never exactly zero.
    """
    x, y = _two_samples(control, treatment)
    if n_permutations < 100:
        raise ValueError(f"n_permutations must be at least 100, got {n_permutations}")

    pooled = np.concatenate([x, y])
    n_t = y.size
    n_c = x.size
    total = pooled.sum()
    observed = y.mean() - x.mean()

    hits = 0
    done = 0
    while done < n_permutations:
        size = min(_CHUNK, n_permutations - done)
        # argpartition on random keys draws a uniform n_t-subset per row in
        # O(n) rather than O(n log n) for a full shuffle
        keys = rng.random((size, pooled.size), dtype=np.float32)
        take = np.argpartition(keys, n_t - 1, axis=1)[:, :n_t]
        sums_t = pooled[take].sum(axis=1)
        perm_diffs = sums_t / n_t - (total - sums_t) / n_c
        hits += int(np.sum(np.abs(perm_diffs) >= abs(observed)))
        done += size

    return PermutationResult(
        observed=float(observed),
        p_value=(1 + hits) / (n_permutations + 1),
        n_permutations=n_permutations,
    )


def trimmed_mean_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    trim: float = 0.01,
) -> float:
    """Difference in trimmed means (treatment minus control).

    A cheap robustness check for whale-dominated metrics: if the trimmed
    and untrimmed differences disagree wildly, a few extreme users are
    carrying the result. ``trim`` is the fraction cut from each tail of
    each arm.
    """
    x, y = _two_samples(control, treatment)
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    return float(stats.trim_mean(y, trim) - stats.trim_mean(x, trim))

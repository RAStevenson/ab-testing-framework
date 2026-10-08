"""Bootstrap and permutation inference for two-sample comparisons.

These are the workhorses for heavy-tailed metrics (revenue per user)
where the normal-theory tests in ``frequentist`` rely on a CLT that has
not kicked in at realistic sample sizes.

Both routines call scipy and process resamples in fixed-size batches, so
peak memory is set by the batch size and stays flat as the resample count
grows. A batch holds a few arrays of ``_BATCH * n`` elements at once, so
budget a small multiple of ``_BATCH * n * 8`` bytes rather than exactly
that.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from abkit.results import BootstrapResult, PermutationResult

_BATCH = 256


def _two_samples(
    control: ArrayLike, treatment: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return both arms as 1-d float arrays, raising on a bad shape, too few values, or a non-finite
    value."""
    control_values = np.asarray(control, dtype=np.float64)
    treatment_values = np.asarray(treatment, dtype=np.float64)
    if control_values.ndim != 1 or treatment_values.ndim != 1:
        raise ValueError(
            f"both arms must be 1-d, got shapes {control_values.shape} and {treatment_values.shape}"
        )
    if control_values.size < 2 or treatment_values.size < 2:
        raise ValueError(
            f"need at least 2 observations per arm, got {control_values.size} "
            f"and {treatment_values.size}"
        )
    # NaN fails every comparison, which in the permutation test means no
    # resample ever counts as extreme: it would return the most significant
    # possible p-value on unusable data. Refuse instead.
    if not np.all(np.isfinite(control_values)) or not np.all(np.isfinite(treatment_values)):
        raise ValueError("outcomes must all be finite; found NaN or infinity")
    return control_values, treatment_values


def _mean_difference(
    control_values: NDArray[np.float64], treatment_values: NDArray[np.float64], axis: int = -1
) -> NDArray[np.float64]:
    """Treatment mean minus control mean along an axis, in the form scipy's resamplers call."""
    return np.mean(treatment_values, axis=axis) - np.mean(control_values, axis=axis)


def bootstrap_mean_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    random_generator: np.random.Generator,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Bootstrap interval for the difference in means, scipy's BCa method.

    Arms are resampled independently with replacement. The
    bias-corrected and accelerated (BCa) interval adjusts the percentile
    interval for the skew of the bootstrap distribution, which matters
    for heavy-tailed metrics at modest n; the coverage test pins it on
    lognormal arms.
    """
    control_values, treatment_values = _two_samples(control, treatment)
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be at least 100, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    result = stats.bootstrap(
        (control_values, treatment_values),
        _mean_difference,
        n_resamples=n_resamples,
        batch=_BATCH,
        vectorized=True,
        confidence_level=confidence,
        method="BCa",
        rng=random_generator,
    )
    return BootstrapResult(
        estimate=float(treatment_values.mean() - control_values.mean()),
        ci_low=float(result.confidence_interval.low),
        ci_high=float(result.confidence_interval.high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def permutation_test_mean_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    random_generator: np.random.Generator,
    n_permutations: int = 10_000,
) -> PermutationResult:
    """Two-sided permutation test for a difference in means, through scipy.

    Exact conditional on the pooled outcomes: no distributional
    assumptions, so it stays valid for heavy-tailed and zero-inflated
    metrics. Also correct for binary outcomes, where the mean difference
    is the rate difference. scipy counts the observed difference among
    the relabelings, so each one-sided p-value is at least
    ``1 / (n_permutations + 1)``, and the two-sided p-value is twice the
    smaller one-sided value, capped at 1: its floor is
    ``2 / (n_permutations + 1)``.
    """
    control_values, treatment_values = _two_samples(control, treatment)
    if n_permutations < 100:
        raise ValueError(f"n_permutations must be at least 100, got {n_permutations}")

    result = stats.permutation_test(
        (control_values, treatment_values),
        _mean_difference,
        permutation_type="independent",
        vectorized=True,
        n_resamples=n_permutations,
        batch=_BATCH,
        alternative="two-sided",
        rng=random_generator,
    )
    return PermutationResult(
        observed=float(treatment_values.mean() - control_values.mean()),
        p_value=float(result.pvalue),
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
    control_values, treatment_values = _two_samples(control, treatment)
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    return float(stats.trim_mean(treatment_values, trim) - stats.trim_mean(control_values, trim))

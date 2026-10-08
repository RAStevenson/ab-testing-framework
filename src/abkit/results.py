"""Result types returned across the library.

Multi-value results are frozen dataclasses rather than tuples so call
sites read as ``result.p_value`` instead of positional unpacking, and so
results can sit in test fixtures or caches without defensive copying.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateComparison:
    """Two-sample comparison of conversion rates (treatment minus control)."""

    n_control: int
    n_treatment: int
    conversions_control: int
    conversions_treatment: int
    rate_control: float
    rate_treatment: float
    absolute_lift: float
    relative_lift: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    confidence: float


@dataclass(frozen=True)
class MeanComparison:
    """Welch two-sample comparison of means (treatment minus control)."""

    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    difference: float
    statistic: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    confidence: float


@dataclass(frozen=True)
class SrmResult:
    """Chi-square sample-ratio-mismatch check of observed arm sizes."""

    n_control: int
    n_treatment: int
    expected_treatment_share: float
    statistic: float
    p_value: float


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a bootstrap confidence interval (scipy's BCa method)."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int


@dataclass(frozen=True)
class PermutationResult:
    """Observed statistic and its permutation p-value."""

    observed: float
    p_value: float
    n_permutations: int


@dataclass(frozen=True)
class SequentialDesign:
    """Group-sequential boundaries produced by an alpha-spending design.

    ``boundaries[k]`` is the two-sided critical value for ``|z|`` at look
    ``k``. It is ``inf`` when the spending function allocates so little
    alpha to that look that no finite z could stop the experiment there,
    which is normal for the first look or two of an O'Brien-Fleming design
    and is a design property rather than a failure.

    ``cumulative_alpha[k]`` is the type I error the spending function
    allocates through look ``k``. Where a boundary is infinite the design
    spends slightly less than this, by at most 1e-10 per such look.
    """

    information_fractions: tuple[float, ...]
    boundaries: tuple[float, ...]
    cumulative_alpha: tuple[float, ...]
    alpha: float
    spending: str


@dataclass(frozen=True)
class SequentialOutcome:
    """Result of monitoring a z-statistic sequence against a design.

    ``stopped_at_look`` is the 1-based index of the first boundary
    crossing, or ``None`` if no look crossed.
    """

    stopped_at_look: int | None
    z_values: tuple[float, ...]
    boundaries: tuple[float, ...]


@dataclass(frozen=True)
class RepeatedIntervals:
    """Adjusted confidence intervals, one per look, that hold their coverage at every look at once
    (repeated confidence intervals).

    ``ci_low[k]`` and ``ci_high[k]`` are ``estimates[k]`` minus and plus
    ``boundaries[k]`` times that look's standard error. An infinite
    boundary gives an unbounded interval.
    """

    estimates: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    boundaries: tuple[float, ...]
    alpha: float


@dataclass(frozen=True)
class CupedResult:
    """Treatment minus control, adjusted by a pre-experiment column (CUPED).

    ``variance_reduction`` is the share of the unadjusted estimate's
    variance the adjustment removed: one minus the squared ratio of the
    adjusted to the unadjusted standard error.
    """

    difference: float
    standard_error: float
    unadjusted_standard_error: float
    variance_reduction: float
    pre_period_coefficient: float
    p_value: float
    ci_low: float
    ci_high: float
    confidence: float

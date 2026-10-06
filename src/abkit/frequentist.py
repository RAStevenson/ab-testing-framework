"""Frequentist tests and multiple-comparison corrections.

All tests are two-sided. Adjustment functions take an array of p-values
and return adjusted p-values in the same order; compare the adjusted
values to the unadjusted alpha.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from abkit.results import MeanComparison, RateComparison, SrmResult


def two_proportion_ztest(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    *,
    confidence: float = 0.95,
) -> RateComparison:
    """Two-sided z-test for a difference in conversion rates.

    The test statistic uses the pooled standard error, which is the
    correct null-hypothesis variance. The confidence interval is an
    unpooled interval, which is what is correct away from the null, in
    its Agresti-Caffo form: one success and one failure are added to each
    arm before forming it. Plain Wald degenerates to zero width when both
    arms sit at 0% or 100%, and undercovers near those boundaries; the
    adjustment costs nothing at experiment sample sizes.

    Because of that adjustment the interval is centered on the adjusted
    difference rather than exactly on ``absolute_lift``, which is reported
    unadjusted. ``relative_lift`` is NaN when the control rate is zero.
    """
    for name, count, total in (
        ("control", conversions_control, n_control),
        ("treatment", conversions_treatment, n_treatment),
    ):
        if total < 1:
            raise ValueError(f"n_{name} must be positive, got {total}")
        if not 0 <= count <= total:
            raise ValueError(f"conversions_{name}={count} outside [0, n_{name}={total}]")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rate_control = conversions_control / n_control
    rate_treatment = conversions_treatment / n_treatment
    difference = rate_treatment - rate_control

    pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    pooled_standard_error = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_control + 1.0 / n_treatment))
    if pooled_standard_error == 0.0:
        raise ValueError("all outcomes identical in both arms; z-test is undefined")
    z_statistic = difference / pooled_standard_error
    p_value = 2.0 * stats.norm.sf(abs(z_statistic))

    # Agresti-Caffo: add one success and one failure to each arm, then form a
    # Wald interval on the adjusted rates. Plain Wald collapses to zero width
    # when both arms sit exactly on 0 or 1, reporting a point as a 95%
    # interval; Agresti-Caffo stays sensible there and has better coverage at
    # rates near the boundary generally. The two are indistinguishable at
    # experiment sample sizes. The interval is therefore centered on the
    # adjusted difference, not on ``absolute_lift``, which stays unadjusted
    # because that is the estimate worth reporting.
    adjusted_n_control = n_control + 2.0
    adjusted_n_treatment = n_treatment + 2.0
    adjusted_rate_control = (conversions_control + 1.0) / adjusted_n_control
    adjusted_rate_treatment = (conversions_treatment + 1.0) / adjusted_n_treatment
    adjusted_difference = adjusted_rate_treatment - adjusted_rate_control
    adjusted_standard_error = np.sqrt(
        adjusted_rate_control * (1.0 - adjusted_rate_control) / adjusted_n_control
        + adjusted_rate_treatment * (1.0 - adjusted_rate_treatment) / adjusted_n_treatment
    )
    z_critical = stats.norm.ppf(0.5 + confidence / 2.0)
    return RateComparison(
        n_control=n_control,
        n_treatment=n_treatment,
        conversions_control=conversions_control,
        conversions_treatment=conversions_treatment,
        rate_control=rate_control,
        rate_treatment=rate_treatment,
        absolute_lift=difference,
        relative_lift=difference / rate_control if rate_control > 0.0 else float("nan"),
        statistic=float(z_statistic),
        p_value=float(p_value),
        # a difference of proportions lives in [-1, 1], and the adjustment can
        # push the interval a hair outside it at extreme rates; clipping to the
        # parameter space can only improve coverage
        ci_low=float(max(-1.0, adjusted_difference - z_critical * adjusted_standard_error)),
        ci_high=float(min(1.0, adjusted_difference + z_critical * adjusted_standard_error)),
        confidence=confidence,
    )


def welch_ttest(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    confidence: float = 0.95,
) -> MeanComparison:
    """Two-sided Welch t-test for a difference in means.

    Welch rather than Student because experiment arms routinely differ in
    both size and variance; Welch costs nothing when they do not.
    """
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
    if not np.all(np.isfinite(control_values)) or not np.all(np.isfinite(treatment_values)):
        raise ValueError("outcomes must all be finite; found NaN or infinity")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    control_variance = control_values.var(ddof=1)
    treatment_variance = treatment_values.var(ddof=1)
    standard_error_squared = (
        control_variance / control_values.size + treatment_variance / treatment_values.size
    )
    if standard_error_squared == 0.0:
        raise ValueError("both arms are constant; t-test is undefined")
    degrees_of_freedom = standard_error_squared**2 / (
        (control_variance / control_values.size) ** 2 / (control_values.size - 1)
        + (treatment_variance / treatment_values.size) ** 2 / (treatment_values.size - 1)
    )
    difference = treatment_values.mean() - control_values.mean()
    t_statistic = difference / np.sqrt(standard_error_squared)
    p_value = 2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
    t_critical = stats.t.ppf(0.5 + confidence / 2.0, degrees_of_freedom)
    half_width = t_critical * np.sqrt(standard_error_squared)
    return MeanComparison(
        n_control=int(control_values.size),
        n_treatment=int(treatment_values.size),
        mean_control=float(control_values.mean()),
        mean_treatment=float(treatment_values.mean()),
        difference=float(difference),
        statistic=float(t_statistic),
        df=float(degrees_of_freedom),
        p_value=float(p_value),
        ci_low=float(difference - half_width),
        ci_high=float(difference + half_width),
        confidence=confidence,
    )


def welch_ttest_from_stats(
    n_control: int,
    mean_control: float,
    variance_control: float,
    n_treatment: int,
    mean_treatment: float,
    variance_treatment: float,
    *,
    confidence: float = 0.95,
) -> MeanComparison:
    """Welch t-test from per-arm summary statistics instead of raw arrays.

    Exists for aggregate-only data (the ASOS shape), where counts, means
    and ddof=1 sample variances are all that was published. Given the
    statistics of the same arrays, this returns exactly what
    ``welch_ttest`` returns; the property tests pin that agreement.
    """
    for name, count in (("n_control", n_control), ("n_treatment", n_treatment)):
        if count < 2:
            raise ValueError(f"need at least 2 observations per arm, got {name}={count}")
    for name, value in (
        ("mean_control", mean_control),
        ("variance_control", variance_control),
        ("mean_treatment", mean_treatment),
        ("variance_treatment", variance_treatment),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if variance_control < 0.0 or variance_treatment < 0.0:
        raise ValueError(
            f"variances must be non-negative, got {variance_control}, {variance_treatment}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    standard_error_squared = variance_control / n_control + variance_treatment / n_treatment
    if standard_error_squared == 0.0:
        raise ValueError("both arms are constant; t-test is undefined")
    degrees_of_freedom = standard_error_squared**2 / (
        (variance_control / n_control) ** 2 / (n_control - 1)
        + (variance_treatment / n_treatment) ** 2 / (n_treatment - 1)
    )
    difference = mean_treatment - mean_control
    t_statistic = difference / np.sqrt(standard_error_squared)
    p_value = 2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
    t_critical = stats.t.ppf(0.5 + confidence / 2.0, degrees_of_freedom)
    half_width = t_critical * np.sqrt(standard_error_squared)
    return MeanComparison(
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        mean_control=float(mean_control),
        mean_treatment=float(mean_treatment),
        difference=float(difference),
        statistic=float(t_statistic),
        df=float(degrees_of_freedom),
        p_value=float(p_value),
        ci_low=float(difference - half_width),
        ci_high=float(difference + half_width),
        confidence=confidence,
    )


def srm_check(
    n_control: int,
    n_treatment: int,
    *,
    expected_treatment_share: float = 0.5,
) -> SrmResult:
    """Chi-square test that observed arm sizes match the intended split.

    Run this before reading any metric: a low p-value (conventionally
    below 1e-3) means users went missing asymmetrically and every
    downstream comparison is suspect.
    """
    if n_control < 0 or n_treatment < 0 or n_control + n_treatment == 0:
        raise ValueError(f"invalid arm sizes: {n_control}, {n_treatment}")
    if not 0.0 < expected_treatment_share < 1.0:
        raise ValueError(
            f"expected_treatment_share must be in (0, 1), got {expected_treatment_share}"
        )
    total = n_control + n_treatment
    expected_treatment = total * expected_treatment_share
    expected_control = total - expected_treatment
    statistic = (n_treatment - expected_treatment) ** 2 / expected_treatment + (
        n_control - expected_control
    ) ** 2 / expected_control
    p_value = stats.chi2.sf(statistic, df=1)
    return SrmResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_treatment_share=expected_treatment_share,
        statistic=float(statistic),
        p_value=float(p_value),
    )


def _validated_pvalues(p_values: ArrayLike) -> NDArray[np.float64]:
    valid_p_values = np.asarray(p_values, dtype=np.float64)
    if valid_p_values.ndim != 1 or valid_p_values.size == 0:
        raise ValueError("p_values must be a non-empty 1-d array")
    if np.any((valid_p_values < 0.0) | (valid_p_values > 1.0)) or np.any(np.isnan(valid_p_values)):
        raise ValueError("p_values must lie in [0, 1]")
    return valid_p_values


def bonferroni(p_values: ArrayLike) -> NDArray[np.float64]:
    """Bonferroni-adjusted p-values. Controls FWER; simple but conservative."""
    valid_p_values = _validated_pvalues(p_values)
    return np.minimum(valid_p_values * valid_p_values.size, 1.0)


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values.

    Controls FWER like Bonferroni but is uniformly more powerful, so
    there is no reason to prefer Bonferroni beyond hand-calculation.
    """
    valid_p_values = _validated_pvalues(p_values)
    n_tests = valid_p_values.size
    order = np.argsort(valid_p_values)
    scaled = valid_p_values[order] * (n_tests - np.arange(n_tests))
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(n_tests)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (q-values).

    Controls the false discovery rate, not FWER: appropriate for screens
    over many secondary metrics where a few false positives are an
    acceptable price for power, not for a single ship decision.
    """
    valid_p_values = _validated_pvalues(p_values)
    n_tests = valid_p_values.size
    order = np.argsort(valid_p_values)
    ranks = np.arange(1, n_tests + 1)
    scaled = valid_p_values[order] * n_tests / ranks
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(n_tests)
    adjusted[order] = adjusted_sorted
    return adjusted

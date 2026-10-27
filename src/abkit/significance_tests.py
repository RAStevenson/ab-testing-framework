"""Frequentist tests and multiple-comparison corrections.

All tests are two-sided. Adjustment functions take an array of p-values
and return adjusted p-values in the same order; compare the adjusted
values to the unadjusted alpha.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from abkit.result_types import MeanComparison, RateComparison, SrmResult


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
    if pooled in (0.0, 1.0):
        raise ValueError("all outcomes identical in both arms; z-test is undefined")
    # treatment first, so the statistic's sign follows treatment minus control;
    # statsmodels pools the variance under the null by default
    z_statistic, p_value = proportions_ztest(
        [conversions_treatment, conversions_control], [n_treatment, n_control]
    )
    ci_low, ci_high = confint_proportions_2indep(
        conversions_treatment,
        n_treatment,
        conversions_control,
        n_control,
        method="agresti-caffo",
        compare="diff",
        alpha=1.0 - confidence,
    )
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
        ci_low=float(max(-1.0, ci_low)),
        ci_high=float(min(1.0, ci_high)),
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

    if control_values.var(ddof=1) == 0.0 and treatment_values.var(ddof=1) == 0.0:
        raise ValueError("both arms are constant; t-test is undefined")
    # treatment first, so the statistic's sign follows treatment minus control
    test = stats.ttest_ind(treatment_values, control_values, equal_var=False)
    interval = test.confidence_interval(confidence_level=confidence)
    difference = treatment_values.mean() - control_values.mean()
    return MeanComparison(
        n_control=int(control_values.size),
        n_treatment=int(treatment_values.size),
        mean_control=float(control_values.mean()),
        mean_treatment=float(treatment_values.mean()),
        difference=float(difference),
        # scipy's stub types the t-test result without these attributes
        statistic=float(test.statistic),  # pyright: ignore[reportAttributeAccessIssue]
        df=float(test.df),  # pyright: ignore[reportAttributeAccessIssue]
        p_value=float(test.pvalue),  # pyright: ignore[reportAttributeAccessIssue]
        ci_low=float(interval.low),
        ci_high=float(interval.high),
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
    statistic, p_value = stats.chisquare(
        [n_control, n_treatment], f_exp=[total - expected_treatment, expected_treatment]
    )
    return SrmResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_treatment_share=expected_treatment_share,
        statistic=float(statistic),
        p_value=float(p_value),
    )


def _validated_pvalues(p_values: ArrayLike) -> NDArray[np.float64]:
    """Return the p-values as a 1-d float array, raising if it is empty or a value lies outside [0,
    1]."""
    valid_p_values = np.asarray(p_values, dtype=np.float64)
    if valid_p_values.ndim != 1 or valid_p_values.size == 0:
        raise ValueError("p_values must be a non-empty 1-d array")
    if np.any((valid_p_values < 0.0) | (valid_p_values > 1.0)) or np.any(np.isnan(valid_p_values)):
        raise ValueError("p_values must lie in [0, 1]")
    return valid_p_values


def bonferroni(p_values: ArrayLike) -> NDArray[np.float64]:
    """Bonferroni-adjusted p-values, in the input order. Controls FWER; simple but conservative."""
    return _adjusted(p_values, "bonferroni")


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values, in the input order.

    Controls FWER like Bonferroni but is uniformly more powerful, so
    there is no reason to prefer Bonferroni beyond hand-calculation.
    """
    return _adjusted(p_values, "holm")


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order.

    Controls the false discovery rate, not FWER: appropriate for screens
    over many secondary metrics where a few false positives are an
    acceptable price for power, not for a single ship decision.
    """
    return _adjusted(p_values, "fdr_bh")


def _adjusted(p_values: ArrayLike, method: str) -> NDArray[np.float64]:
    """Validate the p-values and return statsmodels' adjusted values in the input order."""
    valid_p_values = _validated_pvalues(p_values)
    return np.asarray(multipletests(valid_p_values, method=method)[1], dtype=np.float64)

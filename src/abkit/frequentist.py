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

    Because of that adjustment the interval is centred on the adjusted
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

    rate_c = conversions_control / n_control
    rate_t = conversions_treatment / n_treatment
    diff = rate_t - rate_c

    pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se_pooled = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_control + 1.0 / n_treatment))
    if se_pooled == 0.0:
        raise ValueError("all outcomes identical in both arms; z-test is undefined")
    z = diff / se_pooled
    p_value = 2.0 * stats.norm.sf(abs(z))

    # Agresti-Caffo: add one success and one failure to each arm, then form a
    # Wald interval on the adjusted rates. Plain Wald collapses to zero width
    # when both arms sit exactly on 0 or 1, reporting a point as a 95%
    # interval; Agresti-Caffo stays sensible there and has better coverage at
    # rates near the boundary generally. The two are indistinguishable at
    # experiment sample sizes. The interval is therefore centred on the
    # adjusted difference, not on ``absolute_lift``, which stays unadjusted
    # because that is the estimate worth reporting.
    ac_n_c = n_control + 2.0
    ac_n_t = n_treatment + 2.0
    ac_c = (conversions_control + 1.0) / ac_n_c
    ac_t = (conversions_treatment + 1.0) / ac_n_t
    ac_diff = ac_t - ac_c
    se_ac = np.sqrt(ac_c * (1.0 - ac_c) / ac_n_c + ac_t * (1.0 - ac_t) / ac_n_t)
    z_crit = stats.norm.ppf(0.5 + confidence / 2.0)
    return RateComparison(
        n_control=n_control,
        n_treatment=n_treatment,
        conversions_control=conversions_control,
        conversions_treatment=conversions_treatment,
        rate_control=rate_c,
        rate_treatment=rate_t,
        absolute_lift=diff,
        relative_lift=diff / rate_c if rate_c > 0.0 else float("nan"),
        statistic=float(z),
        p_value=float(p_value),
        # a difference of proportions lives in [-1, 1], and the adjustment can
        # push the interval a hair outside it at extreme rates; clipping to the
        # parameter space can only improve coverage
        ci_low=float(max(-1.0, ac_diff - z_crit * se_ac)),
        ci_high=float(min(1.0, ac_diff + z_crit * se_ac)),
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
    x = np.asarray(control, dtype=np.float64)
    y = np.asarray(treatment, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"both arms must be 1-d, got shapes {x.shape} and {y.shape}")
    if x.size < 2 or y.size < 2:
        raise ValueError(f"need at least 2 observations per arm, got {x.size} and {y.size}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("outcomes must all be finite; found NaN or infinity")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    se_sq = var_x / x.size + var_y / y.size
    if se_sq == 0.0:
        raise ValueError("both arms are constant; t-test is undefined")
    df = se_sq**2 / ((var_x / x.size) ** 2 / (x.size - 1) + (var_y / y.size) ** 2 / (y.size - 1))
    diff = y.mean() - x.mean()
    t = diff / np.sqrt(se_sq)
    p_value = 2.0 * stats.t.sf(abs(t), df)
    t_crit = stats.t.ppf(0.5 + confidence / 2.0, df)
    half_width = t_crit * np.sqrt(se_sq)
    return MeanComparison(
        n_control=int(x.size),
        n_treatment=int(y.size),
        mean_control=float(x.mean()),
        mean_treatment=float(y.mean()),
        difference=float(diff),
        statistic=float(t),
        df=float(df),
        p_value=float(p_value),
        ci_low=float(diff - half_width),
        ci_high=float(diff + half_width),
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
    expected_t = total * expected_treatment_share
    expected_c = total - expected_t
    statistic = (n_treatment - expected_t) ** 2 / expected_t + (
        n_control - expected_c
    ) ** 2 / expected_c
    p_value = stats.chi2.sf(statistic, df=1)
    return SrmResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_treatment_share=expected_treatment_share,
        statistic=float(statistic),
        p_value=float(p_value),
    )


def _validated_pvalues(p_values: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-d array")
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("p_values must lie in [0, 1]")
    return p


def bonferroni(p_values: ArrayLike) -> NDArray[np.float64]:
    """Bonferroni-adjusted p-values. Controls FWER; simple but conservative."""
    p = _validated_pvalues(p_values)
    return np.minimum(p * p.size, 1.0)


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values.

    Controls FWER like Bonferroni but is uniformly more powerful, so
    there is no reason to prefer Bonferroni beyond hand-calculation.
    """
    p = _validated_pvalues(p_values)
    m = p.size
    order = np.argsort(p)
    scaled = p[order] * (m - np.arange(m))
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (q-values).

    Controls the false discovery rate, not FWER: appropriate for screens
    over many secondary metrics where a few false positives are an
    acceptable price for power, not for a single ship decision.
    """
    p = _validated_pvalues(p_values)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    scaled = p[order] * m / ranks
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted

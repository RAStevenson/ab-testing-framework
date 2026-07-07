import numpy as np
import pytest
from scipy import stats

from abkit.frequentist import (
    benjamini_hochberg,
    bonferroni,
    holm,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
)


def test_ztest_matches_chi_square_without_correction():
    result = two_proportion_ztest(480, 10_000, 552, 10_200)
    table = np.array([[480, 10_000 - 480], [552, 10_200 - 552]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    assert result.statistic**2 == pytest.approx(chi2, rel=1e-10)
    assert result.p_value == pytest.approx(p, rel=1e-10)


def test_ztest_estimates_and_ci():
    result = two_proportion_ztest(500, 10_000, 600, 10_000)
    assert result.rate_control == pytest.approx(0.05)
    assert result.rate_treatment == pytest.approx(0.06)
    assert result.absolute_lift == pytest.approx(0.01)
    assert result.relative_lift == pytest.approx(0.20)
    assert result.ci_low < 0.01 < result.ci_high
    # Wald CI half-width, checked against a hand computation
    se = np.sqrt(0.05 * 0.95 / 10_000 + 0.06 * 0.94 / 10_000)
    assert result.ci_high - result.ci_low == pytest.approx(2 * 1.959964 * se, rel=1e-4)


def test_ztest_type_i_error_calibrated():
    rng = np.random.default_rng(21)
    n_sims, n = 4000, 2000
    rejections = 0
    for _ in range(n_sims):
        x = rng.binomial(n, 0.1)
        y = rng.binomial(n, 0.1)
        if x + y in (0, 2 * n):
            continue
        if two_proportion_ztest(x, n, y, n).p_value < 0.05:
            rejections += 1
    # 3 sigma of a binomial(4000, 0.05) is about 0.010
    assert rejections / n_sims == pytest.approx(0.05, abs=0.012)


def test_ztest_rejects_degenerate_input():
    with pytest.raises(ValueError):
        two_proportion_ztest(0, 100, 0, 100)
    with pytest.raises(ValueError):
        two_proportion_ztest(50, 100, 5, 4)
    with pytest.raises(ValueError):
        two_proportion_ztest(5, 0, 5, 100)


def test_welch_matches_scipy():
    rng = np.random.default_rng(22)
    x = rng.normal(0.0, 1.0, 400)
    y = rng.normal(0.2, 2.0, 250)
    result = welch_ttest(x, y)
    expected = stats.ttest_ind(y, x, equal_var=False)
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12)
    assert result.ci_low < result.difference < result.ci_high


def test_welch_ci_coverage():
    rng = np.random.default_rng(23)
    true_diff = 0.5
    covered = 0
    n_sims = 400
    for _ in range(n_sims):
        x = rng.normal(0.0, 1.0, 120)
        y = rng.normal(true_diff, 1.5, 80)
        r = welch_ttest(x, y)
        covered += r.ci_low <= true_diff <= r.ci_high
    assert covered / n_sims == pytest.approx(0.95, abs=0.035)


def test_welch_rejects_degenerate_input():
    with pytest.raises(ValueError):
        welch_ttest([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_ttest([1.0, 1.0], [2.0, 2.0])


def test_srm_detects_mismatch_and_passes_clean_split():
    bad = srm_check(50_000, 51_500)
    assert bad.p_value < 1e-5
    clean = srm_check(50_000, 50_100)
    assert clean.p_value > 0.5
    uneven = srm_check(90_000, 10_050, expected_treatment_share=0.1)
    assert uneven.p_value > 0.5


def test_srm_rejects_bad_input():
    with pytest.raises(ValueError):
        srm_check(0, 0)
    with pytest.raises(ValueError):
        srm_check(100, 100, expected_treatment_share=1.0)


P_VALUES = [0.01, 0.04, 0.03, 0.005]


def test_bonferroni_known_values():
    np.testing.assert_allclose(bonferroni(P_VALUES), [0.04, 0.16, 0.12, 0.02])


def test_holm_known_values():
    np.testing.assert_allclose(holm(P_VALUES), [0.03, 0.06, 0.06, 0.02])


def test_benjamini_hochberg_known_values():
    np.testing.assert_allclose(benjamini_hochberg(P_VALUES), [0.02, 0.04, 0.04, 0.02])


def test_adjustments_dominance_and_bounds():
    rng = np.random.default_rng(24)
    p = rng.random(50) ** 2
    bonf, hol, bh = bonferroni(p), holm(p), benjamini_hochberg(p)
    for adjusted in (bonf, hol, bh):
        assert np.all(adjusted >= p - 1e-15)
        assert np.all(adjusted <= 1.0)
    # Holm is uniformly at least as powerful as Bonferroni, BH at least as Holm
    assert np.all(hol <= bonf + 1e-15)
    assert np.all(bh <= hol + 1e-15)


def test_adjustments_are_monotone_in_p():
    # clipping at 1.0 can create ties, so the order is preserved weakly
    p = np.array([0.9, 0.001, 0.5])
    order = np.argsort(p)
    for fn in (bonferroni, holm, benjamini_hochberg):
        adjusted = fn(p)
        assert adjusted[1] == adjusted.min()
        assert np.all(np.diff(adjusted[order]) >= -1e-15)


def test_adjustments_reject_invalid_pvalues():
    for fn in (bonferroni, holm, benjamini_hochberg):
        with pytest.raises(ValueError):
            fn([0.5, 1.5])
        with pytest.raises(ValueError):
            fn([])

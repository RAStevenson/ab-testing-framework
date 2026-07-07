import numpy as np
import pytest

from abkit.frequentist import welch_ttest
from abkit.resampling import (
    bootstrap_mean_diff,
    permutation_test_mean_diff,
    trimmed_mean_diff,
)


def test_bootstrap_point_estimate_and_reproducibility():
    rng = np.random.default_rng(31)
    x = rng.normal(0.0, 1.0, 500)
    y = rng.normal(0.4, 1.0, 500)
    r1 = bootstrap_mean_diff(x, y, rng=np.random.default_rng(1), n_resamples=2000)
    r2 = bootstrap_mean_diff(x, y, rng=np.random.default_rng(1), n_resamples=2000)
    assert r1 == r2
    assert r1.estimate == pytest.approx(y.mean() - x.mean())
    assert r1.ci_low < r1.estimate < r1.ci_high


def test_bootstrap_ci_coverage_on_skewed_data():
    rng = np.random.default_rng(32)
    true_diff = np.exp(0.5) * (np.exp(0.2) - 1.0)  # lognormal mean difference
    covered = 0
    n_sims = 300
    for _ in range(n_sims):
        x = rng.lognormal(0.0, 1.0, 400)
        y = rng.lognormal(0.2, 1.0, 400)
        r = bootstrap_mean_diff(x, y, rng=rng, n_resamples=1000)
        covered += r.ci_low <= true_diff <= r.ci_high
    # percentile intervals undercover slightly on skewed data; accept 90-98
    assert 0.90 <= covered / n_sims <= 0.98


def test_bootstrap_ci_narrows_with_n():
    rng = np.random.default_rng(33)
    small = rng.normal(0.0, 1.0, 100)
    big = np.tile(small, 25)
    w_small = bootstrap_mean_diff(small, small + 0.1, rng=rng, n_resamples=1000)
    w_big = bootstrap_mean_diff(big, big + 0.1, rng=rng, n_resamples=1000)
    assert (w_big.ci_high - w_big.ci_low) < (w_small.ci_high - w_small.ci_low) / 3


def test_permutation_matches_welch_on_normal_data():
    rng = np.random.default_rng(34)
    x = rng.normal(0.0, 1.0, 400)
    y = rng.normal(0.12, 1.0, 400)
    perm = permutation_test_mean_diff(x, y, rng=rng, n_permutations=20_000)
    welch = welch_ttest(x, y)
    assert perm.p_value == pytest.approx(welch.p_value, abs=0.02)
    assert perm.observed == pytest.approx(welch.difference)


def test_permutation_type_i_error_calibrated():
    rng = np.random.default_rng(35)
    n_sims = 400
    rejections = 0
    for _ in range(n_sims):
        pooled = rng.exponential(1.0, 200)  # skewed null: both arms identical
        x, y = pooled[:100], pooled[100:]
        p = permutation_test_mean_diff(x, y, rng=rng, n_permutations=499).p_value
        rejections += p < 0.05
    assert rejections / n_sims == pytest.approx(0.05, abs=0.03)


def test_permutation_detects_large_effect():
    rng = np.random.default_rng(36)
    x = rng.normal(0.0, 1.0, 300)
    y = rng.normal(1.0, 1.0, 300)
    p = permutation_test_mean_diff(x, y, rng=rng, n_permutations=999).p_value
    assert p == pytest.approx(1 / 1000)  # add-one p-value floor


def test_permutation_works_on_binary_outcomes():
    rng = np.random.default_rng(37)
    x = (rng.random(2000) < 0.10).astype(float)
    y = (rng.random(2000) < 0.14).astype(float)
    p = permutation_test_mean_diff(x, y, rng=rng, n_permutations=2000).p_value
    assert p < 0.01


def test_trimmed_mean_diff_ignores_whales():
    rng = np.random.default_rng(38)
    x = rng.normal(10.0, 1.0, 1000)
    y = rng.normal(10.0, 1.0, 1000)
    y_with_whale = y.copy()
    y_with_whale[0] = 1e6
    assert abs(trimmed_mean_diff(x, y_with_whale, trim=0.01)) < 0.2
    assert abs(np.mean(y_with_whale) - np.mean(x)) > 100


def test_resampling_input_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        bootstrap_mean_diff([1.0], [1.0, 2.0], rng=rng)
    with pytest.raises(ValueError):
        bootstrap_mean_diff([1.0, 2.0], [1.0, 2.0], rng=rng, n_resamples=10)
    with pytest.raises(ValueError):
        permutation_test_mean_diff([1.0, 2.0], [1.0, 2.0], rng=rng, n_permutations=10)
    with pytest.raises(ValueError):
        trimmed_mean_diff([1.0, 2.0], [1.0, 2.0], trim=0.5)

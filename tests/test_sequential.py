import numpy as np
import pytest
from scipy import stats

from abkit.sequential import (
    design,
    monitor,
    obrien_fleming_spending,
    pocock_spending,
    z_from_cumulative_counts,
)


def test_single_look_reduces_to_fixed_test():
    d = design([1.0], alpha=0.05)
    assert d.boundaries[0] == pytest.approx(stats.norm.isf(0.025), abs=1e-6)


def test_spending_functions_hit_alpha_at_full_information():
    for fn in (obrien_fleming_spending, pocock_spending):
        assert fn(1.0, 0.05) == pytest.approx(0.05, abs=1e-12)
        t = np.linspace(0.1, 1.0, 10)
        values = fn(t, 0.05)
        assert np.all(np.diff(values) > 0)


def test_pocock_boundaries_match_published_values():
    # Lan-DeMets Pocock-type spending, 5 equal looks, two-sided alpha 0.05
    d = design([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05, spending="pocock")
    published = [2.4380, 2.4268, 2.4101, 2.3966, 2.3859]
    np.testing.assert_allclose(d.boundaries, published, atol=2e-3)


def test_obf_boundaries_are_steeply_decreasing():
    d = design([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05)
    assert np.all(np.diff(d.boundaries) < 0)
    assert d.boundaries[0] > 4.0
    # the final OBF boundary should cost little relative to the fixed test
    assert d.boundaries[-1] < 2.10
    assert d.cumulative_alpha[-1] == pytest.approx(0.05, abs=1e-12)


def test_two_look_design_against_bivariate_normal():
    # non-crossing probability must equal 1 - alpha; check with scipy's
    # exact bivariate normal rectangle probability, independent of our grid
    d = design([0.5, 1.0], alpha=0.05, spending="pocock")
    c1, c2 = d.boundaries
    rho = np.sqrt(0.5)
    cov = [[1.0, rho], [rho, 1.0]]
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    non_crossing = mvn.cdf([c1, c2]) - mvn.cdf([-c1, c2]) - mvn.cdf([c1, -c2]) + mvn.cdf([-c1, -c2])
    assert non_crossing == pytest.approx(0.95, abs=5e-4)


def test_three_look_design_against_trivariate_normal():
    d = design([1 / 3, 2 / 3, 1.0], alpha=0.05)
    t = np.array(d.information_fractions)
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    b = np.array(d.boundaries)
    # inclusion-exclusion over the 2^3 orthant corners of the rectangle
    non_crossing = 0.0
    mvn = stats.multivariate_normal(mean=np.zeros(3), cov=cov)
    for signs in np.ndindex(2, 2, 2):
        sign_vec = np.where(np.array(signs) == 0, 1.0, -1.0)
        non_crossing += np.prod(sign_vec) * mvn.cdf(sign_vec * b)
    assert non_crossing == pytest.approx(0.95, abs=1e-3)


def test_five_look_type_i_error_by_simulation():
    d = design([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05)
    rng = np.random.default_rng(41)
    n_sims = 200_000
    t = np.array(d.information_fractions)
    increments = rng.normal(size=(n_sims, t.size)) * np.sqrt(np.diff(t, prepend=0.0))
    z = np.cumsum(increments, axis=1) / np.sqrt(t)
    crossed = np.any(np.abs(z) >= np.array(d.boundaries), axis=1)
    # 3 sigma of binomial(200k, 0.05) is about 0.0015
    assert crossed.mean() == pytest.approx(0.05, abs=0.002)


def test_naive_daily_peeking_inflates_type_i_error():
    # the motivating failure: 14 looks at fixed z=1.96 roughly triples alpha
    rng = np.random.default_rng(42)
    n_sims = 50_000
    t = np.linspace(1 / 14, 1.0, 14)
    increments = rng.normal(size=(n_sims, 14)) * np.sqrt(np.diff(t, prepend=0.0))
    z = np.cumsum(increments, axis=1) / np.sqrt(t)
    naive = np.any(np.abs(z) >= 1.96, axis=1).mean()
    assert naive > 0.15

    d = design(t.tolist(), alpha=0.05)
    controlled = np.any(np.abs(z) >= np.array(d.boundaries), axis=1).mean()
    assert controlled == pytest.approx(0.05, abs=0.005)


def test_design_validation():
    with pytest.raises(ValueError):
        design([])
    with pytest.raises(ValueError):
        design([0.5, 0.5, 1.0])
    with pytest.raises(ValueError):
        design([0.5, 1.2])
    with pytest.raises(ValueError):
        design([0.0, 1.0])
    with pytest.raises(ValueError):
        design([1.0], alpha=0.0)
    with pytest.raises(ValueError):
        design([1.0], spending="bayes")
    with pytest.raises(ValueError):
        design([1.0], grid_points=10)


def test_monitor_stops_at_first_crossing():
    d = design([0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="pocock")
    z = [0.5, d.boundaries[1] + 0.1, 5.0, 0.0]
    outcome = monitor(z, d)
    assert outcome.stopped_at_look == 2

    quiet = monitor([0.5, 1.0, -1.2, 1.5], d)
    assert quiet.stopped_at_look is None

    partial = monitor([0.5, 1.0], d)
    assert partial.stopped_at_look is None
    assert len(partial.boundaries) == 2

    with pytest.raises(ValueError):
        monitor([0.0] * 5, d)


def test_z_from_cumulative_counts_matches_scalar_test():
    from abkit.frequentist import two_proportion_ztest

    z = z_from_cumulative_counts([50, 100], [1000, 2000], [70, 130], [1000, 2000])
    expected_final = two_proportion_ztest(100, 2000, 130, 2000).statistic
    assert z[-1] == pytest.approx(expected_final, rel=1e-12)


def test_z_from_cumulative_counts_validation():
    with pytest.raises(ValueError):
        z_from_cumulative_counts([1, 2], [10, 20], [1, 2], [10])
    with pytest.raises(ValueError):
        z_from_cumulative_counts([2, 1], [20, 10], [2, 3], [20, 30])
    with pytest.raises(ValueError):
        z_from_cumulative_counts([0, 0], [10, 20], [0, 0], [10, 20])

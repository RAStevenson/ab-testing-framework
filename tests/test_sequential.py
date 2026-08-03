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


def test_obf_boundaries_match_reference_values():
    # O'Brien-Fleming-type spending, 5 equal looks, two-sided alpha 0.05.
    # Reference computed independently by Gauss-Legendre quadrature on the
    # same recursion, converged to 10 digits; agrees with the standard tables.
    d = design([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05)
    reference = [4.3826127, 3.0997275, 2.5533550, 2.2538479, 2.0635011]
    np.testing.assert_allclose(d.boundaries, reference, atol=1e-4)


def test_first_obf_boundary_matches_the_analytic_identity():
    # OBF spending is defined so that the first boundary is exactly
    # z_{alpha/2} / sqrt(t_1). No tolerance needed beyond machine precision,
    # which makes this a free check on the spending-function definition.
    for t1 in (0.2, 0.35, 0.5):
        d = design([t1, 1.0], alpha=0.05)
        assert d.boundaries[0] == pytest.approx(stats.norm.isf(0.025) / np.sqrt(t1), rel=1e-9)


def test_boundaries_converge_on_grid():
    # pins the accuracy claim in the module docstring: the default grid is
    # within 1e-4 of a four-times-finer one, for a long schedule
    t = np.linspace(1 / 14, 1.0, 14).tolist()
    coarse = design(t, alpha=0.05, grid_points=1024)
    fine = design(t, alpha=0.05, grid_points=4096)
    finite = np.isfinite(coarse.boundaries)
    np.testing.assert_allclose(
        np.array(coarse.boundaries)[finite], np.array(fine.boundaries)[finite], atol=1e-4
    )


def test_unspendable_early_looks_get_infinite_boundaries():
    # under OBF the first daily look of a 14-day run is allocated far too
    # little alpha to be stoppable; that is a design property, not a failure
    d = design(np.linspace(1 / 14, 1.0, 14).tolist(), alpha=0.05)
    assert not np.isfinite(d.boundaries[0])
    assert np.all(np.isfinite(d.boundaries[1:]))
    assert monitor([50.0], d).stopped_at_look is None


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
    # the motivating failure: 14 looks at a fixed z=1.96 reject on about 22%
    # of null runs, roughly four times the nominal 5%
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
    with pytest.raises(ValueError):
        design([1.0], grid_points=100_000)
    with pytest.raises(ValueError):
        design([0.5, np.nan, 1.0])


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
    with pytest.raises(ValueError):
        monitor([0.5, np.nan], d)


def test_monitor_stops_on_a_harmful_treatment():
    # the boundaries are two-sided, so a treatment that hurts must stop the
    # experiment just as readily as one that helps. Comparing z rather than
    # |z| against the boundary would pass every other test in this file.
    d = design([0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="pocock")
    harmful = [-0.5, -(d.boundaries[1] + 0.1), -5.0, 0.0]
    assert monitor(harmful, d).stopped_at_look == 2

    mirrored = [-z for z in harmful]
    assert monitor(mirrored, d).stopped_at_look == monitor(harmful, d).stopped_at_look


def test_monitor_boundary_is_inclusive_and_tight():
    d = design([0.5, 1.0], alpha=0.05, spending="pocock")
    assert monitor([d.boundaries[0], 0.0], d).stopped_at_look == 1
    assert monitor([d.boundaries[0] - 1e-9, 0.0], d).stopped_at_look is None


def test_z_from_cumulative_counts_matches_scalar_test():
    from abkit.frequentist import two_proportion_ztest

    z = z_from_cumulative_counts([50, 100], [1000, 2000], [70, 130], [1000, 2000])
    expected_final = two_proportion_ztest(100, 2000, 130, 2000).statistic
    assert z[-1] == pytest.approx(expected_final, rel=1e-12)


def test_z_from_cumulative_counts_validation():
    with pytest.raises(ValueError):  # shape mismatch
        z_from_cumulative_counts([1, 2], [10, 20], [1, 2], [10])
    with pytest.raises(ValueError):  # arm sizes go backwards
        z_from_cumulative_counts([2, 1], [20, 10], [2, 3], [20, 30])
    with pytest.raises(ValueError):  # no conversions anywhere
        z_from_cumulative_counts([0, 0], [10, 20], [0, 0], [10, 20])
    with pytest.raises(ValueError):  # conversions go backwards
        z_from_cumulative_counts([50, 10], [1000, 2000], [70, 130], [1000, 2000])
    with pytest.raises(ValueError):  # conversions exceed the arm size
        z_from_cumulative_counts([1000, 2000], [50, 100], [70, 130], [1000, 2000])
    with pytest.raises(ValueError):  # negative conversions
        z_from_cumulative_counts([-5, 100], [1000, 2000], [70, 130], [1000, 2000])


def test_backwards_conversions_would_have_forced_a_false_stop():
    # the specific failure the validation above prevents: a cumulative series
    # that goes backwards manufactures a large z and ends the experiment
    good = z_from_cumulative_counts([50, 100], [1000, 2000], [70, 130], [1000, 2000])
    assert np.all(np.abs(good) < 3.0)
    with pytest.raises(ValueError):
        z_from_cumulative_counts([50, 10], [1000, 2000], [70, 130], [1000, 2000])

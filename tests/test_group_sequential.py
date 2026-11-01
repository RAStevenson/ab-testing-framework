import numpy as np
import pytest
from scipy import stats

from abkit.group_sequential import (
    design_boundaries,
    monitor_boundaries,
    obrien_fleming_spending,
    pocock_spending,
    repeated_intervals,
    z_from_cumulative_counts,
)


def test_single_look_reduces_to_fixed_test():
    sequential_design = design_boundaries([1.0], alpha=0.05)
    assert sequential_design.boundaries[0] == pytest.approx(stats.norm.isf(0.025), abs=1e-6)


def test_spending_functions_hit_alpha_at_full_information():
    for spending in (obrien_fleming_spending, pocock_spending):
        assert spending(1.0, 0.05) == pytest.approx(0.05, abs=1e-12)
        fractions = np.linspace(0.1, 1.0, 10)
        values = spending(fractions, 0.05)
        assert np.all(np.diff(values) > 0)


def test_pocock_boundaries_match_published_values():
    # Lan-DeMets Pocock-type spending, 5 equal looks, two-sided alpha 0.05
    sequential_design = design_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05, spending="pocock")
    published = [2.4380, 2.4268, 2.4101, 2.3966, 2.3859]
    np.testing.assert_allclose(sequential_design.boundaries, published, atol=2e-3)


def test_obf_boundaries_are_steeply_decreasing():
    sequential_design = design_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05)
    assert np.all(np.diff(sequential_design.boundaries) < 0)
    assert sequential_design.boundaries[0] > 4.0
    # the final OBF boundary should cost little relative to the fixed test
    assert sequential_design.boundaries[-1] < 2.10
    assert sequential_design.cumulative_alpha[-1] == pytest.approx(0.05, abs=1e-12)


def test_obf_boundaries_match_reference_values():
    # O'Brien-Fleming-type spending, 5 equal looks, two-sided alpha 0.05.
    # Reference computed independently by Gauss-Legendre quadrature on the
    # same recursion, converged to 10 digits; agrees with the standard tables.
    sequential_design = design_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05)
    reference = [4.3826127, 3.0997275, 2.5533550, 2.2538479, 2.0635011]
    np.testing.assert_allclose(sequential_design.boundaries, reference, atol=1e-4)


def test_first_obf_boundary_matches_the_analytic_identity():
    # OBF spending is defined so that the first boundary is exactly
    # z_{alpha/2} / sqrt(t_1). No tolerance needed beyond machine precision,
    # which makes this a free check on the spending-function definition.
    for first_fraction in (0.2, 0.35, 0.5):
        sequential_design = design_boundaries([first_fraction, 1.0], alpha=0.05)
        assert sequential_design.boundaries[0] == pytest.approx(
            stats.norm.isf(0.025) / np.sqrt(first_fraction), rel=1e-9
        )


def test_boundaries_converge_on_grid():
    # pins the accuracy claim in the module docstring: the default grid is
    # within 1e-4 of a four-times-finer one, for a long schedule
    fractions = np.linspace(1 / 14, 1.0, 14).tolist()
    coarse = design_boundaries(fractions, alpha=0.05, grid_points=1024)
    fine = design_boundaries(fractions, alpha=0.05, grid_points=4096)
    finite = np.isfinite(coarse.boundaries)
    np.testing.assert_allclose(
        np.array(coarse.boundaries)[finite], np.array(fine.boundaries)[finite], atol=1e-4
    )


def test_unspendable_early_looks_get_infinite_boundaries():
    # under OBF the first daily look of a 14-day run is allocated far too
    # little alpha to be stoppable; that is a design property, not a failure
    sequential_design = design_boundaries(np.linspace(1 / 14, 1.0, 14).tolist(), alpha=0.05)
    assert not np.isfinite(sequential_design.boundaries[0])
    assert np.all(np.isfinite(sequential_design.boundaries[1:]))
    assert monitor_boundaries([50.0], sequential_design).stopped_at_look is None


def test_two_look_design_against_bivariate_normal():
    # non-crossing probability must equal 1 - alpha; check with scipy's
    # exact bivariate normal rectangle probability, independent of our grid
    sequential_design = design_boundaries([0.5, 1.0], alpha=0.05, spending="pocock")
    first_boundary, second_boundary = sequential_design.boundaries
    correlation = np.sqrt(0.5)
    covariance = [[1.0, correlation], [correlation, 1.0]]
    # scipy's stubs type multivariate_normal's cov as int; it takes a matrix
    normal_distribution = stats.multivariate_normal(mean=[0.0, 0.0], cov=covariance)  # pyright: ignore[reportArgumentType]
    non_crossing = (
        normal_distribution.cdf([first_boundary, second_boundary])
        - normal_distribution.cdf([-first_boundary, second_boundary])
        - normal_distribution.cdf([first_boundary, -second_boundary])
        + normal_distribution.cdf([-first_boundary, -second_boundary])
    )
    assert non_crossing == pytest.approx(0.95, abs=5e-4)


def test_three_look_design_against_trivariate_normal():
    sequential_design = design_boundaries([1 / 3, 2 / 3, 1.0], alpha=0.05)
    fractions = np.array(sequential_design.information_fractions)
    covariance = np.sqrt(
        np.minimum.outer(fractions, fractions) / np.maximum.outer(fractions, fractions)
    )
    boundaries = np.array(sequential_design.boundaries)
    # inclusion-exclusion over the 2^3 orthant corners of the rectangle
    non_crossing = 0.0
    # scipy's stubs type multivariate_normal's cov as int; it takes a matrix
    normal_distribution = stats.multivariate_normal(mean=np.zeros(3), cov=covariance)  # pyright: ignore[reportArgumentType]
    for signs in np.ndindex(2, 2, 2):
        sign_multipliers = np.where(np.array(signs) == 0, 1.0, -1.0)
        non_crossing += np.prod(sign_multipliers) * normal_distribution.cdf(
            sign_multipliers * boundaries
        )
    assert non_crossing == pytest.approx(0.95, abs=1e-3)


def test_five_look_type_i_error_by_simulation():
    sequential_design = design_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05)
    random_generator = np.random.default_rng(41)
    n_sims = 200_000
    fractions = np.array(sequential_design.information_fractions)
    increments = random_generator.normal(size=(n_sims, fractions.size)) * np.sqrt(
        np.diff(fractions, prepend=0.0)
    )
    z_paths = np.cumsum(increments, axis=1) / np.sqrt(fractions)
    crossed = np.any(np.abs(z_paths) >= np.array(sequential_design.boundaries), axis=1)
    # 3 sigma of binomial(200k, 0.05) is about 0.0015
    assert crossed.mean() == pytest.approx(0.05, abs=0.002)


def test_naive_daily_peeking_inflates_type_i_error():
    # the motivating failure: 14 looks at a fixed z=1.96 reject on about 22%
    # of null runs, roughly four times the nominal 5%
    random_generator = np.random.default_rng(42)
    n_sims = 50_000
    fractions = np.linspace(1 / 14, 1.0, 14)
    increments = random_generator.normal(size=(n_sims, 14)) * np.sqrt(
        np.diff(fractions, prepend=0.0)
    )
    z_paths = np.cumsum(increments, axis=1) / np.sqrt(fractions)
    naive = np.any(np.abs(z_paths) >= 1.96, axis=1).mean()
    assert naive > 0.15

    sequential_design = design_boundaries(fractions.tolist(), alpha=0.05)
    controlled = np.any(np.abs(z_paths) >= np.array(sequential_design.boundaries), axis=1).mean()
    assert controlled == pytest.approx(0.05, abs=0.005)
    # the README quotes these two rates from this seed, so they are pinned here
    assert naive == pytest.approx(0.218, abs=0.0005)
    assert controlled == pytest.approx(0.049, abs=0.0005)


def test_design_validation():
    with pytest.raises(ValueError):
        design_boundaries([])
    with pytest.raises(ValueError):
        design_boundaries([0.5, 0.5, 1.0])
    with pytest.raises(ValueError):
        design_boundaries([0.5, 1.2])
    with pytest.raises(ValueError):
        design_boundaries([0.0, 1.0])
    with pytest.raises(ValueError):
        design_boundaries([1.0], alpha=0.0)
    with pytest.raises(ValueError):
        design_boundaries([1.0], spending="bayes")
    with pytest.raises(ValueError):
        design_boundaries([1.0], grid_points=10)
    with pytest.raises(ValueError):
        design_boundaries([1.0], grid_points=100_000)
    with pytest.raises(ValueError):
        design_boundaries([0.5, np.nan, 1.0])


def test_monitor_stops_at_first_crossing():
    sequential_design = design_boundaries([0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="pocock")
    z_scores = [0.5, sequential_design.boundaries[1] + 0.1, 5.0, 0.0]
    outcome = monitor_boundaries(z_scores, sequential_design)
    assert outcome.stopped_at_look == 2

    quiet = monitor_boundaries([0.5, 1.0, -1.2, 1.5], sequential_design)
    assert quiet.stopped_at_look is None

    partial = monitor_boundaries([0.5, 1.0], sequential_design)
    assert partial.stopped_at_look is None
    assert len(partial.boundaries) == 2

    with pytest.raises(ValueError):
        monitor_boundaries([0.0] * 5, sequential_design)
    with pytest.raises(ValueError):
        monitor_boundaries([0.5, np.nan], sequential_design)


def test_monitor_stops_on_a_harmful_treatment():
    # the boundaries are two-sided, so a treatment that hurts must stop the
    # experiment just as readily as one that helps. Comparing z rather than
    # |z| against the boundary would pass every other test in this file.
    sequential_design = design_boundaries([0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="pocock")
    harmful = [-0.5, -(sequential_design.boundaries[1] + 0.1), -5.0, 0.0]
    assert monitor_boundaries(harmful, sequential_design).stopped_at_look == 2

    mirrored = [-z_score for z_score in harmful]
    assert (
        monitor_boundaries(mirrored, sequential_design).stopped_at_look
        == monitor_boundaries(harmful, sequential_design).stopped_at_look
    )


def test_monitor_boundary_is_inclusive_and_tight():
    sequential_design = design_boundaries([0.5, 1.0], alpha=0.05, spending="pocock")
    assert (
        monitor_boundaries(
            [sequential_design.boundaries[0], 0.0], sequential_design
        ).stopped_at_look
        == 1
    )
    assert (
        monitor_boundaries(
            [sequential_design.boundaries[0] - 1e-9, 0.0], sequential_design
        ).stopped_at_look
        is None
    )


def test_z_from_cumulative_counts_matches_scalar_test():
    from abkit.significance_tests import two_proportion_ztest

    z_scores = z_from_cumulative_counts([50, 100], [1000, 2000], [70, 130], [1000, 2000])
    expected_final = two_proportion_ztest(100, 2000, 130, 2000).statistic
    assert z_scores[-1] == pytest.approx(expected_final, rel=1e-12)


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


def test_repeated_intervals_cover_jointly_by_simulation():
    # 14 daily looks under a null effect: the share of paths on which every
    # look's adjusted interval covers zero must be at least 1 - alpha, up to
    # Monte Carlo error. Each path's estimate at look k is the running mean
    # of n_k standard normal draws, with standard error 1 / sqrt(n_k).
    fractions = np.linspace(1 / 14, 1.0, 14)
    sequential_design = design_boundaries(fractions.tolist(), alpha=0.05)
    random_generator = np.random.default_rng(43)
    n_sims = 20_000
    final_n = 14_000
    look_sizes = fractions * final_n
    increments = random_generator.normal(size=(n_sims, 14)) * np.sqrt(
        np.diff(look_sizes, prepend=0.0)
    )
    running_means = np.cumsum(increments, axis=1) / look_sizes
    standard_errors = 1.0 / np.sqrt(look_sizes)
    covered_everywhere = 0
    for path_means in running_means:
        intervals = repeated_intervals(path_means, standard_errors, sequential_design)
        covered_everywhere += all(
            low <= 0.0 <= high
            for low, high in zip(intervals.ci_low, intervals.ci_high, strict=True)
        )
    monte_carlo_standard_error = np.sqrt(0.05 * 0.95 / n_sims)
    assert covered_everywhere / n_sims >= 0.95 - 3 * monte_carlo_standard_error
    # the first look's boundary is infinite under this design, so its interval is unbounded
    assert intervals.ci_low[0] == -np.inf and intervals.ci_high[0] == np.inf
    assert intervals.alpha == 0.05


def test_repeated_intervals_are_the_boundary_times_the_standard_error():
    sequential_design = design_boundaries([0.5, 1.0], alpha=0.05, spending="pocock")
    intervals = repeated_intervals([0.1, 0.2], [0.05, 0.04], sequential_design)
    first_boundary, second_boundary = sequential_design.boundaries
    assert intervals.estimates == (0.1, 0.2)
    assert intervals.ci_low[0] == pytest.approx(0.1 - first_boundary * 0.05)
    assert intervals.ci_high[1] == pytest.approx(0.2 + second_boundary * 0.04)
    # wider than the fixed-sample interval at every look, which is the price of peeking
    assert intervals.ci_high[1] - intervals.ci_low[1] > 2 * stats.norm.isf(0.025) * 0.04


def test_repeated_intervals_reject_mismatched_or_bad_input():
    sequential_design = design_boundaries([0.5, 1.0], alpha=0.05)
    with pytest.raises(ValueError):  # one look short
        repeated_intervals([0.1], [0.05], sequential_design)
    with pytest.raises(ValueError):  # errors and estimates disagree
        repeated_intervals([0.1, 0.2], [0.05], sequential_design)
    with pytest.raises(ValueError):  # a zero standard error
        repeated_intervals([0.1, 0.2], [0.05, 0.0], sequential_design)
    with pytest.raises(ValueError):  # a non-finite estimate
        repeated_intervals([0.1, np.nan], [0.05, 0.04], sequential_design)


def test_a_boundary_never_changes_when_later_looks_are_added():
    # what makes a live read valid: the boundary at each look depends on the
    # looks before it and the plan, never on looks still to come
    fractions = [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    full = design_boundaries(fractions, alpha=0.05)
    for n_looks in range(1, len(fractions)):
        prefix = design_boundaries(fractions[:n_looks], alpha=0.05)
        np.testing.assert_allclose(prefix.boundaries, full.boundaries[:n_looks], atol=1e-6)

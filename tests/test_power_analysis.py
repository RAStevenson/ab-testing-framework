import numpy as np
import pytest
from scipy import stats

from abkit.power_analysis import (
    mde_two_proportions,
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
    simulate_power,
)
from abkit.significance_tests import two_proportion_ztest, welch_ttest


def test_sample_size_proportions_near_textbook_value():
    # 5.0% -> 5.5%, alpha 0.05, power 0.8: the closed form gives ~31,240
    n_per_arm = sample_size_two_proportions(0.05, 0.055, alpha=0.05, power=0.8)
    assert 30_000 <= n_per_arm <= 32_500
    assert power_two_proportions(0.05, 0.055, n_per_arm) >= 0.8
    assert power_two_proportions(0.05, 0.055, n_per_arm - 1) < 0.8


def test_power_proportions_matches_simulation():
    analytic = power_two_proportions(0.10, 0.12, 1500)

    def one_experiment(random_generator: np.random.Generator) -> float:
        control_conversions = random_generator.binomial(1500, 0.10)
        treatment_conversions = random_generator.binomial(1500, 0.12)
        return two_proportion_ztest(control_conversions, 1500, treatment_conversions, 1500).p_value

    simulated = simulate_power(
        one_experiment, random_generator=np.random.default_rng(51), n_sims=3000, alpha=0.05
    )
    assert simulated == pytest.approx(analytic, abs=0.025)


def _exact_power_two_proportions(
    rate_control, rate_treatment, n_per_arm, alpha=0.05, n_treatment=None
):
    """Rejection probability of the pooled z-test, summed over the binomial grid.

    No Monte Carlo error and no shared code with power_analysis.py, so this pins the
    specific variance placement the module documents (pooled under the null,
    unpooled under the alternative) rather than merely the right ballpark.
    """
    n_control = n_per_arm
    n_treatment = n_per_arm if n_treatment is None else n_treatment
    control_success_counts = np.arange(n_control + 1)
    treatment_success_counts = np.arange(n_treatment + 1)
    joint = np.outer(
        stats.binom.pmf(control_success_counts, n_control, rate_control),
        stats.binom.pmf(treatment_success_counts, n_treatment, rate_treatment),
    )
    pooled = (control_success_counts[:, None] + treatment_success_counts[None, :]) / (
        n_control + n_treatment
    )
    standard_error = np.sqrt(pooled * (1.0 - pooled) * (1 / n_control + 1 / n_treatment))
    # se is zero only where both arms are all-failure or all-success, which
    # makes the difference zero too: 0/0 is NaN, NaN >= threshold is False,
    # and that corner correctly counts as no rejection. A recorded run with
    # this block removed (2026-10-05) raised only this invalid-divide warning.
    with np.errstate(invalid="ignore"):
        z_statistics = (
            treatment_success_counts[None, :] / n_treatment
            - control_success_counts[:, None] / n_control
        ) / standard_error
    return float(np.sum(joint * (np.abs(z_statistics) >= stats.norm.isf(alpha / 2))))


def test_power_proportions_matches_exact_binomial_calculation():
    # 0.30 vs 0.45 at n=150 is the discriminating case: using the pooled
    # variance in both places, or the unpooled one in both, moves the answer
    # by 3e-3 to 7e-3, while the documented placement lands within 1e-4.
    analytic = power_two_proportions(0.30, 0.45, 150)
    exact = _exact_power_two_proportions(0.30, 0.45, 150)
    assert analytic == pytest.approx(exact, abs=1e-3)


def test_power_proportions_unequal_arms_matches_exact_binomial_calculation():
    # a 1:3 split in both orientations. The documented formula lands within
    # 2.7e-3 of exact; an unweighted pooled rate misses by 7.7e-3 or more and
    # unpooled variance in both places by 2e-2 or more, so 3e-3 discriminates
    # the weighting as well as the placement.
    for n_control, n_treatment in ((100, 300), (300, 100)):
        analytic = power_two_proportions(0.30, 0.45, n_control, n_treatment=n_treatment)
        exact = _exact_power_two_proportions(0.30, 0.45, n_control, n_treatment=n_treatment)
        assert analytic == pytest.approx(exact, abs=3e-3)


def test_power_is_monotone():
    powers = [power_two_proportions(0.05, 0.06, n_per_arm) for n_per_arm in (1000, 5000, 20_000)]
    assert powers[0] < powers[1] < powers[2]
    assert power_two_proportions(0.05, 0.07, 5000) > power_two_proportions(0.05, 0.06, 5000)


def test_mde_round_trips_with_sample_size():
    n_per_arm = sample_size_two_proportions(0.05, 0.055)
    mde = mde_two_proportions(0.05, n_per_arm)
    assert mde == pytest.approx(0.005, rel=0.02)
    assert power_two_proportions(0.05, 0.05 + mde, n_per_arm) == pytest.approx(0.8, abs=1e-6)


def test_mde_unequal_arms_round_trips_and_sits_between_the_equal_designs():
    mde = mde_two_proportions(0.05, 10_000, n_treatment=100_000)
    assert power_two_proportions(0.05, 0.05 + mde, 10_000, n_treatment=100_000) == pytest.approx(
        0.8, abs=1e-6
    )
    # a big second arm helps, but the small arm still limits what is detectable
    assert mde_two_proportions(0.05, 100_000) < mde < mde_two_proportions(0.05, 10_000)
    assert mde_two_proportions(0.05, 10_000, n_treatment=10_000) == mde_two_proportions(
        0.05, 10_000
    )


def test_mde_raises_when_n_hopeless():
    with pytest.raises(ValueError):
        mde_two_proportions(0.5, 2)


def test_mde_raises_rather_than_leaking_a_solver_error():
    # power tends to alpha/2 as the lift vanishes, so a target at or below
    # that has no root; this used to surface as a raw brentq bracket error
    with pytest.raises(ValueError, match="alpha/2"):
        mde_two_proportions(0.05, 50_000, alpha=0.05, power=0.02)


def test_sample_size_means_is_minimal_on_knife_edge_designs():
    # when the closed form lands exactly on an integer, floating-point error
    # can push ceil() one past it; walking up alone would never come back
    for target_n in (10, 50, 97, 1000):
        z_critical = stats.norm.isf(0.025)
        z_power = stats.norm.isf(1.0 - 0.8)
        difference = (z_critical + z_power) * np.sqrt(2.0 / target_n)
        n_per_arm = sample_size_two_means(difference, 1.0)
        assert n_per_arm == target_n
        assert power_two_means(difference, 1.0, n_per_arm) >= 0.8
        assert power_two_means(difference, 1.0, n_per_arm - 1) < 0.8


def test_sample_size_means_near_textbook_value():
    # delta 0.1 standard deviations, alpha 0.05, power 0.8: classic answer is ~1,571 per arm
    n_per_arm = sample_size_two_means(0.1, 1.0)
    assert 1550 <= n_per_arm <= 1600
    assert power_two_means(0.1, 1.0, n_per_arm) >= 0.8
    # minimality, not just sufficiency. Without this the refinement loop hides
    # a closed form that overshoots, since it only ever walks upward.
    assert power_two_means(0.1, 1.0, n_per_arm - 1) < 0.8


def test_sample_size_means_matches_the_closed_form_exactly():
    # the normal-approximation formula is exact for this power function, so
    # the returned n must equal its ceiling rather than merely bracket it
    for difference, standard_deviation, power in (
        (0.1, 1.0, 0.8),
        (0.5, 2.0, 0.9),
        (2.0, 3.5, 0.95),
    ):
        z_critical = stats.norm.isf(0.025)
        z_power = stats.norm.isf(1.0 - power)
        expected = int(
            np.ceil(2.0 * ((z_critical + z_power) * standard_deviation / abs(difference)) ** 2)
        )
        assert sample_size_two_means(difference, standard_deviation, power=power) == expected


def test_power_means_matches_simulation():
    analytic = power_two_means(0.15, 1.0, 800)

    def one_experiment(random_generator: np.random.Generator) -> float:
        control_values = random_generator.normal(0.0, 1.0, 800)
        treatment_values = random_generator.normal(0.15, 1.0, 800)
        return welch_ttest(control_values, treatment_values).p_value

    simulated = simulate_power(
        one_experiment, random_generator=np.random.default_rng(52), n_sims=2000, alpha=0.05
    )
    assert simulated == pytest.approx(analytic, abs=0.03)


def test_simulate_power_counts_the_boundary_as_a_rejection():
    # a level-alpha test rejects at p == alpha. Discrete p-values can land
    # there exactly, so a strict comparison would silently understate power.
    random_generator = np.random.default_rng(53)
    assert (
        simulate_power(lambda _: 0.05, random_generator=random_generator, n_sims=10, alpha=0.05)
        == 1.0
    )
    assert (
        simulate_power(
            lambda _: 0.0500001, random_generator=random_generator, n_sims=10, alpha=0.05
        )
        == 0.0
    )


def test_validation_errors():
    with pytest.raises(ValueError):
        power_two_proportions(0.05, 0.05, 1000)
    with pytest.raises(ValueError):
        power_two_proportions(0.0, 0.05, 1000)
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.05, 0.06, power=1.0)
    with pytest.raises(ValueError):
        power_two_means(0.0, 1.0, 100)
    with pytest.raises(ValueError):
        sample_size_two_means(0.1, -1.0)
    with pytest.raises(ValueError):
        simulate_power(
            lambda random_generator: 0.5, random_generator=np.random.default_rng(0), n_sims=0
        )

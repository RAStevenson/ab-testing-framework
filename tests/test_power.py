import numpy as np
import pytest

from abkit.frequentist import two_proportion_ztest, welch_ttest
from abkit.power import (
    mde_two_proportions,
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
    simulate_power,
)


def test_sample_size_proportions_near_textbook_value():
    # 5.0% -> 5.5%, alpha 0.05, power 0.8: the closed form gives ~31,240
    n = sample_size_two_proportions(0.05, 0.055, alpha=0.05, power=0.8)
    assert 30_000 <= n <= 32_500
    assert power_two_proportions(0.05, 0.055, n) >= 0.8
    assert power_two_proportions(0.05, 0.055, n - 1) < 0.8


def test_power_proportions_matches_simulation():
    analytic = power_two_proportions(0.10, 0.12, 1500)

    def one_experiment(rng: np.random.Generator) -> float:
        x = rng.binomial(1500, 0.10)
        y = rng.binomial(1500, 0.12)
        return two_proportion_ztest(x, 1500, y, 1500).p_value

    simulated = simulate_power(
        one_experiment, rng=np.random.default_rng(51), n_sims=3000, alpha=0.05
    )
    assert simulated == pytest.approx(analytic, abs=0.025)


def test_power_is_monotone():
    powers = [power_two_proportions(0.05, 0.06, n) for n in (1000, 5000, 20_000)]
    assert powers[0] < powers[1] < powers[2]
    assert power_two_proportions(0.05, 0.07, 5000) > power_two_proportions(0.05, 0.06, 5000)


def test_mde_round_trips_with_sample_size():
    n = sample_size_two_proportions(0.05, 0.055)
    mde = mde_two_proportions(0.05, n)
    assert mde == pytest.approx(0.005, rel=0.02)
    assert power_two_proportions(0.05, 0.05 + mde, n) == pytest.approx(0.8, abs=1e-6)


def test_mde_raises_when_n_hopeless():
    with pytest.raises(ValueError):
        mde_two_proportions(0.5, 2)


def test_sample_size_means_near_textbook_value():
    # delta 0.1 sd, alpha 0.05, power 0.8: classic answer is ~1,571 per arm
    n = sample_size_two_means(0.1, 1.0)
    assert 1550 <= n <= 1600
    assert power_two_means(0.1, 1.0, n) >= 0.8


def test_power_means_matches_simulation():
    analytic = power_two_means(0.15, 1.0, 800)

    def one_experiment(rng: np.random.Generator) -> float:
        x = rng.normal(0.0, 1.0, 800)
        y = rng.normal(0.15, 1.0, 800)
        return welch_ttest(x, y).p_value

    simulated = simulate_power(
        one_experiment, rng=np.random.default_rng(52), n_sims=2000, alpha=0.05
    )
    assert simulated == pytest.approx(analytic, abs=0.03)


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
        simulate_power(lambda rng: 0.5, rng=np.random.default_rng(0), n_sims=0)

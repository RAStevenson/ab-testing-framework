import numpy as np
import pytest

from abkit.frequentist import srm_check
from abkit.generators import generate_binary, generate_continuous


def test_binary_rates_match_configuration():
    rng = np.random.default_rng(7)
    data = generate_binary(400_000, 0.10, 0.02, rng=rng)
    assert data.control.mean() == pytest.approx(0.10, abs=0.004)
    assert data.treatment.mean() == pytest.approx(0.12, abs=0.004)


def test_binary_allocation_respected():
    rng = np.random.default_rng(8)
    data = generate_binary(100_000, 0.05, 0.0, rng=rng, allocation=0.3)
    share = data.n_treatment / (data.n_control + data.n_treatment)
    assert share == pytest.approx(0.3, abs=0.01)


def test_novelty_effect_decays():
    rng = np.random.default_rng(9)
    data = generate_binary(
        600_000, 0.10, 0.0, rng=rng, n_days=14, novelty_boost=0.05, novelty_half_life_days=3.0
    )
    early = data.subset(data.day < 3)
    late = data.subset(data.day >= 10)
    early_lift = early.treatment.mean() - early.control.mean()
    late_lift = late.treatment.mean() - late.control.mean()
    assert early_lift > 0.025
    assert late_lift < 0.01


def test_treatment_loss_inflates_rate_and_trips_srm():
    rng = np.random.default_rng(10)
    data = generate_binary(200_000, 0.10, 0.0, rng=rng, treatment_loss_rate=0.05)
    # dropped users are all non-converters, so the observed rate is biased up
    assert data.treatment.mean() > 0.102
    srm = srm_check(data.n_control, data.n_treatment)
    assert srm.p_value < 1e-6


def test_clean_experiment_passes_srm():
    rng = np.random.default_rng(11)
    data = generate_binary(200_000, 0.10, 0.0, rng=rng)
    assert srm_check(data.n_control, data.n_treatment).p_value > 0.001


def test_continuous_is_zero_inflated_and_heavy_tailed():
    rng = np.random.default_rng(12)
    data = generate_continuous(300_000, rng=rng, spend_rate=0.08)
    spend_share = np.mean(data.outcome > 0)
    assert spend_share == pytest.approx(0.08, abs=0.005)
    spenders = data.outcome[data.outcome > 0]
    # a Pareto(1.5) tail puts the max orders of magnitude above the median
    assert spenders.max() > 100 * np.median(spenders)


def test_continuous_uplift_shifts_spender_mean():
    rng = np.random.default_rng(13)
    data = generate_continuous(800_000, rng=rng, spend_uplift=0.30, whale_rate=0.0, log_sigma=0.8)
    control_spenders = data.control[data.control > 0]
    treatment_spenders = data.treatment[data.treatment > 0]
    ratio = treatment_spenders.mean() / control_spenders.mean()
    assert ratio == pytest.approx(1.30, abs=0.03)


def test_subset_preserves_alignment():
    rng = np.random.default_rng(14)
    data = generate_binary(10_000, 0.5, 0.0, rng=rng, n_days=7)
    early = data.subset(data.day < 3)
    assert np.all(early.day < 3)
    assert early.assignment.size == early.outcome.size == early.day.size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_users": 0},
        {"allocation": 0.0},
        {"allocation": 1.0},
        {"base_rate": 1.2},
        {"absolute_lift": 0.95},  # peak rate above 1
        {"treatment_loss_rate": 1.0},
        {"novelty_half_life_days": 0.0},
        {"n_days": 0},
    ],
)
def test_binary_rejects_bad_parameters(kwargs):
    defaults = {"n_users": 100, "base_rate": 0.1, "absolute_lift": 0.01}
    call = {**defaults, **{name: value for name, value in kwargs.items() if name in defaults}}
    extra = {name: value for name, value in kwargs.items() if name not in defaults}
    with pytest.raises(ValueError):
        generate_binary(
            call["n_users"],
            call["base_rate"],
            call["absolute_lift"],
            rng=np.random.default_rng(0),
            **extra,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spend_rate": -0.1},
        {"spend_rate_lift": 0.95},
        {"whale_shape": 1.0},
        {"whale_scale": 0.0},
        {"log_sigma": 0.0},
        {"spend_uplift": -1.0},
    ],
)
def test_continuous_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_continuous(100, rng=np.random.default_rng(0), **kwargs)

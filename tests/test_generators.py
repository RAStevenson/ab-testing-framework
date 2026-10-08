import numpy as np
import pytest

from abkit.frequentist import srm_check
from abkit.generators import generate_binary, generate_continuous, generate_example_rows


def test_binary_rates_match_configuration():
    random_generator = np.random.default_rng(7)
    experiment_data = generate_binary(400_000, 0.10, 0.02, random_generator=random_generator)
    assert experiment_data.control.mean() == pytest.approx(0.10, abs=0.004)
    assert experiment_data.treatment.mean() == pytest.approx(0.12, abs=0.004)


def test_binary_allocation_respected():
    random_generator = np.random.default_rng(8)
    experiment_data = generate_binary(
        100_000, 0.05, 0.0, random_generator=random_generator, allocation=0.3
    )
    share = experiment_data.n_treatment / (experiment_data.n_control + experiment_data.n_treatment)
    assert share == pytest.approx(0.3, abs=0.01)


def test_novelty_effect_decays():
    random_generator = np.random.default_rng(9)
    experiment_data = generate_binary(
        600_000,
        0.10,
        0.0,
        random_generator=random_generator,
        n_days=14,
        novelty_boost=0.05,
        novelty_half_life_days=3.0,
    )
    early = experiment_data.subset(experiment_data.day < 3)
    late = experiment_data.subset(experiment_data.day >= 10)
    early_lift = early.treatment.mean() - early.control.mean()
    late_lift = late.treatment.mean() - late.control.mean()
    assert early_lift > 0.025
    assert late_lift < 0.01


def test_treatment_loss_inflates_rate_and_trips_srm():
    random_generator = np.random.default_rng(10)
    experiment_data = generate_binary(
        200_000, 0.10, 0.0, random_generator=random_generator, treatment_loss_rate=0.05
    )
    # dropped users are all non-converters, so the observed rate is biased up
    assert experiment_data.treatment.mean() > 0.102
    srm = srm_check(experiment_data.n_control, experiment_data.n_treatment)
    assert srm.p_value < 1e-6


def test_clean_experiment_passes_srm():
    random_generator = np.random.default_rng(11)
    experiment_data = generate_binary(200_000, 0.10, 0.0, random_generator=random_generator)
    assert srm_check(experiment_data.n_control, experiment_data.n_treatment).p_value > 0.001


def test_continuous_is_zero_inflated_and_heavy_tailed():
    random_generator = np.random.default_rng(12)
    experiment_data = generate_continuous(
        300_000, random_generator=random_generator, spend_rate=0.08
    )
    spend_share = np.mean(experiment_data.outcome > 0)
    assert spend_share == pytest.approx(0.08, abs=0.005)
    spenders = experiment_data.outcome[experiment_data.outcome > 0]
    # a Pareto(1.5) tail puts the max orders of magnitude above the median
    assert spenders.max() > 100 * np.median(spenders)


def test_continuous_uplift_shifts_spender_mean():
    random_generator = np.random.default_rng(13)
    experiment_data = generate_continuous(
        800_000, random_generator=random_generator, spend_uplift=0.30, whale_rate=0.0, log_sigma=0.8
    )
    control_spenders = experiment_data.control[experiment_data.control > 0]
    treatment_spenders = experiment_data.treatment[experiment_data.treatment > 0]
    ratio = treatment_spenders.mean() / control_spenders.mean()
    assert ratio == pytest.approx(1.30, abs=0.03)


def test_subset_preserves_alignment():
    random_generator = np.random.default_rng(14)
    experiment_data = generate_binary(10_000, 0.5, 0.0, random_generator=random_generator, n_days=7)
    early = experiment_data.subset(experiment_data.day < 3)
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
            random_generator=np.random.default_rng(0),
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
        generate_continuous(100, random_generator=np.random.default_rng(0), **kwargs)


def _example_rows(n_users: int, seed: int, **overrides: float) -> dict:
    settings = {
        "base_rate": 0.06,
        "absolute_lift": 0.004,
        "n_days": 14,
        "novelty_boost": 0.0,
        "novelty_half_life_days": 3.0,
        "spend_rate": 0.06,
        "spend_rate_lift": 0.02,
        "log_mean": 3.4,
        "log_sigma": 1.3,
        "whale_rate": 0.0,
        "whale_shape": 1.5,
        "whale_scale": 400.0,
        "pre_correlation": 0.6,
    }
    settings.update(overrides)
    return generate_example_rows(n_users, np.random.default_rng(seed), **settings)


def test_example_rows_share_one_assignment_across_columns():
    rows = _example_rows(200_000, seed=15, absolute_lift=0.03)
    assert set(rows) == {"user_id", "group", "day", "converted", "spend", "pre_spend"}
    assert len({column.size for column in rows.values()}) == 1
    assert len(set(rows["user_id"])) == rows["user_id"].size
    treated = rows["group"] == "treatment"
    assert treated.mean() == pytest.approx(0.5, abs=0.01)
    # both planted lifts land on the same group mask, so the assignment is shared
    conversion_lift = rows["converted"][treated].mean() - rows["converted"][~treated].mean()
    spend_rate_lift = (rows["spend"][treated] > 0).mean() - (rows["spend"][~treated] > 0).mean()
    assert conversion_lift == pytest.approx(0.03, abs=0.004)
    assert spend_rate_lift == pytest.approx(0.02, abs=0.004)
    assert np.all((rows["day"] >= 0) & (rows["day"] < 14))


def test_pre_spend_correlates_with_the_spend_propensity_as_planted():
    rows = _example_rows(200_000, seed=16)
    spenders = rows["spend"] > 0
    correlation = np.corrcoef(np.log(rows["spend"][spenders]), np.log(rows["pre_spend"][spenders]))
    assert correlation[0, 1] == pytest.approx(0.6, abs=0.02)
    # everyone has a pre-period value, spenders or not
    assert np.all(rows["pre_spend"] > 0)


def test_example_rows_reject_bad_parameters():
    with pytest.raises(ValueError):
        _example_rows(100, seed=0, pre_correlation=1.0)
    with pytest.raises(ValueError):
        _example_rows(100, seed=0, absolute_lift=0.95)
    with pytest.raises(ValueError):
        _example_rows(0, seed=0)

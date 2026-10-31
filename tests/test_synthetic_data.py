import numpy as np
import pytest

from abkit.significance_tests import srm_check
from abkit.synthetic_data import generate_binary, generate_check_rows, generate_continuous


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


def _check_rows(n_users: int, seed: int, **overrides: float) -> dict:
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
    return generate_check_rows(n_users, np.random.default_rng(seed), **settings)


def test_check_rows_share_one_assignment_across_columns():
    rows = _check_rows(200_000, seed=15, absolute_lift=0.03)
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


def test_pre_spend_shares_the_planted_link_with_spend():
    rows = _check_rows(200_000, seed=16)
    spends_now, spent_before = rows["spend"] > 0, rows["pre_spend"] > 0
    # the amounts: among users who spent in both periods, the log amounts
    # correlate at the planted value (no whales in this call)
    both = spends_now & spent_before
    correlation = np.corrcoef(np.log(rows["spend"][both]), np.log(rows["pre_spend"][both]))
    assert correlation[0, 1] == pytest.approx(0.6, abs=0.02)
    # the decision: a past spender is far likelier to spend now, while the
    # planted rates themselves hold (control 6%, pre-period 6%)
    assert spends_now[spent_before].mean() > 3 * spends_now[~spent_before].mean()
    control = rows["group"] == "control"
    assert spends_now[control].mean() == pytest.approx(0.06, abs=0.003)
    assert spent_before.mean() == pytest.approx(0.06, abs=0.003)


def _expected_cuped_share_without_whales(spend_rate: float, log_sigma: float, rho: float) -> float:
    """The share of spend's variance a pre-experiment column removes, in closed form, for the
    whale-free recipe: spend and pre_spend are each a spend decision (a normal latent below the
    spend-rate quantile) times a lognormal amount, with both latents correlated at rho. The
    squared correlation of the two products is the share; log_mean cancels out of it."""
    from scipy import stats

    cutoff = stats.norm.ppf(spend_rate)
    both_spend = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf(  # pyright: ignore[reportArgumentType]
        [cutoff, cutoff]
    )
    mean = spend_rate * np.exp(log_sigma**2 / 2.0)
    second_moment = spend_rate * np.exp(2.0 * log_sigma**2)
    cross_moment = both_spend * np.exp(log_sigma**2 * (1.0 + rho))
    correlation = (cross_moment - mean**2) / (second_moment - mean**2)
    return float(correlation**2)


def test_cuped_share_matches_the_closed_form_without_whales_and_drops_with_them():
    from abkit.cuped_adjustment import cuped_mean_diff

    def share(whale_rate: float) -> float:
        rows = _check_rows(400_000, seed=17, whale_rate=whale_rate, pre_correlation=0.9)
        treated = rows["group"] == "treatment"
        return cuped_mean_diff(
            rows["spend"][~treated],
            rows["spend"][treated],
            rows["pre_spend"][~treated],
            rows["pre_spend"][treated],
        ).variance_reduction

    expected = _expected_cuped_share_without_whales(0.06, 1.3, 0.9)
    assert expected == pytest.approx(0.296, abs=0.001)
    # measured 2026-10-09 over eight seeds at 400,000 users: 0.273 to 0.337, standard
    # deviation about 0.026, so 0.08 is three standard deviations; the lognormal amount
    # itself is heavy enough that one sample's share wanders this much
    assert share(0.0) == pytest.approx(expected, abs=0.08)
    # whale purchases have no pre-period counterpart and infinite variance, so they can
    # only pull the share down; across the same eight seeds it ran 0.12 to 0.18
    assert 0.05 < share(0.008) < expected


def test_check_rows_reject_bad_parameters():
    with pytest.raises(ValueError):
        _check_rows(100, seed=0, pre_correlation=1.0)
    with pytest.raises(ValueError):
        _check_rows(100, seed=0, absolute_lift=0.95)
    with pytest.raises(ValueError):
        _check_rows(0, seed=0)

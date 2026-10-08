import numpy as np
import pytest

from abkit.cuped_adjustment import cuped_mean_diff
from abkit.frequentist import welch_ttest


def _planted_arms(random_generator: np.random.Generator, n_per_arm: int, slope: float):
    """Two arms whose outcome is slope * pre + noise, with a 0.1 treatment effect planted."""
    control_pre = random_generator.normal(0.0, 1.0, n_per_arm)
    treatment_pre = random_generator.normal(0.0, 1.0, n_per_arm)
    control_values = slope * control_pre + random_generator.normal(0.0, 0.6, n_per_arm)
    treatment_values = slope * treatment_pre + random_generator.normal(0.0, 0.6, n_per_arm) + 0.1
    return control_values, treatment_values, control_pre, treatment_pre


def test_variance_reduction_matches_the_planted_correlation():
    # outcome = 0.8 * pre + N(0, 0.6): the pre column explains 0.64 of the
    # outcome's unit variance, so the adjustment should remove about that share
    control_values, treatment_values, control_pre, treatment_pre = _planted_arms(
        np.random.default_rng(61), 20_000, slope=0.8
    )
    result = cuped_mean_diff(control_values, treatment_values, control_pre, treatment_pre)
    assert result.variance_reduction == pytest.approx(0.64, abs=0.03)
    assert result.pre_period_coefficient == pytest.approx(0.8, abs=0.02)
    assert abs(result.difference - 0.1) < 3 * result.standard_error
    assert result.standard_error < result.unadjusted_standard_error
    assert result.ci_low < result.difference < result.ci_high
    assert result.p_value < 1e-6


def test_uncorrelated_pre_column_removes_nothing():
    control_values, treatment_values, control_pre, treatment_pre = _planted_arms(
        np.random.default_rng(62), 20_000, slope=0.0
    )
    result = cuped_mean_diff(control_values, treatment_values, control_pre, treatment_pre)
    assert result.variance_reduction == pytest.approx(0.0, abs=0.02)
    # with nothing to adjust by, the estimate sits on the plain difference
    welch = welch_ttest(control_values, treatment_values)
    assert result.difference == pytest.approx(welch.difference, abs=3e-3)
    assert result.unadjusted_standard_error == pytest.approx(
        (welch.ci_high - welch.ci_low) / (2 * 1.959964), rel=0.02
    )


def test_confidence_level_widens_the_interval():
    control_values, treatment_values, control_pre, treatment_pre = _planted_arms(
        np.random.default_rng(63), 2_000, slope=0.5
    )
    widths = []
    for confidence in (0.80, 0.95, 0.99):
        result = cuped_mean_diff(
            control_values, treatment_values, control_pre, treatment_pre, confidence=confidence
        )
        assert result.confidence == confidence
        widths.append(result.ci_high - result.ci_low)
    assert widths[0] < widths[1] < widths[2]


def test_cuped_rejects_bad_input():
    random_generator = np.random.default_rng(64)
    control_values = random_generator.normal(size=50)
    treatment_values = random_generator.normal(size=50)
    pre = random_generator.normal(size=50)
    with pytest.raises(ValueError):  # mismatched lengths within an arm
        cuped_mean_diff(control_values, treatment_values, pre[:49], pre)
    with pytest.raises(ValueError):  # constant pre column
        cuped_mean_diff(control_values, treatment_values, np.ones(50), np.ones(50))
    with pytest.raises(ValueError):  # non-finite outcome
        corrupted = control_values.copy()
        corrupted[3] = np.nan
        cuped_mean_diff(corrupted, treatment_values, pre, pre)
    with pytest.raises(ValueError):  # too few per arm
        cuped_mean_diff([1.0], [1.0, 2.0], [0.5], [0.1, 0.2])
    with pytest.raises(ValueError):  # confidence outside (0, 1)
        cuped_mean_diff(control_values, treatment_values, pre, pre, confidence=1.0)

"""Analytic power and sample-size utilities, plus a simulation cross-check.

The proportion formulas use the standard normal approximation with the
pooled variance under the null and the unpooled variance under the
alternative, and ignore the vanishing probability of rejecting in the
wrong direction. Tests pin them against simulation to within Monte Carlo
error.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import optimize, stats


def _validate_alpha_power(alpha: float, power: float | None = None) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")


def _validate_rates(rate_control: float, rate_treatment: float) -> None:
    for name, r in (("rate_control", rate_control), ("rate_treatment", rate_treatment)):
        if not 0.0 < r < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {r}")
    if rate_control == rate_treatment:
        raise ValueError("rates are equal; power against a zero effect is just alpha")


def power_two_proportions(
    rate_control: float,
    rate_treatment: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Power of the two-sided pooled z-test with equal arm sizes."""
    _validate_rates(rate_control, rate_treatment)
    _validate_alpha_power(alpha)
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")

    delta = abs(rate_treatment - rate_control)
    pooled = (rate_control + rate_treatment) / 2.0
    se_null = np.sqrt(2.0 * pooled * (1.0 - pooled) / n_per_arm)
    se_alt = np.sqrt(
        (rate_control * (1.0 - rate_control) + rate_treatment * (1.0 - rate_treatment)) / n_per_arm
    )
    z_crit = stats.norm.isf(alpha / 2.0)
    return float(stats.norm.sf((z_crit * se_null - delta) / se_alt))


def sample_size_two_proportions(
    rate_control: float,
    rate_treatment: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Smallest equal per-arm n reaching the target power for the z-test."""
    _validate_rates(rate_control, rate_treatment)
    _validate_alpha_power(alpha, power)

    delta = abs(rate_treatment - rate_control)
    pooled = (rate_control + rate_treatment) / 2.0
    z_crit = stats.norm.isf(alpha / 2.0)
    z_power = stats.norm.isf(1.0 - power)
    n = (
        (
            z_crit * np.sqrt(2.0 * pooled * (1.0 - pooled))
            + z_power
            * np.sqrt(rate_control * (1.0 - rate_control) + rate_treatment * (1.0 - rate_treatment))
        )
        / delta
    ) ** 2
    n_int = max(2, int(np.ceil(n)))
    # the closed form and the power function differ in how they place the
    # alternative; walk to the exact threshold of the power function we report
    while power_two_proportions(rate_control, rate_treatment, n_int, alpha=alpha) < power:
        n_int += 1
    while (
        n_int > 2
        and power_two_proportions(rate_control, rate_treatment, n_int - 1, alpha=alpha) >= power
    ):
        n_int -= 1
    return n_int


def mde_two_proportions(
    rate_control: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable absolute lift (upward) at the given n and power."""
    if not 0.0 < rate_control < 1.0:
        raise ValueError(f"rate_control must be in (0, 1), got {rate_control}")
    _validate_alpha_power(alpha, power)
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")

    upper = 1.0 - rate_control - 1e-9
    if power_two_proportions(rate_control, rate_control + upper, n_per_arm, alpha=alpha) < power:
        raise ValueError(
            f"n_per_arm={n_per_arm} cannot reach power {power} for any lift "
            f"from rate_control={rate_control}"
        )

    def gap(lift: float) -> float:
        return (
            power_two_proportions(rate_control, rate_control + lift, n_per_arm, alpha=alpha) - power
        )

    return float(optimize.brentq(gap, 1e-12, upper, xtol=1e-12))


def power_two_means(
    difference: float,
    sd: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Power of the two-sided two-sample test for means, common sd, equal n.

    Normal approximation (no t correction), fine for the arm sizes online
    experiments run at.
    """
    _validate_alpha_power(alpha)
    if sd <= 0.0:
        raise ValueError(f"sd must be positive, got {sd}")
    if difference == 0.0:
        raise ValueError("difference is zero; power against a zero effect is just alpha")
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")
    se = sd * np.sqrt(2.0 / n_per_arm)
    z_crit = stats.norm.isf(alpha / 2.0)
    return float(stats.norm.sf(z_crit - abs(difference) / se))


def sample_size_two_means(
    difference: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Smallest equal per-arm n reaching the target power for a mean difference."""
    _validate_alpha_power(alpha, power)
    if sd <= 0.0:
        raise ValueError(f"sd must be positive, got {sd}")
    if difference == 0.0:
        raise ValueError("difference is zero; no finite n reaches any power")
    z_crit = stats.norm.isf(alpha / 2.0)
    z_power = stats.norm.isf(1.0 - power)
    n = 2.0 * ((z_crit + z_power) * sd / abs(difference)) ** 2
    n_int = max(2, int(np.ceil(n)))
    while power_two_means(difference, sd, n_int, alpha=alpha) < power:
        n_int += 1
    return n_int


def simulate_power(
    draw_p_value: Callable[[np.random.Generator], float],
    *,
    rng: np.random.Generator,
    n_sims: int = 1000,
    alpha: float = 0.05,
) -> float:
    """Fraction of simulated experiments with p < alpha.

    ``draw_p_value`` runs one full simulated experiment and analysis and
    returns its p-value. Used to cross-check the analytic formulas and to
    get power for designs they do not cover (heavy tails, unequal arms).
    """
    _validate_alpha_power(alpha)
    if n_sims < 1:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    rejections = sum(draw_p_value(rng) < alpha for _ in range(n_sims))
    return rejections / n_sims

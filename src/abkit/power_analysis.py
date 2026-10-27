"""Analytic power and sample-size utilities, plus a simulation cross-check.

The power functions call statsmodels: the standard normal approximation
with the pooled variance under the null and the unpooled variance under
the alternative for proportions, and a common standard deviation for
means. statsmodels counts both tails of the two-sided test, so power is a
hair above the one-tail-only approximation at low power and the same at
the levels experiments plan for. Tests pin the proportion case against an
exact binomial calculation and both against simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy import optimize, stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import power_proportions_2indep


def _validate_alpha_power(alpha: float, power: float | None = None) -> None:
    """Raise if alpha, or power when given, is outside (0, 1)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")


def _validate_rates(rate_control: float, rate_treatment: float) -> None:
    """Raise if either rate is outside (0, 1) or the two rates are equal."""
    for name, rate in (("rate_control", rate_control), ("rate_treatment", rate_treatment)):
        if not 0.0 < rate < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {rate}")
    if rate_control == rate_treatment:
        raise ValueError("rates are equal; power against a zero effect is just alpha")


def power_two_proportions(
    rate_control: float,
    rate_treatment: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    n_treatment: int | None = None,
) -> float:
    """Power of the two-sided pooled z-test.

    The arms are equal at n_per_arm unless n_treatment is given, in which case
    n_per_arm is the control arm's size and n_treatment the treatment arm's.
    """
    _validate_rates(rate_control, rate_treatment)
    _validate_alpha_power(alpha)
    control_size = n_per_arm
    treatment_size = n_per_arm if n_treatment is None else n_treatment
    if control_size < 2 or treatment_size < 2:
        raise ValueError(
            f"each arm needs at least 2 users, got {control_size} and {treatment_size}"
        )

    # statsmodels calls the first sample the one at prop2 + diff, so the
    # treatment arm goes first and the control arm follows through the ratio;
    # it pools the null variance weighted by arm size and leaves the
    # alternative variance unpooled, the placement the exact-binomial test pins
    # statsmodels leaves the return untyped and pyright infers a union from
    # its branches; without the results object the call returns the power
    power = cast(
        float,
        power_proportions_2indep(
            rate_treatment - rate_control,
            rate_control,
            treatment_size,
            # statsmodels' default of 1 makes pyright read ratio as an int
            ratio=control_size / treatment_size,  # pyright: ignore[reportArgumentType]
            alpha=alpha,
            alternative="two-sided",
            return_results=False,
        ),
    )
    return float(power)


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
    z_critical = stats.norm.isf(alpha / 2.0)
    z_power = stats.norm.isf(1.0 - power)
    n_exact = (
        (
            z_critical * np.sqrt(2.0 * pooled * (1.0 - pooled))
            + z_power
            * np.sqrt(rate_control * (1.0 - rate_control) + rate_treatment * (1.0 - rate_treatment))
        )
        / delta
    ) ** 2
    n_rounded = max(2, int(np.ceil(n_exact)))
    # the closed form is the algebraic inverse of power_two_proportions, so
    # these walks normally take zero steps; they exist to absorb the case
    # where floating-point error puts ceil() on the wrong side of the integer
    while power_two_proportions(rate_control, rate_treatment, n_rounded, alpha=alpha) < power:
        n_rounded += 1
    while (
        n_rounded > 2
        and power_two_proportions(rate_control, rate_treatment, n_rounded - 1, alpha=alpha) >= power
    ):
        n_rounded -= 1
    return n_rounded


def mde_two_proportions(
    rate_control: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    n_treatment: int | None = None,
) -> float:
    """Minimum detectable absolute lift (upward) at the given n and power.

    The arms are equal at n_per_arm unless n_treatment is given, with the same
    meaning as in power_two_proportions.

    Power is unimodal rather than monotone in the lift: at rates near 1 the
    alternative variance collapses and power can fall again. The solve is
    still single-rooted because the feasibility check below requires power at
    the largest admissible lift to clear the target, which on a unimodal
    curve leaves exactly one crossing.
    """
    if not 0.0 < rate_control < 1.0:
        raise ValueError(f"rate_control must be in (0, 1), got {rate_control}")
    _validate_alpha_power(alpha, power)
    if n_per_arm < 2 or (n_treatment is not None and n_treatment < 2):
        raise ValueError(f"each arm needs at least 2 users, got {n_per_arm} and {n_treatment}")
    # power tends to alpha/2 as the lift vanishes, so a target at or below
    # that has no crossing at all and would surface as a raw solver error
    if power <= alpha / 2.0:
        raise ValueError(
            f"power={power} is not above alpha/2={alpha / 2.0}, which every "
            "vanishing lift already achieves; no minimum detectable effect exists"
        )

    upper = 1.0 - rate_control - 1e-9
    if (
        power_two_proportions(
            rate_control, rate_control + upper, n_per_arm, alpha=alpha, n_treatment=n_treatment
        )
        < power
    ):
        raise ValueError(
            f"n_per_arm={n_per_arm} cannot reach power {power} for any lift "
            f"from rate_control={rate_control}"
        )

    def gap(lift: float) -> float:
        """Power at this lift minus the target power, the function brentq solves for zero."""
        return (
            power_two_proportions(
                rate_control, rate_control + lift, n_per_arm, alpha=alpha, n_treatment=n_treatment
            )
            - power
        )

    # brentq returns a float unless full_output=True; scipy leaves it untyped,
    # so the checker infers the tuple form too
    root = cast(float, optimize.brentq(gap, 1e-12, upper, xtol=1e-12))
    return float(root)


def power_two_means(
    difference: float,
    standard_deviation: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Power of the two-sided two-sample test for means, common standard deviation, equal n.

    Normal approximation (no t correction), fine for the arm sizes online
    experiments run at. The effect size handed to statsmodels is the
    difference in standard-deviation units.
    """
    _validate_alpha_power(alpha)
    if standard_deviation <= 0.0:
        raise ValueError(f"standard_deviation must be positive, got {standard_deviation}")
    if difference == 0.0:
        raise ValueError("difference is zero; power against a zero effect is just alpha")
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")
    return float(
        NormalIndPower().power(
            effect_size=abs(difference) / standard_deviation,
            nobs1=n_per_arm,
            alpha=alpha,
            # statsmodels' default of 1 makes pyright read ratio as an int
            ratio=1.0,  # pyright: ignore[reportArgumentType]
            alternative="two-sided",
        )
    )


def sample_size_two_means(
    difference: float,
    standard_deviation: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Smallest equal per-arm n reaching the target power for a mean difference."""
    _validate_alpha_power(alpha, power)
    if standard_deviation <= 0.0:
        raise ValueError(f"standard_deviation must be positive, got {standard_deviation}")
    if difference == 0.0:
        raise ValueError("difference is zero; no finite n reaches any power")
    z_critical = stats.norm.isf(alpha / 2.0)
    z_power = stats.norm.isf(1.0 - power)
    n_exact = 2.0 * ((z_critical + z_power) * standard_deviation / abs(difference)) ** 2
    n_rounded = max(2, int(np.ceil(n_exact)))
    # both directions, for the same reason as sample_size_two_proportions:
    # when n* lands exactly on an integer, floating-point error can push
    # ceil() one past it, and walking up alone would never come back
    while power_two_means(difference, standard_deviation, n_rounded, alpha=alpha) < power:
        n_rounded += 1
    while (
        n_rounded > 2
        and power_two_means(difference, standard_deviation, n_rounded - 1, alpha=alpha) >= power
    ):
        n_rounded -= 1
    return n_rounded


def simulate_power(
    draw_p_value: Callable[[np.random.Generator], float],
    *,
    random_generator: np.random.Generator,
    n_sims: int = 1000,
    alpha: float = 0.05,
) -> float:
    """Fraction of simulated experiments rejecting at ``alpha``.

    ``draw_p_value`` runs one full simulated experiment and analysis and
    returns its p-value. Used to cross-check the analytic formulas and to
    get power for designs they do not cover (heavy tails, unequal arms).

    Rejection is ``p <= alpha``, not ``p < alpha``. The distinction is
    invisible for continuous p-values but matters for a discrete one: a
    p-value that is a multiple of 1/1000 can land on 0.05 exactly, and a
    strict comparison would discard that mass and understate power.
    """
    _validate_alpha_power(alpha)
    if n_sims < 1:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    rejections = sum(draw_p_value(random_generator) <= alpha for _ in range(n_sims))
    return rejections / n_sims

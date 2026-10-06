"""Synthetic experiment data with realistic failure modes built in.

Each generator models a pitfall that routinely bites real experiment
analysis:

- unequal allocation, and sample-ratio mismatch caused by asymmetric data
  loss (``treatment_loss_rate``),
- novelty effects: a treatment lift that decays over the first days of the
  experiment as the user base habituates to the change,
- heavy-tailed continuous outcomes where a handful of very large
  purchases dominate the arm means.

All randomness flows through an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExperimentData:
    """One synthetic experiment: per-user assignment, outcome, exposure day.

    ``assignment`` is 0 for control and 1 for treatment. ``outcome`` is
    float64 even for binary experiments so downstream statistics never
    branch on dtype.
    """

    assignment: NDArray[np.int64]
    outcome: NDArray[np.float64]
    day: NDArray[np.int64]

    @property
    def control(self) -> NDArray[np.float64]:
        return self.outcome[self.assignment == 0]

    @property
    def treatment(self) -> NDArray[np.float64]:
        return self.outcome[self.assignment == 1]

    @property
    def n_control(self) -> int:
        return int(np.sum(self.assignment == 0))

    @property
    def n_treatment(self) -> int:
        return int(np.sum(self.assignment == 1))

    def subset(self, mask: NDArray[np.bool_]) -> ExperimentData:
        """Rows where ``mask`` is True, e.g. an early/late day split."""
        return ExperimentData(
            assignment=self.assignment[mask],
            outcome=self.outcome[mask],
            day=self.day[mask],
        )


def _validate_common(n_users: int, allocation: float, n_days: int) -> None:
    if n_users < 1:
        raise ValueError(f"n_users must be positive, got {n_users}")
    if not 0.0 < allocation < 1.0:
        raise ValueError(f"allocation must be in (0, 1), got {allocation}")
    if n_days < 1:
        raise ValueError(f"n_days must be positive, got {n_days}")


def generate_binary(
    n_users: int,
    base_rate: float,
    absolute_lift: float,
    *,
    rng: np.random.Generator,
    allocation: float = 0.5,
    n_days: int = 14,
    novelty_boost: float = 0.0,
    novelty_half_life_days: float = 3.0,
    treatment_loss_rate: float = 0.0,
) -> ExperimentData:
    """Binary-outcome experiment (e.g. conversion) with optional pitfalls.

    The treatment conversion rate on day ``d`` is::

        base_rate + absolute_lift + novelty_boost * 0.5 ** (d / novelty_half_life_days)

    so ``absolute_lift`` is the persistent effect and ``novelty_boost`` is
    a transient effect that has halved after ``novelty_half_life_days``.
    The decay is over calendar days: the existing user base habituates to
    the change as the experiment runs.

    ``treatment_loss_rate`` models a tracking bug that silently drops that
    fraction of *non-converting* treatment users (e.g. a redirect that
    loses telemetry for users who bounce). It both inflates the observed
    treatment rate and shifts the arm-size ratio, so an SRM check catches
    it. Rows are dropped from the returned data, exactly as they would be
    missing from real logs.
    """
    _validate_common(n_users, allocation, n_days)
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")
    peak_rate = base_rate + absolute_lift + novelty_boost
    if not 0.0 <= peak_rate <= 1.0:
        raise ValueError(
            f"peak treatment rate {peak_rate} outside [0, 1]; "
            "check base_rate, absolute_lift, novelty_boost"
        )
    if base_rate + absolute_lift < 0.0:
        raise ValueError("long-run treatment rate is negative")
    if not 0.0 <= treatment_loss_rate < 1.0:
        raise ValueError(f"treatment_loss_rate must be in [0, 1), got {treatment_loss_rate}")
    if novelty_half_life_days <= 0.0:
        raise ValueError(f"novelty_half_life_days must be positive, got {novelty_half_life_days}")

    day = rng.integers(0, n_days, size=n_users)
    assignment = (rng.random(n_users) < allocation).astype(np.int64)

    rate = np.full(n_users, base_rate)
    decay = 0.5 ** (day / novelty_half_life_days)
    treated = assignment == 1
    rate[treated] = base_rate + absolute_lift + novelty_boost * decay[treated]
    outcome = (rng.random(n_users) < rate).astype(np.float64)

    if treatment_loss_rate > 0.0:
        lost = treated & (outcome == 0.0) & (rng.random(n_users) < treatment_loss_rate)
        keep = ~lost
        return ExperimentData(assignment[keep], outcome[keep], day[keep])
    return ExperimentData(assignment, outcome, day)


def generate_continuous(
    n_users: int,
    *,
    rng: np.random.Generator,
    allocation: float = 0.5,
    n_days: int = 14,
    spend_rate: float = 0.08,
    log_mean: float = 3.2,
    log_sigma: float = 1.1,
    whale_rate: float = 0.01,
    whale_shape: float = 1.5,
    whale_scale: float = 400.0,
    spend_rate_lift: float = 0.0,
    spend_uplift: float = 0.0,
) -> ExperimentData:
    """Revenue-per-user experiment with a realistic heavy tail.

    Most users spend nothing. Spenders draw from a lognormal, except a
    ``whale_rate`` fraction who draw from a Pareto tail with shape
    ``whale_shape``; the default shape 1.5 has finite mean but infinite
    variance, which is what makes naive t-tests on revenue unstable.
    Whale amounts are identical in distribution across arms, so they add
    variance but no signal.

    The treatment can shift the spend probability additively
    (``spend_rate_lift``) and non-whale spend amounts multiplicatively
    (``spend_uplift``; 0.05 means spenders spend 5 percent more at the
    median).
    """
    _validate_common(n_users, allocation, n_days)
    if not 0.0 <= spend_rate <= 1.0:
        raise ValueError(f"spend_rate must be in [0, 1], got {spend_rate}")
    if not 0.0 <= spend_rate + spend_rate_lift <= 1.0:
        raise ValueError("treatment spend rate outside [0, 1]")
    if not 0.0 <= whale_rate <= 1.0:
        raise ValueError(f"whale_rate must be in [0, 1], got {whale_rate}")
    if whale_shape <= 1.0:
        # shape <= 1 has infinite mean, so arm means never converge and no
        # comparison of means is estimating anything
        raise ValueError(f"whale_shape must exceed 1, got {whale_shape}")
    if whale_scale <= 0.0 or log_sigma <= 0.0:
        raise ValueError("whale_scale and log_sigma must be positive")
    if spend_uplift <= -1.0:
        raise ValueError(f"spend_uplift must exceed -1, got {spend_uplift}")

    day = rng.integers(0, n_days, size=n_users)
    assignment = (rng.random(n_users) < allocation).astype(np.int64)
    treated = assignment == 1

    p_spend = np.where(treated, spend_rate + spend_rate_lift, spend_rate)
    spends = rng.random(n_users) < p_spend

    user_log_mean = np.where(treated, log_mean + np.log1p(spend_uplift), log_mean)
    amount = rng.lognormal(mean=user_log_mean, sigma=log_sigma)
    # rng.pareto samples the Lomax distribution; adding 1 gives Pareto I
    # with minimum 1, scaled to whale_scale
    whale_amount = whale_scale * (1.0 + rng.pareto(whale_shape, size=n_users))
    is_whale = rng.random(n_users) < whale_rate
    amount = np.where(is_whale, whale_amount, amount)

    outcome = np.where(spends, amount, 0.0)
    return ExperimentData(assignment, outcome, day)

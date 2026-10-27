"""CUPED: regression adjustment of an experiment's outcome by a pre-experiment column.

CUPED (controlled-experiment using pre-experiment data) removes the part
of each user's outcome that their own past already predicted, so the
treatment effect is estimated against less noise. It changes the
precision of the estimate, not its target: with randomized assignment
the adjusted and unadjusted differences estimate the same effect.

Implemented as an ordinary least squares regression of the outcome on a
treatment indicator and the centered pre-experiment column, with HC2
robust standard errors, through statsmodels. The regression form equals
the textbook CUPED estimator and comes with a standard error that
accounts for the fitted coefficient.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from numpy.typing import ArrayLike, NDArray

from abkit.result_types import CupedResult


def _finite_column(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return the values as a 1-d float array, raising on a bad shape or a non-finite value."""
    column = np.asarray(values, dtype=np.float64)
    if column.ndim != 1:
        raise ValueError(f"{name} must be 1-d, got shape {column.shape}")
    if not np.all(np.isfinite(column)):
        raise ValueError(f"{name} must all be finite; found NaN or infinity")
    return column


def cuped_mean_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    control_pre: ArrayLike,
    treatment_pre: ArrayLike,
    *,
    confidence: float = 0.95,
) -> CupedResult:
    """CUPED-adjusted difference in means, treatment minus control.

    ``control_pre`` and ``treatment_pre`` hold each user's pre-experiment
    value of the same metric (or any column measured before assignment),
    aligned with ``control`` and ``treatment``. The variance reduction is
    one minus the squared ratio of the adjusted to the unadjusted standard
    error, both HC2 robust; it is near zero when the pre-experiment column
    does not predict the outcome.
    """
    control_values = _finite_column(control, "control")
    treatment_values = _finite_column(treatment, "treatment")
    control_pre_values = _finite_column(control_pre, "control_pre")
    treatment_pre_values = _finite_column(treatment_pre, "treatment_pre")
    if control_values.size != control_pre_values.size:
        raise ValueError(
            f"control and control_pre must align, got {control_values.size} "
            f"and {control_pre_values.size}"
        )
    if treatment_values.size != treatment_pre_values.size:
        raise ValueError(
            f"treatment and treatment_pre must align, got {treatment_values.size} "
            f"and {treatment_pre_values.size}"
        )
    if control_values.size < 2 or treatment_values.size < 2:
        raise ValueError(
            f"need at least 2 observations per arm, got {control_values.size} "
            f"and {treatment_values.size}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    outcome = np.concatenate([control_values, treatment_values])
    treated = np.concatenate([np.zeros(control_values.size), np.ones(treatment_values.size)])
    pre_period = np.concatenate([control_pre_values, treatment_pre_values])
    if np.ptp(pre_period) == 0.0:
        raise ValueError("the pre-experiment column is constant; there is nothing to adjust by")
    centered_pre = pre_period - pre_period.mean()
    design_matrix = np.column_stack([np.ones_like(outcome), treated, centered_pre])
    adjusted_fit = sm.OLS(outcome, design_matrix).fit(cov_type="HC2")
    unadjusted_fit = sm.OLS(outcome, design_matrix[:, :2]).fit(cov_type="HC2")
    interval = np.asarray(adjusted_fit.conf_int(alpha=1.0 - confidence), dtype=np.float64)[1]
    adjusted_standard_error = float(adjusted_fit.bse[1])
    unadjusted_standard_error = float(unadjusted_fit.bse[1])
    return CupedResult(
        difference=float(adjusted_fit.params[1]),
        standard_error=adjusted_standard_error,
        unadjusted_standard_error=unadjusted_standard_error,
        variance_reduction=float(1.0 - (adjusted_standard_error / unadjusted_standard_error) ** 2),
        pre_period_coefficient=float(adjusted_fit.params[2]),
        p_value=float(adjusted_fit.pvalues[1]),
        ci_low=float(interval[0]),
        ci_high=float(interval[1]),
        confidence=confidence,
    )

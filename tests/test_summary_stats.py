"""Property tests for the summary-statistic variants: given the statistics
of the same data, each must agree with its raw-array twin. Statistical code
can be wrong and still return plausible numbers, so the agreement is pinned
across many seeded draws, not spot-checked once."""

from __future__ import annotations

import numpy as np
import pytest

from abkit.frequentist import welch_ttest, welch_ttest_from_stats
from abkit.sequential import z_from_cumulative_counts, z_from_cumulative_stats

FIELDS = (
    "n_control",
    "n_treatment",
    "mean_control",
    "mean_treatment",
    "difference",
    "statistic",
    "df",
    "p_value",
    "ci_low",
    "ci_high",
)


class TestWelchFromStats:
    @pytest.mark.parametrize("seed", range(50))
    def test_agrees_with_array_twin(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n_c = int(rng.integers(5, 400))
        n_t = int(rng.integers(5, 400))
        control = rng.lognormal(1.0, rng.uniform(0.2, 1.5), size=n_c)
        treatment = rng.lognormal(1.1, rng.uniform(0.2, 1.5), size=n_t)
        from_arrays = welch_ttest(control, treatment)
        from_stats = welch_ttest_from_stats(
            n_c,
            float(control.mean()),
            float(control.var(ddof=1)),
            n_t,
            float(treatment.mean()),
            float(treatment.var(ddof=1)),
        )
        for field in FIELDS:
            assert getattr(from_stats, field) == pytest.approx(
                getattr(from_arrays, field), rel=1e-12, abs=1e-12
            ), field

    def test_rejects_single_observation_arm(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            welch_ttest_from_stats(1, 0.0, 0.0, 10, 1.0, 1.0)

    def test_rejects_negative_variance(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            welch_ttest_from_stats(10, 0.0, -1.0, 10, 1.0, 1.0)

    def test_rejects_constant_arms(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            welch_ttest_from_stats(10, 1.0, 0.0, 10, 1.0, 0.0)


class TestZFromCumulativeStats:
    @pytest.mark.parametrize("seed", range(20))
    def test_close_to_pooled_counts_twin_on_binary_data(self, seed: int) -> None:
        """On binary data the stats variant (unpooled SE) and the counts
        variant (pooled SE) agree to the bounds the docstring states,
        measured over these exact seeds on 2026-08-10: ratio deviation up
        to 8.1% at early few-hundred-user looks where observed rates are
        noisy, and 1.02% at the final look. The pins are 0.10 and 0.02."""
        rng = np.random.default_rng(seed)
        n_looks = 6
        per_look_c = rng.integers(400, 900, size=n_looks)
        per_look_t = rng.integers(400, 900, size=n_looks)
        outcomes_c = rng.random(int(per_look_c.sum())) < 0.06
        outcomes_t = rng.random(int(per_look_t.sum())) < 0.066
        nc = np.cumsum(per_look_c).astype(np.float64)
        nt = np.cumsum(per_look_t).astype(np.float64)
        xc = np.array([outcomes_c[: int(n)].sum() for n in nc], dtype=np.float64)
        xt = np.array([outcomes_t[: int(n)].sum() for n in nt], dtype=np.float64)
        rate_c, rate_t = xc / nc, xt / nt
        # ddof=1 sample variance of a 0/1 array with mean p is n*p*(1-p)/(n-1)
        var_c = nc * rate_c * (1.0 - rate_c) / (nc - 1.0)
        var_t = nt * rate_t * (1.0 - rate_t) / (nt - 1.0)

        z_stats = z_from_cumulative_stats(nc, rate_c, var_c, nt, rate_t, var_t)
        z_counts = z_from_cumulative_counts(xc, nc, xt, nt)
        # same numerator, so z_stats / z_counts is exactly the SE ratio and
        # a ratio bound is the fair comparison even where z is near zero
        ratio_dev = np.abs(z_stats / z_counts - 1.0)
        assert float(ratio_dev.max()) <= 0.10
        assert float(ratio_dev[-1]) <= 0.02

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            z_from_cumulative_stats(
                np.array([10.0, 20.0]),
                np.array([0.5, 0.5]),
                np.array([0.25, 0.25]),
                np.array([10.0]),
                np.array([0.5]),
                np.array([0.25]),
            )

    def test_rejects_decreasing_counts(self) -> None:
        with pytest.raises(ValueError, match="cumulative"):
            z_from_cumulative_stats(
                np.array([20.0, 10.0]),
                np.array([0.5, 0.5]),
                np.array([0.25, 0.25]),
                np.array([10.0, 20.0]),
                np.array([0.5, 0.5]),
                np.array([0.25, 0.25]),
            )

    def test_rejects_zero_variance_look(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            z_from_cumulative_stats(
                np.array([10.0]),
                np.array([0.5]),
                np.array([0.0]),
                np.array([10.0]),
                np.array([0.5]),
                np.array([0.0]),
            )

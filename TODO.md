# TODO

Deferred polish, roughly in the order I would pick it up.

- BCa bootstrap intervals. Percentile is fine for mean differences at
  experiment scale but undercovers on small skewed samples; the coverage
  test in `test_resampling.py` documents the current tolerance.
- Bootstrap over arbitrary statistics (median, trimmed mean, quantiles).
  The current implementation is mean-only so it can stay fully vectorized;
  a general version needs a chunked apply path.
- CUPED / regression adjustment. The single biggest practical win for
  experiment sensitivity; needs the generators to emit a pre-experiment
  covariate first.
- Futility (beta-spending) boundaries in `sequential.py`, so monitoring can
  stop hopeless experiments early, not just clear winners.
- Confidence interval for relative lift via the delta method; the point
  estimate is reported today but without an interval.
- Confidence interval for the effect estimate after a sequential stop
  (stage-wise ordering); the report currently just warns that the naive
  estimate at the stopping look is biased.
- Stratified and clustered assignment in the generators, to demonstrate
  variance estimation when the randomization unit is not the analysis unit.
- Property-based tests with hypothesis for the adjustment functions
  (monotonicity, idempotence on sorted inputs, permutation invariance).
- Coverage measurement in CI. Line coverage is currently about 96% measured
  by hand with the stdlib `trace` module; `pytest-cov` plus a floor in the
  workflow would keep it from sliding.
- Cap the grid widening in `sequential._propagate`. When early boundaries are
  infinite the support grows additively per look, which wastes resolution on
  long schedules: at 100 looks the boundary error reaches 9e-5, near the 1e-4
  the docstring claims. Fine for the schedules this is used on, but the fix
  is to widen relative to `sqrt(t_k)` instead.

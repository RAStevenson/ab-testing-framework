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
- CI workflow (GitHub Actions): pytest + ruff check + ruff format --check,
  and a job that regenerates the report and fails on drift.
- Property-based tests with hypothesis for the adjustment functions
  (monotonicity, idempotence on sorted inputs, permutation invariance).

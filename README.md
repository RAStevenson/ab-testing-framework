# abkit

A small A/B testing and experiment-analysis toolkit I built to show how I
approach experimentation work end to end: not just running a z-test, but
handling the parts that actually go wrong in practice — broken assignment,
novelty effects, peeking, multiple comparisons, and revenue metrics with
tails heavy enough to break the CLT at realistic sample sizes.

If you only read one thing, read [analysis/report.md](analysis/report.md).
It walks a synthetic checkout-redesign experiment from power analysis to a
ship recommendation, including a metric where three valid methods give three
different answers and the right move is to say so. Because the data are
synthetic, the report ends by comparing every conclusion against the effects
the generator actually planted.

## What is here

- `src/abkit/generators.py` — synthetic experiment data with the failure
  modes built in as parameters: unequal allocation, sample-ratio mismatch
  from asymmetric tracking loss, a novelty effect that decays over the run,
  and zero-inflated lognormal revenue with a Pareto whale tail (infinite
  variance by default, which is not a joke — real revenue data behaves this
  way).
- `src/abkit/frequentist.py` — pooled two-proportion z-test, Welch t-test,
  chi-square SRM check, and Bonferroni / Holm / Benjamini-Hochberg adjusted
  p-values.
- `src/abkit/resampling.py` — percentile bootstrap and a permutation test,
  both vectorized in fixed-size chunks so memory stays flat regardless of
  resample count. These are the tools I actually trust on revenue metrics.
- `src/abkit/sequential.py` — group-sequential monitoring with O'Brien-Fleming
  and Pocock alpha spending. Boundaries come from the standard recursive
  numerical-integration construction, not a lookup table, so arbitrary look
  schedules work (the report uses the 14 realized daily information
  fractions).
- `src/abkit/power.py` — analytic power, sample size, and MDE, plus a
  simulation harness that the tests use to keep the formulas honest.

## How it is tested

Statistical code lies in ways ordinary code does not: it can be wrong and
still return plausible numbers. So the suite tests statistical properties,
not just plumbing. Type I error rates are pinned by simulation with fixed
seeds. Confidence interval coverage is measured, not assumed. The sequential
boundaries are checked three independent ways: against published Lan-DeMets
values, against exact bivariate and trivariate normal probabilities from
scipy (which share no code with my recursion), and against Monte Carlo
crossing rates for five looks. One test exists purely to demonstrate the
motivating failure: fourteen naive daily peeks at z = 1.96 inflate a nominal
5% false-positive rate past 15%.

## Running it

```
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]   # .venv/bin/python on Unix
.venv/Scripts/python -m pytest
.venv/Scripts/python analysis/run_analysis.py   # regenerates the report
```

Python 3.11+. The library depends only on numpy and scipy; matplotlib is a
dev dependency used by the analysis script. The report is deterministic:
fixed seeds throughout, no hidden global RNG state anywhere in the library
(every function that draws randomness takes an explicit
`numpy.random.Generator`).

## Decisions worth flagging

The z-test uses the pooled standard error for the statistic and the unpooled
one for the confidence interval, because each is correct for its own job.
The permutation test draws subsets with `argpartition` on random keys rather
than full shuffles, which is O(n) per permutation instead of O(n log n) and
matters at 10,000 permutations on 120,000 users. The bootstrap interval is
percentile rather than BCa — adequate for mean differences at these sample
sizes, and BCa is on the list below. Results come back as frozen dataclasses
because `result.p_value` reads better in a review than `result[3]`.

Deferred work lives in [TODO.md](TODO.md); conventions for contributing are
in [CLAUDE.md](CLAUDE.md).

# abkit

A small A/B testing and experiment-analysis toolkit I wrote around the parts
of experimentation that actually go wrong in practice: broken assignment,
novelty effects, peeking, multiple comparisons, and revenue metrics with
tails so heavy that the normal approximation behind the usual tests (the
central limit theorem) breaks down at realistic sample sizes.

Fourteen daily peeks at a fixed |z| > 1.96 turn a nominal 5% false-positive
rate into 21.9%. An O'Brien-Fleming spending design, built on the same
fourteen looks at the sample sizes they actually reached, holds it at 4.9%.
Both figures come from
100,000 simulated null runs at a fixed seed, and `python scripts/run_build_checkout_report.py`
reproduces them.

![Daily monitoring against spending boundaries](analysis/checkout/figures/sequential.png)

If you only read one thing, read [analysis/checkout/report.md](analysis/checkout/report.md).
It walks a synthetic checkout-redesign experiment from power analysis to a
ship recommendation, including a revenue metric where Welch, the bootstrap,
and a permutation test return three different verdicts on the same data and
the right move is to say so rather than pick one. Because the data are
synthetic, the report closes by putting its headline conclusions next to the
effects the generator actually planted.

The same engine also runs on real data, and on yours. Two real-data reports
are committed: [analysis/asos/report.md](analysis/asos/report.md) reads a
real ASOS.com experiment (aggregate statistics with a time axis, so it
carries the sequential replay and a 78-experiment archive sweep, and its
final checkpoint turned out to be internally inconsistent, which the report
excludes and says so), and
[analysis/hillstrom/report.md](analysis/hillstrom/report.md) reads the
classic 2008 retail e-mail experiment (user-level with a heavy-tailed spend
metric and a genuinely misspelled category the validator normalizes on the
record). Sections render exactly where the data can support them; the two
opposite shapes are the demonstration that this is a mechanism, not a
special case.

## What is here

- `src/abkit/generators.py` - synthetic experiment data with the failure
  modes built in as parameters: unequal allocation, sample-ratio mismatch
  (SRM) from asymmetric tracking loss, a novelty effect that decays over the
  run, and zero-inflated lognormal revenue with a Pareto whale tail (infinite
  variance by default).
- `src/abkit/frequentist.py` - pooled two-proportion z-test, Welch t-test,
  chi-square SRM check, and Bonferroni / Holm / Benjamini-Hochberg adjusted
  p-values.
- `src/abkit/resampling.py` - percentile bootstrap, a permutation test, and a
  trimmed-mean difference for diagnosing whale-driven results. The bootstrap
  and permutation loops vectorize over resamples in 256-row chunks, so peak
  memory is set by the chunk size rather than the resample count. The
  permutation test assumes only that when the treatment does nothing, the
  group labels are interchangeable (exchangeability under the null), which is
  why it is the one I lean on for revenue.
- `src/abkit/sequential.py` - group-sequential monitoring with O'Brien-Fleming
  and Pocock alpha spending. Boundaries come from the standard recursive
  numerical-integration construction, not a lookup table, so arbitrary look
  schedules work (the report uses the 14 realized daily information
  fractions).
- `src/abkit/power.py` - analytic power, sample size, and minimum detectable
  effect (MDE), plus a simulated-power routine that the tests use to keep the
  formulas honest.
- `src/abkit/io.py` - the intake: a TOML experiment description, loaders for
  user-level rows and for per-arm cumulative aggregates, and a validator
  whose every cleaning action lands in the report rather than happening
  silently.
- `src/abkit/report.py` and `src/abkit/report_sections.py` - the report
  engine (numbering and cross-references resolved at render time, build
  fails on an uncited figure) and the presence-gated sections it renders.

## How it is tested

Statistical code can be wrong and still return plausible numbers, so the suite
tests statistical properties rather than plumbing. I simulate each test's
type I error rate, the share of runs that flag an effect when none exists, at
a fixed seed, and I measure confidence interval coverage instead of assuming
it.

The sequential boundaries get the most scrutiny, because they are the part I
could most easily have got subtly wrong. They are checked against a published
Lan-DeMets Pocock table for five equal looks, against exact bivariate and
trivariate normal rectangle probabilities from scipy (which share no code with
my recursion), against Monte Carlo crossing rates, and against a grid
refinement that pins the accuracy the module docstring claims. One test exists
purely to demonstrate the motivating failure and its fix. Fourteen naive daily
peeks at z = 1.96 reject on 22% of null runs, while the spending boundaries on
those same simulated paths hold 5%.

The suite also covers the directions that are easy to leave untested. A
two-sided boundary must stop a treatment that *hurts*, not just one that
helps, and a cumulative count series that goes backwards must raise an error
rather than manufacture a significant result.

## Running it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt  # .venv/bin/python on Unix
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_build_checkout_report.py # regenerates the flagship report
.venv/Scripts/python scripts/run_analyze_experiment.py --config analysis/asos/config.toml --sweep
.venv/Scripts/python scripts/run_fetch_hillstrom.py       # fetches the 2008 dataset (not redistributed)
.venv/Scripts/python scripts/run_analyze_experiment.py --config analysis/hillstrom/config.toml
```

To analyze your own experiment, write a config describing it and point the
door at your file:

```bash
.venv/Scripts/python scripts/run_analyze_experiment.py --config my_experiment.toml --data my_data.csv
```

The config declares the arm column and labels, the metrics and which one is
primary, the intended split if you know it, and optional value aliases; the
two committed demo configs (`analysis/asos/config.toml` for aggregate data,
`analysis/hillstrom/config.toml` for user-level rows) are complete working
examples of both input shapes. Validation is never silent: every row the
intake drops or normalizes is counted in the report, an unrecognized arm
label stops the run, and a config may pin its data file's md5 so an analysis
cannot quietly run against the wrong file.

Python 3.11+. No install step: the entry scripts and `tests/conftest.py` put
`src/` on the import path themselves, so a fresh clone runs as-is once the
dependencies are present. An editable install
(`pip install -e '.[dev]'`) still works if you prefer one. The library
depends only on numpy and scipy; matplotlib is a
dev dependency used by the report script. Rerunning the analysis reproduces
`report.md` byte for byte, though figures can shift across matplotlib
versions. CI regenerates the report on every push and fails if the committed
copy has drifted. There is no hidden global random-number state anywhere in
the library, because every function that draws randomness takes an explicit
`numpy.random.Generator`.

## What it does not do

It has no CUPED or regression adjustment, the standard ways of using
pre-experiment data to shrink variance. That is the largest sensitivity win
available for this kind of analysis and the first thing I would add, but it
needs the generators to emit a pre-experiment covariate first. There are no
futility boundaries, so the monitoring code can stop a clear winner early but
not a hopeless run. There is no confidence interval for the effect after a
sequential stop, which is why the report warns that the estimate at the
stopping look is biased upward rather than correcting it. The bootstrap
handles mean differences only, so it can stay fully vectorized. Users are
assigned independently of one another, so nothing here demonstrates variance
estimation when the randomization unit differs from the analysis unit. The
relative lift is reported as a point estimate without a confidence interval.

## Decisions worth flagging

The z-test uses the pooled standard error for the statistic and the unpooled
one for the confidence interval. The pooled variance is the right one under
the null the test is evaluating, and the unpooled one is right away from the
null, where the interval lives.

The permutation test gives each user a random key and takes `argpartition` of
those keys to pick a treatment-sized subset. That keeps a whole chunk of
permutations inside one vectorized call, and costs O(n) per permutation where
`argsort` of the same keys would cost O(n log n). The script
`scripts/run_bench_permutation.py` times both against a per-row Fisher-Yates
shuffle at the report's scale of 10,000 permutations over 120,000 pooled
users; on a 2026-08-07 run (Windows 11, 32-core AMD, numpy 2.4.6)
argpartition measured 3.0x faster than `argsort` and 2.5x faster than the
shuffle, which is also O(n) but has to materialize and permute the full
block.

The bootstrap interval is percentile rather than BCa. Percentile undercovers
slightly on skewed data; the coverage test measures 90-98% against a nominal
95% on lognormal arms at n = 400. That is acceptable for mean differences at
experiment scale.

Results come back as frozen dataclasses because `result.p_value` reads better
in a review than `result[3]`.

## Conventions

The library holds itself to a few rules throughout. Every public function
carries full type hints, with `numpy.typing.NDArray[np.float64]` for
arrays, and returns a frozen dataclass rather than a tuple. Randomness
always flows through an explicit `numpy.random.Generator` parameter; there
is no module-level seeding and none of the legacy `np.random.*` API. There
is no error swallowing. The library has no `except` blocks, inputs fail
with an explicit `ValueError`, and the report builder is the one documented
place a failure may be caught, logged, and skipped. Comments explain
non-obvious decisions only.

On the statistics side, tests are two-sided unless the function name says
otherwise, p-value adjustment functions take and return arrays in the
original order, and anything approximate states its accuracy in its
docstring and has a test pinning it against a published or simulated value.

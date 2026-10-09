# abkit

A command-line tool that reads an exported experiment file and prints a
readout. The readout puts the data checks first and the result after them,
which is the order an experiment has to be read.

I built it for an analyst who has an exported file and no experimentation
platform, or who wants a second opinion on a dashboard. statsmodels and
scipy give you one test at a time. This runs the checks a readout needs, in
order, on the exported file, and it adds the one part those libraries lack,
which is daily monitoring. It calls statsmodels and scipy for the z-test,
Welch's t-test, the multiple-comparison corrections, power, the bootstrap,
the permutation test, and the CUPED regression, which removes the part of
each user's outcome that their own pre-experiment value already predicted.
The repo's own code is the intake with its checks and repairs, the
ordering, the readout, and the monitoring boundaries with their adjusted
intervals. Neither statsmodels nor scipy computes those boundaries at the
sample sizes each daily check, each look, actually reached.

Checking a running experiment every day against the usual 1.96 threshold is
the failure the monitoring part exists for. Fourteen daily looks at a null
effect reject 21.8% of runs, not 5%. An O'Brien-Fleming spending design, a
preset rule that spreads the 5% false-positive budget across the looks,
built on those same fourteen looks holds 4.9%, inside the budget. Both
figures come from 50,000 simulated null runs at seed 42, pinned in
`tests/test_sequential.py` by `test_naive_daily_peeking_inflates_type_i_error`.

![One real experiment replayed (the ASOS experiment below): the observed |z| path against the spending boundaries, and the effect at each look inside its adjusted interval](analysis/asos/figures/monitoring.png)

## What the readout contains

The readout is a text file of labeled blocks. Each heading carries a few
words on how to read its block, results are short labeled lines, and the
file closes with a decision line and the figures' paths. There is no
generated prose. This is the committed readout of a real ASOS.com
experiment, [analysis/asos/readout.txt](analysis/asos/readout.txt), whole:

```text
ASOS experiment 54a85a: per-group totals
Data: what the intake did before any statistic
  users   control 3,320,536   variant_0 102,316   variant_1 3,316,730   variant_2 3,320,435
  repairs and drops   none; the file passed every rule as delivered

Limitations (*): read every result below with these in view
  * aggregate data carries no per-user pre-experiment values, so CUPED cannot run
  * 3 gaps in the checkpoint sequence (the largest 1 day against a typical 0.5); the time-based
      blocks treat the checkpoints present as the whole run
  * the final 1 checkpoint is internally inconsistent and left out of the time-based blocks; an
      increment's recovered variance is negative, which no real accumulation of users produces; the
      end-of-run results still use the final totals

Group split: does assignment match the plan?
  intended split not on record; testing whether the share of new users per checkpoint is constant
  variant_0 vs control   96 checkpoints   chi-square 78.23   df 95   p 0.894
  result   stable across checkpoints

Sensitivity: smallest lift this sample could detect, 80% power
  control 25.1%   3,320,536 vs 102,316 users   detectable 0.39 pp (1.5% of control)

Primary metric_1: effect, 95% interval, p; interval first
  variant_0 vs control   -0.07 pp   -0.33 to +0.20 pp   p 0.635   not detected
  variant_1 vs control   +0.10 pp   +0.04 to +0.17 pp   p 0.003   detected, treatment higher
  variant_2 vs control   -0.06 pp   -0.12 to +0.01 pp   p 0.085   not detected

Other metrics: corrected for 9 comparisons
  metric_2   variant_0 vs control   +0.001   p 0.893   Holm 1.000   BH 0.893
  metric_2   variant_1 vs control   +0.002   p 0.106   Holm 0.703   BH 0.238
  metric_2   variant_2 vs control   -0.002   p 0.100   Holm 0.703   BH 0.238
  metric_3   variant_0 vs control   -0.003   p 0.848   Holm 1.000   BH 0.893
  metric_3   variant_1 vs control   +0.009   p 0.033   Holm 0.296   BH 0.238
  metric_3   variant_2 vs control   -0.007   p 0.088   Holm 0.703   BH 0.238
  metric_4   variant_0 vs control   -0.185   p 0.628   Holm 1.000   BH 0.808
  metric_4   variant_1 vs control   +0.065   p 0.497   Holm 1.000   BH 0.745
  metric_4   variant_2 vs control   -0.110   p 0.251   Holm 1.000   BH 0.451
  result   unadjusted 1 of 9 clear alpha; after Holm 0 survive

Early versus late: did the effect fade?
  first 31 checkpoints against the last 31, of 95 usable
  variant_0 vs control   early -0.01 pp   late -1.26 pp   z 2.88   p 0.004   effect fell over the
      run
  variant_1 vs control   early +0.09 pp   late +0.08 pp   z 0.09   p 0.924   no change detected
  variant_2 vs control   early -0.17 pp   late +0.23 pp   z -3.70   p 0.0002   effect rose over the
      run

Daily monitoring: 95 looks; crossing a boundary = safe to stop
  variant_0 vs control on metric_1, O'Brien-Fleming spending at the sizes reached
  spending design   never crosses in 95 looks; no early stop in either direction
  naive 1.96 rule   never fires
  final look   effect -0.07 pp   adjusted 95% -0.38 to +0.24 pp   boundary 2.25

Decision: no detectable effect; this test could have seen 0.39 pp or more
Figures:
  figures/corrections.png
  figures/early_late.png
  figures/monitoring.png
```

The blocks come in the order an experiment has to be read. The data block
says what the intake did to the file. The limitations block lists what the
checks flagged, and every result below is read with those in view. The
group-split check comes before any result. A split that is off from the
plan means users went missing or were assigned unevenly, and no result can
repair that. When the intended split is not on record, as here, the check
becomes a test that the share of new users per checkpoint stayed constant.
Sensitivity states the smallest effect this sample could have
detected, so a "not detected" result says what it rules out. The primary
metric comes next, interval first, with the CUPED-adjusted result beside it
when a pre-experiment column exists. Secondary metrics are corrected two
ways, Holm for a ship decision and Benjamini-Hochberg for a screen. With
user-level rows and a continuous metric such as spend, a cross-check runs
Welch, a BCa (bias-corrected and accelerated) bootstrap, a permutation
test, and a trimmed mean on the same question. With a day column, the early-versus-late block tests whether the effect faded,
and the monitoring block replays the run as daily looks against boundaries
computed at the sizes reached, with an adjusted interval that holds at
every look at once. On the synthetic example only, a final block prints what
the generator planted beside what the readout found.

Two more readouts are committed. [analysis/hillstrom/readout.txt](analysis/hillstrom/readout.txt)
reads the 2008 MineThatData e-mail experiment, user-level rows with three
arms, a heavy-tailed spend metric, a misspelled category value the intake
normalizes and counts, and a past-spending column that CUPED adjusts by.
[analysis/synthetic/readout.txt](analysis/synthetic/readout.txt) is the
check that the tool recovers what was planted. It is one generated
experiment, read through the identical intake. The generator plants a
conversion lift, a novelty boost that fades over the first days, a few
users who spend far more than the rest (a whale tail), and a
pre-experiment column. The planted
end-of-run lift is +0.67 pp with the novelty averaged in, and the readout
finds +0.67 pp with an interval of +0.39 to +0.95 pp. CUPED on the
pre-experiment column removes 14% of the noise on spend, well below the
planted correlation squared, because the whale tail is unpredictable from
the past.

## Data checks and repairs

The intake never cleans a file silently. Every repair and drop is counted
in the data block, and a fault the intake cannot repair stops the run
instead.

Repaired and counted:

- stray spaces trimmed, and group labels matched regardless of capitals
- Excel's byte-order mark at the start of a CSV accepted
- value replacements the config declares applied

Dropped and counted, with a stop above 1% of rows:

- a row whose metric value is blank, not a number, or infinite
- a negative value where the config says the metric cannot be negative
- a value other than 0 or 1 in a yes/no metric

The 1% stop exists because a file past that share is not clean enough to
read, and a readout of the survivors would describe a different experiment.

Stops the run, naming what failed:

- a required column missing, or a group label the config does not list
- a user appearing twice
- daily totals going backwards, or a negative count or variance

Flagged in the limitations block:

- group sizes off from the intended split
- a metric whose top 1% of users carry more than a quarter of its total
- fewer than 10 successes or failures in an arm
- a smallest detectable effect above a 5% relative lift
- no pre-experiment column
- gaps in the checkpoint sequence
- a final checkpoint whose totals are internally inconsistent

## How to run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt  # .venv/bin/python on Unix
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_analyze_experiment.py --config analysis/asos/config.toml
.venv/Scripts/python scripts/run_plan_experiment.py --baseline-rate 0.05 --minimum-lift 0.005
.venv/Scripts/python scripts/run_synthetic_example.py
.venv/Scripts/python scripts/run_fetch_hillstrom.py   # fetches the 2008 dataset (not redistributed)
.venv/Scripts/python scripts/run_analyze_experiment.py --config analysis/hillstrom/config.toml
```

The analyze command prints the readout and saves it as `readout.txt` beside
the config, with the figures in a `figures/` folder. The `--out` flag
chooses another directory and `--seed` fixes the resampling seed. The
planning
command prints the users needed per group for a baseline rate and the
smallest lift worth detecting, or for a standard deviation and the smallest
difference, at the chosen alpha and power. To read your own experiment,
write a config and point the analyze command at your file:

```bash
.venv/Scripts/python scripts/run_analyze_experiment.py --config my_experiment.toml --data my_data.csv
```

The config names the group column and its labels, the metrics and which
one is primary, the intended split if you know it, alpha, any value
replacements, and any pre-experiment column. Two of the three committed
configs, `analysis/hillstrom/config.toml` for user-level rows and
`analysis/asos/config.toml` for per-group cumulative totals, are complete
working examples of both input shapes. A few conventions matter when
writing yours. The first label in `arms.labels` is the control, and
ordering them the other way flips the sign of every effect. A `day`
column under `[columns]`, numeric time since the experiment started, is
what turns on the early-versus-late block and the monitoring replay. A
`pre_column` on a metric, a value measured before assignment, is what
turns on CUPED for that metric. Columns the config does not name are
ignored. A config may pin its data file's md5 so a readout cannot quietly
describe the wrong file.

It needs Python 3.11 or newer. There is no install step, because the entry
scripts put `src/` on the import path themselves, so a fresh clone runs
once the dependencies are present. The library depends on numpy, scipy, and
statsmodels. Matplotlib draws the figures and installs with the dev
dependencies. Rerunning a readout reproduces `readout.txt` byte for byte,
and CI regenerates the ASOS and synthetic readouts on every push to master
and fails if a committed copy has drifted. The Hillstrom readout is outside that
check on purpose, since its data is fetched rather than committed and CI
should not depend on a 2008 URL staying up.

## What it does not do

It assumes users were randomized one by one. An experiment that randomized
groups of users, such as stores or communities, needs a variance estimate
this tool does not have, and nothing in the file lets it detect that. It
draws no causal conclusions beyond the randomized comparison. There are no
futility boundaries, so the monitoring replay can stop early on a clear
effect but not on a hopeless run. After a sequential stop the only interval
is the adjusted one, which is built to hold at every look at once rather
than for the stopping look alone. The bootstrap handles mean differences
only. The relative lift is a point estimate without an interval.

The bootstrap interval is scipy's BCa, and on skewed data it still runs a
little under its nominal coverage. Measured on 2026-10-07, outside the
suite, on lognormal arms of 400 users over 1,000 simulations at each of
three seeds, it covered 92.6 to 93.7% against a nominal 95%, and the
percentile interval it replaced covered 93.9 to 94.7%. I kept BCa because
it is scipy's default and the gap is about a point. The coverage test in
`tests/test_resampling.py` holds the interval to a 90 to 98% band.

## Related tools

Spotify's confidence library covers the same ground in a notebook, with
summary statistics in and sequential tests out. I ran it on the same ASOS
experiment on 2026-10-06 as a cross-check before deciding to call
statsmodels directly. The primary metric's z-test p-value was 0.6345
against this repo's 0.635. The Holm-adjusted p-values on the other three
metrics were identical. Its sequential replay over the 95 looks flagged no
stop, and neither does this one. Its final adjusted interval was -0.38 to
+0.25 pp against this repo's -0.377 to +0.241 pp, which the readout prints
as -0.38 to +0.24 pp. The gap comes from how the boundaries are placed.
That library needs the final total sample size in advance, while this tool
computes them at the sizes the looks actually reached. I did not take it as a dependency because it installs 68
packages, pins pandas below 3, and is marked beta, and because the
readout's intake and ordering are the part I wanted to own.

## Decisions worth flagging

The permutation test calls scipy although the hand-built loop it replaced
measured 2.8 times faster (7.5 s against 20.7 s for 10,000 permutations
over 120,000 users, 2026-10-06). Speed is not a concern at the sizes
experiments run at, and a scipy call is one less thing to defend. The
resampling on a real file costs a few seconds. On the Hillstrom spend
contrast, 21,306 and 21,307 users, the bootstrap and the permutation test
each took 7 seconds at 10,000 resamples, measured 2026-10-07.

The monitoring boundaries are computed at the sample sizes the looks
actually reached rather than on a planned schedule. The method is the
standard recursive numerical integration of Armitage, McPherson, and Rowe,
which carries forward the probability that no earlier look has crossed and
solves each look's cutoff from it. That is why they stay the repo's own
code. Neither statsmodels nor scipy computes them, and Spotify's library
needs the final sample size in advance. The adjusted
intervals are each look's estimate plus or minus that look's boundary
times its standard error, which is what makes them hold at every look at
once.

The z-test uses the pooled standard error for the statistic and the
unpooled Agresti-Caffo interval for the estimate, which stays a real
interval when an arm sits at 0% or 100%, both through statsmodels. The pooled variance is the right one under the null the test
evaluates, and the unpooled one is right away from the null, where the
interval lives.

Results come back as frozen dataclasses because `result.p_value` reads
better in a review than `result[3]`.

## How it is tested

Statistical code can be wrong and still return plausible numbers, so the
suite tests statistical properties. The false-positive rates of the
z-test, the permutation test, and the spending boundaries are simulated at
fixed seeds, interval coverage is measured rather than assumed, and the
proportion power function is pinned against an exact binomial calculation. The monitoring boundaries are checked against a published
Lan-DeMets Pocock table, against exact bivariate and trivariate normal
rectangle probabilities from scipy, against Monte Carlo crossing rates,
and against a grid refinement. The adjusted intervals are checked for
joint coverage over 20,000 simulated null paths. The intake has one test
per rule on injected bad data and one per limitation code on data built
to raise it, and the readout text is pinned to two golden files that are
read by eye when they change. Three standard checks also run: every
script does its work inside `main()` with an argument parser, every
definition has a docstring, and no variable name is one or two letters
outside a short allowed list.

## Repo layout

- `src/abkit/io.py`: the config, the two loaders, and the checks and repairs.
- `src/abkit/readout_analysis.py`: every number the readout reports, computed once, no text.
- `src/abkit/readout_text.py` and `readout_figures.py`: the lines and the figures.
- `src/abkit/frequentist.py`, `power.py`, `resampling.py`, `cuped_adjustment.py`: thin wrappers over statsmodels and scipy that return the repo's result dataclasses.
- `src/abkit/sequential.py`: the spending boundaries and the adjusted intervals.
- `src/abkit/generators.py`: the synthetic experiment with its planted effects.
- `scripts/`: the analyze, plan, synthetic, and fetch commands.
- `analysis/`: the three committed readouts with their configs and figures.

## Conventions

Every public function carries full type hints and returns a frozen
dataclass rather than a tuple. Randomness always flows through an explicit
`numpy.random.Generator` parameter, with no module-level seeding. Inputs
fail with an explicit `ValueError`. Two exceptions exist. The intake's
coercion catch turns an unreadable value into a counted and reported row
drop, and the two time-based blocks omit themselves on a data error after
printing it in full. Tests are two-sided unless the
function name says otherwise, p-value adjustments take and return arrays
in the original order, and anything approximate states its accuracy in
its docstring and has a test pinning it against a published or simulated
value. Spelling is US throughout.

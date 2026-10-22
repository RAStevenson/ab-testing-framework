# abkit

abkit is a command-line program that reads an exported experiment file, one
row per user, and prints a readout. Its goal is to make the evaluation of an
A/B test one ordered pass instead of one test at a time. The readout puts
the data checks first and the result after. The standard tests come from
statsmodels and scipy. The intake that checks and repairs the file is the
repo's own code, and so is the order of the readout. The readout also
replays a finished run day by day, as if the experiment were still being
watched, which those libraries do not do. It is built for an analyst who has
an exported file and no experimentation platform, or who wants a second
opinion on a dashboard.

A look is one check of an experiment's running totals. Checking at every
look against the usual cutoff, a z (the effect divided by its standard
error) past 1.96, raises the false-positive rate well above the 5% it is
meant to hold. On fourteen daily looks at a null effect, that rule rejects
21.8% of runs. An O'Brien-Fleming spending design treats the 5% as a budget
divided across the looks, so each look gets its own cutoff, the spending
boundary. It spends almost none of the budget early and most of it at the
end, and rejects 4.9% on the same fourteen looks. Where a file has a day
column, the replay applies that design to the finished run. Both figures
come from 50,000 simulated null runs at seed 42, pinned by a test in
`tests/test_sequential.py`.

![The synthetic check replayed look by look](analysis/synthetic/figures/monitoring.png)

*The replay from the synthetic check, a generated experiment with a planted
lift on conversion, described below. The observed |z| passes the naive 1.96
line at look 1, where the design allows no stop, because stops that early
are what produce the 21.8% on runs with no effect. It crosses the spending
boundary at look 4 of 14. The effect on the right falls across the looks
because the generator gives the earliest users an extra lift.*

## The readout

The readout is a text file of labeled blocks, with no generated prose. Each
heading carries a few words on how to read its block, and the file closes
with one decision line per treatment and the figures' paths. The committed
readout of the Hillstrom e-mail experiment, a public 2008 dataset described
below, follows whole, with a key to its blocks after it.

The file is
[analysis/hillstrom/readout.txt](analysis/hillstrom/readout.txt).

```text
Hillstrom e-mail experiment
Data: what the intake did before any statistic
  users   No E-Mail 21,306 (33.3%)   Mens E-Mail 21,307 (33.3%)   Womens E-Mail 21,387 (33.4%)
  alias_normalized   28,776 rows   zip_code 'Surburban' to 'Suburban', first at data row 2

Limitations (*): read every result below with these in view
  * the top 1% of users carry 100% of spend, because fewer than 1% have any spend at all
  * the smallest detectable effect on visit is above a modest 5% lift for Mens E-Mail vs No E-Mail
      (8.8% of control) and Womens E-Mail vs No E-Mail (8.8% of control); a 'not detected' result
      there rules out little

Group split: does assignment match the plan?
  Mens E-Mail vs No E-Mail   21,307 vs 21,306   chi-square 0.00   p 0.971
  Womens E-Mail vs No E-Mail   21,387 vs 21,306   chi-square 0.18   p 0.672
  result   consistent with the plan

Sensitivity: smallest lift this sample could detect, 80% power
  each at alpha 0.05/2, the strictest level the Holm correction applies
  Mens E-Mail vs No E-Mail   21,307 vs 21,306 users   control 10.6%   detectable 0.94 pp (8.8% of
      control)
  Womens E-Mail vs No E-Mail   21,387 vs 21,306 users   control 10.6%   detectable 0.94 pp (8.8% of
      control)

Primary visit: effect, 95% interval, p; interval first
  Mens E-Mail vs No E-Mail   +7.66 pp   +6.99 to +8.32 pp   p 5.7e-112   Holm 1.1e-111   detected,
      treatment higher
  Womens E-Mail vs No E-Mail   +4.52 pp   +3.89 to +5.16 pp   p 3.2e-44   Holm 3.2e-44   detected,
      treatment higher
  Holm-corrected across the 2 treatments; detection judged on the Holm p; intervals unadjusted

CUPED on spend: adjusted by its pre-experiment column; narrower interval = less noise
  Mens E-Mail vs No E-Mail   +0.767   +0.483 to +1.052   p 1.2e-07   noise removed 0%
  Womens E-Mail vs No E-Mail   +0.423   +0.167 to +0.678   p 0.001   noise removed 0%

Other metrics: corrected for 4 comparisons (Holm, and BH = Benjamini-Hochberg)
  conversion   Mens E-Mail vs No E-Mail   +0.68 pp   p 1.5e-13   Holm 6.1e-13   BH 6.1e-13
  conversion   Womens E-Mail vs No E-Mail   +0.31 pp   p 0.0002   Holm 0.0003   BH 0.0002
  spend   Mens E-Mail vs No E-Mail   +0.770   p 1.2e-07   Holm 3.5e-07   BH 2.3e-07
  spend   Womens E-Mail vs No E-Mail   +0.424   p 0.001   Holm 0.001   BH 0.001
  result   unadjusted 4 of 4 below 0.05; after Holm 4 survive

Decision for Mens E-Mail vs No E-Mail: higher on visit by +7.66 pp (+6.99 to +8.32 pp)
Decision for Womens E-Mail vs No E-Mail: higher on visit by +4.52 pp (+3.89 to +5.16 pp)
Figures:
  figures/corrections.png
```

How to read it, block by block:

- **Data.** Each group's user count and its share of the total, then what
  the intake repaired or dropped, with counts.
- **Limitations.** What the checks flagged. A flag does not stop the run;
  every result below is read with these in view.
- **Group split.** Whether users landed in the groups as planned, tested
  against the intended split in the config. A split counts as broken when
  its chi-square test gives p below 0.001. A broken split means users went
  missing or were assigned unevenly, and no statistic repairs that, so the
  decision lines are replaced by one stop line. With no intended split on
  record, the block says there is no check, and the Data block's shares are
  left to the reader.
- **Sensitivity.** The smallest lift this sample could have detected with
  80% power. A "not detected" result means an effect at least that large
  would have been seen with 80% probability and says nothing about smaller
  ones. With several treatments it is computed at the false-positive rate,
  alpha, divided by their number, so it holds under the Holm correction
  below.
- **Primary metric.** The effect, its 95% interval, and p for each
  treatment. Read the interval before the p. With several treatments the
  comparisons are Holm-corrected, which caps the chance of any false
  positive among them at alpha, and detection is judged on the corrected p.
- **CUPED** (controlled-experiment using pre-experiment data). It runs on
  each metric with a pre-experiment column, which in both committed readouts
  is spend, not the primary. It reports that metric's effect after a
  regression adjustment removes the part of each user's outcome their
  pre-experiment value already predicted. "Noise removed" is the share by
  which the adjustment shrank the estimate's variance, so more removed means
  a narrower interval.
- **Other metrics.** The secondary metrics, corrected two ways. Holm, as on
  the primary metric, suits a ship decision. Benjamini-Hochberg caps the
  share of false positives instead, which is enough for a screen of what to
  test next.
- **Monitoring replay** (files with a day column only, so the synthetic
  readout and not this one). The finished run checked look by look against
  the spending boundary. A crossing means a stop at that look would have
  kept the false-positive rate, and nothing more. "Never crosses" means no
  look allowed a stop. On a run that ended on schedule, the primary block's
  interval is the result. The adjusted interval is the one a stop would have
  had to report. The readout prints the final look's, which is a little
  wider than the primary block's interval because it is built to hold at
  every look at once. The naive 1.96 line is printed for contrast. With
  several treatments the replay runs on the first treatment listed in the
  config, a rule that does not look at the results.
- **Planted versus found** (the synthetic check only). What the generator
  planted beside what the readout found.
- **Decision.** One line per treatment, judged on the primary metric's p,
  Holm-corrected when there are several treatments. A "no detectable effect"
  line carries the sensitivity figure, so it says what was ruled out.

## What the intake does to a file

The intake never cleans a file silently. A fault it can neither repair nor
drop with a count stops the run.

Repaired and counted:

- stray spaces trimmed from group labels, and group labels matched
  regardless of capitals
- value replacements the config declares applied

A byte-order mark at the start of a CSV, which Excel writes, is accepted
without a count.

Dropped and counted, with a stop above 1% of rows:

- a row whose metric value is blank, not a number, or infinite, and the same
  for a pre-experiment column or the day column
- a row whose group label is blank
- a negative value where the config says the metric cannot be negative
- a value other than 0 or 1 in a binary metric

A file past that share is not clean enough to read, and a readout of the
remaining rows would describe a different experiment. Some exports leave a
metric blank for users who never did the thing it counts, such as spend for
a user who bought nothing. Fill those blanks with zero before running the
tool, or the drops will pass 1% and stop the run.

Stops the run, naming what failed:

- a column the config names missing from the file, or a group label the
  config does not list
- a user appearing twice, when the config names a user id column
- a data file that does not match the md5 the config pins

Flagged in the limitations block:

- group sizes off from the intended split
- the first continuous metric (the primary, if it is continuous) with its
  top 1% of users carrying more than a quarter of its total
- fewer than 10 successes or failures in an arm on a binary primary metric
- a smallest detectable effect above a 5% relative lift
- no pre-experiment column
- gaps in the day sequence
- a replay with more looks than the 14 its boundary accuracy is checked on

## Running it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt  # .venv/bin/python on Unix
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_synthetic_check.py
.venv/Scripts/python scripts/run_fetch_hillstrom.py   # fetches the 2008 dataset (not redistributed)
.venv/Scripts/python scripts/run_analyze_experiment.py --config analysis/hillstrom/config.toml
.venv/Scripts/python scripts/run_plan_experiment.py --baseline-rate 0.05 --minimum-lift 0.005
```

The analyze command prints the readout and saves it as `readout.txt` beside
the config, with the figures in a `figures/` folder. The `--out` flag
chooses another directory. The planning command prints the users needed per
group for a baseline rate and the smallest lift worth detecting, or for a
standard deviation and the smallest difference, at the chosen alpha and
power.

To read your own experiment, write a config and point the analyze command at
your file:

```bash
.venv/Scripts/python scripts/run_analyze_experiment.py --config my_experiment.toml --data my_data.csv
```

`analysis/hillstrom/config.toml` is a complete working example. Columns the
config does not name are ignored. These are the keys a config uses.

| Table and key | Needed | What it does |
|---|---|---|
| `[experiment]` `name` | yes | the readout's title |
| `[experiment]` `alpha` | no, 0.05 | the false-positive rate for the effect tests, the sensitivity, the corrections, and the replay, whose adjusted interval follows it; the other intervals stay 95% and the split check uses 0.001 |
| `[input]` `path` | no | the data file, relative to the config; `--data` overrides it |
| `[input]` `md5` | no | the data file's checksum, so a readout cannot quietly describe the wrong file; a different file stops the run |
| `[columns]` `arm` | yes | the column holding each user's group |
| `[columns]` `day` | no | the day each user entered the experiment, as a number; a date string is dropped as unreadable, and each distinct value is one look of the replay; turns on the replay |
| `[columns]` `user_id` | no | turns on the stop for a user appearing twice |
| `[columns.aliases.<column>]` | no | value replacements for the group column or another text column, old = new, counted as repairs; metric, pre-experiment, day, and user id values are read as numbers, and an alias table on one of those stops the run |
| `[arms]` `labels` | yes | the group labels, control first; reversing them flips the sign of every effect |
| `[arms]` `intended_split` | no | the planned share per group; turns on the check against the plan |
| `[[metrics]]` `name`, `kind` | yes | one entry per metric; `kind` is `binary` or `continuous` |
| `[[metrics]]` `primary` | one metric | the metric the decisions are made on |
| `[[metrics]]` `column` | no, the name | the metric's column in the file |
| `[[metrics]]` `nonnegative` | no | drops a row with a negative value |
| `[[metrics]]` `pre_column` | no | a value measured before assignment; turns on CUPED for that metric |
| `[truth]` | no | the synthetic check's generation settings; prints the planted-versus-found block |

The tool needs Python 3.11 or newer. There is no install step, because the
entry scripts put `src/` on the import path themselves, so a fresh clone
runs once the dependencies in `requirements.txt` are present. The library
depends on numpy, scipy, and statsmodels, and the figures need matplotlib,
which that file pins with the dev tools. Rerunning a readout with the
versions pinned there reproduces `readout.txt` byte for byte. CI regenerates
the synthetic readout on every push to master, with the newest versions
`pyproject.toml` allows, and fails if the committed copy has drifted. The
Hillstrom readout is outside that check, since its data is fetched rather
than committed and CI should not depend on a 2008 URL staying up.

## The two committed readouts

[analysis/hillstrom/readout.txt](analysis/hillstrom/readout.txt), quoted
above, reads the 2008 MineThatData e-mail experiment, in which 64,000
customers of a retailer were split in thirds between a men's e-mail, a
women's e-mail, and no e-mail. Fewer than 1% of its users spent anything,
which the limitations block flags, and a misspelled zip-code category is
normalized and counted by the intake. Both e-mail arms raise visits. Both
also raise spend after the Holm correction, though that result rests on the
fewer than 1% of users who spent. CUPED on the past-spending column removes
none of the noise, because spend in the year before the e-mail does not
predict spend in the two weeks after it.

[analysis/synthetic/readout.txt](analysis/synthetic/readout.txt) is not an
example of analysis. It is a test of the tool. One generated experiment with
planted effects runs through the same intake and readout as the real file,
and the readout is checked against what was planted. The generator plants a
lasting lift of +0.40 percentage points (pp) on conversion. It adds an extra
lift for the earliest users, a novelty effect, which halves with every three
days later a user enters. Averaged over the run, the expected lift at the
end is +0.67 pp, which the readout prints with "novelty averaged in". The
readout finds +0.67 pp with an interval of +0.39 to +0.95 pp, and its replay
crosses the spending boundary at look 4 of 14. The planted rise in spend,
+0.32 per user, is too small for this sample to detect under its heavy tail,
a few users spending far more than the rest. The generator plants that tail
through its whale settings, its name for those users, which the planted
block lists. The readout reports it as not detected, with an interval that
contains the planted value, rather than as found. CUPED removes 14% of the
noise on spend at this seed. Over eight seeds at a larger sample, measured
once and recorded in docs/NOTES.md, it removed 12% to 18%. The settings are
in the config beside the readout, and its planted-versus-found block prints
each one beside what was found.

## What it does not do

It reads one row per user, so a file of per-group summary totals is not
read. It assumes users were randomized one by one. An experiment that
randomized groups of users, such as stores or communities, needs a variance
estimate this tool does not have, and nothing in the file lets it detect
that. It draws no causal conclusion beyond the randomized comparison. The
replay reads a finished file and cannot watch a running experiment, because
each boundary depends on the final look's sample size. There are no futility
boundaries, so the replay can show an early stop on a clear effect but never
on a hopeless run. Each look holds the users enrolled by then with their
final outcomes, so the replay sees outcomes a live look would not yet have
had. After a sequential stop the only interval is the adjusted one. Every
metric is one value per user and every comparison is a difference in means,
so a ratio such as revenue per order, whose denominator varies by user, is
out of scope. The relative lift is a point estimate without an interval.

## For the technical reader

Statistical code can be wrong and still return plausible numbers, so the
suite tests statistical properties. The false-positive rates of the z-test
and the spending boundaries are simulated at fixed seeds, interval coverage
is measured rather than assumed, and the proportion power function is pinned
against an exact binomial calculation. The boundary calculation, run with
Lan and DeMets's Pocock-type spending, is checked against published values.
The boundaries are also checked against exact bivariate and trivariate
normal rectangle probabilities from scipy, against Monte Carlo crossing
rates, and against a grid refinement. The adjusted intervals are checked for
joint coverage over 20,000 simulated null paths. The intake has one test per
rule on injected bad data and one per limitation code on data built to raise
it. The readout text is pinned to a saved copy of the expected text that is
read by eye when it changes.

The monitoring boundaries are computed at the sample sizes the looks
actually reached rather than on a planned schedule. Each look's share of the
run's final precision is taken from user counts rather than from the
observed variances, so where the variance drifts across looks the boundaries
are approximate. The method is the recursive numerical integration of
Armitage, McPherson, and Rowe, which carries forward the probability that no
earlier look has crossed and solves each look's cutoff from it. The adjusted
intervals are each look's estimate plus or minus that look's boundary times
its standard error.

[docs/NOTES.md](docs/NOTES.md) holds the dated measurements the README
cites, the history of the synthetic generator, and the record of what was
removed on 2026-10-21.

## Repo layout

- `src/abkit/io.py`: the config, the loader, and the checks and repairs.
- `src/abkit/readout_analysis.py`: every number the readout reports,
  computed once, no text.
- `src/abkit/readout_text.py` and `readout_figures.py`: the lines and the
  figures.
- `src/abkit/frequentist.py`, `power.py`, and `cuped_adjustment.py`: thin
  wrappers over statsmodels and scipy. The tests among them return the
  repo's result dataclasses.
- `src/abkit/sequential.py`: the spending boundaries and the adjusted
  intervals.
- `src/abkit/results.py`: the frozen result dataclasses the statistical
  functions return.
- `src/abkit/generators.py`: the generated experiment the synthetic pipeline
  check runs on.
- `scripts/`: the analyze, plan, synthetic, and fetch commands.
- `analysis/`: the two committed readouts with their configs and figures.
- `tests/`: the suite, mostly one file per module.
- `docs/NOTES.md`: dated measurements and decisions.

## Conventions

Every public function carries full type hints, and every statistical test
returns a frozen dataclass rather than a tuple. Randomness always flows
through an explicit `numpy.random.Generator` parameter, with no module-level
seeding. Inputs fail with an explicit `ValueError`, except in two places.
The intake's coercion catch turns an unreadable value into a counted and
reported row drop, and the replay omits itself on a data error after
printing it in full. Tests are two-sided unless the function name says
otherwise, p-value adjustments take and return arrays in the original order,
and anything approximate states its accuracy in its docstring and has a test
pinning it against a published or simulated value. Spelling is US
throughout.

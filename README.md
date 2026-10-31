# abkit

abkit is a command-line program that reads an exported A/B test file and
prints a readout, a text report of the data checks and the test results. Its
goal is to make the evaluation of an A/B test one ordered pass instead of
one test at a time. The readout puts the data checks first and the result
after. The standard tests come from statsmodels and scipy. The repo's own
code checks and repairs the file first, a step called the intake. The same
code sets the order of the readout. The readout also replays a finished run
day by day, as if the experiment were still being watched, which those
libraries do not do. It is built for an analyst who has an exported file and
no experimentation platform, or who wants a second opinion on a dashboard.

## The readout

The readout is a text file of labeled blocks rather than paragraphs. Each
heading carries a few words on how to read its block, and the file closes
with one decision line per treatment and the figures' paths. The committed
readout of the Hillstrom e-mail experiment follows whole, with a key to its
blocks after it. The experiment's data was published by MineThatData
in 2008. In the experiment, 64,000 customers of a retailer were split in thirds
between a men's e-mail, a women's e-mail, and no e-mail.

The file is
[analysis/hillstrom/readout.txt](analysis/hillstrom/readout.txt).

```text
Hillstrom e-mail experiment
Data: what the intake did before any statistic
  users   No E-Mail 21,306 (33.3%)   Mens E-Mail 21,307 (33.3%)   Womens E-Mail 21,387 (33.4%)
  alias normalized   28,776 rows   zip_code 'Surburban' to 'Suburban', first at data row 2

Limitations (*): read every result below with these in view
  * the top 1% of users carry 100% of spend, because fewer than 1% have any spend at all
  * the smallest detectable effect on visit is above a modest 5% lift for Mens E-Mail vs No E-Mail
      (8.8% of control) and Womens E-Mail vs No E-Mail (8.8% of control); a 'not detected' result
      there would rule out little

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

Primary metric, visit: effect, 95% interval, p; interval first
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
  every result below is read with these in view. Each flag comes from the
  data alone, never from an effect the readout found, so a flag can sit
  beside a result it does not affect, such as a sensitivity flag above an
  effect that was detected.
- **Group split.** Whether users landed in the groups as planned, tested
  against the intended split in the config. This is a sample ratio mismatch
  (SRM) check. A split counts as broken when its chi-square test gives p
  below 0.001. A broken split means users went missing or were assigned
  unevenly, and no statistic repairs that, so the decision lines are
  replaced by one stop line. With no intended split on record, the block
  says there is no check, and the Data block's shares are left to the
  reader.
- **Sensitivity.** The smallest lift on the primary metric this sample could
  have detected with 80% power, known as the minimum detectable effect
  (MDE). Lifts on a yes/no metric are in percentage points (pp). A "not
  detected" result means an effect at least that large would have been seen
  with at least 80% probability and says nothing about smaller ones.
- **Primary metric.** The effect, its 95% interval, and p for each
  treatment. For a yes/no metric the test is a two-proportion z-test, which
  compares the two groups' rates. For a continuous metric, such as spend, it
  is Welch's t-test, which compares the two groups' means and still works
  when their variances differ. With several treatments the comparisons are
  Holm-corrected, which caps the chance of any false positive among them at
  alpha, the false-positive rate the config sets. Holm tests the smallest p
  against alpha divided by the number of comparisons, the next against alpha
  divided by one fewer, and so on, stopping at the first that fails. The
  readout's Holm column restates that test as a corrected p, and an effect
  counts as detected when its corrected p is below alpha. The Sensitivity
  block uses the strictest of these levels, alpha divided by the number of
  treatments, so an effect that large has an 80% chance of detection however
  the other comparisons come out.
- **CUPED** (controlled-experiment using pre-experiment data). It reports a
  metric's effect with less noise, after a regression adjustment removes the
  part of each user's outcome their pre-experiment value already predicted.
  It runs on each metric that has a pre-experiment column. In the quoted
  readout that metric is spend, not the primary metric. "Noise removed" is
  the share by which the adjustment shrank the estimate's variance, so more
  removed means a narrower interval. A 0% means the pre-experiment column
  predicts almost none of the outcome, as on Hillstrom. Its p is not
  corrected for the other comparisons.
- **Other metrics.** The secondary metrics, corrected two ways, printed
  when they give two or more comparisons. Holm, as on the primary metric,
  suits a ship decision. Benjamini-Hochberg instead caps the false discovery
  rate, the expected share of false positives among the results it flags.
  That is enough for a screen of what to test next.
- **Decision.** One line per treatment, judged on the primary metric's p,
  Holm-corrected when there are several treatments. A "no detectable effect"
  line carries the sensitivity figure, so it says what was ruled out.

## The monitoring replay

The replay shows where a finished run could have been stopped early while
holding the false-positive rate. The quoted readout has no replay block,
because the Hillstrom file has no day column. A file whose day column holds
at least three distinct days gets the block, printed after the other
results. When a day cannot be tested, such as a first day with no
conversions in either group, the readout leaves the block out and the
console prints the error.

A look is one check of the primary metric's running totals. The z-test
divides the effect by its standard error to get a score called z, and the
replay computes that score for a continuous metric too. The test calls the
effect detected when z passes 1.96 in either direction, the same as p below
0.05. On a single check, that cutoff holds the false-positive rate at 5%.
Checking that cutoff at every look raises the rate well above 5%. A spending
design treats the 5% as a budget divided across the looks. That gives each
look its own cutoff, called the spending boundary. The O'Brien-Fleming
design used here spends almost none of the budget early and most of it at
the end. The replay applies that design to the finished run, checking it
look by look against the spending boundary.

The block prints the look where z first crosses the spending boundary, if it
does, and the final look's effect with its adjusted interval, an interval
widened to allow for the repeated looks. A crossing means the design allows
a stop at that look. For a run checked at exactly these looks, stopping at
the first crossing holds the false-positive rate at 5%. A crossing says
nothing about the effect's size. "Never crosses" means no look allowed a
stop. The fixed 1.96 cutoff, labeled naive, is printed for contrast.

On a run that ended on schedule, the primary block's interval is the result.
The adjusted interval at a look is the one a stop there would have had to
report. The readout prints the final look's adjusted interval, which is a
little wider than the primary block's interval because the adjusted interval
is built to cover the true effect at every look at once. With several
treatments the replay runs on the first treatment listed in the config, a
rule that does not look at the results.

![The replay drawn from the synthetic check](analysis/synthetic/figures/monitoring.png)

*This figure draws the replay from the synthetic check, which runs on
generated data, not a real experiment. The observed |z| passes the fixed
1.96 cutoff at look 1, but the spending design gives that look almost none
of the 5% and allows no stop there. The observed |z| crosses the spending
boundary at look 4 of 14. The panel on the right shows each look's effect
and its adjusted interval. No interval is drawn at look 1, where no stop is
allowed.*

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
- a negative value where the config says the metric cannot be negative,
  in the metric or its pre-experiment column
- a value other than 0 or 1 in a binary metric or its pre-experiment column

A file past that share is not clean enough to read, and a readout of the
remaining rows would describe a different experiment. Some exports leave a
metric blank for users who never did the thing it counts, such as spend for
a user who bought nothing. Fill those blanks with zero before running the
tool, or the drops will pass 1% and stop the run.

Stops the run, naming what failed:

- a column the config names missing from the file, or a group label the
  config does not list
- a user appearing twice, when the config names a user id column
- a data file whose checksum does not match the md5 the config pins,
  checked by the analyze command before the intake reads the file

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
your file, a comma-separated CSV in UTF-8 with a header row:

```bash
.venv/Scripts/python scripts/run_analyze_experiment.py --config my_experiment.toml --data my_data.csv
```

`analysis/hillstrom/config.toml` is a complete working example. Data columns
the config does not name are ignored. The table lists the keys a config
uses.

| Table and key | Needed | What it does |
|---|---|---|
| `[experiment]` `name` | yes | the readout's title |
| `[experiment]` `alpha` | no, 0.05 | the false-positive rate for the effect tests, the sensitivity, the corrections, and the replay, whose adjusted interval follows it; the other intervals stay 95% and the split check uses 0.001 |
| `[input]` `path` | no | the data file, relative to the config; `--data` overrides it |
| `[input]` `md5` | no | the data file's checksum, so a readout cannot be made from the wrong file; a different file stops the run |
| `[columns]` `arm` | yes | the column holding each user's group |
| `[columns]` `day` | no | the day each user entered the experiment, as a whole number such as days since the start; a date string is dropped as unreadable, and each distinct value is one look of the replay, so a timestamp stored as a number, such as seconds, would make nearly every user its own look; with at least three distinct days, turns on the replay |
| `[columns]` `user_id` | no | turns on the stop for a user appearing twice |
| `[columns.aliases.<column>]` | no | value replacements for the group column or another text column, old = new, counted as repairs; metric, pre-experiment, and day values are read as numbers and user ids as text, and an alias table on any of those columns stops the run |
| `[arms]` `labels` | yes | the group labels, control first; reversing them flips the sign of every effect |
| `[arms]` `intended_split` | no | the planned share per group; turns on the check against the plan |
| `[[metrics]]` `name`, `kind` | yes | one entry per metric; `kind` is `binary` or `continuous` |
| `[[metrics]]` `primary` | one metric | the metric the decisions are made on |
| `[[metrics]]` `column` | no, the name | the metric's column in the file |
| `[[metrics]]` `nonnegative` | no | drops a row with a negative value; set it for a metric that cannot go below zero, such as a count, and leave it off where negatives are real, such as revenue net of refunds |
| `[[metrics]]` `pre_column` | no | a value measured before assignment; turns on CUPED for that metric |
| `[truth]` | no | the synthetic check's generation settings; prints the planted-versus-found block, labels the closing lines as check output, and marks the replay figure as synthetic |

The tool needs Python 3.11 or newer. There is no package install step,
because the entry scripts put `src/` on the import path themselves, so a
fresh clone runs once the dependencies in `requirements.txt` are present.
The library depends on numpy, scipy, and statsmodels, and the figures need
matplotlib, which that file pins with the dev tools. Rerunning a readout
with the versions pinned there reproduces `readout.txt` byte for byte. The
continuous integration checks regenerate the synthetic readout on every push
to master, with the newest versions `pyproject.toml` allows, and fail if the
committed copy has drifted. The Hillstrom readout is outside that check,
since its data is fetched rather than committed and the checks should not
depend on a URL from 2008 staying up.

## The two committed readouts

[analysis/hillstrom/readout.txt](analysis/hillstrom/readout.txt), quoted
above, reads the Hillstrom e-mail experiment. Fewer than 1% of its users
spent anything, which the limitations block flags. The intake also
normalizes and counts a misspelled zip-code category. Nothing else in the
readout reads that column, so the config's alias for it is there to show the
repair on real data. Both e-mail arms raise visits. Both also raise spend
after the Holm correction, though that result rests on the fewer than 1% of
users who spent. CUPED on the past-spending column removes under 1% of the
noise, because spend in the year before the e-mail does not predict spend in
the two weeks after it.

[analysis/synthetic/readout.txt](analysis/synthetic/readout.txt) is a purely
synthetic check, not an example of analysis. One generated experiment with
planted effects runs through the same intake and readout as a real file. The
check's purpose is to show the pipeline works from start to finish. This one
run is not evidence that the statistics are right. That evidence is the test
suite, described under the section "For the technical reader". The readout's
block headed "Planted versus found" lists the generation settings, with what
the readout found beside the ones it can compare.

## What it does not do

It reads one row per user, so a file of per-group summary totals is not
read. It assumes users were randomized one by one. An experiment that
randomized groups of users, such as stores or communities, needs a variance
estimate this tool does not have, and nothing in the file lets it detect
that. It draws no causal conclusion beyond the randomized comparison. The
replay reads a finished file and cannot watch a running experiment, because
each look's boundary depends on its share of the final sample size, which is
known only once the run ends. A live version would need a planned final size
set before the start. There are no futility boundaries, so the replay can
show an early stop on a clear effect but never on a hopeless run. Each look
holds the users enrolled by then with their final outcomes, so the replay
sees outcomes a live look would not yet have had. Every metric is one value
per user and every comparison is a difference in means, so a ratio such as
revenue per order, whose denominator varies by user, is out of scope.

## For the technical reader

Statistical code can be wrong and still return plausible numbers, so the
suite tests statistical properties. The false-positive rates of the z-test
and the spending boundaries are simulated at fixed seeds. On fourteen daily
looks with no true effect, checking the fixed 1.96 cutoff at every look
detects an effect on 21.8% of runs, and the O'Brien-Fleming design on 4.9%.
Both rates come from 50,000 simulated runs at seed 42 and are pinned by a
test in `tests/test_group_sequential.py`. Interval coverage is measured
rather than assumed, and the proportion power function is pinned against an
exact binomial calculation. The boundary calculation is checked against
published values for a second spending design, against exact probabilities
from scipy for two- and three-look designs, against simulated crossing
rates, and against a finer integration grid. Across 20,000 simulated runs
with no effect, the adjusted intervals cover zero at every look at once on
at least 95% of runs, allowing for simulation error. The intake has one test
per rule on injected bad data and one per limitation code on data built to
raise it. The readout text is pinned to a saved copy of the expected text
that is read by eye when it changes.

The monitoring boundaries are computed at the sample sizes the looks
actually reached rather than on a planned schedule. Each look's share of the
run's final precision is taken from user counts rather than from the
observed variances, so where the variance drifts across looks the boundaries
are approximate. The method is the recursive numerical integration of
Armitage, McPherson, and Rowe, which carries forward the probability that no
earlier look has crossed and solves each look's cutoff from it. The adjusted
intervals are each look's estimate plus or minus that look's boundary times
its standard error.

[docs/NOTES.md](docs/NOTES.md) holds dated measurements and decisions, and
the record of what was removed on 2026-10-21.

## Repo layout

- `src/abkit/experiment_intake.py`: the config, the loader, and the checks
  and repairs.
- `src/abkit/readout_analysis.py`: every number the readout reports,
  computed once, no text.
- `src/abkit/readout_text.py` and `readout_figures.py`: the lines and the
  figures.
- `src/abkit/significance_tests.py`, `power_analysis.py`, and
  `cuped_adjustment.py`: thin wrappers over statsmodels and scipy. Their
  statistical tests return the repo's result dataclasses.
- `src/abkit/group_sequential.py`: the spending boundaries and the adjusted
  intervals.
- `src/abkit/result_types.py`: the frozen result dataclasses the statistical
  functions return.
- `src/abkit/synthetic_data.py`: the generated experiment the synthetic
  pipeline check runs on.
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

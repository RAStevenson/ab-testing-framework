# abkit

abkit is a command-line program that reads an exported A/B test file and
prints a readout, a text report of the data checks and the test results. Its
goal is to make the evaluation of an A/B test one ordered pass instead of
one test at a time. The readout puts the data checks first and the result
after. The standard tests come from statsmodels and scipy. The project
checks and repairs the file before any test runs, a step called the intake.
The project also sets the order of the readout. The readout replays a
finished run day by day, as if the experiment were still being watched.
Neither statsmodels nor scipy offers that replay. It is built for an analyst
who has an exported file and no experimentation platform, or who wants a
second opinion on a dashboard.

## Setup and run

Requires git and Python 3.11 or later.

In a terminal, run the following commands.

1. Clone the repository:

   ```
   git clone https://github.com/RAStevenson/ab-testing-framework.git
   ```

2. Navigate to the root directory:

   ```
   cd ab-testing-framework
   ```

3. Create a new environment and activate it (on Mac or Linux the second
   line is `source .venv/bin/activate`):

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Fetch the Hillstrom dataset, which is the real-data example, and read it:

   ```
   python scripts/run_fetch_hillstrom.py
   python scripts/run_analyze_experiment.py --config analysis/hillstrom/config.toml
   ```

The first command downloads the 2008 dataset from its original URL into
`data/hillstrom/`, which git ignores, and checks it against a pinned
checksum. The dataset is not redistributed here because its publisher never
stated a license. The second command prints the readout quoted under the
section "The readout" and saves it as `readout.txt` beside the config, with
the figures in a `figures/` folder. Reading the file takes a few seconds
after the download. A measured run is recorded in `docs/NOTES.md`.

The other entry points, with every argument listed under [Script
arguments](#script-arguments) at the end:

```
python scripts/run_synthetic_check.py
python scripts/run_plan_experiment.py --baseline-rate 0.05 --minimum-lift 0.005
pytest
```

The first command regenerates the synthetic check in `analysis/synthetic/`,
a run of the pipeline on generated data. The second prints the users needed
per group for a baseline rate and the smallest lift worth detecting, or for
a standard deviation and the smallest difference, at the chosen alpha and
power. The third runs the test suite.

abkit itself is not installed. The entry scripts put `src/` on the import
path, so a fresh clone runs once the dependencies are installed. The
versions in `requirements.txt` are the exact ones the committed readouts
were produced with. The library depends on numpy, scipy, and statsmodels,
and the figures need matplotlib, which that file pins with the dev tools.
Rerunning a readout with those versions reproduces `readout.txt` byte for
byte. The continuous integration checks regenerate the synthetic readout on
every push to master, with the newest versions `pyproject.toml` allows, and
fail if the committed copy has drifted. The Hillstrom readout is outside
that check, since its data is fetched rather than committed and the checks
should not depend on a URL from 2008 staying up.

### Using it on your own data

To read your own experiment, write a config and point the analyze command at
your file, a comma-separated CSV in UTF-8 with a header row:

```
python scripts/run_analyze_experiment.py --config my_experiment.toml --data my_data.csv
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
| `[[metrics]]` `pre_column` | no | a column measured before assignment; turns on CUPED for that metric; a row whose cell there is not a number is dropped, and the metric's `nonnegative` and binary rules do not apply to that column |
| `[truth]` | no | the synthetic check's generation settings; prints the planted-versus-found block, labels the closing lines as check output, and marks the replay figure as synthetic |

## How it works

A run has three stages. The intake, in `src/abkit/experiment_intake.py`,
reads the config and the CSV, repairs and drops what its rules allow with a
count of each, and stops on a fault it cannot count. The analysis, in
`src/abkit/readout_analysis.py`, computes every number the readout reports.
The writers, `src/abkit/readout_text.py` and `src/abkit/readout_figures.py`,
print the lines and draw the figures. The tests, the power calculation,
and CUPED are thin wrappers over statsmodels and scipy, in the modules
`significance_tests.py`, `power_analysis.py`, and `cuped_adjustment.py`.
The replay's spending boundaries and adjusted intervals are the project's
own, in `src/abkit/group_sequential.py`.

## The readout

The readout is a text file of labeled blocks rather than paragraphs. Each
heading carries a few words on how to read its block, and the file closes
with one decision line per treatment and the figures' paths.

The committed real-data example is the Hillstrom e-mail experiment, whose
data was published by MineThatData in 2008. In the experiment, 64,000
customers of a retailer were split in thirds between a men's e-mail, a
women's e-mail, and no e-mail. Its readout follows whole, with a key to its
blocks after it.

The file is [analysis/hillstrom/readout.txt](analysis/hillstrom/readout.txt).

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
  alpha, the false-positive rate the config sets. The readout's Holm column
  is the corrected p, and an effect counts as detected when that p is below
  alpha. The Sensitivity block is computed at alpha divided by the number
  of treatments, the strictest level Holm applies, so an effect that large
  has an 80% chance of detection however the other comparisons come out.
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
  line carries the sensitivity figure, so it says what size of effect the
  run could have seen.

On Hillstrom, fewer than 1% of users spent anything, which the limitations
block flags. The intake also normalizes and counts a misspelled zip-code
category. Nothing else in the readout reads that column, so the config's
alias for it is there to show the repair on real data. Both e-mail arms
raise visits. Both also raise spend after the Holm correction, though that
result rests on the fewer than 1% of users who spent. CUPED on the
past-spending column removes under 1% of the noise, because spend in the
year before the e-mail barely predicts spend in the two weeks after it.

## The monitoring replay

The replay shows where a finished run could have been stopped early while
holding the false-positive rate. The quoted readout has no replay block,
because the Hillstrom file has no day column. A file whose day column holds
at least three distinct days gets the block, printed after the other
results. When the data cannot support a test on one of those days, such as a
first day with no conversions in either group, the readout leaves the block
out and the console prints the error.

A look is one check of the primary metric's running totals, one per day in
the file. The z-test divides the effect by its standard error to get a score
called z. The replay computes that score for a continuous metric too. The
test calls the effect detected when z passes 1.96 in either direction, the
same as p below 0.05. On a single check, that cutoff holds the
false-positive rate at 5%. Checking that cutoff at every look raises the
rate well above 5%. A spending design treats the 5% as a budget divided
across the looks. That gives each look its own cutoff, called the spending
boundary. The O'Brien-Fleming design used here spends almost none of the
budget early and most of it at the end. The replay applies that design to
the finished run, checking it look by look against the spending boundary.

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

The run behind the figure is a purely synthetic check, not an example of
analysis. One generated experiment with planted effects, from
`src/abkit/synthetic_data.py`, runs through the same intake and readout as a
real file. The check's purpose is to show the pipeline works from start to
finish. This one run is not evidence that the statistics are right. That
evidence is the test suite, described under the section "How the statistics
are tested". The readout's block headed "Planted versus found" lists the
generation settings, with what the readout found beside the ones it can
compare.

The file is [analysis/synthetic/readout.txt](analysis/synthetic/readout.txt).

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
- a row with a negative value where the config says the metric cannot be
  negative
- a row with a value other than 0 or 1 in a binary metric

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
- the top 1% of users carrying more than a quarter of a continuous
  metric's total, checked on the first continuous metric (the primary, if it
  is continuous) over the users in the control group and the first treatment
- fewer than 10 successes or failures in an arm on a binary primary metric
- a smallest detectable effect above a 5% relative lift
- no pre-experiment column
- gaps in the day sequence
- a replay with more looks than the 14 its boundary accuracy is checked on

## How the statistics are tested

A statistical routine can be wrong and still return plausible numbers, so
the suite tests statistical properties. The false-positive rates of the
z-test and the spending boundaries are simulated at fixed seeds. On
fourteen daily looks with no true effect, checking the fixed 1.96 cutoff at
every look detects an effect on 21.8% of runs, and the O'Brien-Fleming
design on 4.9%. Both rates come from 50,000 simulated runs at seed 42 and
are pinned by a test in `tests/test_group_sequential.py`. Interval coverage
is measured rather than assumed, and the proportion power function is
pinned against an exact binomial calculation. The boundary calculation is
checked against published values for a second spending design, against
exact probabilities from scipy for two- and three-look designs, against
simulated crossing rates, and against a finer integration grid. Across
20,000 simulated runs with no effect at seed 43, the adjusted intervals
cover zero at every look at once on at least 95% of runs, allowing for
simulation error. The intake has one test per rule on injected bad data and
one per limitation code on data built to raise it. The readout text is
pinned to a saved copy of the expected text that is read by eye when it
changes.

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

## Limitations

The tool reads one row per user, so a file of per-group summary totals is
not read. It assumes users were randomized one by one. An experiment that
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

## Conventions

Every public function carries full type hints, and every statistical test
returns a frozen dataclass from `src/abkit/result_types.py` rather than a
tuple. Randomness always flows through an explicit `numpy.random.Generator`
parameter, with no module-level seeding. Inputs fail with an explicit
`ValueError`, except in two places. The intake's coercion catch turns an
unreadable value into a counted and reported row drop, and the replay omits
itself on a data error after printing it in full. Tests are two-sided unless
the function name says otherwise, p-value adjustments take and return arrays
in the original order, and anything approximate states its accuracy in its
docstring and has a test pinning it against a published or simulated value.
Spelling is US throughout.

## Script arguments

This lists every argument of every user-facing script, mirroring `--help`,
which each script prints with `-h`.

### `run_fetch_hillstrom.py`

Takes no arguments. It downloads the Hillstrom file from its original URL
into `data/hillstrom/`, checks the download against the SHA-256 pinned in
the script, and refuses a file that does not match, keeping the rejected
bytes beside the cache for review. A cached file that matches is not
downloaded again.

### `run_analyze_experiment.py`

Required:

- `--config`: the experiment TOML, with the keys listed under "Using it on
  your own data". `analysis/hillstrom/config.toml` is a complete example.

Optional:

- `--data`: the data file to read, overriding the config's `[input]`
  `path`. The config's `md5`, when set, is still checked against this file.
- `--out`: the directory for `readout.txt` and the `figures/` folder.
  Default is the config's directory.

### `run_synthetic_check.py`

Optional:

- `--config`: the recipe TOML, an experiment config with a `[truth]` table
  of generation settings. Default `analysis/synthetic/config.toml`.
- `--out`: the directory for the readout and figures. Default is the
  config's directory. The generated rows go to the config's `[input]`
  `path`, which for the default recipe is `data/synthetic/`, ignored by
  git.

### `run_plan_experiment.py`

Required, one pair or the other:

- `--baseline-rate` with `--minimum-lift`: for a yes/no metric, the
  control rate and the smallest absolute lift worth detecting, as fractions
  such as 0.05 and 0.005.
- `--standard-deviation` with `--minimum-difference`: for a continuous
  metric, its standard deviation and the smallest difference in means worth
  detecting, in the metric's own units.

Passing both pairs, or neither, stops the command with an error naming the
choice.

Optional:

- `--alpha`: the false-positive rate. Default 0.05.
- `--power`: the chance of detecting an effect of the stated size. Default
  0.8.

# Notes: measurements and decisions

Dated measurements, decisions, and history that do not belong in the README.
Each measurement names its date, its conditions, and whether the loop that
produced it is in the repo.

## How the monitoring boundaries are computed (moved from the README, 2026-11-03)

The monitoring boundaries are computed at the sample sizes the looks
actually reached, as shares of the planned size, rather than on a planned
schedule of looks. A test pins that a run read short of its plan gets the
same boundaries for its looks as the same run read once it reached its plan.
Each look's share of the plan is taken from user counts rather than from the
observed variances, so where the variance drifts across looks the boundaries
are approximate. The method is the recursive numerical integration of
Armitage, McPherson, and Rowe, which carries forward the probability that no
earlier look has crossed and solves each look's cutoff from it. The adjusted
intervals are each look's estimate plus or minus that look's boundary times
its standard error.

## The committed runs use the same function as a user's run (a design decision, 2026-11-03)

Robert asked on his structural read whether the Hillstrom demonstration and
the synthetic check had paths of their own, which would make neither a test
of what a user runs. Hillstrom already ran the user's command,
`run_analyze_experiment.py`, on its config. The synthetic script called the
same library steps but not that script's two checks, that the data file
exists and that it matches the md5 its config pins. Neither check could have
stopped the synthetic run, because the script writes the file just before
reading it and its config pins no md5, so the change gives both scripts one
function rather than closing a gap.

Both checks and the library calls now sit in `write_experiment_readout`, in
`src/abkit/experiment_readout.py`, which both scripts call. The synthetic
script only generates and writes the rows before that call. The only
Hillstrom-specific code left in `run_analyze_experiment.py` is the hint to
run the fetch script when the Hillstrom file is missing. Robert ruled to
keep it. It sits in that script's error handler rather than in the library,
and it fires only when the missing file is the data file.

## What the test suite pins (moved from the README, 2026-11-03)

Robert cut the README's testing section on his structural read of
2026-11-03 as padding. Its measurements keep their record here, verbatim.

A statistical routine can be wrong and still return plausible numbers, so
the suite tests statistical properties. The false-positive rates of the
z-test and the spending boundaries are simulated at fixed seeds. On fourteen
daily looks with no true effect, checking the fixed 1.96 cutoff at every
look detects an effect on 21.8% of runs, and the spending design on 4.9%.
Both rates come from 50,000 simulated runs at seed 42 and are pinned by a
test in `tests/test_group_sequential.py`. Interval coverage is measured
rather than assumed, and the proportion power function is pinned against an
exact binomial calculation. The boundary calculation is checked against
published values for a second spending design, against exact probabilities
from scipy for two- and three-look designs, against simulated crossing
rates, and against a finer integration grid. Across 20,000 simulated runs
with no effect at seed 43, the adjusted intervals cover zero at every look
at once on at least 95% of runs, allowing for simulation error. The intake
has one test per rule on injected bad data and one per limitation code on
data built to raise it. The readout text is pinned to a saved copy of the
expected text that is read by eye when it changes.

## The power cost of monitoring against the plan (2026-11-03)

The planning command sizes a test for one check at the planned size. Over
fourteen equally spaced daily looks against that plan, the final look's
boundary from `design_boundaries` is |z| 2.145 rather than 1.96, so the test
detects a real effect a little less often. With an effect sized for 80%
power on a single check, 200,000 simulated runs at seed 42 detected it on
77.8% of runs when monitored and on 80.2% when checked once, where the 0.2
above 80 is simulation error. A plan about 5.6% larger restored 80% under
monitoring. The final boundary is lower with fewer looks, 1.979 at two
equally spaced looks and 2.063 at five, so fewer looks should cost less, but
only fourteen were simulated. The runs draw z-scores across the looks
directly rather than simulating users. The loop was run by hand on
2026-11-03 and is not in the repo.

## The monitoring method's name (2026-11-01)

The monitoring is Lan-DeMets alpha spending with an O'Brien-Fleming-style
spending function, not the O'Brien-Fleming design itself. The design O'Brien
and Fleming published in 1979 assumes a fixed number of equally spaced
looks. Lan and DeMets (Biometrika, 1983) replaced that schedule with a
spending function, which gives how much of the alpha may be used by each
share of the plan, so the looks need no schedule in advance. The function
`obrien_fleming_spending` in `src/abkit/group_sequential.py` follows the
shape of the 1979 boundaries. Both papers were written for clinical trials.

The plan and the function have to be fixed before the data is seen. Sources
checked on 2026-11-01 disagree on how much harm follows when the results
decide when to look. The monitoring takes one look per day in the file, so
the results do not decide it.

## The Hillstrom readout's running time (2026-11-01)

```
python scripts/run_analyze_experiment.py --config analysis/hillstrom/config.toml
```

The command above took about 3 seconds on 2026-11-01 on a Windows 11
desktop, with the data already fetched and the versions pinned in
`requirements.txt`. The synthetic check took about the same. Both were
measured once by hand, and no timing loop is in the repo.

## Scope reduced (2026-10-21)

The scope was cut because the project had grown past what its author could
learn and defend in an interview, and a smaller process is quicker to learn
and to publish. What stays is the data checks, the group split, the sample's
sensitivity, the primary effect with its interval, the corrections for
several comparisons, CUPED, and the replay of a run checked at every look.
Three things were removed, with the modules that only they used. The intake no
longer reads per-group summary totals, so the ASOS example and its committed
dataset are gone. The early-versus-late block, which tested whether an
effect changed over the run, is gone. The cross-check on a skewed metric,
which compared Welch's t-test with a bootstrap and a permutation test, is
gone with the resampling module. Git history keeps all of it. The entries in
the history section below record measurements from those modules and stay as
history.

## The heavy-tail flag on zero-inflated metrics (2026-10-09)

The limitation that flags a metric whose top 1% of users carry more than a
quarter of its total fires on any metric most users have none of, because
the few who have any carry all of it. On Hillstrom, fewer than 1% of users
spent anything, so the top 1% carry 100% of spend. The flag's wording names
that case ("because fewer than 1% have any spend at all") rather than
calling the metric heavy-tailed.

## The first-treatment rule (2026-10-09)

With several treatments, the monitoring replay runs on one contrast, and the
readout names which. The rule is the first treatment in `groups.labels`. It
does not look at the results, so the choice cannot favor whichever treatment
looks best, and the replay block prints the value "first treatment in the
group order, not by its result" on its "chosen" line. Until 2026-10-21 the
same rule also chose the contrast for the cross-check and for a split test
across checkpoints on per-group totals.

## The z-test's two standard errors (a design decision, 2026-10-07)

The z-test uses the pooled standard error for the statistic, through
statsmodels. The pooled variance is the right one under the null the test
evaluates. The interval for the estimate is the unpooled Agresti-Caffo
interval, also through statsmodels, because the unpooled variance is right
away from the null, where the interval lives. It also stays wide enough when
a group sits at or near 0% or 100%, where the plain unpooled interval comes
out too narrow.

## Removed features and the ASOS example (history)

### Bootstrap interval coverage (2026-10-07)

The bootstrap interval was scipy's BCa (bias-corrected and accelerated).
Measured with a one-off loop that is not in the repo, on lognormal groups of
400 users over 1,000 simulations at each of three seeds, BCa covered 92.6 to
93.7% against a nominal 95%. The percentile interval it replaced covered
93.9 to 94.7%. BCa was kept because it is scipy's default and the gap is
about a point.

### Permutation test timing (2026-10-06 and 2026-10-07)

The hand-built permutation loop that scipy's `permutation_test` replaced ran
10,000 permutations over 120,000 users in 7.5 s against scipy's 20.7 s, 2.8
times faster, measured 2026-10-06 with a benchmark script since removed. The
scipy call was kept because speed was not a concern at experiment sizes and
it is one less thing to defend. On the Hillstrom spend contrast, 21,306 and
21,307 users, the bootstrap and the permutation test each took 7 seconds at
10,000 resamples, measured 2026-10-07 with a one-off loop that is not in the
repo.

### The Spotify confidence trial (2026-10-06)

Spotify's confidence library, version 4.1.0, was run on the ASOS experiment
54a85a as a cross-check before the decision to call statsmodels directly.
The primary metric's z-test p-value was 0.6345 against this repo's 0.635.
The Holm-adjusted p-values on the other three metrics were identical. Its
sequential test over the 95 looks flagged no stop, and neither did this
repo's replay. Its final adjusted interval was -0.38 to +0.25 pp against
this repo's -0.377 to +0.241 pp. The source of that gap was not traced. The
run was made in a separate environment, not in the repo. The install brought
68 packages, pinned pandas below 3, and the package is marked beta.

### The early-versus-late users line (2026-10-20)

The ASOS readout printed variant_0 at -0.01 pp early and -1.26 pp late
against -0.07 pp over the whole run, and a reader panel could not reconcile
the three. The late window (the last 31 of 95 usable checkpoints) held
14,188 variant_0 users against 450,624 control, where the early window held
69,687 against 2,275,261, and both groups' rates rose from about 0.20 to about
0.36 across the run. The block then printed each window's users under every
contrast.

# Notes: measurements and decisions

Dated measurements and history the README refers to but does not carry. Each
entry names its date, its conditions, and whether the loop that produced it
is in the repo.

## Scope reduced (2026-10-21)

The scope was cut because the project had grown past what its author could
learn and defend in an interview, and a smaller process is quicker to learn
and to publish. What stays is the data checks, the group split, the sample's
sensitivity, the primary effect with its interval, the corrections for
several comparisons, CUPED, and the replay of a run checked at every look.
Three things were removed, with the code that only they used. The intake no
longer reads per-group summary totals, so the ASOS example and its committed
dataset are gone. The early-versus-late block, which tested whether an
effect changed over the run, is gone. The cross-check on a skewed metric,
which compared Welch's t-test with a bootstrap and a permutation test, is
gone with the resampling module. Git history keeps all of it. The entries in
the history section below record measurements from that code and stay as
history.

## CUPED on the synthetic check (2026-10-08 and 2026-10-09)

The generator decides whether a user spends, and how much, from hidden
random draws. The pre-experiment column comes from matching draws for the
period before, each correlated with its present one at 0.9. Spend itself
then correlates with past spend at only about 0.54, and CUPED removes the
square of that, about 30% when there are no whale purchases. That value,
0.296, is computed in closed form and checked against simulation by the
generator's tests. The whale purchases, a few users who spend far more than
the rest, have no counterpart in the pre-experiment column, so they add
noise CUPED cannot remove. The whale tail's variance is infinite, so how
much they add depends on how large the sample's biggest purchases happen to
be. Measured on 2026-10-09 with a one-off loop that is not in the repo, over
eight seeds at 400,000 users, the share CUPED removed ran from 12% to 18%.
The committed readout, seed 38 at 120,000 users, shows 14%.

The generator was revised once, on 2026-10-08. The first version tied past
to present spend only through the log of the amount, among the few users who
spend, so CUPED, which works on spend itself, removed 0%. The revision also
ties whether a user spent before to whether they spend now, and the setting
for the hidden draws' correlation rose from 0.6 to 0.9.

Since 2026-10-21 the planted-versus-found block checks the planted spend
effect against the CUPED estimate, because the cross-check that held the
unadjusted comparison was removed. CUPED estimates the same difference in
means with less noise.

## The heavy-tail flag on zero-inflated metrics (2026-10-09)

The limitation that flags a metric whose top 1% of users carry more than a
quarter of its total fires on any metric most users have none of, because
the few who have any carry all of it. On Hillstrom, fewer than 1% of users
spent anything, so the top 1% carry 100% of spend. The flag's wording names
that case ("because fewer than 1% have any spend at all") rather than
calling the metric heavy-tailed.

## The first-treatment rule (2026-10-09)

With several treatments, the monitoring replay runs on one contrast, and the
readout names which. The rule is the first treatment in `arms.labels`. It
does not look at the results, so the choice cannot favor whichever treatment
looks best, and the readout prints "chosen first treatment in the arm order,
not by its result" in the replay block. Until 2026-10-21 the same rule also
chose the contrast for the cross-check and for a split test across
checkpoints on per-group totals.

## The z-test's two standard errors (a design decision, 2026-10-07)

The z-test uses the pooled standard error for the statistic and the unpooled
Agresti-Caffo interval for the estimate, which stays a real interval when an
arm sits at 0% or 100%, both through statsmodels. The pooled variance is the
right one under the null the test evaluates, and the unpooled one is right
away from the null, where the interval lives.

## Removed code and the ASOS example (history)

### Bootstrap interval coverage (2026-10-07)

The bootstrap interval was scipy's BCa (bias-corrected and accelerated).
Measured with a one-off loop that is not in the repo, on lognormal arms of
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
10,000 resamples, measured 2026-10-07.

### The Spotify confidence trial (2026-10-06)

Spotify's confidence library, version 4.1.0, was run on the ASOS experiment
54a85a as a cross-check before the decision to call statsmodels directly.
The primary metric's z-test p-value was 0.6345 against this repo's 0.635.
The Holm-adjusted p-values on the other three metrics were identical. Its
sequential test over the 95 looks flagged no stop, and neither did this
repo's replay. Its final adjusted interval was -0.38 to +0.25 pp against
this repo's -0.377 to +0.241 pp. The source of that gap was not traced. The
install brought 68 packages, pinned pandas below 3, and the package is
marked beta.

### The early-versus-late users line (2026-10-20)

The ASOS readout printed variant_0 at -0.01 pp early and -1.26 pp late
against -0.07 pp over the whole run, and a reader panel could not reconcile
the three. The late window (the last 31 of 95 usable checkpoints) held
14,188 variant_0 users against 450,624 control, where the early window held
69,687 against 2,275,261, and both arms' rates rose from about 0.20 to about
0.36 across the run. The block then printed each window's users under every
contrast.

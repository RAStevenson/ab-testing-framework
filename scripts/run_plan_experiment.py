"""Plan an experiment: the users needed per group to detect the smallest effect worth acting on.

Give it the baseline conversion rate and the smallest lift worth
detecting, and it prints the users needed per group and in total at the
chosen power and alpha, in the readout's style. For a measured metric
such as spend, give the standard deviation and the smallest difference
worth detecting instead.

Run with:
    python scripts/run_plan_experiment.py --baseline-rate 0.05 --minimum-lift 0.005
    python scripts/run_plan_experiment.py --standard-deviation 12 --minimum-difference 0.5

Flags: --alpha (default 0.05) and --power (default 0.8) set the test's
false-positive rate and the chance of detecting the effect when it is
there. Nothing is written; the plan prints to the terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import traceback

from abkit.power import sample_size_two_means, sample_size_two_proportions


def plan_lines(args: argparse.Namespace) -> list[str]:
    """The plan as printable lines, from the parsed arguments."""
    rate_mode = args.baseline_rate is not None or args.minimum_lift is not None
    mean_mode = args.standard_deviation is not None or args.minimum_difference is not None
    if rate_mode == mean_mode:
        raise ValueError(
            "give either --baseline-rate with --minimum-lift (a yes/no metric) or "
            "--standard-deviation with --minimum-difference (a measured metric), not both"
        )
    lines = [f"Planning: users needed per group, {args.power:.0%} power, alpha {args.alpha:g}"]
    if rate_mode:
        if args.baseline_rate is None or args.minimum_lift is None:
            raise ValueError("--baseline-rate and --minimum-lift go together")
        per_group = sample_size_two_proportions(
            args.baseline_rate,
            args.baseline_rate + args.minimum_lift,
            alpha=args.alpha,
            power=args.power,
        )
        lines.append(
            f"  baseline {100 * args.baseline_rate:.1f}%   smallest lift worth detecting "
            f"{100 * args.minimum_lift:+.2f} pp ({args.minimum_lift / args.baseline_rate:.1%} "
            "of baseline)"
        )
    else:
        if args.standard_deviation is None or args.minimum_difference is None:
            raise ValueError("--standard-deviation and --minimum-difference go together")
        per_group = sample_size_two_means(
            args.minimum_difference, args.standard_deviation, alpha=args.alpha, power=args.power
        )
        lines.append(
            f"  standard deviation {args.standard_deviation:g}   smallest difference worth "
            f"detecting {args.minimum_difference:+g}"
        )
    lines += [
        f"  per group   {per_group:,}   total   {2 * per_group:,}",
        "",
        f"Decision: run until each group has {per_group:,} users, then read the experiment once",
    ]
    return lines


def main() -> None:
    """Parse the planning inputs and print the users needed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--baseline-rate", type=float, help="control conversion rate, e.g. 0.05")
    parser.add_argument(
        "--minimum-lift", type=float, help="smallest absolute lift worth detecting, e.g. 0.005"
    )
    parser.add_argument(
        "--standard-deviation", type=float, help="the measured metric's standard deviation"
    )
    parser.add_argument(
        "--minimum-difference", type=float, help="smallest difference in means worth detecting"
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="false-positive rate (0.05)")
    parser.add_argument("--power", type=float, default=0.8, help="chance of detection (0.8)")
    args = parser.parse_args()
    try:
        print("\n".join(plan_lines(args)))
    except ValueError:
        traceback.print_exc()
        print(
            "error: nothing was planned; fix the input named in the error above and rerun.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

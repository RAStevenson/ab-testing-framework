"""Generate the synthetic pipeline check and write its readout.

The config's [truth] table is the recipe: one experiment with a planted
lift on conversion, a transient novelty boost, a lift on the share of
users who spend, a whale tail, and a pre-experiment spend column for
CUPED. The rows are written to data/synthetic/ (never committed) and then
read through the identical intake, analysis, figure, and text path that
run_analyze_experiment.py uses, so the readout's planted-versus-found
block confirms the pipeline finds effects of known size when they are
large enough to detect, and reports smaller ones as not detected with
intervals that contain the planted value. The run tests
the system; it is not an example of analysis.

Run with:
    python scripts/run_synthetic_check.py

Flags: --config picks another recipe (default analysis/synthetic/config.toml),
and --out another output directory (default: the config's directory). The
generation seed is the recipe's own.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import traceback

import numpy as np

from abkit.generators import generate_example_rows
from abkit.io import load_config, load_user_rows
from abkit.readout_analysis import analyze_experiment
from abkit.readout_figures import draw_figures
from abkit.readout_text import save_readout

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "analysis" / "synthetic" / "config.toml"
COLUMNS = ("user_id", "group", "day", "converted", "spend", "pre_spend")


def write_rows(rows: dict[str, np.ndarray], data_path: Path) -> int:
    """Write the generated columns as a CSV with fixed formatting and return the row count."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = int(rows["user_id"].size)
    with open(data_path, "w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file, lineterminator="\n")
        writer.writerow(COLUMNS)
        for index in range(n_rows):
            writer.writerow(
                [
                    rows["user_id"][index],
                    rows["group"][index],
                    int(rows["day"][index]),
                    int(rows["converted"][index]),
                    f"{rows['spend'][index]:.2f}",
                    f"{rows['pre_spend'][index]:.2f}",
                ]
            )
    return n_rows


def main() -> None:
    """Generate the rows from the recipe, then load, analyze, draw, print, and save the readout."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="recipe TOML")
    parser.add_argument("--out", type=Path, help="output directory (default: config's directory)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        truth = config.get("truth")
        if truth is None:
            raise ValueError(f"{args.config} has no [truth] table to generate from")
        rows = generate_example_rows(
            int(truth["n_users"]),
            np.random.default_rng(int(truth["seed"])),
            base_rate=float(truth["base_rate"]),
            absolute_lift=float(truth["absolute_lift"]),
            n_days=int(truth["n_days"]),
            novelty_boost=float(truth["novelty_boost"]),
            novelty_half_life_days=float(truth["novelty_half_life_days"]),
            spend_rate=float(truth["spend_rate"]),
            spend_rate_lift=float(truth["spend_rate_lift"]),
            log_mean=float(truth["log_mean"]),
            log_sigma=float(truth["log_sigma"]),
            whale_rate=float(truth["whale_rate"]),
            whale_shape=float(truth["whale_shape"]),
            whale_scale=float(truth["whale_scale"]),
            pre_correlation=float(truth["pre_correlation"]),
        )
        data_path = args.config.parent / config["input"]["path"]
        n_rows = write_rows(rows, data_path)
        print(f"wrote {n_rows:,} rows to {data_path}\n")
        experiment = load_user_rows(config, data_path)
        readout = analyze_experiment(experiment)
        out_dir = args.out or args.config.parent
        figure_paths = draw_figures(readout, out_dir / "figures")
        lines = save_readout(readout, out_dir, figure_paths)
        print("\n".join(lines))
        print(f"\nsaved {out_dir / 'readout.txt'}")
        return
    except ImportError as error:
        # only a missing matplotlib has a known fix; any other failed import
        # is a broken install or a bug and keeps Python's own report
        if (error.name or "").partition(".")[0] != "matplotlib":
            raise
        traceback.print_exc()
        print(
            "error: the readout's figures are drawn with matplotlib, which "
            "installs with the pinned dependencies: pip install -r requirements.txt. "
            "No readout was written.",
            file=sys.stderr,
        )
        sys.exit(2)
    except (ValueError, OSError):
        traceback.print_exc()
        print(
            "error: the run stopped before readout.txt was written; the generated rows "
            "and figures written before the failure may already be replaced. Fix the "
            "problem named in the error above and rerun the same command.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

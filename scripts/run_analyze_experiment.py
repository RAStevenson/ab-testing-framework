"""Read an experiment from a config and a data file, and print its readout.

Point it at a TOML experiment description and a CSV with one row per
user. It checks and gently repairs the data, runs every check and analysis
the data can support, prints the readout, and saves it as readout.txt with
the figures beside it. The committed Hillstrom readout is this script run
on its committed config; a user's own experiment goes through the
identical path.

Run with:
    python scripts/run_analyze_experiment.py --config <experiment.toml>

Flags: --data overrides the config's input path, and --out chooses the
output directory (default: the config's directory).

When the config declares `input.md5`, the data file's digest is checked
before anything reads it, so a readout can never silently describe a
file other than the one its config pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import hashlib
import traceback

from abkit.experiment_intake import load_config, load_user_rows
from abkit.readout_analysis import analyze_experiment
from abkit.readout_figures import draw_figures
from abkit.readout_text import save_readout


def main() -> None:
    """Load the config and data, analyze, draw the figures, print the readout, and save it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", required=True, type=Path, help="experiment TOML")
    parser.add_argument("--data", type=Path, help="data file; overrides the config's input path")
    parser.add_argument("--out", type=Path, help="output directory (default: config's directory)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        data_path = args.data or args.config.parent / config["input"].get("path", "")
        if not data_path.is_file():
            hint = (
                " For the committed hillstrom example, run scripts/run_fetch_hillstrom.py first."
                if "hillstrom" in str(data_path).lower()
                else ""
            )
            raise ValueError(
                f"data file not found: {data_path} (set input.path in the config "
                f"or pass --data).{hint}"
            )
        pinned = config["input"].get("md5")
        if pinned:
            digest = hashlib.md5(data_path.read_bytes()).hexdigest()
            if digest != pinned:
                raise ValueError(
                    f"data file {data_path.name} has md5 {digest}, but the config "
                    f"pins {pinned}; refusing to read a file other than the "
                    "one the config was written against"
                )
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
            "error: the analysis stopped before readout.txt was written; figures "
            "drawn before the failure may already be replaced. Fix the problem named "
            "in the error above and rerun the same command.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

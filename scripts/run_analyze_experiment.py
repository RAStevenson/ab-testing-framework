"""Read an experiment from a config and a data file, and print its readout.

Point it at a TOML experiment description and a CSV with one row per
user. It checks and gently repairs the data, runs every check and analysis
the data can support, prints the readout, and saves it as readout.txt with
the figures beside it. The work is abkit.experiment_readout's, the one path
every run takes: the committed Hillstrom readout is this script run on its
committed config, and the synthetic check calls the same function.

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
import traceback

from abkit.experiment_readout import MissingDataFileError, write_experiment_readout


def main() -> None:
    """Parse the arguments, write the readout and its figures, and print it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", required=True, type=Path, help="experiment TOML")
    parser.add_argument("--data", type=Path, help="data file; overrides the config's input path")
    parser.add_argument("--out", type=Path, help="output directory (default: config's directory)")
    args = parser.parse_args()

    out_dir = args.out or args.config.parent
    try:
        lines = write_experiment_readout(args.config, out_dir, args.data)
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
    except (ValueError, OSError) as error:
        traceback.print_exc()
        # the Hillstrom file is fetched, never committed, so a fresh clone
        # running the committed config hits this first
        fetch_hint = (
            " For the committed Hillstrom example, run scripts/run_fetch_hillstrom.py first."
            if isinstance(error, MissingDataFileError)
            and "hillstrom" in str(error.data_path).lower()
            else ""
        )
        print(
            "error: the analysis stopped before readout.txt was written. Fix the problem "
            f"named in the error above and rerun the same command.{fetch_hint}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

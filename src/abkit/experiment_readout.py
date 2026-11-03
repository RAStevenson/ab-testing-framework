"""One experiment's readout from a config and a data file: the path every run takes.

``write_experiment_readout`` checks the data file against the config, loads
and cleans it, analyzes it, draws the figures, and writes readout.txt. A
user's run, the committed Hillstrom readout, and the synthetic check all go
through it, so no committed run has a path of its own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from abkit.experiment_intake import load_config, load_user_rows
from abkit.readout_analysis import analyze_experiment
from abkit.readout_figures import draw_figures
from abkit.readout_text import save_readout


class MissingDataFileError(FileNotFoundError):
    """The data file a run was pointed at does not exist.

    The path is kept on ``data_path`` rather than OSError's ``filename``, which
    would reprint it as an escaped string after the message.
    """

    def __init__(self, data_path: Path) -> None:
        """Name the missing file in the message and keep its path."""
        super().__init__(
            f"data file not found: {data_path} (set input.path in the config or pass --data)"
        )
        self.data_path = data_path


def input_data_path(config_path: Path, config: dict) -> Path:
    """The data file the config's ``input.path`` names, relative to the config's folder.

    Raises ValueError when the config names no data file.
    """
    relative = config["input"].get("path")
    if not relative:
        raise ValueError(f"{config_path} names no data file; set input.path or pass --data")
    return config_path.parent / relative


def write_experiment_readout(
    config_path: Path, out_dir: Path, data_path: Path | None = None
) -> list[str]:
    """Read the experiment, write readout.txt and its figures into out_dir, and return the
    readout's lines for printing.

    ``data_path`` overrides the config's ``input.path``. A missing data file
    raises MissingDataFileError naming the path. When the config pins
    ``input.md5``, the file's digest is checked before anything reads it, and a
    mismatch raises ValueError, so a readout cannot describe a file other than
    the one its config was written against.
    """
    config = load_config(config_path)
    data_path = data_path or input_data_path(config_path, config)
    if not data_path.is_file():
        raise MissingDataFileError(data_path)
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
    figure_paths = draw_figures(readout, out_dir / "figures")
    return save_readout(readout, out_dir, figure_paths)

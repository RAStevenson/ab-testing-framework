"""The shared readout path: config and data in, readout.txt and figures out,
with the data file checked against the config before anything reads it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from abkit.experiment_intake import load_config
from abkit.experiment_readout import (
    MissingDataFileError,
    input_data_path,
    write_experiment_readout,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixtures(tmp_path: Path, md5: str | None = None) -> Path:
    """Copy the fixture config and rows into tmp_path, optionally pinning an md5."""
    (tmp_path / "user_rows_ok.csv").write_bytes((FIXTURES / "user_rows_ok.csv").read_bytes())
    config_text = (FIXTURES / "experiment_ok.toml").read_text(encoding="utf-8")
    if md5 is not None:
        config_text = config_text.replace(
            'path = "user_rows_ok.csv"', f'path = "user_rows_ok.csv"\nmd5 = "{md5}"'
        )
    config = tmp_path / "experiment_ok.toml"
    config.write_text(config_text, encoding="utf-8")
    return config


def test_input_data_path_is_relative_to_the_config(tmp_path: Path) -> None:
    config = _copy_fixtures(tmp_path)
    assert input_data_path(config, load_config(config)) == tmp_path / "user_rows_ok.csv"


def test_writes_the_readout_it_returns(tmp_path: Path) -> None:
    config = _copy_fixtures(tmp_path)
    lines = write_experiment_readout(config, tmp_path / "out")
    saved = (tmp_path / "out" / "readout.txt").read_text(encoding="utf-8")
    assert saved == "\n".join(lines) + "\n"
    assert lines[0] == "fixture-experiment"


def test_a_matching_md5_is_read(tmp_path: Path) -> None:
    digest = hashlib.md5((FIXTURES / "user_rows_ok.csv").read_bytes()).hexdigest()
    config = _copy_fixtures(tmp_path, md5=digest)
    assert write_experiment_readout(config, tmp_path / "out")[0] == "fixture-experiment"


def test_a_mismatched_md5_stops_before_any_output(tmp_path: Path) -> None:
    config = _copy_fixtures(tmp_path, md5="0" * 32)
    with pytest.raises(ValueError, match="but the config pins"):
        write_experiment_readout(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_missing_data_file_names_its_path(tmp_path: Path) -> None:
    config = _copy_fixtures(tmp_path)
    missing = tmp_path / "elsewhere.csv"
    with pytest.raises(MissingDataFileError) as raised:
        write_experiment_readout(config, tmp_path / "out", missing)
    assert raised.value.data_path == missing
    assert str(raised.value).startswith(f"data file not found: {missing} (")


def test_a_config_with_no_input_path_says_so(tmp_path: Path) -> None:
    config = _copy_fixtures(tmp_path)
    parsed = load_config(config)
    parsed["input"].pop("path")
    with pytest.raises(ValueError, match="names no data file"):
        input_data_path(config, parsed)

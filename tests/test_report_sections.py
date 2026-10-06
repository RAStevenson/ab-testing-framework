"""Section-library tests: the right sections render for each data shape,
the engine accepts every default report without a citation error, and a
spot-pinned number proves the analyses run on the real fixture values."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from abkit.io import capabilities, load_aggregates, load_config, load_user_rows
from abkit.report import render_report
from abkit.report_sections import (
    _consistent_prefix,
    default_sections,
    fmt_perm_p,
    sec_novelty,
    sec_sequential,
)
from abkit.results import PermutationResult

FIXTURES = Path(__file__).parent / "fixtures"


def _user_experiment(tmp_path: Path, extra_metric: bool = False, truth: bool = False):
    """The user-rows fixture, optionally with a third metric (so the
    corrections family has 2+ members) and a [truth] table."""
    text = (
        '[experiment]\nname = "t"\nalpha = 0.05\n\n'
        '[input]\nshape = "user_rows"\npath = "user_rows_ok.csv"\n\n'
        '[columns]\narm = "group"\nday = "day"\nuser_id = "user_id"\n\n'
        '[arms]\nlabels = ["control", "treatment"]\nintended_split = [0.5, 0.5]\n\n'
        '[[metrics]]\nname = "converted"\ncolumn = "converted"\nkind = "binary"\nprimary = true\n\n'
        '[[metrics]]\nname = "spend"\ncolumn = "spend"\nkind = "continuous"\nnonnegative = true\n'
    )
    if extra_metric:
        text += '\n[[metrics]]\nname = "converted_again"\ncolumn = "converted"\nkind = "binary"\n'
    if truth:
        text += "\n[truth]\nbase_rate = 0.27\ntrue_lift = 0.13\n"
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    return load_user_rows(config, FIXTURES / "user_rows_ok.csv")


class TestUserRowsReport:
    def test_section_set_and_render(self, tmp_path: Path) -> None:
        experiment = _user_experiment(tmp_path, extra_metric=True)
        sections = default_sections(experiment, np.random.default_rng(0), tmp_path / "figs")
        keys = [section.key for section in sections]
        assert keys == [
            "validation",
            "design_power",
            "srm",
            "primary",
            "novelty",
            "sequential",
            "corrections",
            "heavy_tail",
        ]
        report_text = render_report("# Fixture", sections, capabilities(experiment), "figs")
        assert "truth" not in keys
        assert "## 1. Data validation" in report_text

    def test_primary_number_pinned(self, tmp_path: Path) -> None:
        # control 4/15 vs treatment 6/15 -> absolute lift 13.33 pp
        experiment = _user_experiment(tmp_path)
        sections = default_sections(experiment, np.random.default_rng(0), tmp_path / "figs")
        primary = next(section for section in sections if section.key == "primary")
        assert "13.33 pp" in primary.body

    def test_figures_written_and_cited(self, tmp_path: Path) -> None:
        experiment = _user_experiment(tmp_path, extra_metric=True)
        figure_dir = tmp_path / "figs"
        sections = default_sections(experiment, np.random.default_rng(0), figure_dir)
        report_text = render_report("# Fixture", sections, capabilities(experiment), "figs")
        for name in ("novelty.png", "sequential.png", "corrections.png", "heavy_tail.png"):
            assert (figure_dir / name).exists()
            assert f"figs/{name}" in report_text

    def test_truth_section_renders_only_with_truth(self, tmp_path: Path) -> None:
        experiment = _user_experiment(tmp_path, truth=True)
        sections = default_sections(experiment, np.random.default_rng(0), tmp_path / "figs")
        truth = next(section for section in sections if section.key == "truth")
        assert "base_rate" in truth.body
        assert "0.27" in truth.body

    def test_corrections_family_of_one_is_omitted(self, tmp_path: Path) -> None:
        # two metrics, two arms -> one non-primary comparison -> no family
        experiment = _user_experiment(tmp_path, extra_metric=False)
        sections = default_sections(experiment, np.random.default_rng(0), tmp_path / "figs")
        assert "corrections" not in [section.key for section in sections]


class TestAggregatesReport:
    @pytest.fixture()
    def sections_and_exp(self, tmp_path: Path):
        config = load_config(FIXTURES / "aggregates_config.toml")
        experiment = load_aggregates(config, FIXTURES / "aggregates_ok.csv")
        return default_sections(experiment, np.random.default_rng(0), tmp_path / "figs"), experiment

    def test_section_set(self, sections_and_exp) -> None:
        sections, _ = sections_and_exp
        keys = [section.key for section in sections]
        # no user rows: heavy_tail absent; no intended split: design_power
        # requires a control rate, which exists, so it renders on the binary
        # primary; srm renders in its ratio-stability form
        assert "heavy_tail" not in keys
        assert "truth" not in keys
        assert {"validation", "srm", "primary", "novelty", "sequential"} <= set(keys)

    def test_srm_uses_ratio_stability_form(self, sections_and_exp) -> None:
        sections, _ = sections_and_exp
        srm = next(section for section in sections if section.key == "srm")
        assert "RATIO STABILITY" in srm.body
        assert "not on record" in srm.body

    def test_render_passes_citation_check(self, sections_and_exp) -> None:
        sections, experiment = sections_and_exp
        report_text = render_report("# Agg", sections, capabilities(experiment), "figs")
        assert "## 1." in report_text

    def test_sequential_narrates_first_contrast(self, sections_and_exp) -> None:
        sections, _ = sections_and_exp
        sequential_section = next(section for section in sections if section.key == "sequential")
        assert "variant_2 vs control" in sequential_section.body


class TestConsistentPrefix:
    def test_clean_series_keeps_everything(self) -> None:
        from abkit.report_sections import _consistent_prefix

        random_generator = np.random.default_rng(3)
        observations = random_generator.lognormal(1.0, 0.6, size=300)
        splits = [100, 200, 300]
        count = np.array([float(end) for end in splits])
        mean = np.array([observations[:end].mean() for end in splits])
        variance = np.array([observations[:end].var(ddof=1) for end in splits])
        assert _consistent_prefix(count, mean, variance) == 3

    def test_corrupt_final_checkpoint_is_cut(self) -> None:
        # final checkpoint's mean drops while n barely grows, the settled-
        # recompute shape found in the real ASOS archive: the implied
        # increment variance goes hugely negative and the prefix stops
        from abkit.report_sections import _consistent_prefix

        count = np.array([1000.0, 2000.0, 2010.0])
        mean = np.array([0.10, 0.11, 0.05])
        variance = mean * (1 - mean) * count / (count - 1)
        assert _consistent_prefix(count, mean, variance) == 2


class TestArchiveSweep:
    def test_sweep_counts_and_accounting_on_fixture(self) -> None:
        from abkit.report_sections import sweep_aggregate_archive

        config = load_config(FIXTURES / "aggregates_config.toml")
        section = sweep_aggregate_archive(config, FIXTURES / "aggregates_ok.csv")
        # fixture holds expA (3 checkpoints, testable) and expB (1
        # checkpoint, must be counted as untestable, never hidden)
        assert "(2 total)" in section.body
        assert "1 experiments were testable. 1 could not be tested" in section.body
        assert "flagged at the conventional p < 1e-3 alarm: none" in section.body


class TestCommittedAsosReport:
    """Pins on the committed real-data demo: the presence-keying and the
    stated exclusion must hold in the artifact a reader actually sees."""

    REPORT = Path(__file__).parents[1] / "analysis" / "asos" / "report.md"

    def test_no_user_level_sections_render(self) -> None:
        text = self.REPORT.read_text(encoding="utf-8")
        assert "Heavy-tailed" not in text
        assert "What the generator actually did" not in text

    def test_srm_takes_ratio_stability_form(self) -> None:
        text = self.REPORT.read_text(encoding="utf-8")
        assert "RATIO STABILITY" in text

    def test_inconsistent_final_checkpoint_is_stated_not_silent(self) -> None:
        text = self.REPORT.read_text(encoding="utf-8")
        assert "EXCLUDED" in text
        assert "internally inconsistent" in text


class TestHillstromShapedFixture:
    """The Hillstrom demo's logic, exercised entirely on a FABRICATED
    30-row fixture with the real file's header, so no test ever touches
    the network or the gitignored cache."""

    def _experiment(self, tmp_path: Path):
        text = (
            '[experiment]\nname = "hillstrom-fixture"\nalpha = 0.05\n\n'
            '[input]\nshape = "user_rows"\n\n'
            '[columns]\narm = "segment"\n\n'
            '[columns.aliases.zip_code]\n"Surburban" = "Suburban"\n\n'
            '[arms]\nlabels = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]\n'
            "intended_split = [0.3334, 0.3333, 0.3333]\n\n"
            '[[metrics]]\nname = "visit"\nkind = "binary"\nprimary = true\n\n'
            '[[metrics]]\nname = "conversion"\nkind = "binary"\n\n'
            '[[metrics]]\nname = "spend"\nkind = "continuous"\nnonnegative = true\n'
        )
        config_path = tmp_path / "config.toml"
        config_path.write_text(text, encoding="utf-8")
        config = load_config(config_path)
        return load_user_rows(config, FIXTURES / "hillstrom_shaped.csv")

    def test_three_arms_and_no_time_axis(self, tmp_path: Path) -> None:
        experiment = self._experiment(tmp_path)
        assert experiment.arms == ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
        assert capabilities(experiment) == frozenset(
            {"user_rows", "multi_metric", "continuous_metric", "intended_split"}
        )

    def test_alias_normalization_fires_on_the_misspelling(self, tmp_path: Path) -> None:
        experiment = self._experiment(tmp_path)
        by_rule = {event.rule: event.count for event in experiment.validation}
        assert by_rule["alias_normalized"] == 7
        assert np.sum(experiment.rows["zip_code"] == "Surburban") == 0
        assert np.sum(experiment.rows["zip_code"] == "Suburban") == 7

    def test_sections_render_without_time_axis(self, tmp_path: Path) -> None:
        experiment = self._experiment(tmp_path)
        sections = default_sections(experiment, np.random.default_rng(0), tmp_path / "figs")
        keys = [section.key for section in sections]
        assert "sequential" not in keys
        assert "novelty" not in keys
        assert "corrections" in keys  # 2 non-primary metrics x 2 contrasts
        assert "heavy_tail" in keys


class TestCommittedHillstromReport:
    REPORT = Path(__file__).parents[1] / "analysis" / "hillstrom" / "report.md"

    def test_time_sections_absent_and_tail_present(self) -> None:
        text = self.REPORT.read_text(encoding="utf-8")
        assert "sequential monitoring" not in text.lower()
        assert "Heavy-tailed metric: spend" in text
        assert "alias_normalized" in text


def test_permutation_p_at_its_floor_is_stated_as_a_bound():
    """A p-value at the add-one floor names the floor, anything above prints as a plain value."""
    floor = PermutationResult(observed=0.5, p_value=1 / 10_001, n_permutations=10_000)
    assert fmt_perm_p(floor) == "p = 1.0e-04, the smallest value 10,000 relabelings can give"
    above = PermutationResult(observed=0.5, p_value=2 / 10_001, n_permutations=10_000)
    assert fmt_perm_p(above) == "p = 0.0002"


class TestOmittedSections:
    """A section the data cannot support is omitted, the run carries on, and
    the console says which section and why, with the error in full."""

    def test_missing_series_omits_both_time_sections_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = load_config(FIXTURES / "aggregates_config.toml")
        experiment = load_aggregates(config, FIXTURES / "aggregates_ok.csv")
        treatment_arm = experiment.arms[1]
        # keep the end-of-run cells, drop that arm's per-checkpoint cells
        damaged = dataclasses.replace(
            experiment,
            aggregates=tuple(
                cell
                for cell in experiment.aggregates
                if not (cell.arm == treatment_arm and cell.checkpoint is not None)
            ),
        )
        assert sec_novelty(damaged, tmp_path / "figs") is None
        assert sec_sequential(damaged, tmp_path / "figs") is None
        stderr = capsys.readouterr().err
        assert stderr.count("Traceback") == 2
        assert "no per-checkpoint aggregates" in stderr
        assert "omits the section 'Effect stability over time'" in stderr
        assert "omits the section 'What sequential monitoring would have decided'" in stderr


def test_consistent_prefix_needs_no_warning_block_for_short_increments():
    """One-observation and zero increments are skipped by the division itself,
    so the check runs clean with every RuntimeWarning turned into an error."""
    observations = np.random.default_rng(4).lognormal(1.0, 0.6, size=60)
    splits = [30, 31, 31, 60]
    counts = np.array([float(end) for end in splits])
    means = np.array([observations[:end].mean() for end in splits])
    variances = np.array([observations[:end].var(ddof=1) for end in splits])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _consistent_prefix(counts, means, variances) == len(splits)

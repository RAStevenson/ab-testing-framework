"""Section-library tests: the right sections render for each data shape,
the engine accepts every default report without a citation error, and a
spot-pinned number proves the analyses run on the real fixture values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abkit.io import capabilities, load_aggregates, load_config, load_user_rows
from abkit.report import render_report
from abkit.report_sections import default_sections

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
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    return load_user_rows(cfg, FIXTURES / "user_rows_ok.csv")


class TestUserRowsReport:
    def test_section_set_and_render(self, tmp_path: Path) -> None:
        exp = _user_experiment(tmp_path, extra_metric=True)
        sections = default_sections(exp, np.random.default_rng(0), tmp_path / "figs")
        keys = [s.key for s in sections]
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
        doc = render_report("# Fixture", sections, capabilities(exp), "figs")
        assert "truth" not in keys
        assert "## 1. Data validation" in doc

    def test_primary_number_pinned(self, tmp_path: Path) -> None:
        # control 4/15 vs treatment 6/15 -> absolute lift 13.33 pp
        exp = _user_experiment(tmp_path)
        sections = default_sections(exp, np.random.default_rng(0), tmp_path / "figs")
        primary = next(s for s in sections if s.key == "primary")
        assert "13.33 pp" in primary.body

    def test_figures_written_and_cited(self, tmp_path: Path) -> None:
        exp = _user_experiment(tmp_path, extra_metric=True)
        figdir = tmp_path / "figs"
        sections = default_sections(exp, np.random.default_rng(0), figdir)
        doc = render_report("# Fixture", sections, capabilities(exp), "figs")
        for name in ("novelty.png", "sequential.png", "corrections.png", "heavy_tail.png"):
            assert (figdir / name).exists()
            assert f"figs/{name}" in doc

    def test_truth_section_renders_only_with_truth(self, tmp_path: Path) -> None:
        exp = _user_experiment(tmp_path, truth=True)
        sections = default_sections(exp, np.random.default_rng(0), tmp_path / "figs")
        truth = next(s for s in sections if s.key == "truth")
        assert "base_rate" in truth.body
        assert "0.27" in truth.body

    def test_corrections_family_of_one_is_omitted(self, tmp_path: Path) -> None:
        # two metrics, two arms -> one non-primary comparison -> no family
        exp = _user_experiment(tmp_path, extra_metric=False)
        sections = default_sections(exp, np.random.default_rng(0), tmp_path / "figs")
        assert "corrections" not in [s.key for s in sections]


class TestAggregatesReport:
    @pytest.fixture()
    def sections_and_exp(self, tmp_path: Path):
        cfg = load_config(FIXTURES / "aggregates_config.toml")
        exp = load_aggregates(cfg, FIXTURES / "aggregates_ok.csv")
        return default_sections(exp, np.random.default_rng(0), tmp_path / "figs"), exp

    def test_section_set(self, sections_and_exp) -> None:
        sections, _ = sections_and_exp
        keys = [s.key for s in sections]
        # no user rows: heavy_tail absent; no intended split: design_power
        # requires a control rate, which exists, so it renders on the binary
        # primary; srm renders in its ratio-stability form
        assert "heavy_tail" not in keys
        assert "truth" not in keys
        assert {"validation", "srm", "primary", "novelty", "sequential"} <= set(keys)

    def test_srm_uses_ratio_stability_form(self, sections_and_exp) -> None:
        sections, _ = sections_and_exp
        srm = next(s for s in sections if s.key == "srm")
        assert "RATIO STABILITY" in srm.body
        assert "not on record" in srm.body

    def test_render_passes_citation_check(self, sections_and_exp) -> None:
        sections, exp = sections_and_exp
        doc = render_report("# Agg", sections, capabilities(exp), "figs")
        assert "## 1." in doc

    def test_sequential_narrates_first_contrast(self, sections_and_exp) -> None:
        sections, _ = sections_and_exp
        seq = next(s for s in sections if s.key == "sequential")
        assert "variant_2 vs control" in seq.body


class TestConsistentPrefix:
    def test_clean_series_keeps_everything(self) -> None:
        from abkit.report_sections import _consistent_prefix

        rng = np.random.default_rng(3)
        observations = rng.lognormal(1.0, 0.6, size=300)
        splits = [100, 200, 300]
        n = np.array([float(k) for k in splits])
        mean = np.array([observations[:k].mean() for k in splits])
        var = np.array([observations[:k].var(ddof=1) for k in splits])
        assert _consistent_prefix(n, mean, var) == 3

    def test_corrupt_final_checkpoint_is_cut(self) -> None:
        # final checkpoint's mean drops while n barely grows, the settled-
        # recompute shape found in the real ASOS archive: the implied
        # increment variance goes hugely negative and the prefix stops
        from abkit.report_sections import _consistent_prefix

        n = np.array([1000.0, 2000.0, 2010.0])
        mean = np.array([0.10, 0.11, 0.05])
        var = mean * (1 - mean) * n / (n - 1)
        assert _consistent_prefix(n, mean, var) == 2


class TestArchiveSweep:
    def test_sweep_counts_and_accounting_on_fixture(self) -> None:
        from abkit.report_sections import sweep_aggregate_archive

        cfg = load_config(FIXTURES / "aggregates_config.toml")
        section = sweep_aggregate_archive(cfg, FIXTURES / "aggregates_ok.csv")
        # fixture holds expA (3 checkpoints, testable) and expB (1
        # checkpoint, must be counted as untestable, never hidden)
        assert "(2 total)" in section.body
        assert "1 experiments were testable; 1 could not be tested" in section.body
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
        p = tmp_path / "config.toml"
        p.write_text(text, encoding="utf-8")
        cfg = load_config(p)
        return load_user_rows(cfg, FIXTURES / "hillstrom_shaped.csv")

    def test_three_arms_and_no_time_axis(self, tmp_path: Path) -> None:
        exp = self._experiment(tmp_path)
        assert exp.arms == ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
        assert capabilities(exp) == frozenset(
            {"user_rows", "multi_metric", "continuous_metric", "intended_split"}
        )

    def test_alias_normalization_fires_on_the_misspelling(self, tmp_path: Path) -> None:
        exp = self._experiment(tmp_path)
        by_rule = {e.rule: e.count for e in exp.validation}
        assert by_rule["alias_normalized"] == 7
        assert np.sum(exp.rows["zip_code"] == "Surburban") == 0
        assert np.sum(exp.rows["zip_code"] == "Suburban") == 7

    def test_sections_render_without_time_axis(self, tmp_path: Path) -> None:
        exp = self._experiment(tmp_path)
        sections = default_sections(exp, np.random.default_rng(0), tmp_path / "figs")
        keys = [s.key for s in sections]
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

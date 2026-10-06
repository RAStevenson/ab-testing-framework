"""Report engine tests: presence-keying, render-time numbering, token
resolution, and the citation rules that fail a build rather than ship a
structural error."""

from __future__ import annotations

import pytest

from abkit.report import FigureSpec, ReportBuildError, SectionSpec, render_report


def _sections() -> list[SectionSpec]:
    return [
        SectionSpec(key="always", title="Always", body="Runs everywhere."),
        SectionSpec(
            key="timed",
            title="Timed",
            body="Needs a time axis; see section {{sec:always}} and {{fig:decay}}.",
            requires=frozenset({"time_axis"}),
            figures=(FigureSpec("decay", "decay.png", "Daily decay"),),
        ),
        SectionSpec(
            key="tailed",
            title="Tailed",
            body="Needs user rows.",
            requires=frozenset({"user_rows", "continuous_metric"}),
        ),
    ]


class TestPresenceKeying:
    def test_full_caps_render_all(self) -> None:
        report_text = render_report(
            "# T",
            _sections(),
            frozenset({"time_axis", "user_rows", "continuous_metric"}),
            "figures",
        )
        assert "## 1. Always" in report_text
        assert "## 2. Timed" in report_text
        assert "## 3. Tailed" in report_text

    def test_missing_caps_drop_sections_and_keep_numbering_contiguous(self) -> None:
        report_text = render_report(
            "# T", _sections(), frozenset({"user_rows", "continuous_metric"}), "f"
        )
        assert "Timed" not in report_text
        assert "## 2. Tailed" in report_text  # not 3: numbering is contiguous

    def test_tokens_resolve_to_assigned_numbers(self) -> None:
        report_text = render_report("# T", _sections(), frozenset({"time_axis"}), "figures")
        assert "see section 1 and Figure 1." in report_text
        assert "![Figure 1: Daily decay](figures/decay.png)" in report_text

    def test_figure_lands_after_first_citing_paragraph(self) -> None:
        report_text = render_report("# T", _sections(), frozenset({"time_axis"}), "figures")
        assert (
            report_text.index("see section 1") < report_text.index("![Figure 1") < len(report_text)
        )


class TestFailures:
    def test_uncited_figure_fails_build(self) -> None:
        sections = [
            SectionSpec(
                key="s",
                title="S",
                body="No citation here.",
                figures=(FigureSpec("orphan", "o.png", "Orphan"),),
            )
        ]
        with pytest.raises(ReportBuildError, match="never cited"):
            render_report("# T", sections, frozenset(), "f")

    def test_reference_to_dropped_section_fails_build(self) -> None:
        sections = [
            SectionSpec(key="a", title="A", body="See section {{sec:timed}}."),
            SectionSpec(key="timed", title="B", body="x", requires=frozenset({"time_axis"})),
        ]
        with pytest.raises(ReportBuildError, match="did not render"):
            render_report("# T", sections, frozenset(), "f")

    def test_reference_to_unknown_section_fails_build(self) -> None:
        sections = [SectionSpec(key="a", title="A", body="See section {{sec:ghost}}.")]
        with pytest.raises(ReportBuildError, match="does not exist"):
            render_report("# T", sections, frozenset(), "f")

    def test_reference_to_unknown_figure_fails_build(self) -> None:
        sections = [SectionSpec(key="a", title="A", body="See {{fig:ghost}}.")]
        with pytest.raises(ReportBuildError, match="does not exist"):
            render_report("# T", sections, frozenset(), "f")

    def test_reference_to_dropped_sections_figure_names_the_gate(self) -> None:
        sections = [
            SectionSpec(key="a", title="A", body="See {{fig:decay}}."),
            SectionSpec(
                key="timed",
                title="B",
                body="{{fig:decay}}",
                requires=frozenset({"time_axis"}),
                figures=(FigureSpec("decay", "d.png", "Decay"),),
            ),
        ]
        with pytest.raises(ReportBuildError, match="did not render under the current data"):
            render_report("# T", sections, frozenset(), "f")

    def test_duplicate_section_keys_fail_build(self) -> None:
        sections = [
            SectionSpec(key="a", title="A", body="x"),
            SectionSpec(key="a", title="B", body="y"),
        ]
        with pytest.raises(ReportBuildError, match="duplicate section keys"):
            render_report("# T", sections, frozenset(), "f")

    def test_duplicate_figure_keys_fail_build(self) -> None:
        sections = [
            SectionSpec(
                key="a",
                title="A",
                body="{{fig:f1}} and {{fig:f1}} again.",
                figures=(FigureSpec("f1", "1.png", "One"),),
            ),
            SectionSpec(
                key="b",
                title="B",
                body="{{fig:f1}}.",
                figures=(FigureSpec("f1", "2.png", "Two"),),
            ),
        ]
        with pytest.raises(ReportBuildError, match="duplicate figure key"):
            render_report("# T", sections, frozenset(), "f")


class TestNumberingUnderOmission:
    def test_figure_numbers_follow_citation_order_across_sections(self) -> None:
        sections = [
            SectionSpec(
                key="one",
                title="One",
                body="Cites {{fig:late}} first.",
                figures=(FigureSpec("early", "e.png", "Early"),),
            ),
            SectionSpec(
                key="two",
                title="Two",
                body="Then {{fig:early}}.",
                figures=(FigureSpec("late", "l.png", "Late"),),
            ),
        ]
        report_text = render_report("# T", sections, frozenset(), "f")
        # "late" is cited first, so it is Figure 1 wherever its file lives
        assert "![Figure 1: Late](f/l.png)" in report_text
        assert "![Figure 2: Early](f/e.png)" in report_text

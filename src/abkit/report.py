"""Report rendering: presence-keyed sections, render-time numbering,
token cross-references, and figure citation enforced at build.

The rules this module exists to enforce (settled 2026-08-08 after the
survival repo's report review, recorded in its project standard):

- Section and figure numbers are assigned at render time, never written
  in prose. Body text references sections and figures through tokens
  (``{{sec:key}}`` and ``{{fig:key}}``), so a section that does not
  render because the data cannot support it never leaves a hole in the
  numbering or a dangling "see section 5".
- A section renders exactly when the experiment's capability set covers
  the section's ``requires``. That is the entire variant mechanism: one
  template, keyed off what is PRESENT in the data, never a mode flag.
- Every figure is cited from body prose or the build fails. The citation
  is what forces the author to say what the figure is evidence FOR.

Failures raise ``ReportBuildError`` and never skip silently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\{\{(sec|fig):([A-Za-z0-9_-]+)\}\}")


class ReportBuildError(ValueError):
    """A structural report error: bad token, uncited figure, duplicate key."""


@dataclass(frozen=True)
class FigureSpec:
    """One figure: its token key, the file the builder wrote, its caption."""

    key: str
    filename: str
    caption: str


@dataclass(frozen=True)
class SectionSpec:
    """One report section.

    ``requires`` names the capabilities (see ``abkit.io.capabilities``)
    the experiment must have for this section to render; empty means the
    section always renders. ``body`` is markdown whose cross-references
    are ``{{sec:key}}`` / ``{{fig:key}}`` tokens, resolved at render
    time. Figures attach to the section that owns them and are emitted
    beneath the paragraph that first cites them.
    """

    key: str
    title: str
    body: str
    requires: frozenset[str] = field(default_factory=frozenset)
    figures: tuple[FigureSpec, ...] = ()


def render_report(
    header: str,
    sections: Sequence[SectionSpec],
    data_capabilities: frozenset[str],
    figure_dir_relative: str,
) -> str:
    """Render the sections the capability set supports, in order.

    Returns the full markdown document: ``header`` verbatim, then each
    rendered section as ``## N. Title`` with tokens resolved and figures
    placed after their first citing paragraph.
    """
    keys = [section.key for section in sections]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ReportBuildError(f"duplicate section keys: {duplicates}")

    rendered = [section for section in sections if section.requires <= data_capabilities]
    section_number = {section.key: i + 1 for i, section in enumerate(rendered)}

    figure_keys: list[str] = []
    figure_by_key: dict[str, FigureSpec] = {}
    for section in rendered:
        for fig in section.figures:
            if fig.key in figure_by_key:
                raise ReportBuildError(f"duplicate figure key {fig.key!r}")
            figure_by_key[fig.key] = fig
            figure_keys.append(fig.key)

    # figure numbers follow first-citation order across the whole document,
    # so "Figure 1" is the first figure a reader meets, not an accident of
    # which section owns it
    citation_order: list[str] = []
    for section in rendered:
        for kind, key in _TOKEN.findall(section.body):
            if kind == "fig" and key in figure_by_key and key not in citation_order:
                citation_order.append(key)
    uncited = [key for key in figure_keys if key not in citation_order]
    if uncited:
        raise ReportBuildError(
            f"figures never cited from any rendered body: {uncited}; every figure "
            "must be evidence for a stated claim or it does not ship"
        )
    figure_number = {key: i + 1 for i, key in enumerate(citation_order)}

    def resolve(match: re.Match[str], *, where: str) -> str:
        kind, key = match.group(1), match.group(2)
        if kind == "sec":
            if key not in section_number:
                dropped = key in {section.key for section in sections}
                raise ReportBuildError(
                    f"{where} references section {key!r} which "
                    + ("did not render under the current data" if dropped else "does not exist")
                    + "; gate the referencing text on the same requirements"
                )
            return str(section_number[key])
        if key not in figure_number:
            supplied = {figure.key for section in sections for figure in section.figures}
            raise ReportBuildError(
                f"{where} references figure {key!r} which "
                + (
                    "belongs to a section that did not render under the current data"
                    if key in supplied
                    else "does not exist"
                )
                + "; gate the referencing text on the same requirements"
            )
        return f"Figure {figure_number[key]}"

    report_lines: list[str] = [header.rstrip() + "\n"]
    emitted: set[str] = set()
    for section in rendered:
        report_lines.append(f"\n## {section_number[section.key]}. {section.title}\n")
        for paragraph in section.body.split("\n\n"):
            where = f"section {section.key!r}"
            resolved = _TOKEN.sub(
                lambda match, location=where: resolve(match, where=location), paragraph
            ).rstrip()
            if resolved:
                report_lines.append("\n" + resolved + "\n")
            for kind, key in _TOKEN.findall(paragraph):
                if kind == "fig" and key in figure_by_key and key not in emitted:
                    emitted.add(key)
                    fig = figure_by_key[key]
                    report_lines.append(
                        f"\n![Figure {figure_number[key]}: {fig.caption}]"
                        f"({figure_dir_relative}/{fig.filename})\n"
                    )
    return "".join(report_lines)

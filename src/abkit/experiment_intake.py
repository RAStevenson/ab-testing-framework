"""Experiment intake: config, loader, validation, the Experiment object.

The readout reads one shape of file: one row per user, with a group
column, one or more metric columns, and optionally a day column and a
pre-experiment column per metric. The rows come from a user's own export,
a fetched public dataset, or the synthetic generator, and normalize into
the ``Experiment`` object defined here, which also carries per-group
summary statistics at the end of the run and, with a day column, at each
checkpoint. ``experiment_capabilities`` reports which data properties a given
Experiment has; the readout's analysis layer reads its ``time_axis`` flag to
decide whether the monitoring block runs.

The experiment description comes from a TOML config (stdlib ``tomllib``,
so the library's dependencies stay numpy, scipy, and statsmodels).
``load_config`` validates the config's shape; ``load_user_rows`` validates the
data, recording every cleaning action as a ``ValidationEvent`` that the
readout's data block prints before any statistic.
"""

from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_METRIC_KINDS = ("binary", "continuous")
_RESERVED_COLUMN_KEYS = ("group", "day", "user_id")
# share of data rows the drop rules may remove before the run stops; values
# missing for a reason tied to the outcome can bias the comparison, and the
# file cannot show why a value is blank or invalid
DROPPED_ROW_STOP = 0.01


@dataclass(frozen=True)
class MetricSpec:
    """One outcome metric: where it lives and how to treat it."""

    name: str
    column: str
    kind: str  # "binary" | "continuous"
    primary: bool = False
    nonnegative: bool = False
    pre_column: str | None = None  # a pre-experiment column for CUPED


@dataclass(frozen=True)
class ValidationEvent:
    """One cleaning or rejection action taken by a loader, for the readout."""

    rule: str
    count: int
    detail: str


@dataclass(frozen=True)
class ArmAggregate:
    """Summary statistics for one (arm, metric) cell, cumulative through
    ``checkpoint`` when one is given, end-of-run when ``checkpoint`` is None.

    ``variance`` is the ddof=1 sample variance, and is NaN when ``n < 2``.
    """

    arm: str
    metric: str
    checkpoint: float | None
    n: int
    mean: float
    variance: float


@dataclass(frozen=True)
class Experiment:
    """A normalized experiment.

    ``arms[0]`` is the control. ``rows`` maps each declared column to a numpy
    array aligned across columns. ``pre_period`` maps each metric NAME that
    declared a ``pre_column`` to that column's values, aligned with ``rows``,
    and is None when no metric did. ``aggregates`` holds the per-group summary
    statistics built from the rows. ``planned_users_per_group`` is the size
    each arm was planned to reach, the number the planning command prints,
    and None when the config sets none. ``truth`` carries a generator config
    verbatim for synthetic data and is None otherwise.
    """

    name: str
    arms: tuple[str, ...]
    intended_split: tuple[float, ...] | None
    metrics: tuple[MetricSpec, ...]
    alpha: float
    planned_users_per_group: int | None
    rows: dict[str, NDArray]
    pre_period: dict[str, NDArray] | None
    aggregates: tuple[ArmAggregate, ...]
    checkpoints: tuple[float, ...] | None
    truth: dict | None
    validation: tuple[ValidationEvent, ...]


def experiment_capabilities(experiment: Experiment) -> frozenset[str]:
    """The data properties this experiment can support a readout block with."""
    data_capabilities = set()
    if experiment.checkpoints is not None and len(experiment.checkpoints) >= 3:
        data_capabilities.add("time_axis")
    if len(experiment.metrics) >= 2 or len(experiment.arms) > 2:
        data_capabilities.add("multi_metric")
    if any(metric.kind == "continuous" for metric in experiment.metrics):
        data_capabilities.add("continuous_metric")
    if experiment.intended_split is not None:
        data_capabilities.add("intended_split")
    if experiment.truth is not None:
        data_capabilities.add("truth")
    return frozenset(data_capabilities)


def load_config(path: Path) -> dict:
    """Parse and shape-check an experiment TOML config.

    Returns the parsed dict with defaults filled. Raises ValueError naming
    the offending key on any structural problem; nothing is silently
    defaulted except ``alpha`` (0.05), ``planned_users_per_group``
    (None), the optional tables, and each metric's ``column`` (its name),
    ``primary`` and ``nonnegative`` (False), and ``pre_column`` (None).
    ``input.shape`` is optional and defaults to "user_rows"; any other value
    stops the run, so an old config written for per-group totals fails
    loudly instead of being misread.
    """
    with open(path, "rb") as config_file:
        config = tomllib.load(config_file)

    experiment = config.get("experiment")
    if (
        not isinstance(experiment, dict)
        or not isinstance(experiment.get("name"), str)
        or not experiment["name"]
    ):
        raise ValueError("config needs [experiment] with a non-empty string 'name'")
    alpha = experiment.setdefault("alpha", 0.05)
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise ValueError(f"experiment.alpha must be a float in (0, 1), got {alpha!r}")
    planned = experiment.setdefault("planned_users_per_group", None)
    # bool is an int subclass, so `true` would pass an isinstance check alone
    if planned is not None and (
        isinstance(planned, bool) or not isinstance(planned, int) or planned < 2
    ):
        raise ValueError(
            "experiment.planned_users_per_group must be a whole number of at least 2 when "
            f"present, got {planned!r}"
        )

    input_table = config.setdefault("input", {})
    if not isinstance(input_table, dict):
        raise ValueError("[input] must be a table when present")
    if input_table.setdefault("shape", "user_rows") != "user_rows":
        raise ValueError(
            "input.shape must be 'user_rows' when present; the readout reads one row per user"
        )
    if "md5" in input_table and not (
        isinstance(input_table["md5"], str) and len(input_table["md5"]) == 32
    ):
        raise ValueError("input.md5 must be a 32-character hex digest when present")

    arms = config.get("groups")
    if not isinstance(arms, dict) or not isinstance(arms.get("labels"), list):
        raise ValueError("config needs [groups] with a 'labels' list, control first")
    labels = arms["labels"]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"groups.labels needs at least 2 unique labels, got {labels!r}")
    if not all(isinstance(value, str) and value for value in labels):
        raise ValueError(f"groups.labels must be non-empty strings, got {labels!r}")
    folded = [value.casefold() for value in labels]
    if len(set(folded)) != len(folded):
        # row arm values are matched casefolded, so labels differing only
        # by case would collapse into one arm with no error
        raise ValueError(f"groups.labels must be unique ignoring case, got {labels!r}")
    split = arms.get("intended_split")
    if split is not None and (
        not isinstance(split, list)
        or len(split) != len(labels)
        or not all(isinstance(value, (int, float)) and value > 0 for value in split)
        or abs(sum(split) - 1.0) > 1e-9
    ):
        raise ValueError(
            "groups.intended_split must be positive shares, one per label, summing to 1"
        )
    # shares are typed by hand, so a third written as 0.3333 counts as equal
    if (
        planned is not None
        and split is not None
        and any(abs(share - 1.0 / len(labels)) > 1e-3 for share in split)
    ):
        raise ValueError(
            "experiment.planned_users_per_group is one size every group reaches, so it needs "
            f"an equal groups.intended_split (shares within 0.001 of equal), got {split!r}"
        )

    metrics = config.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("config needs at least one [[metrics]] entry")
    seen_names: set[str] = set()
    n_primary = 0
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError(f"every [[metrics]] entry must be a table, got {metric!r}")
        if not isinstance(metric.get("name"), str) or not metric["name"]:
            raise ValueError(f"every [[metrics]] entry needs a string 'name', got {metric!r}")
        if metric["name"] in _RESERVED_COLUMN_KEYS:
            # metric arrays are stored under their names next to the
            # structural arrays; a collision would silently corrupt them
            raise ValueError(
                f"metric name {metric['name']!r} is reserved for a structural column; rename it"
            )
        if metric["name"] in seen_names:
            raise ValueError(f"duplicate metric name {metric['name']!r}")
        seen_names.add(metric["name"])
        if metric.get("kind") not in _METRIC_KINDS:
            raise ValueError(f"metric {metric['name']!r} kind must be one of {_METRIC_KINDS}")
        if not isinstance(metric.get("primary", False), bool):
            raise ValueError(f"metric {metric['name']!r} primary must be a boolean, not a string")
        metric.setdefault("column", metric["name"])
        metric.setdefault("primary", False)
        metric.setdefault("nonnegative", False)
        pre_column = metric.setdefault("pre_column", None)
        if pre_column is not None and (not isinstance(pre_column, str) or not pre_column):
            raise ValueError(
                f"metric {metric['name']!r} pre_column must be a non-empty column name"
            )
        n_primary += bool(metric["primary"])
    if n_primary != 1:
        raise ValueError(f"exactly one metric must set primary = true, found {n_primary}")

    columns = config.setdefault("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("[columns] must be a table")
    if not isinstance(columns.get("group"), str):
        raise ValueError("config needs [columns] with a 'group' column name")
    for key in ("day", "user_id"):
        # an empty string here silently half-declares the column; normalize
        # to absent so every later check sees one representation
        if not columns.get(key):
            columns.pop(key, None)
    aliases = columns.setdefault("aliases", {})
    alias_tables_ok = isinstance(aliases, dict) and all(
        isinstance(value, dict) and all(isinstance(canonical, str) for canonical in value.values())
        for value in aliases.values()
    )
    if not alias_tables_ok:
        raise ValueError("[columns.aliases.<col>] tables must map strings to strings")
    # Metric, pre-experiment, and day cells are read as numbers and user id cells are
    # kept as text, but none of them pass through the alias tables, so a table on one
    # of them would be accepted and do nothing; the README promises nothing is ignored
    # silently.
    numeric_columns = {str(metric["column"]) for metric in config["metrics"]}
    numeric_columns |= {
        metric["pre_column"] for metric in config["metrics"] if metric.get("pre_column")
    }
    numeric_columns |= {columns[key] for key in ("day", "user_id") if key in columns}
    ignored = sorted(numeric_columns & set(aliases))
    if ignored:
        raise ValueError(
            f"[columns.aliases] names {ignored}, which the alias tables never reach (metric, "
            "pre-experiment, day, and user id columns); aliases apply to the group column and "
            "other text columns only"
        )

    truth = config.get("truth")
    if truth is not None and not isinstance(truth, dict):
        raise ValueError("[truth] must be a table when present")
    return config


def _metric_specs(config: dict) -> tuple[MetricSpec, ...]:
    """Return the config's metrics as MetricSpec records, in config order."""
    return tuple(
        MetricSpec(
            name=metric["name"],
            column=str(metric["column"]),
            kind=metric["kind"],
            primary=metric["primary"],
            nonnegative=metric["nonnegative"],
            pre_column=metric.get("pre_column"),
        )
        for metric in config["metrics"]
    )


def _aggregate(values: NDArray[np.float64]) -> tuple[int, float, float]:
    """Return the count, mean, and ddof=1 variance of a value array, NaN where there are too few
    values."""
    count = int(values.size)
    if count == 0:
        return 0, float("nan"), float("nan")
    mean = float(values.mean())
    variance = float(values.var(ddof=1)) if count >= 2 else float("nan")
    return count, mean, variance


def _aggregates_from_rows(
    rows: dict[str, NDArray],
    arm_column: str,
    day_column: str | None,
    arms: tuple[str, ...],
    metrics: tuple[MetricSpec, ...],
    checkpoints: tuple[float, ...] | None,
) -> tuple[ArmAggregate, ...]:
    """Summarize user rows into end-of-run cells per arm and metric, plus cumulative cells at each
    checkpoint when there is a day column."""
    aggregates: list[ArmAggregate] = []
    for arm in arms:
        arm_mask = rows[arm_column] == arm
        for metric in metrics:
            values = rows[metric.name][arm_mask]
            count, mean, variance = _aggregate(values)
            aggregates.append(ArmAggregate(arm, metric.name, None, count, mean, variance))
            if checkpoints is not None and day_column is not None:
                days = rows[day_column][arm_mask]
                for checkpoint in checkpoints:
                    count, mean, variance = _aggregate(values[days <= checkpoint])
                    aggregates.append(
                        ArmAggregate(arm, metric.name, float(checkpoint), count, mean, variance)
                    )
    return tuple(aggregates)


def _coercion_failure(raw: str, metric: MetricSpec | None = None) -> str | None:
    """Return the drop rule a raw numeric cell breaks, or None when it is usable. With a metric,
    the cell must also satisfy that metric's nonnegative and binary rules; a pre-experiment cell
    is passed without one."""
    if not raw:
        return "null_row_dropped"
    try:
        value = float(raw)
    except ValueError:
        return "uncoercible_dropped"
    if not np.isfinite(value):
        # float("nan") and float("inf") parse without error and would
        # poison every mean while validation reported a clean load
        return "uncoercible_dropped"
    if metric is None:
        return None
    if metric.nonnegative and value < 0.0:
        return "negative_value_dropped"
    if metric.kind == "binary" and value not in (0.0, 1.0):
        return "nonbinary_dropped"
    return None


def load_user_rows(config: dict, data_path: Path) -> Experiment:
    """Load, clean, and normalize a user-level CSV into an Experiment.

    Cleaning rules, each recorded as a ValidationEvent when it fires:

    - ``alias_normalized``: arm and aliased categorical values are stripped
      and mapped through ``[columns.aliases]``.
    - ``group_label_normalized``: an arm label matched to ``groups.labels`` only
      after trimming spaces or ignoring capitals.
    - ``null_row_dropped``: a row with an empty arm or an empty value in
      any declared metric or pre-experiment column is dropped. Where a
      metric or pre-experiment column caused a drop, the rule's first
      occurrence records that column with its line in the file, the header being
      line 1. A blank day cell is recorded as ``uncoercible_dropped`` instead.
    - ``negative_value_dropped``: a row with a negative value in a metric
      declared ``nonnegative`` is dropped.
    - ``uncoercible_dropped``: a row whose metric, pre-experiment, or day
      value does not parse as a finite number is dropped.
    - ``nonbinary_dropped``: a row whose binary metric is not 0 or 1.

    Three things stop the run with a ValueError instead of being recorded:
    an arm label that is not in ``groups.labels`` after normalization
    (``unmapped_group_error``: silently dropping an arm would fabricate a
    split); a non-blank user id that appears twice when a user_id column is
    declared (``duplicate_user_error``, naming the id and both rows); and
    the drop rules removing more than ``DROPPED_ROW_STOP`` of the data
    rows (``too_many_dropped_error``, with the count per rule). Past that
    share, values missing for a reason tied to the outcome could bias the
    comparison, and the file cannot show why a value is blank or invalid.

    A metric's ``pre_column`` (its pre-experiment value, for CUPED) must be
    a finite number and is returned in ``pre_period`` keyed by metric name.
    The metric's nonnegative and binary rules do not reach it.

    The returned ``rows`` dict uses CANONICAL keys, not the file's column
    names: ``"group"``, ``"day"`` and ``"user_id"`` for the structural
    columns, each metric under its metric NAME, and any extra aliased
    covariate columns under their raw names. Consumers therefore never
    need the config to interpret an Experiment.
    """
    columns = config["columns"]
    arm_column: str = columns["group"]
    day_column: str | None = columns.get("day")
    user_column: str | None = columns.get("user_id")
    aliases: dict[str, dict[str, str]] = columns["aliases"]
    labels = tuple(config["groups"]["labels"])
    metrics = _metric_specs(config)

    declared = [arm_column, *(metric.column for metric in metrics)]
    pre_columns = [metric.pre_column for metric in metrics if metric.pre_column is not None]
    declared.extend(column for column in pre_columns if column not in declared)
    if day_column:
        declared.append(day_column)
    if user_column:
        declared.append(user_column)
    extra_alias_cols = [column for column in aliases if column not in declared]
    declared.extend(extra_alias_cols)

    # utf-8-sig: Excel's "CSV UTF-8" writes a byte-order mark that would
    # otherwise glue itself onto the first column name
    with open(data_path, newline="", encoding="utf-8-sig") as data_file:
        reader = csv.DictReader(data_file)
        header = reader.fieldnames or []
        missing = [column for column in declared if column not in header]
        if missing:
            raise ValueError(f"data file {data_path.name} is missing declared columns: {missing}")
        raw_rows = list(reader)

    label_by_fold = {value.casefold(): value for value in labels}
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    kept: dict[str, list] = {column: [] for column in declared}
    first_row_of_user: dict[str, int] = {}

    def event(rule: str, index: int, note: str | None = None) -> None:
        """Count one occurrence of a validation rule and keep the first file line it happened
        at, with the note (which column, or which value became which) for that first line."""
        counts[rule] = counts.get(rule, 0) + 1
        examples.setdefault(rule, f"first at line {index}" + (f", {note}" if note else ""))

    for index, row in enumerate(raw_rows, start=2):  # row 1 is the header
        arm_raw = (row[arm_column] or "").strip()
        arm_aliased = aliases.get(arm_column, {}).get(arm_raw, arm_raw)
        if arm_aliased != arm_raw:
            event("alias_normalized", index, f"{arm_column} {arm_raw!r} to {arm_aliased!r}")
        if not arm_aliased:
            event("null_row_dropped", index, f"column {arm_column!r}")
            continue
        arm = label_by_fold.get(arm_aliased.casefold())
        # whitespace or case differences are normalization too, and the
        # README promises none of it happens silently; alias rows are
        # already counted under their own rule
        if arm is not None and (row[arm_column] or "") != arm and arm_aliased == arm_raw:
            event("group_label_normalized", index)
        if arm is None:
            raise ValueError(
                f"unmapped_group_error: group value {arm_raw!r} at line {index} matches no "
                f"label in {labels}; add it to groups.labels or [columns.aliases.{arm_column}]"
            )

        user_id = (row[user_column] or "").strip() if user_column else ""
        if user_id:
            # a BLANK id is treated as missing, never as a duplicate of
            # every other blank id. A repeated id is a join or export error
            # that no rule can repair: keeping either copy guesses which
            # outcome is the user's, so the run stops.
            if user_id in first_row_of_user:
                raise ValueError(
                    f"duplicate_user_error: user id {user_id!r} appears at lines "
                    f"{first_row_of_user[user_id]} and {index}; each user must appear once"
                )
            first_row_of_user[user_id] = index

        values: dict[str, float] = {}
        drop = False
        for metric in metrics:
            columns_to_coerce: list[tuple[str, MetricSpec | None]] = [(metric.column, metric)]
            if metric.pre_column is not None:
                # a past value can fall outside the metric's own range: spend
                # that cannot be negative now can have a past that nets refunds,
                # and a yes/no metric can be adjusted by a past count
                columns_to_coerce.append((metric.pre_column, None))
            for column, range_rules in columns_to_coerce:
                rule = _coercion_failure((row[column] or "").strip(), range_rules)
                if rule is not None:
                    event(rule, index, f"column {column!r}")
                    drop = True
                    break
                values[column] = float((row[column] or "").strip())
            if drop:
                break
        if drop:
            continue

        if day_column:
            day_raw = (row[day_column] or "").strip()
            try:
                day_value = float(day_raw)
            except ValueError:
                event("uncoercible_dropped", index, f"column {day_column!r}")
                continue
            if not np.isfinite(day_value):
                event("uncoercible_dropped", index, f"column {day_column!r}")
                continue

        if user_column:
            kept[user_column].append(user_id)
        kept[arm_column].append(arm)
        for column, value in values.items():
            kept[column].append(value)
        if day_column:
            kept[day_column].append(day_value)
        for column in extra_alias_cols:
            raw = (row[column] or "").strip()
            mapped = aliases[column].get(raw, raw)
            if mapped != raw:
                event("alias_normalized", index, f"{column} {raw!r} to {mapped!r}")
            kept[column].append(mapped)

    dropped_by_rule = {rule: count for rule, count in counts.items() if rule.endswith("_dropped")}
    n_dropped = sum(dropped_by_rule.values())
    if not kept[arm_column]:
        if not raw_rows:
            raise ValueError(f"no_usable_rows_error: {data_path.name} has a header and no rows")
        # the same accounting the 1% stop prints, because a file every row of
        # which fails one rule (a date in the day column, yes/no in a binary
        # column) needs the rule, the column, and the line more than any other
        raise ValueError(
            f"no_usable_rows_error: the drop rules removed every one of the {len(raw_rows):,} "
            f"data rows in {data_path.name}; by rule: {dropped_by_rule}; where each first "
            "fired: "
            + "; ".join(f"{rule} {examples[rule]}" for rule in dropped_by_rule)
            + ". Clean the file before reading it"
        )
    if n_dropped > DROPPED_ROW_STOP * len(raw_rows):
        raise ValueError(
            f"too_many_dropped_error: the drop rules removed {n_dropped:,} of {len(raw_rows):,} "
            f"data rows ({n_dropped / len(raw_rows):.1%}), above the {DROPPED_ROW_STOP:.0%} "
            f"stop; by rule: {dropped_by_rule}; where each first fired: "
            + "; ".join(f"{rule} {examples[rule]}" for rule in dropped_by_rule)
            + ". Clean the file before reading it"
        )

    # canonical keys so consumers never need the config to read an Experiment;
    # metrics are keyed per metric NAME, so two metrics sharing one source
    # column each get their own entry
    rows: dict[str, NDArray] = {"arm": np.asarray(kept[arm_column], dtype=object)}
    if day_column:
        rows["day"] = np.asarray(kept[day_column], dtype=np.float64)
    if user_column:
        rows["user_id"] = np.asarray(kept[user_column], dtype=object)
    for metric in metrics:
        rows[metric.name] = np.asarray(kept[metric.column], dtype=np.float64)
    for column in extra_alias_cols:
        rows[column] = np.asarray(kept[column], dtype=object)
    pre_period: dict[str, NDArray] | None = (
        {
            metric.name: np.asarray(kept[metric.pre_column], dtype=np.float64)
            for metric in metrics
            if metric.pre_column is not None
        }
        if pre_columns
        else None
    )

    checkpoints = (
        tuple(float(value) for value in np.unique(rows["day"])) if day_column is not None else None
    )
    validation = tuple(
        ValidationEvent(rule, counts[rule], examples[rule]) for rule in sorted(counts)
    )
    split = config["groups"].get("intended_split")
    return Experiment(
        name=config["experiment"]["name"],
        arms=labels,
        intended_split=tuple(split) if split is not None else None,
        metrics=metrics,
        alpha=config["experiment"]["alpha"],
        planned_users_per_group=config["experiment"]["planned_users_per_group"],
        rows=rows,
        pre_period=pre_period,
        aggregates=_aggregates_from_rows(
            rows, "arm", "day" if day_column else None, labels, metrics, checkpoints
        ),
        checkpoints=checkpoints,
        truth=config.get("truth"),
        validation=validation,
    )

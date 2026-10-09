"""Experiment intake: config, loaders, validation, the Experiment object.

Every analysis in the library can run on data from three origins - the
synthetic generators, a committed demo dataset, or a user's own file -
provided the data arrives in one of two shapes:

- user-level rows: one row per user, with an arm column and one or more
  metric columns, optionally a day column;
- per-arm aggregates: count, mean and variance per arm, optionally per
  checkpoint.

Both shapes normalize into the ``Experiment`` object defined here.
Aggregates are always derivable from user-level rows; the reverse is
impossible, and that asymmetry is what decides which blocks the readout
can compute: ``capabilities`` reports what a given Experiment can
support, and the readout's analysis layer reads it.

The experiment description comes from a TOML config (stdlib ``tomllib``,
so the library's dependencies stay numpy, scipy, and statsmodels).
``load_config`` validates the config's shape; the loaders validate the
data, recording every cleaning action as a ``ValidationEvent`` that the
readout's data block prints before any statistic.
"""

from __future__ import annotations

import csv
import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_METRIC_KINDS = ("binary", "continuous")
_INPUT_SHAPES = ("user_rows", "aggregates")
_RESERVED_COLUMN_KEYS = ("arm", "day", "user_id")
# share of data rows the drop rules may remove before the run stops: past
# this the file is not cleaned-ish and the readout would describe the
# survivors rather than the experiment
DROPPED_ROW_STOP = 0.01


@dataclass(frozen=True)
class MetricSpec:
    """One outcome metric: where it lives and how to treat it."""

    name: str
    column: str
    kind: str  # "binary" | "continuous"
    primary: bool = False
    nonnegative: bool = False
    pre_column: str | None = None  # a pre-experiment column for CUPED, user rows only


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
    """A normalized experiment, whatever shape it arrived in.

    ``arms[0]`` is the control. ``rows`` is None for aggregate-only data;
    when present it maps each declared column to a numpy array aligned
    across columns. ``pre_period`` maps each metric NAME that declared a
    ``pre_column`` to that column's values, aligned with ``rows``, and is
    None when no metric did or the data is aggregate-only. ``aggregates``
    is always populated. ``truth`` carries a generator config verbatim for
    synthetic data and is None otherwise.
    """

    name: str
    arms: tuple[str, ...]
    intended_split: tuple[float, ...] | None
    metrics: tuple[MetricSpec, ...]
    alpha: float
    rows: dict[str, NDArray] | None
    pre_period: dict[str, NDArray] | None
    aggregates: tuple[ArmAggregate, ...]
    checkpoints: tuple[float, ...] | None
    truth: dict | None
    validation: tuple[ValidationEvent, ...]


def capabilities(experiment: Experiment) -> frozenset[str]:
    """The data properties this experiment can support a readout block with."""
    data_capabilities = set()
    if experiment.rows is not None:
        data_capabilities.add("user_rows")
    if experiment.checkpoints is not None and len(experiment.checkpoints) >= 3:
        data_capabilities.add("time_axis")
    if len(experiment.metrics) >= 2 or len(experiment.arms) > 2:
        data_capabilities.add("multi_metric")
    if experiment.rows is not None and any(
        metric.kind == "continuous" for metric in experiment.metrics
    ):
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
    defaulted except ``alpha`` (0.05) and the optional tables.
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
    if "preamble" in experiment and not isinstance(experiment["preamble"], str):
        raise ValueError("experiment.preamble must be a string when present")

    input_table = config.get("input")
    if not isinstance(input_table, dict) or input_table.get("shape") not in _INPUT_SHAPES:
        raise ValueError(f"config needs [input] with shape in {_INPUT_SHAPES}")
    if "md5" in input_table and not (
        isinstance(input_table["md5"], str) and len(input_table["md5"]) == 32
    ):
        raise ValueError("input.md5 must be a 32-character hex digest when present")

    arms = config.get("arms")
    if input_table["shape"] == "user_rows":
        # user-level data needs the arm labels declared up front; aggregate
        # data derives its arms from the file (control plus each variant),
        # so [arms] is optional there and intended_split is checked at load
        if not isinstance(arms, dict) or not isinstance(arms.get("labels"), list):
            raise ValueError("config needs [arms] with a 'labels' list, control first")
        labels = arms["labels"]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f"arms.labels needs at least 2 unique labels, got {labels!r}")
        if not all(isinstance(value, str) and value for value in labels):
            raise ValueError(f"arms.labels must be non-empty strings, got {labels!r}")
        folded = [value.casefold() for value in labels]
        if len(set(folded)) != len(folded):
            # row arm values are matched casefolded, so labels differing only
            # by case would collapse into one arm with no error
            raise ValueError(f"arms.labels must be unique ignoring case, got {labels!r}")
        split = arms.get("intended_split")
        if split is not None and (
            not isinstance(split, list)
            or len(split) != len(labels)
            or not all(isinstance(value, (int, float)) and value > 0 for value in split)
            or abs(sum(split) - 1.0) > 1e-9
        ):
            raise ValueError(
                "arms.intended_split must be positive shares, one per label, summing to 1"
            )
    else:
        aggregate_columns = config.get("aggregates")
        required = (
            "variant",
            "metric",
            "checkpoint",
            "count_control",
            "mean_control",
            "variance_control",
            "count_treatment",
            "mean_treatment",
            "variance_treatment",
        )
        if not isinstance(aggregate_columns, dict):
            raise ValueError("aggregates input needs an [aggregates] column-mapping table")
        missing_keys = [key for key in required if not isinstance(aggregate_columns.get(key), str)]
        if missing_keys:
            raise ValueError(f"[aggregates] is missing column names for: {missing_keys}")
        has_id_column = isinstance(aggregate_columns.get("experiment_id_column"), str)
        has_id = "experiment_id" in aggregate_columns
        if has_id_column != has_id:
            raise ValueError(
                "[aggregates] experiment_id_column and experiment_id must be set together"
            )
        arms_config = config.get("arms")
        if arms_config is not None and not isinstance(arms_config, dict):
            raise ValueError("[arms] must be a table when present")
        split = (arms_config or {}).get("intended_split")
        if split is not None and (
            not isinstance(split, list)
            or not all(isinstance(value, (int, float)) and value > 0 for value in split)
            or abs(sum(split) - 1.0) > 1e-9
        ):
            raise ValueError(
                "arms.intended_split must be positive shares summing to 1 "
                "(one per arm in control-first, sorted-variant order)"
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
        if pre_column is not None:
            if not isinstance(pre_column, str) or not pre_column:
                raise ValueError(
                    f"metric {metric['name']!r} pre_column must be a non-empty column name"
                )
            if input_table["shape"] != "user_rows":
                raise ValueError(
                    f"metric {metric['name']!r} declares pre_column, which needs user-level rows"
                )
        n_primary += bool(metric["primary"])
    if n_primary != 1:
        raise ValueError(f"exactly one metric must set primary = true, found {n_primary}")

    columns = config.setdefault("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("[columns] must be a table")
    if input_table["shape"] == "user_rows" and not isinstance(columns.get("arm"), str):
        raise ValueError("user_rows input needs [columns] with an 'arm' column name")
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


def _coercion_failure(metric: MetricSpec, raw: str) -> str | None:
    """Return the drop rule a raw cell of this metric (or its pre-experiment column) breaks, or None
    when it parses to a usable value."""
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
    if metric.nonnegative and value < 0.0:
        return "negative_value_dropped"
    if metric.kind == "binary" and value not in (0.0, 1.0):
        return "nonbinary_dropped"
    return None


def load_user_rows(config: dict, data_path: Path) -> Experiment:
    """Load, clean, and normalize a user-level CSV into an Experiment.

    Cleaning rules, each recorded as a ValidationEvent when it fires:

    - ``alias_normalized``: arm and aliased categorical values are
      stripped, casefolded, and mapped through ``[columns.aliases]``.
    - ``null_row_dropped``: a row with an empty arm or an empty value in
      any declared metric or pre-experiment column is dropped.
    - ``negative_value_dropped``: a row with a negative value in a metric
      declared ``nonnegative`` (or its pre-experiment column) is dropped.
    - ``uncoercible_dropped``: a row whose metric, pre-experiment, or day
      value does not parse as a finite number is dropped.
    - ``nonbinary_dropped``: a row whose binary metric is not 0 or 1.

    Three things stop the run with a ValueError instead of being recorded:
    an arm label that is not in ``arms.labels`` after normalization
    (``unmapped_arm_error``: silently dropping an arm would fabricate a
    split); a non-blank user id that appears twice when a user_id column is
    declared (``duplicate_user_error``, naming the id and both rows); and
    the drop rules removing more than ``DROPPED_ROW_STOP`` of the data
    rows (``too_many_dropped_error``, with the count per rule). A file past
    that share is not cleaned-ish, and a readout of its survivors would
    describe a different experiment.

    A metric's ``pre_column`` (its pre-experiment value, for CUPED) is read
    under the same coercion rules as the metric and returned in
    ``pre_period`` keyed by metric name.

    The returned ``rows`` dict uses CANONICAL keys, not the file's column
    names: ``"arm"``, ``"day"`` and ``"user_id"`` for the structural
    columns, each metric under its metric NAME, and any extra aliased
    covariate columns under their raw names. Consumers therefore never
    need the config to interpret an Experiment.
    """
    columns = config["columns"]
    arm_column: str = columns["arm"]
    day_column: str | None = columns.get("day")
    user_column: str | None = columns.get("user_id")
    aliases: dict[str, dict[str, str]] = columns["aliases"]
    labels = tuple(config["arms"]["labels"])
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
        """Count one occurrence of a validation rule and keep the first data row it happened at,
        with the note (which value became which) when the rule changed a value."""
        counts[rule] = counts.get(rule, 0) + 1
        examples.setdefault(rule, (f"{note}, " if note else "") + f"first at data row {index}")

    for index, row in enumerate(raw_rows, start=2):  # row 1 is the header
        arm_raw = (row[arm_column] or "").strip()
        arm_aliased = aliases.get(arm_column, {}).get(arm_raw, arm_raw)
        if arm_aliased != arm_raw:
            event("alias_normalized", index, f"{arm_column} {arm_raw!r} to {arm_aliased!r}")
        if not arm_aliased:
            event("null_row_dropped", index)
            continue
        arm = label_by_fold.get(arm_aliased.casefold())
        # whitespace or case differences are normalization too, and the
        # README promises none of it happens silently; alias rows are
        # already counted under their own rule
        if arm is not None and (row[arm_column] or "") != arm and arm_aliased == arm_raw:
            event("arm_label_normalized", index)
        if arm is None:
            raise ValueError(
                f"unmapped_arm_error: arm value {arm_raw!r} at data row {index} matches no "
                f"label in {labels}; add it to arms.labels or [columns.aliases.{arm_column}]"
            )

        user_id = (row[user_column] or "").strip() if user_column else ""
        if user_id:
            # a BLANK id is treated as missing, never as a duplicate of
            # every other blank id. A repeated id is a join or export error
            # that no rule can repair: keeping either copy guesses which
            # outcome is the user's, so the run stops.
            if user_id in first_row_of_user:
                raise ValueError(
                    f"duplicate_user_error: user id {user_id!r} appears at data rows "
                    f"{first_row_of_user[user_id]} and {index}; each user must appear once"
                )
            first_row_of_user[user_id] = index

        values: dict[str, float] = {}
        drop = False
        for metric in metrics:
            columns_to_coerce = [metric.column]
            if metric.pre_column is not None:
                columns_to_coerce.append(metric.pre_column)
            for column in columns_to_coerce:
                rule = _coercion_failure(metric, (row[column] or "").strip())
                if rule is not None:
                    event(rule, index)
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
                event("uncoercible_dropped", index)
                continue
            if not np.isfinite(day_value):
                event("uncoercible_dropped", index)
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

    if not kept[arm_column]:
        raise ValueError(f"no usable rows in {data_path.name} after validation")
    dropped_by_rule = {rule: count for rule, count in counts.items() if rule.endswith("_dropped")}
    n_dropped = sum(dropped_by_rule.values())
    if n_dropped > DROPPED_ROW_STOP * len(raw_rows):
        raise ValueError(
            f"too_many_dropped_error: the drop rules removed {n_dropped:,} of {len(raw_rows):,} "
            f"data rows ({n_dropped / len(raw_rows):.1%}), above the {DROPPED_ROW_STOP:.0%} "
            f"stop; by rule: {dropped_by_rule}. Clean the file before reading it"
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
    split = config["arms"].get("intended_split")
    return Experiment(
        name=config["experiment"]["name"],
        arms=labels,
        intended_split=tuple(split) if split is not None else None,
        metrics=metrics,
        alpha=config["experiment"]["alpha"],
        rows=rows,
        pre_period=pre_period,
        aggregates=_aggregates_from_rows(
            rows, "arm", "day" if day_column else None, labels, metrics, checkpoints
        ),
        checkpoints=checkpoints,
        truth=config.get("truth"),
        validation=validation,
    )


def load_aggregates(config: dict, data_path: Path) -> Experiment:
    """Load per-arm cumulative summary statistics into an Experiment.

    Expects the side-by-side shape the ASOS Digital Experiments Dataset
    uses: one row per (variant, metric, checkpoint) carrying count, mean
    and variance for the control group and for that treatment variant.
    Arms are derived from the file as ``control`` plus each distinct
    variant value, in sorted order; ``rows`` is None because no
    user-level data exists in this shape.

    Hard errors (the data is unusable, so these raise rather than log):
    a nonpositive count, a negative variance, or a cumulative count that
    decreases across checkpoints (``cumulative_regression_error``).
    """
    aggregate_columns = config["aggregates"]
    metrics = _metric_specs(config)
    filter_column = aggregate_columns.get("experiment_id_column")
    filter_value = aggregate_columns.get("experiment_id")

    with open(data_path, newline="", encoding="utf-8-sig") as data_file:
        reader = csv.DictReader(data_file)
        header = reader.fieldnames or []
        needed = [
            aggregate_columns[config_key] for config_key in ("variant", "metric", "checkpoint")
        ] + [
            aggregate_columns[config_key]
            for config_key in (
                "count_control",
                "mean_control",
                "variance_control",
                "count_treatment",
                "mean_treatment",
                "variance_treatment",
            )
        ]
        if filter_column:
            needed.append(filter_column)
        missing = [column for column in needed if column not in header]
        if missing:
            raise ValueError(f"data file {data_path.name} is missing declared columns: {missing}")
        raw_rows = [
            row
            for row in reader
            if not filter_column or (row[filter_column] or "").strip() == str(filter_value)
        ]
    if not raw_rows:
        raise ValueError(
            f"no rows in {data_path.name}"
            + (f" for experiment_id {filter_value!r}" if filter_column else "")
        )

    metric_by_column = {metric.column: metric for metric in metrics}
    variants = sorted({(row[aggregate_columns["variant"]] or "").strip() for row in raw_rows})
    if any(not variant for variant in variants):
        raise ValueError("empty variant value in aggregate data")
    arms = ("control", *(f"variant_{variant}" for variant in variants))

    # cells[(arm, metric_name)] -> list of (checkpoint, n, mean, variance)
    cells: dict[tuple[str, str], list[tuple[float, int, float, float]]] = {}
    control_seen: dict[tuple[str, float], tuple[int, float, float]] = {}

    def _number(row: dict, key: str, index: int) -> float:
        """Return one cell of the aggregate file as a float, raising with the row and column named
        if it does not parse."""
        raw = (row[aggregate_columns[key]] or "").strip()
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"unparseable {key} value {raw!r} at data row {index}") from error
        return value

    for index, row in enumerate(raw_rows, start=2):
        metric = metric_by_column.get((row[aggregate_columns["metric"]] or "").strip())
        if metric is None:
            continue  # metrics not declared in config are simply not analyzed
        checkpoint = _number(row, "checkpoint", index)
        variant_arm = f"variant_{(row[aggregate_columns['variant']] or '').strip()}"
        for arm, prefix in ((("control"), "control"), ((variant_arm), "treatment")):
            count_value = _number(row, f"count_{prefix}", index)
            if count_value != int(count_value):
                raise ValueError(f"non-integer count {count_value} at data row {index} ({arm})")
            count = int(count_value)
            mean = _number(row, f"mean_{prefix}", index)
            variance = _number(row, f"variance_{prefix}", index)
            if count <= 0:
                raise ValueError(f"nonpositive count {count} at data row {index} ({arm})")
            if not np.isfinite(mean) or not np.isfinite(variance):
                # NaN answers False to every comparison, so without this it
                # would sail through and detonate inside a test far away
                raise ValueError(
                    f"non-finite statistics at data row {index} ({arm}); "
                    "clean or exclude the row before analysis"
                )
            if variance < 0.0:
                raise ValueError(f"negative variance {variance} at data row {index} ({arm})")
            if arm == "control":
                # the control column repeats per variant; keep one copy and
                # insist repeats agree, because a disagreement means the rows
                # are not describing the same control group
                key = (metric.name, checkpoint)
                previous = control_seen.get(key)
                if previous is not None:
                    if previous != (count, mean, variance):
                        raise ValueError(
                            f"control stats disagree across variants at data row {index} "
                            f"for metric {metric.name!r}, checkpoint {checkpoint}"
                        )
                    continue
                control_seen[key] = (count, mean, variance)
            cells.setdefault((arm, metric.name), []).append((checkpoint, count, mean, variance))

    aggregates: list[ArmAggregate] = []
    for (arm, metric_name), series in cells.items():
        series.sort(key=lambda item: item[0])
        counts = [count for _, count, _, _ in series]
        if any(later < earlier for earlier, later in itertools.pairwise(counts)):
            raise ValueError(
                f"cumulative_regression_error: counts for ({arm}, {metric_name}) "
                f"decrease across checkpoints: {counts}"
            )
        for checkpoint, count, mean, variance in series:
            aggregates.append(ArmAggregate(arm, metric_name, checkpoint, count, mean, variance))
        _, count, mean, variance = series[-1]
        aggregates.append(ArmAggregate(arm, metric_name, None, count, mean, variance))

    # every declared metric must actually be present for every arm, and every
    # arm must share the same checkpoint grid per metric: sections combine
    # arm series positionally, so a missing metric or an offset grid would
    # surface as a far-away crash or, worse, silently misaligned statistics
    grids: dict[str, tuple[float, ...]] = {}
    for metric in metrics:
        for arm in arms:
            cell_checkpoints = tuple(
                sorted(
                    cell.checkpoint
                    for cell in aggregates
                    if cell.arm == arm
                    and cell.metric == metric.name
                    and cell.checkpoint is not None
                )
            )
            if not any(cell.arm == arm and cell.metric == metric.name for cell in aggregates):
                raise ValueError(
                    f"declared metric {metric.name!r} (column {metric.column!r}) has no "
                    f"data for arm {arm!r}"
                    + (f" in experiment {filter_value!r}" if filter_column else "")
                )
            reference = grids.setdefault(metric.name, cell_checkpoints)
            if cell_checkpoints != reference:
                raise ValueError(
                    f"checkpoint grids differ across arms for metric {metric.name!r}: "
                    f"{arm!r} has {len(cell_checkpoints)} checkpoints where "
                    f"{arms[0]!r} has {len(reference)}; the archive rows are not "
                    "aligned and positional analysis would be wrong"
                )

    checkpoints = tuple(
        sorted({cell.checkpoint for cell in aggregates if cell.checkpoint is not None})
    )
    arms_config = config.get("arms") or {}
    split = arms_config.get("intended_split")
    if split is not None and len(split) != len(arms):
        raise ValueError(
            f"intended_split has {len(split)} shares but the data derives {len(arms)} arms"
        )
    return Experiment(
        name=config["experiment"]["name"],
        arms=arms,
        intended_split=tuple(split) if split is not None else None,
        metrics=metrics,
        alpha=config["experiment"]["alpha"],
        rows=None,
        pre_period=None,
        aggregates=tuple(aggregates),
        checkpoints=checkpoints if checkpoints else None,
        truth=config.get("truth"),
        validation=(),
    )


def interval_from_cumulative(
    count: NDArray, mean: NDArray, variance: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Recover per-interval n, mean and variance from cumulative statistics.

    Inputs are aligned 1-d arrays of cumulative counts, means and ddof=1
    sample variances at each checkpoint. Means are recovered exactly;
    variances are recovered exactly UNDER THE ddof=1 CONVENTION via the
    sum-of-squares identity ``SS_k = (n_k - 1) * var_k + n_k * mean_k**2``.
    If the source computed variance with a different ddof the recovered
    interval variances are approximate to O(1/n).

    An interval with fewer than 2 observations gets a NaN variance. A
    recovered variance that lands a hair negative from float cancellation
    is clamped to 0.0; anything materially negative raises, because it
    means the inputs were not really cumulative statistics.
    """
    counts = np.asarray(count, dtype=np.float64)
    means = np.asarray(mean, dtype=np.float64)
    variances = np.asarray(variance, dtype=np.float64)
    if not (counts.shape == means.shape == variances.shape) or counts.ndim != 1:
        raise ValueError("n, mean and variance must be 1-d arrays of the same length")
    if counts.size == 0:
        raise ValueError("need at least one checkpoint")
    if np.any(counts <= 0) or np.any(np.diff(counts) < 0):
        raise ValueError(f"counts must be positive and non-decreasing, got {counts.tolist()}")
    if np.any(variances < 0.0):
        raise ValueError("variances must be non-negative")

    sums = counts * means
    squares = (counts - 1.0) * variances + counts * means**2
    interval_counts = np.diff(counts, prepend=0.0)
    interval_sums = np.diff(sums, prepend=0.0)
    interval_squares = np.diff(squares, prepend=0.0)
    if np.any(interval_counts == 0):
        raise ValueError("an interval has zero observations; drop repeated checkpoints first")
    interval_means = interval_sums / interval_counts
    interval_variances = np.divide(
        interval_squares - interval_counts * interval_means**2,
        interval_counts - 1.0,
        out=np.full_like(interval_counts, np.nan),
        where=interval_counts >= 2,
    )
    # float cancellation can push a true-zero variance slightly negative;
    # a materially negative value means the inputs were inconsistent
    bad = interval_variances < -1e-6 * np.maximum(variances, 1.0)
    if np.any(bad):
        where = np.nonzero(bad)[0].tolist()
        raise ValueError(
            f"recovered interval variance is negative at checkpoints {where}; "
            "the cumulative statistics are inconsistent"
        )
    interval_variances = np.where(
        np.isnan(interval_variances), np.nan, np.maximum(interval_variances, 0.0)
    )
    return interval_counts, interval_means, interval_variances

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
impossible, and that asymmetry is what drives the report's
presence-keying: ``capabilities`` reports what a given Experiment can
support, and each report section declares what it requires.

The experiment description comes from a TOML config (stdlib ``tomllib``,
so the library's numpy + scipy dependency claim holds). ``load_config``
validates the config's shape; the loaders validate the data, recording
every cleaning action as a ``ValidationEvent`` so the report can show
what was done to the data before any statistic was computed.
"""

from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_METRIC_KINDS = ("binary", "continuous")
_INPUT_SHAPES = ("user_rows", "aggregates")


@dataclass(frozen=True)
class MetricSpec:
    """One outcome metric: where it lives and how to treat it."""

    name: str
    column: str
    kind: str  # "binary" | "continuous"
    primary: bool = False
    nonnegative: bool = False


@dataclass(frozen=True)
class ValidationEvent:
    """One cleaning or rejection action taken by a loader, for the report."""

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
    across columns. ``aggregates`` is always populated. ``truth`` carries
    a generator config verbatim for synthetic data and is None otherwise.
    """

    name: str
    arms: tuple[str, ...]
    intended_split: tuple[float, ...] | None
    metrics: tuple[MetricSpec, ...]
    alpha: float
    rows: dict[str, NDArray] | None
    aggregates: tuple[ArmAggregate, ...]
    checkpoints: tuple[float, ...] | None
    truth: dict | None
    validation: tuple[ValidationEvent, ...]


def capabilities(exp: Experiment) -> frozenset[str]:
    """The data properties this experiment can support a section with."""
    caps = set()
    if exp.rows is not None:
        caps.add("user_rows")
    if exp.checkpoints is not None and len(exp.checkpoints) >= 3:
        caps.add("time_axis")
    if len(exp.metrics) >= 2 or len(exp.arms) > 2:
        caps.add("multi_metric")
    if exp.rows is not None and any(m.kind == "continuous" for m in exp.metrics):
        caps.add("continuous_metric")
    if exp.intended_split is not None:
        caps.add("intended_split")
    if exp.truth is not None:
        caps.add("truth")
    return frozenset(caps)


def load_config(path: Path) -> dict:
    """Parse and shape-check an experiment TOML config.

    Returns the parsed dict with defaults filled. Raises ValueError naming
    the offending key on any structural problem; nothing is silently
    defaulted except ``alpha`` (0.05) and the optional tables.
    """
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)

    exp = cfg.get("experiment")
    if not isinstance(exp, dict) or not isinstance(exp.get("name"), str) or not exp["name"]:
        raise ValueError("config needs [experiment] with a non-empty string 'name'")
    alpha = exp.setdefault("alpha", 0.05)
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise ValueError(f"experiment.alpha must be a float in (0, 1), got {alpha!r}")

    inp = cfg.get("input")
    if not isinstance(inp, dict) or inp.get("shape") not in _INPUT_SHAPES:
        raise ValueError(f"config needs [input] with shape in {_INPUT_SHAPES}")

    arms = cfg.get("arms")
    if inp["shape"] == "user_rows":
        # user-level data needs the arm labels declared up front; aggregate
        # data derives its arms from the file (control plus each variant),
        # so [arms] is optional there and intended_split is checked at load
        if not isinstance(arms, dict) or not isinstance(arms.get("labels"), list):
            raise ValueError("config needs [arms] with a 'labels' list, control first")
        labels = arms["labels"]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f"arms.labels needs at least 2 unique labels, got {labels!r}")
        if not all(isinstance(v, str) and v for v in labels):
            raise ValueError(f"arms.labels must be non-empty strings, got {labels!r}")
        split = arms.get("intended_split")
        if split is not None and (
            not isinstance(split, list)
            or len(split) != len(labels)
            or not all(isinstance(v, (int, float)) and v > 0 for v in split)
            or abs(sum(split) - 1.0) > 1e-9
        ):
            raise ValueError(
                "arms.intended_split must be positive shares, one per label, summing to 1"
            )
    else:
        agg = cfg.get("aggregates")
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
        if not isinstance(agg, dict):
            raise ValueError("aggregates input needs an [aggregates] column-mapping table")
        missing_keys = [k for k in required if not isinstance(agg.get(k), str)]
        if missing_keys:
            raise ValueError(f"[aggregates] is missing column names for: {missing_keys}")

    metrics = cfg.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("config needs at least one [[metrics]] entry")
    seen_names: set[str] = set()
    n_primary = 0
    for m in metrics:
        if not isinstance(m.get("name"), str) or not m["name"]:
            raise ValueError(f"every [[metrics]] entry needs a string 'name', got {m!r}")
        if m["name"] in seen_names:
            raise ValueError(f"duplicate metric name {m['name']!r}")
        seen_names.add(m["name"])
        if m.get("kind") not in _METRIC_KINDS:
            raise ValueError(f"metric {m['name']!r} kind must be one of {_METRIC_KINDS}")
        m.setdefault("column", m["name"])
        m.setdefault("primary", False)
        m.setdefault("nonnegative", False)
        n_primary += bool(m["primary"])
    if n_primary != 1:
        raise ValueError(f"exactly one metric must set primary = true, found {n_primary}")

    columns = cfg.setdefault("columns", {})
    if inp["shape"] == "user_rows" and not isinstance(columns.get("arm"), str):
        raise ValueError("user_rows input needs [columns] with an 'arm' column name")
    aliases = columns.setdefault("aliases", {})
    alias_tables_ok = isinstance(aliases, dict) and all(
        isinstance(v, dict) and all(isinstance(x, str) for x in v.values())
        for v in aliases.values()
    )
    if not alias_tables_ok:
        raise ValueError("[columns.aliases.<col>] tables must map strings to strings")

    truth = cfg.get("truth")
    if truth is not None and not isinstance(truth, dict):
        raise ValueError("[truth] must be a table when present")
    return cfg


def _metric_specs(cfg: dict) -> tuple[MetricSpec, ...]:
    return tuple(
        MetricSpec(
            name=m["name"],
            column=str(m["column"]),
            kind=m["kind"],
            primary=m["primary"],
            nonnegative=m["nonnegative"],
        )
        for m in cfg["metrics"]
    )


def _aggregate(values: NDArray[np.float64]) -> tuple[int, float, float]:
    n = int(values.size)
    if n == 0:
        return 0, float("nan"), float("nan")
    mean = float(values.mean())
    variance = float(values.var(ddof=1)) if n >= 2 else float("nan")
    return n, mean, variance


def _aggregates_from_rows(
    rows: dict[str, NDArray],
    arm_col: str,
    day_col: str | None,
    arms: tuple[str, ...],
    metrics: tuple[MetricSpec, ...],
    checkpoints: tuple[float, ...] | None,
) -> tuple[ArmAggregate, ...]:
    out: list[ArmAggregate] = []
    for arm in arms:
        arm_mask = rows[arm_col] == arm
        for metric in metrics:
            values = rows[metric.column][arm_mask]
            n, mean, variance = _aggregate(values)
            out.append(ArmAggregate(arm, metric.name, None, n, mean, variance))
            if checkpoints is not None and day_col is not None:
                days = rows[day_col][arm_mask]
                for cp in checkpoints:
                    n, mean, variance = _aggregate(values[days <= cp])
                    out.append(ArmAggregate(arm, metric.name, float(cp), n, mean, variance))
    return tuple(out)


def load_user_rows(config: dict, data_path: Path) -> Experiment:
    """Load, clean, and normalize a user-level CSV into an Experiment.

    Cleaning rules, each recorded as a ValidationEvent when it fires:

    - ``alias_normalized``: arm and aliased categorical values are
      stripped, casefolded, and mapped through ``[columns.aliases]``.
    - ``null_row_dropped``: a row with an empty arm or an empty value in
      any declared metric column is dropped.
    - ``duplicate_user_dropped``: when a user_id column is declared, the
      first occurrence is kept and later ones dropped.
    - ``negative_value_dropped``: a row with a negative value in a metric
      declared ``nonnegative`` is dropped.
    - ``uncoercible_dropped``: a row whose metric value does not parse as
      a number is dropped.

    An arm label that is not in ``arms.labels`` after normalization
    raises ValueError (``unmapped_arm_error``): silently dropping an arm
    would fabricate a split.
    """
    columns = config["columns"]
    arm_col: str = columns["arm"]
    day_col: str | None = columns.get("day")
    user_col: str | None = columns.get("user_id")
    aliases: dict[str, dict[str, str]] = columns["aliases"]
    labels = tuple(config["arms"]["labels"])
    metrics = _metric_specs(config)

    declared = [arm_col, *(m.column for m in metrics)]
    if day_col:
        declared.append(day_col)
    if user_col:
        declared.append(user_col)
    extra_alias_cols = [c for c in aliases if c not in declared]
    declared.extend(extra_alias_cols)

    with open(data_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in declared if c not in header]
        if missing:
            raise ValueError(f"data file {data_path.name} is missing declared columns: {missing}")
        raw_rows = list(reader)

    label_by_fold = {v.casefold(): v for v in labels}
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    kept: dict[str, list] = {c: [] for c in declared}
    seen_users: set[str] = set()

    def event(rule: str, index: int) -> None:
        counts[rule] = counts.get(rule, 0) + 1
        examples.setdefault(rule, f"first at data row {index}")

    for index, row in enumerate(raw_rows, start=2):  # row 1 is the header
        arm_raw = (row[arm_col] or "").strip()
        arm_aliased = aliases.get(arm_col, {}).get(arm_raw, arm_raw)
        if arm_aliased != arm_raw:
            event("alias_normalized", index)
        if not arm_aliased:
            event("null_row_dropped", index)
            continue
        arm = label_by_fold.get(arm_aliased.casefold())
        if arm is None:
            raise ValueError(
                f"unmapped_arm_error: arm value {arm_raw!r} at data row {index} matches no "
                f"label in {labels}; add it to arms.labels or [columns.aliases.{arm_col}]"
            )

        if user_col:
            uid = (row[user_col] or "").strip()
            if uid in seen_users:
                event("duplicate_user_dropped", index)
                continue

        values: dict[str, float] = {}
        drop = False
        for metric in metrics:
            raw = (row[metric.column] or "").strip()
            if not raw:
                event("null_row_dropped", index)
                drop = True
                break
            try:
                value = float(raw)
            except ValueError:
                event("uncoercible_dropped", index)
                drop = True
                break
            if metric.nonnegative and value < 0.0:
                event("negative_value_dropped", index)
                drop = True
                break
            values[metric.column] = value
        if drop:
            continue

        if user_col:
            seen_users.add(uid)
            kept[user_col].append(uid)
        kept[arm_col].append(arm)
        for column, value in values.items():
            kept[column].append(value)
        if day_col:
            kept[day_col].append(float(row[day_col]))
        for column in extra_alias_cols:
            raw = (row[column] or "").strip()
            mapped = aliases[column].get(raw, raw)
            if mapped != raw:
                event("alias_normalized", index)
            kept[column].append(mapped)

    if not kept[arm_col]:
        raise ValueError(f"no usable rows in {data_path.name} after validation")

    rows: dict[str, NDArray] = {}
    for column in declared:
        if column in (arm_col, user_col) or column in extra_alias_cols:
            rows[column] = np.asarray(kept[column], dtype=object)
        else:
            rows[column] = np.asarray(kept[column], dtype=np.float64)

    checkpoints = tuple(float(v) for v in np.unique(rows[day_col])) if day_col is not None else None
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
        aggregates=_aggregates_from_rows(rows, arm_col, day_col, labels, metrics, checkpoints),
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
    agg = config["aggregates"]
    metrics = _metric_specs(config)
    filter_col = agg.get("experiment_id_column")
    filter_val = agg.get("experiment_id")

    with open(data_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        needed = [agg[k] for k in ("variant", "metric", "checkpoint")] + [
            agg[k]
            for k in (
                "count_control",
                "mean_control",
                "variance_control",
                "count_treatment",
                "mean_treatment",
                "variance_treatment",
            )
        ]
        if filter_col:
            needed.append(filter_col)
        missing = [c for c in needed if c not in header]
        if missing:
            raise ValueError(f"data file {data_path.name} is missing declared columns: {missing}")
        raw_rows = [
            row
            for row in reader
            if not filter_col or (row[filter_col] or "").strip() == str(filter_val)
        ]
    if not raw_rows:
        raise ValueError(
            f"no rows in {data_path.name}"
            + (f" for experiment_id {filter_val!r}" if filter_col else "")
        )

    metric_by_column = {m.column: m for m in metrics}
    variants = sorted({(row[agg["variant"]] or "").strip() for row in raw_rows})
    if any(not v for v in variants):
        raise ValueError("empty variant value in aggregate data")
    arms = ("control", *(f"variant_{v}" for v in variants))

    # cells[(arm, metric_name)] -> list of (checkpoint, n, mean, variance)
    cells: dict[tuple[str, str], list[tuple[float, int, float, float]]] = {}
    control_seen: dict[tuple[str, float], tuple[int, float, float]] = {}
    for index, row in enumerate(raw_rows, start=2):
        metric = metric_by_column.get((row[agg["metric"]] or "").strip())
        if metric is None:
            continue  # metrics not declared in config are simply not analyzed
        checkpoint = float(row[agg["checkpoint"]])
        variant_arm = f"variant_{(row[agg['variant']] or '').strip()}"
        for arm, prefix in ((("control"), "control"), ((variant_arm), "treatment")):
            n = int(float(row[agg[f"count_{prefix}"]]))
            mean = float(row[agg[f"mean_{prefix}"]])
            variance = float(row[agg[f"variance_{prefix}"]])
            if n <= 0:
                raise ValueError(f"nonpositive count {n} at data row {index} ({arm})")
            if variance < 0.0:
                raise ValueError(f"negative variance {variance} at data row {index} ({arm})")
            if arm == "control":
                # the control column repeats per variant; keep one copy and
                # insist repeats agree, because a disagreement means the rows
                # are not describing the same control group
                key = (metric.name, checkpoint)
                previous = control_seen.get(key)
                if previous is not None:
                    if previous != (n, mean, variance):
                        raise ValueError(
                            f"control stats disagree across variants at data row {index} "
                            f"for metric {metric.name!r}, checkpoint {checkpoint}"
                        )
                    continue
                control_seen[key] = (n, mean, variance)
            cells.setdefault((arm, metric.name), []).append((checkpoint, n, mean, variance))

    aggregates: list[ArmAggregate] = []
    for (arm, metric_name), series in cells.items():
        series.sort(key=lambda item: item[0])
        ns = [n for _, n, _, _ in series]
        if any(b < a for a, b in zip(ns, ns[1:], strict=False)):
            raise ValueError(
                f"cumulative_regression_error: counts for ({arm}, {metric_name}) "
                f"decrease across checkpoints: {ns}"
            )
        for checkpoint, n, mean, variance in series:
            aggregates.append(ArmAggregate(arm, metric_name, checkpoint, n, mean, variance))
        final_cp, n, mean, variance = series[-1]
        aggregates.append(ArmAggregate(arm, metric_name, None, n, mean, variance))

    checkpoints = tuple(sorted({a.checkpoint for a in aggregates if a.checkpoint is not None}))
    arms_cfg = config.get("arms") or {}
    split = arms_cfg.get("intended_split")
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
        aggregates=tuple(aggregates),
        checkpoints=checkpoints if checkpoints else None,
        truth=config.get("truth"),
        validation=(),
    )


def interval_from_cumulative(
    n: NDArray, mean: NDArray, variance: NDArray
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
    n_arr = np.asarray(n, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    var_arr = np.asarray(variance, dtype=np.float64)
    if not (n_arr.shape == mean_arr.shape == var_arr.shape) or n_arr.ndim != 1:
        raise ValueError("n, mean and variance must be 1-d arrays of the same length")
    if n_arr.size == 0:
        raise ValueError("need at least one checkpoint")
    if np.any(n_arr <= 0) or np.any(np.diff(n_arr) < 0):
        raise ValueError(f"counts must be positive and non-decreasing, got {n_arr.tolist()}")
    if np.any(var_arr < 0.0):
        raise ValueError("variances must be non-negative")

    sums = n_arr * mean_arr
    squares = (n_arr - 1.0) * var_arr + n_arr * mean_arr**2
    int_n = np.diff(n_arr, prepend=0.0)
    int_sums = np.diff(sums, prepend=0.0)
    int_squares = np.diff(squares, prepend=0.0)
    if np.any(int_n == 0):
        raise ValueError("an interval has zero observations; drop repeated checkpoints first")
    int_mean = int_sums / int_n
    with np.errstate(invalid="ignore", divide="ignore"):
        int_var = (int_squares - int_n * int_mean**2) / (int_n - 1.0)
    int_var = np.where(int_n < 2, np.nan, int_var)
    # float cancellation can push a true-zero variance slightly negative;
    # a materially negative value means the inputs were inconsistent
    bad = int_var < -1e-6 * np.maximum(var_arr, 1.0)
    if np.any(bad):
        where = np.nonzero(bad)[0].tolist()
        raise ValueError(
            f"recovered interval variance is negative at checkpoints {where}; "
            "the cumulative statistics are inconsistent"
        )
    int_var = np.where(np.isnan(int_var), np.nan, np.maximum(int_var, 0.0))
    return int_n, int_mean, int_var

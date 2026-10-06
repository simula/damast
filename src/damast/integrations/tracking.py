"""
Backend-agnostic interface for reporting pipeline runs to an experiment tracker
(e.g. MLflow, W&B): the "metadata contract" (units, value ranges, computed
statistics) alongside per-step timing, for audit trails of scientific runs.
"""

from __future__ import annotations

import json
import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from damast.core.dataframe import AnnotatedDataFrame
    from damast.core.metadata import MetaData

__all__ = [
    "ExperimentTracker",
    "flatten_metadata",
    "flatten_step_params",
    "flatten_step_stats",
]

#: Characters MLflow rejects in a metric or parameter name - it permits alphanumerics,
#: underscore, dash, period, space and slash
_UNSAFE_NAME = re.compile(r"[^0-9a-zA-Z_\-./ ]")


def _safe_name(path: str) -> str:
    """
    Make a dotted path usable as a metric or parameter name.

    A tracker names its own statistics, and a name can reach here from data - a tracked column,
    for instance, which damast itself may label `latitude (deg)`. Sanitising here keeps such a
    name from failing the run when it is logged.

    Args:
        path: The dotted path to sanitise.

    Returns:
        The path with every character MLflow rejects replaced by an underscore.
    """
    return _UNSAFE_NAME.sub("_", path)


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """
    Every non-mapping leaf of a nested mapping, with its dotted path.

    A list is a leaf rather than something to descend into, so that e.g. a sample of removed
    values stays a single entry.

    Args:
        value: The value to walk.
        prefix: The path accumulated so far.

    Yields:
        A `(dotted path, leaf value)` pair per leaf.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _leaves(child, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def _is_metric(value: Any) -> bool:
    """
    Whether a leaf is a measurement rather than something to record as a parameter.

    `bool` is excluded although `isinstance(True, int)` holds in Python: a flag such as "the
    sample was truncated" is not a quantity to plot. :func:`flatten_step_params` takes exactly
    the leaves this rejects, so nothing a tracker reports is dropped.

    Args:
        value: The leaf value to classify.

    Returns:
        `True` if the value belongs in metrics.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_metadata(
    metadata: MetaData,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """
    Flatten a `MetaData` contract into tracker-agnostic params/metrics/tags.

    Per-column fields that describe the declared contract (unit, representation type,
    value range, ...) become params; the computed `value_stats` (mean, stddev, ...) become
    metrics, since they are numeric and specific to this run; scalar-valued annotations
    (e.g. institution, license) become tags.

    Example:

    ```python
    params, metrics, tags = flatten_metadata(adf.metadata)
    ```

    Args:
        metadata: The metadata to flatten.

    Returns:
        A `(params, metrics, tags)` tuple.
    """
    params: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}

    for spec in metadata.columns:
        spec_dict = dict(spec)
        name = spec_dict.pop("name")
        col_prefix = f"col.{name}"

        value_stats = spec_dict.pop("value_stats", None)
        if value_stats:
            for stat_name, value in value_stats.items():
                if isinstance(value, (int, float)):
                    metrics[f"{col_prefix}.{stat_name}"] = float(value)

        for field_name, value in spec_dict.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            params[f"{col_prefix}.{field_name}"] = value

    for annotation_name, annotation in metadata.annotations.items():
        if isinstance(annotation.value, (str, int, float, bool)):
            tags[annotation_name] = str(annotation.value)

    return params, metrics, tags


def flatten_step_stats(processing_stats: dict[str, dict[str, Any]]) -> dict[str, float]:
    """
    Flatten the numeric part of `DataProcessingPipeline.processing_stats` into metrics.

    Every numeric leaf becomes `step.<step name>.<dotted path>`, whatever
    `damast.core.tracking.PipelineElementTracker` produced it - so a new tracker's measurements
    are reported without a change here. Non-numeric leaves and flags go to
    :func:`flatten_step_params`.

    Example:

    ```python
    metrics = flatten_step_stats(pipeline.processing_stats)
    # {"step.valid_mmsi.rows_removed": 4.0,
    #  "step.valid_mmsi.keys.mmsi.unique_removed": 4.0, ...}
    ```

    Args:
        processing_stats: Per-step stats, as returned by
            `DataProcessingPipeline.processing_stats`.

    Returns:
        A flat mapping of metric name to numeric value.
    """
    metrics: dict[str, float] = {}
    for step_name, stats in processing_stats.items():
        for path, value in _leaves(stats):
            if _is_metric(value):
                metrics[_safe_name(f"step.{step_name}.{path}")] = float(value)

    return metrics


def flatten_step_params(processing_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Flatten the non-numeric part of `DataProcessingPipeline.processing_stats` into parameters.

    The counterpart of :func:`flatten_step_stats`: it takes exactly the leaves that are not
    measurements - timestamps, flags such as whether a sample was truncated, and the sampled
    values themselves - so that nothing a tracker reports is lost. Containers are JSON-encoded,
    as in :func:`flatten_metadata`.

    Args:
        processing_stats: Per-step stats, as returned by
            `DataProcessingPipeline.processing_stats`.

    Returns:
        A flat mapping of parameter name to value.
    """
    params: dict[str, Any] = {}
    for step_name, stats in processing_stats.items():
        for path, value in _leaves(stats):
            if _is_metric(value):
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            params[_safe_name(f"step.{step_name}.{path}")] = value

    return params


class ExperimentTracker(ABC):
    """
    Minimal interface a pipeline run can be reported to.

    Concrete backends (see `damast.integrations.mlflow_tracker.MLflowTracker`) implement the
    six primitives below; `log_result` is shared and builds on them.
    """

    @abstractmethod
    def start_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        """Start a new run."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of (immutable, contract-like) parameters."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a batch of numeric metrics."""

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """Set a batch of free-form tags."""

    @abstractmethod
    def log_artifact(self, path: str | Path) -> None:
        """Attach a local file to the run."""

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""

    def log_result(self, adf: AnnotatedDataFrame) -> None:
        """
        Log an `AnnotatedDataFrame`'s metadata contract to the current run.

        Logs the flattened params/metrics/tags (see `flatten_metadata`) plus the full
        metadata, saved as a YAML artifact for anyone who needs more than the flattened view.

        Args:
            adf: The (typically pipeline output) dataframe whose metadata to log.
        """
        params, metrics, tags = flatten_metadata(adf.metadata)
        if params:
            self.log_params(params)
        if metrics:
            self.log_metrics(metrics)
        if tags:
            self.set_tags(tags)

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "metadata.damast.yaml"
            adf.metadata.save_yaml(metadata_path)
            self.log_artifact(metadata_path)

"""
Backend-agnostic interface for reporting pipeline runs to an experiment tracker
(e.g. MLflow, W&B): the "metadata contract" (units, value ranges, computed
statistics) alongside per-step timing, for audit trails of scientific runs.
"""

from __future__ import annotations

import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from damast.core.dataframe import AnnotatedDataFrame
    from damast.core.metadata import MetaData

__all__ = ["ExperimentTracker", "flatten_metadata", "flatten_step_stats"]


def flatten_metadata(
    metadata: "MetaData",
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, str]]:
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
    params: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}
    tags: Dict[str, str] = {}

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


def flatten_step_stats(processing_stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Flatten `DataProcessingPipeline.processing_stats` into tracker-agnostic metrics.

    Args:
        processing_stats: Per-step stats, as returned by
            `DataProcessingPipeline.processing_stats`.

    Returns:
        A flat mapping of metric name to numeric value.
    """
    metrics: Dict[str, float] = {}
    for step_name, stats in processing_stats.items():
        for key in ("processing_time_in_s", "output_dataframe_length"):
            if key in stats:
                metrics[f"step.{step_name}.{key}"] = float(stats[key])

        for source, length in stats.get("input_dataframe_length", {}).items():
            metrics[f"step.{step_name}.input_dataframe_length.{source}"] = float(length)

    return metrics


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
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of (immutable, contract-like) parameters."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log a batch of numeric metrics."""

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set a batch of free-form tags."""

    @abstractmethod
    def log_artifact(self, path: str | Path) -> None:
        """Attach a local file to the run."""

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""

    def log_result(self, adf: "AnnotatedDataFrame") -> None:
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

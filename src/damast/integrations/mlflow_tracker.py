"""
MLflow backend for `damast.integrations.tracking.ExperimentTracker`.

Example:

```python
from damast.integrations.mlflow_tracker import track_pipeline

with track_pipeline(pipeline, run_name="ais-cleaning") as tracker:
    adf = pipeline.transform(df)
    tracker.log_result(adf)
```

Requires the optional `mlflow` dependency (`uv add --optional mlflow mlflow`).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator

from damast.utils import ensure_packages

from .tracking import ExperimentTracker, flatten_step_stats

if TYPE_CHECKING:
    from damast.core.dataprocessing import DataProcessingPipeline

__all__ = ["MLflowTracker", "track_pipeline"]


class MLflowTracker(ExperimentTracker):
    """`ExperimentTracker` backed by MLflow. Requires the optional `mlflow` dependency."""

    def __init__(self):
        ensure_packages(["mlflow"], required_for="Logging pipeline runs to MLflow")
        import mlflow

        self._mlflow = mlflow

    def start_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        self._mlflow.start_run(run_name=run_name, **kwargs)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def set_tags(self, tags: Dict[str, str]) -> None:
        self._mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def end_run(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)


@contextmanager
def track_pipeline(
    pipeline: "DataProcessingPipeline",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    **run_kwargs: Any,
) -> Iterator[MLflowTracker]:
    """
    Report a `DataProcessingPipeline` run to MLflow.

    Starts an MLflow run, yields a `MLflowTracker` to log the result against (call
    `tracker.log_result(adf)` on the transformed dataframe), and on exit always logs the
    pipeline's per-step timing/row-count stats (`DataProcessingPipeline.processing_stats`,
    partial if the pipeline raised) and ends the run - `FAILED` if the `with` block raised,
    `FINISHED` otherwise.

    Example:

    ```python
    with track_pipeline(pipeline, run_name="ais-cleaning") as tracker:
        adf = pipeline.transform(df)
        tracker.log_result(adf)
    ```

    Args:
        pipeline: The pipeline being run.
        run_name: Name for the MLflow run.
        tracking_uri: Passed to `mlflow.set_tracking_uri`, if given.
        experiment_name: Passed to `mlflow.set_experiment`, if given.
        run_kwargs: Forwarded to `mlflow.start_run`.

    Yields:
        A `MLflowTracker` for the active run.
    """
    ensure_packages(["mlflow"], required_for="Logging pipeline runs to MLflow")
    import mlflow

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    tracker = MLflowTracker()
    tracker.start_run(run_name=run_name, **run_kwargs)
    tracker.set_tags({"damast.pipeline": pipeline.name})

    status = "FINISHED"
    try:
        yield tracker
    except Exception:
        status = "FAILED"
        raise
    finally:
        step_metrics = flatten_step_stats(pipeline.processing_stats)
        if step_metrics:
            tracker.log_metrics(step_metrics)
        tracker.end_run(status=status)

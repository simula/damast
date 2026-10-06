"""
Trackers that observe what each step of a :class:`damast.core.dataprocessing.DataProcessingPipeline`
does to the data.

A pipeline is given a list of :class:`PipelineElementTracker` instances and calls each one before
and after every step. What a tracker returns at the end of a step is merged into the pipeline's
:func:`damast.core.dataprocessing.DataProcessingPipeline.processing_stats`, so a new kind of
measurement is a new class here rather than a change to the pipeline.

Not to be confused with :class:`damast.integrations.tracking.ExperimentTracker`, which reports a
finished run to an experiment tracker such as MLflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.transformations import PipelineElement

__all__ = [
    "KeyTracker",
    "PipelineElementTracker",
    "RowCountTracker",
    "TimingTracker",
]


class PipelineElementTracker(ABC):
    """
    Observes a single :class:`damast.core.transformations.PipelineElement` as a pipeline runs it.

    A tracker is called twice per step - :func:`on_step_start` with the step's inputs, and
    :func:`on_step_end` with its output - and returns the statistics for that step from
    :func:`on_step_end`. The pipeline merges the returned dictionaries of all its trackers, so two
    trackers must not report under the same key.

    A tracker that needs to compare the two ends correlates them by ``step.uuid`` and keeps its own
    state; it is never told the step's name, since the pipeline keys the merged result by that.
    """

    def on_step_start(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> None:
        """
        Called before a step runs. Does nothing by default, so a stateless tracker can ignore it.

        :param step: The step that is about to run
        :param dataframes: The step's input, keyed by datasource - several entries for a join
        """

    @abstractmethod
    def on_step_end(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> dict[str, Any]:
        """
        Called after a step ran.

        :param step: The step that just ran
        :param dataframes: The step's output, keyed by datasource
        :return: The statistics for this step, merged into the pipeline's processing stats
        """


class TimingTracker(PipelineElementTracker):
    """
    How long each step took.

    Reports ``start_time``, ``end_time`` and ``processing_time_in_s``.
    """

    def __init__(self):
        #: step uuid -> when it started, held only while that step runs
        self._started: dict[Any, datetime] = {}

    def on_step_start(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> None:
        self._started[step.uuid] = datetime.now(timezone.utc)

    def on_step_end(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> dict[str, Any]:
        end_time = datetime.now(timezone.utc)
        start_time = self._started.pop(step.uuid, None)
        if start_time is None:
            return {"end_time": end_time}

        return {
            "start_time": start_time,
            "end_time": end_time,
            "processing_time_in_s": (end_time - start_time).total_seconds(),
        }


class RowCountTracker(PipelineElementTracker):
    """
    How many rows each step consumed and produced.

    Reports ``input_dataframe_length`` (per datasource), ``output_dataframe_length`` and
    ``rows_removed`` - the latter against the total of all inputs, so for a join it states what the
    step reduced rather than comparing against one side.

    The row count materialises the frame, which the pipeline does anyway in order to run the next
    step.
    """

    def __init__(self):
        #: step uuid -> total input rows, held only while that step runs
        self._input_lengths: dict[Any, int] = {}

    def on_step_start(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> None:
        lengths = {label: adf.shape[0] for label, adf in dataframes.items()}
        self._input_lengths[step.uuid] = lengths

    def on_step_end(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> dict[str, Any]:
        output_length = sum(adf.shape[0] for adf in dataframes.values())
        lengths = self._input_lengths.pop(step.uuid, None)
        if lengths is None:
            return {"output_dataframe_length": output_length}

        return {
            "input_dataframe_length": lengths,
            "output_dataframe_length": output_length,
            "rows_removed": sum(lengths.values()) - output_length,
        }


class KeyTracker(PipelineElementTracker):
    """
    How many distinct entities each step consumed and produced.

    For every tracked column this reports, under ``keys``, the distinct counts ``unique_in``,
    ``unique_out`` and ``unique_removed``, plus ``removed_sample`` - at most
    :attr:`max_removed_keys` of the removed values, with ``removed_truncated`` saying whether the
    sample is partial. The removed set itself is unbounded, hence the sample.

    Example:

        .. highlight:: python
        .. code-block:: python

            pipeline = DataProcessingPipeline(
                name="prepare", base_dir=out,
                trackers=[TimingTracker(), RowCountTracker(), KeyTracker(["mmsi"])])

    :param keys: Columns identifying an entity, e.g. ``["mmsi"]``
    :param max_removed_keys: How many removed values to keep per step and column; 0 keeps the
        counts but no values
    :raise ValueError: If ``max_removed_keys`` is negative

    .. note::
        The distinct values of each tracked column are held in memory while a step runs, so a
        very high-cardinality key is not free. The aggregation itself runs over a frame the
        pipeline has already materialised to count its rows.
    """

    def __init__(self, keys: list[str], max_removed_keys: int = 10):
        if max_removed_keys < 0:
            raise ValueError(f"{self.__class__.__name__}.__init__: max_removed_keys must not be"
                             f" negative, got {max_removed_keys}")

        self.keys = list(keys)
        self.max_removed_keys = max_removed_keys
        #: step uuid -> column -> distinct input values, held only while that step runs
        self._values_in: dict[Any, dict[str, set[Any]]] = {}

    @staticmethod
    def _distinct_values(adf: AnnotatedDataFrame, column: str) -> set[Any]:
        """
        The distinct values of a column of an already collected dataframe.

        :param adf: The dataframe holding the column
        :param column: Name of the column
        :return: The distinct values, with null left out - it identifies no entity
        """
        return set(adf.lazyframe.compat.collected().get_column(column).drop_nulls().unique().to_list())

    def _collect(self, dataframes: dict[str, AnnotatedDataFrame]) -> dict[str, set[Any]]:
        """
        The distinct values of every tracked column present in the given dataframes.

        :param dataframes: The dataframes to scan, keyed by datasource
        :return: Column name to its distinct values, union-ed over the dataframes that carry it.
            A column no dataframe carries is left out.
        """
        values = {}
        for key in self.keys:
            for adf in dataframes.values():
                if key in adf.column_names:
                    values.setdefault(key, set()).update(self._distinct_values(adf, key))
        return values

    def on_step_start(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> None:
        # Materialise now: with 'inplace_transformation' the input frame is the same object the
        # step mutates, so a lazy 'before' would resolve to the state afterwards.
        self._values_in[step.uuid] = self._collect(dataframes)

    def on_step_end(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> dict[str, Any]:
        values_in = self._values_in.pop(step.uuid, {})
        if not values_in:
            return {}

        values_out = self._collect(dataframes)
        statistics = {}
        for key, before in values_in.items():
            if key not in values_out:
                # the step is allowed to drop the column - that is not 'every entity was removed'
                continue

            removed = before - values_out[key]
            sample = sorted(removed)[:self.max_removed_keys]
            statistics[key] = {
                "unique_in": len(before),
                "unique_out": len(values_out[key]),
                "unique_removed": len(removed),
                "removed_sample": sample,
                "removed_truncated": len(sample) < len(removed),
            }

        return {"keys": statistics} if statistics else {}

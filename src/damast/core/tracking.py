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

import polars

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.deferred import Deferred
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
    def on_step_end(self, step: PipelineElement,
                    dataframes: dict[str, AnnotatedDataFrame]) -> dict[str, Any] | Deferred:
        """
        Called after a step ran.

        A tracker that needs the data returns a :class:`damast.core.deferred.Deferred` of its
        statistics instead of computing them, so the pipeline can compute them together with the
        result in a single pass rather than running the pipeline for every step.

        :param step: The step that just ran
        :param dataframes: The step's output, keyed by datasource
        :return: The statistics for this step, or a deferred of them, merged into the pipeline's
            processing stats
        """


class TimingTracker(PipelineElementTracker):
    """
    How long each step took.

    Reports ``start_time``, ``end_time`` and ``processing_time_in_s``. Since a pipeline run only
    builds a lazy plan and executes it in one pass at the end, this is the time a step took to
    build its part of the plan - plus whatever a step executes eagerly itself.
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

    The counts are deferred: within a pipeline run they are computed in the pass that produces the
    result, so counting runs neither the pipeline nor materialises a frame per step.
    """

    def __init__(self):
        #: step uuid -> row count queries of the inputs, held only while that step runs
        self._input_counts: dict[Any, dict[str, polars.LazyFrame]] = {}

    def on_step_start(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> None:
        self._input_counts[step.uuid] = {label: adf.lazyframe.select(polars.len())
                                         for label, adf in dataframes.items()}

    def on_step_end(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> Deferred:
        inputs = self._input_counts.pop(step.uuid, {})
        outputs = [adf.lazyframe.select(polars.len()) for adf in dataframes.values()]

        def report(results: list[polars.DataFrame]) -> dict[str, Any]:
            counts = [result.item() for result in results]
            output_length = sum(counts[len(inputs):])
            if not inputs:
                return {"output_dataframe_length": output_length}

            lengths = dict(zip(inputs, counts[:len(inputs)]))
            return {
                "input_dataframe_length": lengths,
                "output_dataframe_length": output_length,
                "rows_removed": sum(lengths.values()) - output_length,
            }

        return Deferred([*inputs.values(), *outputs], report)


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
        The distinct values of each tracked column are held in memory when they are computed, so a
        very high-cardinality key is not free. Like :class:`RowCountTracker`'s counts they are
        deferred to the pass that produces the result.
    """

    def __init__(self, keys: list[str], max_removed_keys: int = 10):
        if max_removed_keys < 0:
            raise ValueError(f"{self.__class__.__name__}.__init__: max_removed_keys must not be"
                             f" negative, got {max_removed_keys}")

        self.keys = list(keys)
        self.max_removed_keys = max_removed_keys
        #: step uuid -> (column, distinct input values query), held only while that step runs
        self._values_in: dict[Any, list[tuple[str, polars.LazyFrame]]] = {}

    def _queries(self, dataframes: dict[str, AnnotatedDataFrame]) -> list[tuple[str, polars.LazyFrame]]:
        """
        Queries for the distinct values of every tracked column present in the given dataframes.

        Built from the frames as they are now - a LazyFrame is an immutable plan, so a query for
        a step's input still describes the input after the step replaced its frame.

        :param dataframes: The dataframes, keyed by datasource
        :return: (column, query) per tracked column and dataframe carrying it; null is left out,
            it identifies no entity
        """
        return [(key, adf.lazyframe.select(polars.col(key).drop_nulls().unique().implode()))
                for key in self.keys for adf in dataframes.values() if key in adf.column_names]

    @staticmethod
    def _union(queries: list[tuple[str, polars.LazyFrame]],
               results: list[polars.DataFrame]) -> dict[str, set[Any]]:
        values = {}
        for (key, _), result in zip(queries, results):
            values.setdefault(key, set()).update(result.item().to_list())
        return values

    def on_step_start(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> None:
        self._values_in[step.uuid] = self._queries(dataframes)

    def on_step_end(self, step: PipelineElement, dataframes: dict[str, AnnotatedDataFrame]) -> Deferred:
        queries_in = self._values_in.pop(step.uuid, [])
        queries_out = self._queries(dataframes) if queries_in else []

        def report(results: list[polars.DataFrame]) -> dict[str, Any]:
            values_in = self._union(queries_in, results[:len(queries_in)])
            values_out = self._union(queries_out, results[len(queries_in):])
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

        return Deferred([query for _, query in [*queries_in, *queries_out]], report)

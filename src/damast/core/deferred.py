"""
Checks and statistics that a pipeline run collects without executing the pipeline for each of them.

Every collect re-runs the whole lazy plan from the input files, so checking value ranges, counting
rows and computing statistics step by step would execute the pipeline once per check. Instead a
:class:`damast.core.dataprocessing.DataProcessingPipeline` run registers them as queries in a
:class:`PendingQueries`, and they are all computed in the single streaming pass that eventually
produces the result - e.g. :func:`damast.core.dataframe.AnnotatedDataFrame.export`.
"""
from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import polars

__all__ = ["Deferred", "PendingQueries", "current", "running"]


class Deferred:
    """
    A value computed from one or more queries once they are executed.

    :param queries: The queries, each expected to yield a small result (an aggregate)
    :param finish: Turns the queries' results, in the same order, into the value
    """

    def __init__(self, queries: list[polars.LazyFrame], finish: Callable[[list[polars.DataFrame]], Any]):
        self.queries = queries
        self.finish = finish

    def resolve(self) -> Any:
        """Compute the value right away, executing each query separately."""
        return self.finish(polars.collect_all(self.queries))


class PendingQueries:
    """
    Queries waiting to be executed together, each with what to do with its result.

    Results are handed on in registration order, so of several failing checks the earliest one -
    i.e. the first step whose output violates its metadata - is the one reported.
    """

    def __init__(self):
        self._entries: list[tuple[Deferred, Callable[[Any], None], str]] = []
        self._on_complete: list[Callable[[], None]] = []
        #: Label of what is currently registering queries, e.g. the step being run
        self.label: str = ""
        self._done: bool = False
        #: Set by :func:`absorb` - the queries that took over this one's
        self._absorbed_by: PendingQueries | None = None

    @property
    def done(self) -> bool:
        """Whether :func:`run` has computed the registered queries."""
        if self._absorbed_by is not None:
            return self._absorbed_by.done
        return self._done

    def __len__(self) -> int:
        return len(self._entries)

    def absorb(self, other: PendingQueries) -> None:
        """
        Take over the queries and callbacks of another, not yet executed run.

        Used when a pipeline runs on the result of another one, so that both runs are computed
        in a single pass. Running `other` afterwards runs this one.

        :param other: The other run's queries
        """
        if other is self or other.done:
            return
        while other._absorbed_by is not None:
            other = other._absorbed_by
        self._entries = other._entries + self._entries
        self._on_complete = other._on_complete + self._on_complete
        other._entries, other._on_complete = [], []
        other._absorbed_by = self

    def add(self, deferred: Deferred, on_value: Callable[[Any], None] = lambda value: None) -> None:
        """
        Register a deferred value.

        :param deferred: The queries and how to turn their results into a value
        :param on_value: Receives the value once it is computed
        """
        self._entries.append((deferred, on_value, self.label))

    def on_complete(self, callback: Callable[[], None]) -> None:
        """
        Call something once all registered values have been handed on.

        :param callback: Called without arguments
        """
        self._on_complete.append(callback)

    def run(self, *outputs: polars.LazyFrame) -> list[polars.DataFrame]:
        """
        Execute all registered queries together with the given outputs in one pass.

        Polars runs the plan the queries and outputs share only once, so this costs a single
        execution of the pipeline however many checks and statistics were registered. Once
        done, a further call only computes `outputs`.

        :param outputs: Further frames to compute in the same pass, e.g. a lazy
            ``sink_parquet(..., lazy=True)`` or a frame to collect
        :return: The results of `outputs`, in the given order
        :raise RuntimeError: If a registered check fails, naming what registered it
        """
        if self._absorbed_by is not None:
            return self._absorbed_by.run(*outputs)
        if self.done:
            return polars.collect_all(list(outputs), engine="streaming") if outputs else []
        self._done = True

        queries = [query for deferred, _, _ in self._entries for query in deferred.queries]
        if not queries and not outputs:
            results = []
        else:
            results = polars.collect_all([*queries, *outputs], engine="streaming")

        position = 0
        for deferred, on_value, label in self._entries:
            count = len(deferred.queries)
            try:
                on_value(deferred.finish(results[position:position + count]))
            except ValueError as e:
                raise RuntimeError(f"{label}: {e}" if label else str(e)) from e
            position += count

        for callback in self._on_complete:
            callback()
        return results[position:]


_current: contextvars.ContextVar[PendingQueries | None] = contextvars.ContextVar(
    "damast_pending_queries", default=None
)


def current() -> PendingQueries | None:
    """The queries of the pipeline run in progress, or None outside a run."""
    return _current.get()


@contextmanager
def running(pending: PendingQueries) -> Iterator[PendingQueries]:
    """
    Make `pending` the run in progress, so that checks register there instead of executing.

    :param pending: The run's queries
    """
    token = _current.set(pending)
    try:
        yield pending
    finally:
        _current.reset(token)

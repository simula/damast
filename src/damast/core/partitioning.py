"""
Strategies for splitting an :class:`AnnotatedDataFrame` into multiple export files (e.g. per
day, per hour, per mmsi group) - see :meth:`AnnotatedDataFrame.export_partitioned`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Callable

import polars

__all__ = ["PartitionStrategy", "ByColumn", "ByTime", "ByExpr"]


class PartitionStrategy(ABC):
    """
    Splits a dataframe into named partitions for export as separate files.

    A strategy only has to say how to derive a per-row partition key and how to turn one
    key value into a filename - the actual splitting and writing is handled by
    :meth:`AnnotatedDataFrame.export_partitioned`.
    """

    @abstractmethod
    def key_expr(self) -> polars.Expr:
        """A polars expression computing this strategy's per-row partition key."""

    def filename(self, key: Any) -> str:
        """
        Format one partition's key value into a file stem (without extension).

        :param key: One distinct value of :meth:`key_expr` for the source dataframe
        :return: A filesystem-safe file stem for that partition
        """
        return str(key)


class ByColumn(PartitionStrategy):
    """One file per distinct value of an existing column - e.g. one file per mmsi."""

    def __init__(self, column: str, *, prefix: str | None = None):
        """
        :param column: Name of the column to partition by
        :param prefix: Filename prefix (default: ``column``), e.g. ``prefix="mmsi"``
            produces filenames like ``mmsi_257123456.parquet``
        """
        self.column = column
        self.prefix = column if prefix is None else prefix

    def key_expr(self) -> polars.Expr:
        return polars.col(self.column)

    def filename(self, key: Any) -> str:
        return f"{self.prefix}_{key}"


class ByTime(PartitionStrategy):
    """
    One file per time bucket of a timestamp column, truncated to a given interval.

    ``every`` is any interval polars' :meth:`polars.Expr.dt.truncate` accepts, e.g. ``"1h"``
    for one file per hour, ``"1d"`` for one file per day, ``"1w"`` per week.
    """

    def __init__(
        self,
        timestamp_column: str,
        every: str,
        *,
        format: str | None = None,
        prefix: str | None = None,
    ):
        """
        :param timestamp_column: Name of the datetime column to partition by
        :param every: Truncation interval, e.g. ``"1h"``, ``"1d"``, ``"1w"``
        :param format: Optional ``strftime`` format for the filename stem (default:
            automatically pick the coarsest representation matching the truncated value, e.g.
            ``"2026-09-14"`` for a day, ``"2026-09-14T13"`` for an hour). Use e.g.
            ``format="%Y/%m/%d"`` for a nested ``year/month/day`` directory layout - missing
            parent directories are created as needed.
        :param prefix: Optional filename prefix, e.g. ``prefix="ais"`` produces
            ``"ais_2026-09-14.parquet"``
        """
        self.timestamp_column = timestamp_column
        self.every = every
        self.format = format
        self.prefix = prefix

    def key_expr(self) -> polars.Expr:
        return polars.col(self.timestamp_column).dt.truncate(self.every)

    def filename(self, key: Any) -> str:
        stem = (
            key.strftime(self.format)
            if self.format is not None
            else self._default_stem(key)
        )
        return f"{self.prefix}_{stem}" if self.prefix else stem

    @staticmethod
    def _default_stem(key: Any) -> str:
        # Pick the coarsest clean representation that still matches the truncated value,
        # rather than parsing `every` ourselves - "2026-09-14", not "2026-09-14T00-00-00".
        if isinstance(key, date) and not isinstance(key, datetime):
            return key.isoformat()
        if key.microsecond or key.second:
            return key.strftime("%Y-%m-%dT%H-%M-%S")
        if key.minute:
            return key.strftime("%Y-%m-%dT%H-%M")
        if key.hour:
            return key.strftime("%Y-%m-%dT%H")
        return key.date().isoformat()


class ByExpr(PartitionStrategy):
    """
    Escape hatch: any custom key expression plus filename formatter.

    Example:

    .. code-block:: python

        # Bucket mmsis into 10 groups instead of one file per mmsi
        adf.export_partitioned(
            "out/",
            ByExpr(
                polars.col("mmsi") % 10,
                filename_fn=lambda group: f"mmsi_group_{group}",
            ),
        )
    """

    def __init__(self, expr: polars.Expr, filename_fn: Callable[[Any], str] = str):
        """
        :param expr: Polars expression computing the per-row partition key
        :param filename_fn: Formats one key value into a file stem (default: ``str``)
        """
        self._expr = expr
        self._filename_fn = filename_fn

    def key_expr(self) -> polars.Expr:
        return self._expr

    def filename(self, key: Any) -> str:
        return self._filename_fn(key)

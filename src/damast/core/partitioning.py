"""
Strategies for splitting an :class:`AnnotatedDataFrame` into multiple export files (e.g. per
day, per hour, per mmsi group) - see :meth:`AnnotatedDataFrame.export_partitioned`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars

if TYPE_CHECKING:
    from .dataframe import AnnotatedDataFrame

__all__ = ["ByColumn", "ByExpr", "ByTime", "PartitionStrategy", "SaveAs"]


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


#: Friendly names for common `polars.Expr.dt.truncate` intervals, accepted by `SaveAs.parse`
#: alongside any raw interval string (e.g. `"3h"`) it doesn't recognize.
_INTERVAL_ALIASES = {
    "hourly": "1h",
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1mo",
}

#: `<strategy>:` prefixes `SaveAs.parse` recognizes - anything else is a plain output path.
_SAVE_AS_STRATEGIES = ("time+column", "time", "column")


class SaveAs:
    """
    A parsed ``--save-as``-style string: either a plain output path (:attr:`strategy` is
    `None`), or a base directory plus `PartitionStrategy` - see :meth:`parse` and
    :meth:`export`.
    """

    def __init__(self, path: Path, strategy: PartitionStrategy | None = None):
        """
        :param path: Plain output file path, or the base directory for a partitioned export
        :param strategy: `None` for a plain single-file export
        """
        self.path = path
        self.strategy = strategy

    @classmethod
    def parse(cls, value: str) -> SaveAs:
        """
        Parse a `--save-as`-style string, so a single CLI-facing argument can select
        :meth:`AnnotatedDataFrame.export_partitioned` over the default single-file
        :meth:`AnnotatedDataFrame.export` - see :meth:`export`.

        Without a recognized `<strategy>:` prefix, `value` is a plain output file path, exactly
        as before - ``SaveAs(Path(value))``.

        With a `<strategy>:<spec>:<template>` prefix, `value` selects partitioned export.
        `template` is the filename stem for every partition (no extension - `.parquet` is added
        automatically by `export_partitioned`) and may contain:

        - any `strftime` code (``%Y``, ``%m``, ``%d``, ``%H``, ...) - the truncated timestamp
        - ``{<column>}`` - the literal value of that column for this partition (curly braces,
          not a `%` code, so it never collides with a `strftime` code)

        plus arbitrary literal text, including ``/`` for a nested output directory - missing
        parent directories are created automatically.

        Supported strategies:

        - ``time:<timestamp_column>+<interval>:<template>`` - one file per time bucket
        - ``column:<column>:<template>`` - one file per distinct value of ``<column>``
        - ``time+column:<timestamp_column>+<interval>+<column>:<template>`` - one file per
          (time bucket, column value) pair

        ``<interval>`` is one of ``hourly``, ``daily``, ``weekly``, ``monthly``, or any raw
        `polars.Expr.dt.truncate` interval (e.g. ``"3h"``, ``"15m"``).

        Example:

        .. code-block:: python

            SaveAs.parse("out/result.parquet")
            # -> SaveAs(Path("out/result.parquet"), strategy=None)

            SaveAs.parse("time:timestamp+daily:out/AIS_%Y_%m_%d")
            # -> SaveAs(Path("."), ByExpr(...))  # one file per day

            SaveAs.parse("column:mmsi:out/vessel_{mmsi}")
            # -> SaveAs(Path("."), ByExpr(...))  # one file per mmsi

            SaveAs.parse("time+column:timestamp+daily+mmsi:out/{mmsi}/AIS_%Y_%m_%d")
            # -> SaveAs(Path("."), ByExpr(...))  # one file per (day, mmsi) pair

        :param value: A plain output path, or a `<strategy>:<spec>:<template>` partitioning string
        :raise ValueError: If a recognized `<strategy>:` prefix is used with a malformed spec
        """
        for strategy_name in _SAVE_AS_STRATEGIES:
            prefix = f"{strategy_name}:"
            if value.startswith(prefix):
                spec, _, template = value[len(prefix) :].partition(":")
                if not template:
                    raise ValueError(
                        f"SaveAs.parse: '{strategy_name}:' requires '{strategy_name}:<spec>:<template>',"
                        f" but got {value!r}"
                    )
                return cls(
                    Path("."), cls._build_strategy(strategy_name, spec, template, value)
                )

        return cls(Path(value))

    def export(self, adf: AnnotatedDataFrame) -> Path | list[Path]:
        """
        Export `adf` per this spec: a single file via :meth:`AnnotatedDataFrame.save` for a
        plain path (parquet with a sidecar `.spec.yaml`, or hdf5), or one file per partition
        via :meth:`AnnotatedDataFrame.export_partitioned` otherwise.

        :param adf: The dataframe to export
        :return: The single written path, or the list of per-partition paths
        """
        if self.strategy is None:
            adf.save(filename=self.path)
            return self.path

        return adf.export_partitioned(self.path, self.strategy)

    @classmethod
    def expected_paths(cls, value: str, *, start: datetime, end: datetime) -> list[Path]:
        """
        Enumerate the file paths a ``--save-as``-style spec (see :meth:`parse`) is expected to create for
        data in ``[start, end]`` - the read-side counterpart to :meth:`export`.
        Can be used to identify files that already exist rather than write new ones, e.g., selecting
        the partitions of an existing archive that overlap a time range.

        The following cases are covered:
        (a) a plain path (no recognized strategy prefix) returns ``[Path(value)]`` unchanged.
        (b) for a partitioned spec, one path is returned per time bucket touched by ``[start, end]``;
        (c) any part of the template that cannot be derived from time alone - a ``{column}``
        placeholder, or the whole template for a ``column:``-only spec - is rendered as a glob
        wildcard ``*``, so the caller can ``Path(...).glob(...)`` for the files that actually
        exist there instead of assuming every candidate was written.

        Example:

        .. code-block:: python

            SaveAs.expected_paths("out/result.parquet", start=d1, end=d2)
            # -> [Path("out/result.parquet")]

            SaveAs.expected_paths("time:timestamp+daily:out/AIS_%Y_%m_%d", start=d1, end=d2)
            # -> one Path per day in [d1, d2], e.g. [Path("out/AIS_2026-01-01.parquet"), ...]

            SaveAs.expected_paths("column:mmsi:out/vessel_{mmsi}", start=d1, end=d2)
            # -> [Path("out/vessel_*.parquet")]

        :param value: A plain output path, or a `<strategy>:<spec>:<template>` partitioning string
        :param start: Start of the time range (inclusive)
        :param end: End of the time range (inclusive)
        :return: Candidate paths - exact for a plain path or a purely time-based spec, glob
            patterns (containing ``*``) wherever a value can't be derived from ``start``/``end`` alone
        :raise ValueError: If a recognized `<strategy>:` prefix is used with a malformed spec
        """
        for strategy_name in _SAVE_AS_STRATEGIES:
            prefix = f"{strategy_name}:"
            if not value.startswith(prefix):
                continue
            spec, _, template = value[len(prefix) :].partition(":")
            if not template:
                raise ValueError(
                    f"SaveAs.expected_paths: '{strategy_name}:' requires"
                    f" '{strategy_name}:<spec>:<template>', but got {value!r}"
                )
            return cls._expected_paths_for_strategy(strategy_name, spec, template, start, end, value)

        return [Path(value)]

    @classmethod
    def _build_strategy(cls,
        strategy_name: str, spec: str, template: str, value: str
    ) -> PartitionStrategy:
        """Construct the partition(ing) strategy """
        if strategy_name == "column":
            column = spec
            if not column:
                raise ValueError(
                    f"SaveAs.parse: 'column:' requires 'column:<column>:<template>', but got {value!r}"
                )
            return ByExpr(
                polars.col(column), filename_fn=lambda key: template.format(**{column: key})
            )

        if strategy_name == "time":
            timestamp_column, _, interval = spec.partition("+")
            if not timestamp_column or not interval:
                raise ValueError(
                    f"SaveAs.parse: 'time:' requires 'time:<timestamp_column>+<interval>:<template>',"
                    f" but got {value!r}"
                )
            interval = _INTERVAL_ALIASES.get(interval, interval)
            key_expr = polars.col(timestamp_column).dt.truncate(interval)
            return ByExpr(key_expr, filename_fn=lambda key: key.strftime(template))

        # "time+column"
        timestamp_column, _, rest = spec.partition("+")
        interval, _, column = rest.partition("+")
        if not timestamp_column or not interval or not column:
            raise ValueError(
                "SaveAs.parse: 'time+column:' requires"
                f" 'time+column:<timestamp_column>+<interval>+<column>:<template>', but got {value!r}"
            )
        interval = _INTERVAL_ALIASES.get(interval, interval)
        time_field, column_field = "__time", column
        key_expr = polars.struct(
            [
                polars.col(timestamp_column).dt.truncate(interval).alias(time_field),
                polars.col(column).alias(column_field),
            ]
        )

        def filename_fn(key: dict) -> str:
            substituted = template.format(**{column_field: key[column_field]})
            return key[time_field].strftime(substituted)

        return ByExpr(key_expr, filename_fn=filename_fn)


    @classmethod
    def _expected_paths_for_strategy(cls,
        strategy_name: str, spec: str, template: str, start: datetime, end: datetime, value: str
    ) -> list[Path]:
        """Read-side counterpart to :meth:`SaveAs._build_strategy`, for :meth:`SaveAs.expected_paths`."""
        if strategy_name == "column":
            column = spec
            if not column:
                raise ValueError(
                    f"SaveAs.expected_paths: 'column:' requires 'column:<column>:<template>', but got {value!r}"
                )
            wildcard = template.format(**{column: "*"})
            return [Path(f"{wildcard}.parquet")]

        if strategy_name == "time":
            timestamp_column, _, interval = spec.partition("+")
            if not timestamp_column or not interval:
                raise ValueError(
                    f"SaveAs.expected_paths: 'time:' requires 'time:<timestamp_column>+<interval>:<template>',"
                    f" but got {value!r}"
                )
            interval = _INTERVAL_ALIASES.get(interval, interval)
            return [Path(f"{bucket.strftime(template)}.parquet") for bucket in cls._time_buckets(start, end, interval)]

        # "time+column"
        timestamp_column, _, rest = spec.partition("+")
        interval, _, column = rest.partition("+")
        if not timestamp_column or not interval or not column:
            raise ValueError(
                "SaveAs.expected_paths: 'time+column:' requires"
                f" 'time+column:<timestamp_column>+<interval>+<column>:<template>', but got {value!r}"
            )
        interval = _INTERVAL_ALIASES.get(interval, interval)
        wildcard_template = template.format(**{column: "*"})
        return [Path(f"{bucket.strftime(wildcard_template)}.parquet") for bucket in cls._time_buckets(start, end, interval)]


    @classmethod
    def _time_buckets(cls, start: datetime, end: datetime, interval: str) -> list[datetime]:
        """Return the sorted, deduplicated `dt.truncate(interval)` bucket starts covering `[start, end]`."""
        samples = polars.datetime_range(start, end, interval=interval, eager=True)
        return sorted(samples.dt.truncate(interval).unique().to_list())

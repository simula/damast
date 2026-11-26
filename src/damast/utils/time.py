"""Time helpers: naive UTC datetimes, in Python and in polars frames."""

from __future__ import annotations

import datetime as dt

import polars as pl

__all__ = ["Time"]


class Time:
    """Conversions to naive UTC datetimes, for Python values and polars columns."""

    @staticmethod
    def naive_utc(value: dt.datetime) -> dt.datetime:
        """The datetime in UTC without a time zone; a naive one is taken as UTC already."""
        if value.tzinfo is not None:
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value

    @classmethod
    def utc_day(cls, value: dt.datetime | dt.date | None) -> dt.date | None:
        """The UTC day of a datetime (see :meth:`naive_utc`); a date as it is."""
        if isinstance(value, dt.datetime):
            return cls.naive_utc(value).date()
        return value

    @staticmethod
    def parse_datetime(column: str, dtype: pl.DataType, format: str | None = None) -> pl.Expr:
        """The column as a datetime (microseconds): strings parsed with the
        strftime-like format (inferred, naive ones read as UTC, if None;
        unparsable ones null), dates at midnight, numbers as epoch seconds."""
        col = pl.col(column)
        if dtype == pl.String:
            if format is None:
                # Inferred: read as UTC, so strings with an offset parse too.
                return col.str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
            return col.str.to_datetime(format=format, time_unit="us", strict=False)
        if isinstance(dtype, pl.Datetime):
            return col
        if dtype == pl.Date:
            return col.cast(pl.Datetime("us"))
        return pl.from_epoch(col.cast(pl.Int64), time_unit="s").alias(column)

    @classmethod
    def with_naive_utc(cls, lf: pl.LazyFrame, column: str, format: str | None = None) -> pl.LazyFrame:
        """The frame with the column as a naive UTC datetime in microseconds
        (parsed by :meth:`parse_datetime`; time zone aware ones converted to UTC)."""
        lf = lf.with_columns(cls.parse_datetime(column, lf.collect_schema()[column], format))
        dtype = lf.collect_schema()[column]
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            lf = lf.with_columns(pl.col(column).dt.convert_time_zone("UTC").dt.replace_time_zone(None))
        return lf.with_columns(pl.col(column).dt.cast_time_unit("us"))

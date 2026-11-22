import datetime as dt

import polars as pl
import pytest

from damast.utils.time import Time

CET = dt.timezone(dt.timedelta(hours=1))


def test_naive_utc():
    assert Time.naive_utc(dt.datetime(2026, 6, 1, 1, 30, tzinfo=CET)) == dt.datetime(2026, 6, 1, 0, 30)
    assert Time.naive_utc(dt.datetime(2026, 6, 1, 1, 30)) == dt.datetime(2026, 6, 1, 1, 30)


def test_utc_day():
    assert Time.utc_day(dt.datetime(2026, 6, 1, 0, 30, tzinfo=CET)) == dt.date(2026, 5, 31)
    assert Time.utc_day(dt.date(2026, 6, 1)) == dt.date(2026, 6, 1)
    assert Time.utc_day(None) is None


@pytest.mark.parametrize(
    "values, format",
    [
        (["01/06/2026 10:00:00"], "%d/%m/%Y %H:%M:%S"),
        (["2026-06-01T12:00:00.000+02:00"], "%Y-%m-%dT%H:%M:%S%.f%:z"),
        (["2026-06-01T12:00:00+02:00"], None),
        (["2026-06-01 10:00:00"], None),
        ([dt.datetime(2026, 6, 1, 10)], None),
        ([1780308000], None),
    ],
)
def test_with_naive_utc(values, format):
    lf = Time.with_naive_utc(pl.LazyFrame({"t": values}), "t", format)
    out = lf.collect()
    assert out.schema["t"] == pl.Datetime("us")
    assert out["t"].to_list() == [dt.datetime(2026, 6, 1, 10)]


def test_with_naive_utc_converts_aware_datetimes():
    aware = pl.LazyFrame({"t": [dt.datetime(2026, 6, 1, 12)]}).with_columns(
        pl.col("t").dt.replace_time_zone("Europe/Oslo")
    )
    assert Time.with_naive_utc(aware, "t").collect()["t"].to_list() == [dt.datetime(2026, 6, 1, 10)]


def test_unparsable_strings_are_null():
    out = Time.with_naive_utc(pl.LazyFrame({"t": ["not a time"]}), "t", "%d/%m/%Y %H:%M:%S").collect()
    assert out["t"].null_count() == 1

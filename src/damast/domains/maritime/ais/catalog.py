"""An AIS data lake: daily Parquet files of several sources in a hive layout,

    <base>/source=<name>/date=YYYY-MM-DD/*.parquet
    <base>/source-mapping.yaml

with each source's columns mapped to :data:`CANONICAL_AIS_SCHEMA` by the
mapping file::

    <source>:
      timestamp_format: "%d/%m/%Y %H:%M:%S"  # optional: of a string timestamp column
      scale:                                 # optional: factor per canonical column
        draught: 0.1
      columns:                               # canonical column -> source column
        mmsi: MMSI
        timestamp: '# Timestamp'
        ...
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, Field, field_validator

from damast.utils.time import Time

logger = logging.getLogger(__name__)

__all__ = ["CANONICAL_AIS_SCHEMA", "AISDataLake", "SourceMapping", "TimeInterval", "VesselCoverage"]

CANONICAL_AIS_SCHEMA: list[str] = [
    "mmsi",
    "timestamp",
    "latitude",
    "longitude",
    "sog",
    "cog",
    "heading",
    "nav_status",
    "ship_type",
    "imo",
    "call_sign",
    "name",
    "length",
    "width",
    "draught",
    "report_class",
]

TRACK_COLUMNS = ["mmsi", "timestamp", "latitude", "longitude"]


class TimeInterval(BaseModel):
    """The first and last day a source has data for."""

    from_date: dt.date
    to_date: dt.date


class VesselCoverage(BaseModel):
    """The days, and their files, a vessel appears in for one source."""

    source: str
    dates: list[dt.date] = Field(default_factory=list)
    files: list[Path] = Field(default_factory=list)


class SourceMapping(BaseModel):
    """How one source's columns map to the canonical AIS schema."""

    columns: dict[str, str]
    timestamp_format: str | None = None
    # In case a rescaling of values is necessary
    scale: dict[str, float] = Field(default_factory=dict)

    @field_validator("columns", "scale")
    @classmethod
    def _canonical_keys(cls, value: dict) -> dict:
        unknown = sorted(set(value) - set(CANONICAL_AIS_SCHEMA))
        if unknown:
            raise ValueError(f"SourceMapping._canonical_keys: not in the canonical AIS schema: {unknown}")
        return value

    @property
    def missing(self) -> list[str]:
        """The canonical columns this source doesn't map."""
        return [c for c in CANONICAL_AIS_SCHEMA if c not in self.columns]

    @property
    def renames(self) -> dict[str, str]:
        """Compute the mapping from source column to canonical column, for use in `rename`."""
        return {raw: canonical for canonical, raw in self.columns.items()}


class AISDataLake:
    """The files and column mappings of an AIS data lake (see the module docstring).

    Directory listings and the mapping file are read on first use and by
    :meth:`refresh`.
    """

    SOURCE_MAPPING_FILENAME = "source-mapping.yaml"

    SOURCE_DIR_REGEX: re.Pattern = re.compile(r"^source=(?P<source>.+)$")
    DATE_DIR_REGEX: re.Pattern = re.compile(r"^date=(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})$")

    def __init__(self, base_dir: Path | str):
        base_dir = Path(base_dir)
        if not base_dir.is_dir():
            raise FileNotFoundError(f"AISDataLake: no such directory {base_dir}")

        self.base_dir = base_dir

        self._sources: set[str] | None = None
        self._time_intervals: dict[str, TimeInterval] = {}
        self._mappings: dict[str, SourceMapping] = {}

    def refresh(self) -> None:
        """Read the sources, their days and the mapping file again.

        :raises ValueError: if the mapping file is malformed
        """
        self._sources = self.discover_sources()
        self._time_intervals = self.available_timeintervals()
        self._mappings = self._read_mappings()

    def _ensure_loaded(self) -> None:
        if self._sources is None:
            self.refresh()

    def _read_mappings(self) -> dict[str, SourceMapping]:
        mapfile = self.base_dir / self.SOURCE_MAPPING_FILENAME
        if not mapfile.exists():
            logger.warning(f"AISDataLake: no mapping file {mapfile}")
            return {}

        with open(mapfile) as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(f"AISDataLake: {mapfile} must map source names to their mappings")

        mappings = {}
        for source, entry in data.items():
            try:
                mapping = SourceMapping.model_validate(entry)
            except ValueError as e:
                raise ValueError(f"AISDataLake: {mapfile}: source {source!r}: {e}") from e

            if mapping.missing:
                logger.warning(f"AISDataLake: {mapfile}: source {source!r} doesn't map {mapping.missing}")
            mappings[str(source)] = mapping
        return mappings

    @property
    def sources(self) -> set[str]:
        self._ensure_loaded()

        assert self._sources is not None
        return self._sources

    @property
    def time_intervals(self) -> dict[str, TimeInterval]:
        """The first and last day of each source with at least one day."""
        self._ensure_loaded()
        return self._time_intervals

    def source_mapping(self, source: str) -> SourceMapping | None:
        """The source's column mapping, None if the mapping file has none."""
        self._ensure_loaded()
        return self._mappings.get(source)

    def get_source_mapping(self, source: str) -> dict[str, str]:
        """Source column -> canonical column.

        :raises ValueError: if the mapping file has no entry for the source
        """
        mapping = self.source_mapping(source)
        if mapping is None:
            raise ValueError(f"AISDataLake.get_source_mapping: could not find {source=} in mapping file")
        return mapping.renames

    def discover_sources(self) -> set[str]:
        sources = set()
        for source_dir in self.base_dir.iterdir():
            m = self.SOURCE_DIR_REGEX.match(source_dir.name)
            if m and source_dir.is_dir():
                sources.add(m.group("source"))
        return sources

    def source_dir(self, source: str) -> Path:
        return self.base_dir / f"source={source}"

    def days(self, source: str) -> list[tuple[dt.date, Path]]:
        """The source's day directories, sorted by day."""
        try:
            entries = list(self.source_dir(source).iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return []

        days = []
        for date_dir in entries:
            m = self.DATE_DIR_REGEX.match(date_dir.name)
            if not m or not date_dir.is_dir():
                continue
            try:
                days.append((dt.date.fromisoformat(m.group("date")), date_dir))
            except ValueError:
                logger.warning(f"AISDataLake: not a valid day: {date_dir}")
        return sorted(days)

    def available_timeintervals(self) -> dict[str, TimeInterval]:
        """The first and last day of each source; sources without days are left out."""
        intervals = {}
        for source in self.sources:
            days = self.days(source)
            if days:
                intervals[source] = TimeInterval(from_date=days[0][0], to_date=days[-1][0])
        return intervals

    def matching_files(
        self,
        from_time: dt.datetime | dt.date | None = None,
        to_time: dt.datetime | dt.date | None = None,
        sources: Iterable[str] | None = None,
    ) -> Iterator[tuple[str, dt.date, Path]]:
        """(source, day, file) of every Parquet file of the given sources
        (all if None) on the days from from_time to to_time, both included
        (unbounded if None), sorted by source, day and file name."""

        from_day = Time.utc_day(from_time)
        to_day = Time.utc_day(to_time)

        selected = self.sources if sources is None else set(sources) & self.sources
        for source in sorted(selected):
            for day, date_dir in self.days(source):
                if (from_day is not None and day < from_day) or (to_day is not None and day > to_day):
                    continue

                for file in sorted(date_dir.glob("*.parquet")):
                    if file.is_file():
                        yield source, day, file

    def scan_source(self, source: str, files: list[Path], columns: list[str] | None = None) -> pl.LazyFrame:
        """The files of one source in the canonical schema: renamed, the
        timestamp parsed to a naive UTC datetime and scale applied. columns:
        the canonical columns to keep (all mapped ones if None).

        :raises ValueError: if the mapping file has no entry for the source
        """
        mapping = self.source_mapping(source)
        if mapping is None:
            raise ValueError(f"AISDataLake: could not find source={source!r} in mapping file")

        wanted = [c for c in (columns or CANONICAL_AIS_SCHEMA) if c in mapping.columns]
        lf = pl.scan_parquet(files).select([pl.col(mapping.columns[c]).alias(c) for c in wanted])

        if "timestamp" in wanted:
            lf = Time.with_naive_utc(lf, "timestamp", mapping.timestamp_format)

        scaled = [(pl.col(c) * f).alias(c) for c, f in mapping.scale.items() if c in wanted]
        return lf.with_columns(scaled) if scaled else lf

    def by_mmsi_quickview(
        self,
        sources: Iterable[str] | None = None,
        from_time: dt.datetime | dt.date | None = None,
        to_time: dt.datetime | dt.date | None = None,
    ) -> dict[int, dict[str, VesselCoverage]]:
        """mmsi -> source -> the days and files the vessel appears in."""
        coverage: dict[int, dict[str, VesselCoverage]] = {}
        for source, day, file in self.matching_files(from_time=from_time, to_time=to_time, sources=sources):
            mmsis = self.scan_source(source, [file], columns=["mmsi"]).unique().collect()["mmsi"]
            for mmsi in mmsis:
                if mmsi is None:
                    continue

                entry = coverage.setdefault(mmsi, {}).setdefault(source, VesselCoverage(source=source))
                if not entry.dates or entry.dates[-1] != day:
                    entry.dates.append(day)
                entry.files.append(file)
        return coverage

    def get_track(
        self,
        mmsi: int,
        from_time: dt.datetime | None = None,
        to_time: dt.datetime | None = None,
        sources: Iterable[str] | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """The vessel's reports from all (or the given) sources between
        from_time and to_time (both included; aware times are converted to
        UTC), sorted by time, with a `source` column. columns: the canonical
        columns (default: mmsi, timestamp, latitude, longitude)."""
        columns = list(columns or TRACK_COLUMNS)
        for required in ("mmsi", "timestamp"):
            if required not in columns:
                columns.append(required)

        files: dict[str, list[Path]] = {}
        for source, _, file in self.matching_files(from_time=from_time, to_time=to_time, sources=sources):
            files.setdefault(source, []).append(file)

        frames = []
        for source, source_files in files.items():
            lf = self.scan_source(source, source_files, columns).filter(pl.col("mmsi") == mmsi)
            if from_time is not None:
                lf = lf.filter(pl.col("timestamp") >= Time.naive_utc(from_time))
            if to_time is not None:
                lf = lf.filter(pl.col("timestamp") <= Time.naive_utc(to_time))
            frames.append(lf.with_columns(pl.lit(source).alias("source")))

        if not frames:
            return pl.DataFrame(schema={"mmsi": pl.Int64, "timestamp": pl.Datetime("us"), "source": pl.String})
        return pl.concat(frames, how="diagonal_relaxed").sort("timestamp").collect(engine="streaming")


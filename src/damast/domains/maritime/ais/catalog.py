from pathlib import Path
import datetime as dt
import re
import yaml
from pydantic import BaseModel

import polars as pl
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

class TimeInterval(BaseModel):
    from_time: dt.datetime
    to_time: dt.datetime

class VesselCoverage(BaseModel):
    source: str
    dates: list[dt.datetime] = []
    files: list[Path] = []

class AISDataLake:
    base_dir: Path

    CANONICAL_AIS_SCHEMA = [
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
        "report_class"
    ]
    SOURCE_MAPPING_FILENAME = "source-mapping.yaml"
    _sources: set[str]
    _sources_column_mapping: dict[str, dict[str, str]]

    time_intervals: dict[str, TimeInterval]

    SOURCE_DIR_REGEX : re.Pattern = re.compile("source=(?P<source>.*)")
    DATE_DIR_REGEX : re.Pattern = re.compile("date=(?P<date>[0-9]{4}-[0-2][0-9]-[0-3][0-9])")

    def __init__(self, base_dir: Path | str):
        base_dir = Path(base_dir)

        if not base_dir.exists():
            raise FileNotFoundError("AISDataLake: could not find {base_dir=}")

        self.base_dir = base_dir
        self._sources = None
        self._sources_column_mapping = {}

    def refresh(self):
        self._sources = self.discover_sources()
        self.time_intervals = self.available_timeintervals()

        mapfile = (self.base_dir / self.SOURCE_MAPPING_FILENAME)
        if mapfile.exists():
            with open(mapfile, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                breakpoint()
                for source in data:
                    self._sources_column_mapping[source] = {y: x for x,y in data[source]["columns"].items()}
        else:
            logger.warning("AISDataLake.refresh: no mapping file {mapfile} exists")

    def get_source_mapping(self, source: str) -> str:
        if source in self._sources_column_mapping:
            return self._sources_column_mapping[source]

        raise ValueError(f"AISDataLake.get_source_mapping: could not find {source=} in mapping file")

    @property
    def sources(self):
        if self._sources:
            return self._sources

        self.refresh()
        return self._sources

    def discover_sources(self) -> set[str]:
        sources = set()

        source_dirs = self.base_dir.glob("source=*")
        for source_dir in source_dirs:
            m = self.SOURCE_DIR_REGEX.search(str(source_dir))
            sources.add(m.group("source"))

        return sources


    def available_timeintervals(self) -> dict[str, TimeInterval]:
        timeintervals = {}

        for source in self.sources:
            date_dirs = self.base_dir.glob(f"source={source}/date=*")
            from_time, to_time = None, None
            sorted_date_dirs = sorted(date_dirs)

            for date_dir in sorted_date_dirs:
                m = self.DATE_DIR_REGEX.search(str(date_dir))
                if m:
                    from_time = dt.datetime.fromisoformat(m.group("date"))
                    break

            for date_dir in reversed(sorted_date_dirs):
                m = self.DATE_DIR_REGEX.search(str(date_dir))
                if m:
                    to_time = dt.datetime.fromisoformat(m.group("date"))
                    break

            if from_time > to_time:
                raise RuntimeError("AISDataLake.available_timeintervals: {from_time=} cannot be greater than {to_time=}")
            timeintervals[source] = TimeInterval(from_time=from_time, to_time=to_time)

        return timeintervals


    def matching_files(self,
            from_time: dt.datetime | None,
            to_time: dt.datetime | None,
            sources: list[str] | None = None,
    ) -> list[Path]:

        for source in self.sources:
            if from_time is None:
                start_ordinal = self.time_intervals[source].from_time.toordinal()
            else:
                start_ordinal = from_time.toordinal()

            if to_time is None:
                end_ordinal = self.time_intervals[source].to_time.toordinal()
            else:
                end_ordinal = to_time.toordinal()


            for ordinal in range(start_ordinal, end_ordinal + 1):
                date = dt.datetime.fromordinal(ordinal).strftime("%Y-%m-%d")
                for file in self.base_dir.glob(f"source={source}/date={date}/*.parquet"):
                    yield source, date, file

    def by_mmsi_quickview(self, sources: list[str] | None = None, from_time: dt.datetime | None = None, to_time: dt.datetime | None = None):

        coverage: dict[str, list[VesselCoverage]] = {}
        for source, date, file in tqdm(self.matching_files(sources=sources, from_time=from_time, to_time=to_time), desc="Loading files", unit="file"):
            mmsis = pl.scan_parquet([file]).rename(self.get_source_mapping(source)).select(pl.col("mmsi")).collect(engine="streaming")

            for m in mmsis.iter_rows():
                mmsi = m[0]
                if mmsi not in mmsis:
                    coverage[mmsi] = { source: VesselCoverage(source=source, dates=[date], files=[file]) }
                else:
                    mmsi_coverage = coverage[mmsi]
                    if source not in mmsi_coverage:
                        mmsi_coverage[source] = VesselCoverage(source=source, dates=[date], files=[file])
                    else:
                        mmsi_coverage[source].dates.append(date)
                        mmsi_coverage[source].files.append(file)

        return coverage

    def get_track(self,
                  mmsi: int,
                  from_time: dt.datetime | None = None,
                  to_time: dt.datetime | None = None,
                  sources: list[str] | None = None
                  ) -> pl.LazyFrame:

        sources_files = {}
        for source,_,file in self.matching_files(from_time=from_time, to_time=to_time, sources=sources):
            if source not in sources_files:
                sources_files[source] = [file]
            else:
                sources_files[source].append(file)

        dataframes = []
        for source, files in sources_files.items():
            dataframes.append(pl.scan_parquet(files)
              .rename(self.get_source_mapping(source))
              .select([pl.col(x) for x in ["mmsi", "latitude", "longitude", "timestamp"]])
              .filter(pl.col("mmsi") == mmsi)
            )

        return pl.concat(dataframes).collect(engine="streaming")

        #def _run_track_query(mmsi: int, files: list[Path]):

        #with ProcessPoolExecutor(max_workers=1) as pool:
        #    results = pool.submit(_run_track_query, files, mmsi).result()

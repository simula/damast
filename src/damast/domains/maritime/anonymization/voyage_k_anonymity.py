"""
Anonymization strategy `VoyageKAnonymity`: per-voyage pseudonyms and k-anonymity in space and time.

Replacing the MMSI alone is pseudonymization only: AIS is broadcast publicly, so a track can be
re-identified by matching it against another AIS source, and the berth where it starts or ends
identifies the vessel. The strategy chains these elements, in this order:

1. `SegmentVoyages` - drop fixes at rest and split each vessel's track into voyages
2. `SuppressVoyageEndpoints` - drop the fixes close to where a voyage starts or ends
3. `GeneralizeSpaceTime` - snap positions to H3 cells, bucket time, round speed and course
4. `KAnonymityFilter` - drop the fixes of every (cell, time bucket) with fewer than k vessels
5. `PseudonymizeVoyages` - replace the voyage by a keyed pseudonym and keep only whitelisted
   columns, which removes the MMSI

The MMSI is kept until the last step, so that k counts vessels: a vessel whose track is split
into several voyages still counts once.
"""
import hashlib
import hmac
import logging
import os
import secrets
import tempfile
from pathlib import Path

import polars as pl
import polars_h3

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement
from damast.domains.maritime.ais.vessel_types import ShipType
from damast.domains.maritime.anonymization.strategy import AISColumns, AnonymizationStrategy, Step

__all__ = [
    "GeneralizeSpaceTime",
    "KAnonymityFilter",
    "PseudonymizeVoyages",
    "SegmentVoyages",
    "SuppressVoyageEndpoints",
    "VoyageKAnonymity",
]

_log = logging.getLogger(__name__)

#: Navigational status codes of a vessel at rest: 1 - at anchor, 5 - moored
AT_REST_STATUS: list[int] = [1, 5]


def voyage_pseudonym(key: bytes, voyage: str, length: int = 16) -> str:
    """
    Pseudonym of a voyage - a keyed hash, so it cannot be reversed by enumerating MMSIs.

    Args:
        key: Secret key of the HMAC
        voyage: Key of the voyage, as `SegmentVoyages` makes it: "mmsi|start time"
        length: Number of hex digits to keep

    Returns:
        The first `length` hex digits of HMAC-SHA256(key, voyage)
    """
    return hmac.new(key, voyage.encode(), hashlib.sha256).hexdigest()[:length]


def load_key(key_env: str | None, user: str) -> bytes:
    """
    The secret key of a keyed step, read as hex from the environment variable `key_env`.

    Args:
        key_env: Name of the environment variable holding the key, or None for a random key
        user: Name of the step using the key, for the log message

    Returns:
        The key, or 32 random bytes if `key_env` is None

    Raises:
        KeyError: If the environment variable `key_env` is not set
    """
    if key_env is None:
        _log.info(f"{user}: no key_env - using a random key, the output is not reproducible")
        return secrets.token_bytes(32)
    return bytes.fromhex(os.environ[key_env])


#: Ship type code -> code of its category, see `ShipType.category`
SHIP_TYPE_CATEGORIES: dict[int, int | None] = {code: ShipType.category(code) for code in range(100)}


class SegmentVoyages(PipelineElement):
    """
    Drop the fixes at rest and split each vessel's track into voyages.

    Each vessel's fixes are sorted by time. A fix is at rest if its navigational status is
    anchored or moored, or its speed is below `rest_speed`; those fixes are dropped, since a
    berth or anchorage identifies a vessel. A new voyage starts at the first fix after a stretch
    at rest, or after a gap of more than `max_gap_s` seconds.

    The added column `voyage` identifies a voyage as "mmsi|start time" - it contains the MMSI, so
    it must not be released; `PseudonymizeVoyages` replaces it.

    Args:
        max_gap_s: Gap between two fixes, in seconds, after which a new voyage starts
        rest_speed: Speed over ground, in knots, below which a fix counts as at rest
    """

    def __init__(self, max_gap_s: float = 3600.0, rest_speed: float = 0.5):
        self.max_gap_s = max_gap_s
        self.rest_speed = rest_speed

    @describe("Drop the fixes at rest and split each vessel's track into voyages")
    @input({"mmsi": {"representation_type": int},
            "timestamp": {},
            "speed": {},
            "status": {"representation_type": int}})
    @output({"voyage": {"representation_type": str, "description": "voyage as 'mmsi|start time'"}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        mmsi = self.get_name("mmsi")
        timestamp = self.get_name("timestamp")
        speed = self.get_name("speed")
        status = self.get_name("status")

        at_rest = (pl.col(status).is_in(AT_REST_STATUS) | (pl.col(speed) < self.rest_speed)).fill_null(False)
        moving = df.lazyframe.sort(mmsi, timestamp).with_columns(
            at_rest.shift(1).over(mmsi).fill_null(False).alias("_after_rest"),
            at_rest.alias("_at_rest"),
        ).filter(~pl.col("_at_rest"))

        # gaps are measured between moving fixes only - a stretch at rest is a break of its own
        gap = pl.col(timestamp).diff().over(mmsi).dt.total_seconds(fractional=True)
        voyage_start = (gap.is_null() | (gap > self.max_gap_s) | pl.col("_after_rest")).cast(pl.UInt32)
        moving = moving.with_columns(voyage_start.cum_sum().over(mmsi).alias("_voyage"))
        df.lazyframe = moving.with_columns(
            pl.concat_str(pl.col(mmsi), pl.col(timestamp).first().over(mmsi, "_voyage").cast(pl.String),
                          separator="|").alias(self.get_name("voyage"))
        ).drop("_after_rest", "_at_rest", "_voyage")
        return df


class PseudonymizeVoyages(PipelineElement):
    """
    Replace the voyage by a keyed pseudonym, and keep only whitelisted columns.

    The last step of the anonymization: a voyage's pseudonym `voyage_id` is the HMAC of the
    `voyage` column of `SegmentVoyages` - see `voyage_pseudonym`. The result holds only
    `voyage_id` and the mapped columns `timestamp`, `lat`, `lon`, `speed`, `course`, `status` and
    `ship_type` - every other column (MMSI, voyage, name, call sign, IMO number, destination,
    dimensions, ...) is dropped, so an unexpected identifying column cannot slip through.
    `ship_type` is coarsened to its category, see `ShipType.category`.

    The key is read from the environment variable named by `key_env`, as hex. It is never a
    parameter, since parameters are saved with the pipeline. Without `key_env`, a random key is
    drawn on every run: pseudonyms then differ between runs and cannot be linked across releases.

    Example:

    ```python
    pipeline.add("pseudonymize", PseudonymizeVoyages(key_env="AIS_PSEUDONYM_KEY"),
                 name_mappings={"voyage": "voyage", "timestamp": "msgtime",
                                "lat": "latitude", "lon": "longitude",
                                "speed": "speedOverGround", "course": "courseOverGround",
                                "status": "navigationalStatus", "ship_type": "shipType"})
    ```

    Args:
        key_env: Name of the environment variable holding the HMAC key as hex, or None for a
            random key per run

    Raises:
        KeyError: On transform, if the environment variable `key_env` is not set
    """

    def __init__(self, key_env: str | None = None):
        self.key_env = key_env

    @describe("Replace the voyage by a keyed pseudonym and drop all non-whitelisted columns")
    @input({"voyage": {},
            "timestamp": {},
            "lat": {"unit": "deg"},
            "lon": {"unit": "deg"},
            "speed": {},
            "course": {},
            "status": {"representation_type": int},
            "ship_type": {"representation_type": int}})
    @output({"voyage_id": {"representation_type": str, "description": "keyed pseudonym of the voyage"},
             "timestamp": {}, "lat": {}, "lon": {}, "speed": {}, "course": {}, "status": {}, "ship_type": {}},
            exclusive=True)
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        voyage = self.get_name("voyage")
        key = load_key(self.key_env, self.__class__.__name__)

        # each voyage is hashed once and joined back - there are far fewer voyages than fixes
        pseudonyms = df.lazyframe.select(voyage).unique().with_columns(
            pl.col(voyage).map_elements(lambda v: voyage_pseudonym(key, v), return_dtype=pl.String)
            .alias(self.get_name("voyage_id"))
        )
        df.lazyframe = df.lazyframe.join(pseudonyms, on=voyage, how="left", maintain_order="left").with_columns(
            # a code outside 0-99 is not a ship type
            pl.col(self.get_name("ship_type")).replace_strict(SHIP_TYPE_CATEGORIES, default=None,
                                                              return_dtype=pl.Int64)
        )
        return df


class SuppressVoyageEndpoints(PipelineElement):
    """
    Drop the fixes within `radius_km` of the first or last fix of their voyage.

    Where a voyage starts or ends - a berth, an anchorage, a fishing ground - identifies the
    vessel. Voyages left with fewer than `min_fixes` fixes are dropped entirely.

    Args:
        radius_km: Distance to a voyage's first or last fix within which fixes are dropped
        min_fixes: Minimum number of fixes a voyage must keep
    """

    def __init__(self, radius_km: float = 2.0, min_fixes: int = 2):
        self.radius_km = radius_km
        self.min_fixes = min_fixes

    @describe("Drop the fixes close to where a voyage starts or ends")
    @input({"group": {},
            "sort": {},
            "lat": {"unit": "deg"},
            "lon": {"unit": "deg"}})
    @output({"group": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        group = self.get_name("group")
        lat = self.get_name("lat")
        lon = self.get_name("lon")

        # polars_h3 takes column names only, so the voyage's ends go into temporary columns
        ends = ["_first_lat", "_first_lon", "_last_lat", "_last_lon"]
        lazyframe = df.lazyframe.sort(group, self.get_name("sort")).with_columns(
            pl.col(lat).first().over(group).alias("_first_lat"),
            pl.col(lon).first().over(group).alias("_first_lon"),
            pl.col(lat).last().over(group).alias("_last_lat"),
            pl.col(lon).last().over(group).alias("_last_lon"),
        )
        lazyframe = lazyframe.filter(
            (polars_h3.great_circle_distance(lat, lon, "_first_lat", "_first_lon", unit="km") >= self.radius_km)
            & (polars_h3.great_circle_distance(lat, lon, "_last_lat", "_last_lon", unit="km") >= self.radius_km)
        ).drop(ends)
        df.lazyframe = lazyframe.filter(pl.len().over(group) >= self.min_fixes)
        return df


class GeneralizeSpaceTime(PipelineElement):
    """
    Coarsen positions, time, speed and course.

    A position is replaced by the centre of its H3 cell at `h3_resolution`, which is added as
    column `cell`. The time is truncated to `time_bucket`, speed and course are rounded to
    `speed_step` and `course_step`.

    Args:
        h3_resolution: H3 resolution of the cells - 6 is about 5.6 km across (36 km²), each
            step finer or coarser divides or multiplies the area by 7
        time_bucket: Interval the time is truncated to, as a polars duration string, e.g. "10m"
        speed_step: Step speed is rounded to, in the speed's unit
        course_step: Step course is rounded to, in degrees
    """

    def __init__(self, h3_resolution: int = 6, time_bucket: str = "10m",
                 speed_step: float = 1.0, course_step: float = 10.0):
        self.h3_resolution = h3_resolution
        self.time_bucket = time_bucket
        self.speed_step = speed_step
        self.course_step = course_step

    @describe("Snap positions to H3 cells, bucket time, round speed and course")
    @input({"timestamp": {},
            "lat": {"unit": "deg"},
            "lon": {"unit": "deg"},
            "speed": {},
            "course": {"unit": "deg"}})
    @output({"cell": {"description": "H3 cell of the position"}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat = self.get_name("lat")
        lon = self.get_name("lon")
        speed = self.get_name("speed")
        course = self.get_name("course")
        cell = self.get_name("cell")

        lazyframe = df.lazyframe.with_columns(
            polars_h3.latlng_to_cell(lat, lon, self.h3_resolution).alias(cell)
        )
        df.lazyframe = lazyframe.with_columns(
            polars_h3.cell_to_lat(cell).alias(lat),
            polars_h3.cell_to_lng(cell).alias(lon),
            pl.col(self.get_name("timestamp")).dt.truncate(self.time_bucket),
            ((pl.col(speed) / self.speed_step).round() * self.speed_step).alias(speed),
            (((pl.col(course) / self.course_step).round() * self.course_step) % 360).alias(course),
        )
        return df


class KAnonymityFilter(PipelineElement):
    """
    Drop the fixes of every (cell, time) combination with fewer than `k` distinct ids.

    Use it after `GeneralizeSpaceTime`, so cells and times are coarse, and map `id` to the MMSI,
    so that k counts vessels - a vessel with several voyages in one cell would otherwise count
    several times. It protects single fixes only: a long voyage can still be linked to external
    data by its sequence of cells, which `SegmentVoyages` limits by keeping voyages short.

    Args:
        k: Minimum number of distinct ids (vessels) per (cell, time)
    """

    def __init__(self, k: int = 5):
        self.k = k

    @describe("Drop the fixes of (cell, time) combinations with fewer than k distinct ids")
    @input({"cell": {}, "timestamp": {}, "id": {}})
    @output({"id": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        cell_and_time = [self.get_name("cell"), self.get_name("timestamp")]
        df.lazyframe = df.lazyframe.filter(
            pl.col(self.get_name("id")).n_unique().over(cell_and_time) >= self.k
        )
        return df


class VoyageKAnonymity(AnonymizationStrategy):
    """
    Anonymize AIS data for public release with per-voyage pseudonyms and k-anonymity.

    Chains `SegmentVoyages`, `SuppressVoyageEndpoints`, `GeneralizeSpaceTime`, `KAnonymityFilter`
    (counting vessels) and `PseudonymizeVoyages` - the parameters are theirs, see there; `columns`,
    `name` and `base_dir` are those of `AnonymizationStrategy`.

    Example:

    ```python
    released = VoyageKAnonymity(key_env="AIS_PSEUDONYM_KEY", time_bucket="1h").transform(df=adf)
    ```
    """

    def __init__(self, *, key_env: str | None = None, max_gap_s: float = 3600.0, rest_speed: float = 0.5,
                 radius_km: float = 2.0, min_fixes: int = 2, h3_resolution: int = 6, time_bucket: str = "10m",
                 speed_step: float = 1.0, course_step: float = 10.0, k: int = 5,
                 columns: AISColumns = AISColumns(), name: str = "anonymize",
                 base_dir: str | Path = tempfile.gettempdir()):
        self.key_env = key_env
        self.max_gap_s = max_gap_s
        self.rest_speed = rest_speed
        self.radius_km = radius_km
        self.min_fixes = min_fixes
        self.h3_resolution = h3_resolution
        self.time_bucket = time_bucket
        self.speed_step = speed_step
        self.course_step = course_step
        self.k = k
        super().__init__(columns=columns, name=name, base_dir=base_dir)

    def steps(self, columns: AISColumns) -> list[Step]:
        c = columns
        motion = {"timestamp": c.timestamp, "lat": c.lat, "lon": c.lon, "speed": c.speed, "course": c.course}
        return [
            Step("segment voyages", SegmentVoyages(max_gap_s=self.max_gap_s, rest_speed=self.rest_speed),
                 {"mmsi": c.mmsi, "timestamp": c.timestamp, "speed": c.speed, "status": c.status}),
            Step("suppress voyage endpoints", SuppressVoyageEndpoints(radius_km=self.radius_km,
                                                                      min_fixes=self.min_fixes),
                 {"group": "voyage", "sort": c.timestamp, "lat": c.lat, "lon": c.lon}),
            Step("generalize space and time",
                 GeneralizeSpaceTime(h3_resolution=self.h3_resolution, time_bucket=self.time_bucket,
                                     speed_step=self.speed_step, course_step=self.course_step),
                 motion),
            Step("k-anonymity", KAnonymityFilter(k=self.k),
                 {"cell": "cell", "timestamp": c.timestamp, "id": c.mmsi}),
            Step("pseudonymize voyages", PseudonymizeVoyages(key_env=self.key_env),
                 {**motion, "voyage": "voyage", "status": c.status, "ship_type": c.ship_type}),
        ]

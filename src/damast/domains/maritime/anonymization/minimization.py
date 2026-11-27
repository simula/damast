"""
Strategy `DataMinimization`: keep fewer and coarser AIS data, but keep the MMSI.

Unlike `VoyageKAnonymity` and `VoyageRelocation` the result is not anonymous: the MMSI stays, so it is
for internal use, where vessels must be told apart, and not for a public release. The strategy
chains, in this order:

0. `OwnerRisk`, `DropHighOwnerRisk` - drop the vessels likely to identify a private person
   (unless `drop_high_owner_risk=False`)
1. `DropFixesAtRest` - drop the fixes at anchor or moored, since a berth identifies a vessel
2. `CoarsenPositionTime` - round positions to `decimals`, truncate time to `time_resolution`

Columns are not dropped: a pipeline step may only drop all columns it does not declare, see
`PseudonymizeVoyages`, so identifiers such as name and call sign are dropped after the pipeline
(as `damast convert --privacy minimize` does).
"""
import tempfile
from pathlib import Path

import polars as pl

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement
from damast.domains.maritime.anonymization.strategy import (
    AISColumns,
    AnonymizationStrategy,
    Step,
    owner_risk_steps,
)
from damast.domains.maritime.anonymization.voyage_k_anonymity import at_rest

__all__ = [
    "CoarsenPositionTime",
    "DataMinimization",
    "DropFixesAtRest",
]


class DropFixesAtRest(PipelineElement):
    """
    Drop the fixes at rest: at anchor, moored, or slower than `rest_speed`, see `SegmentVoyages`.

    Args:
        rest_speed: Speed over ground, in knots, below which a fix counts as at rest
    """

    def __init__(self, rest_speed: float = 0.5):
        self.rest_speed = rest_speed

    @describe("Drop the fixes at anchor, moored or slower than rest_speed")
    @input({"speed": {}, "status": {"representation_type": int}})
    @output({"speed": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        df.lazyframe = df.lazyframe.filter(
            ~at_rest(pl.col(self.get_name("status")), pl.col(self.get_name("speed")), self.rest_speed))
        return df


class CoarsenPositionTime(PipelineElement):
    """
    Round positions to `decimals` and truncate the time to `time_resolution`.

    4 decimals are about 11 m in latitude, AIS itself reports about 0.2 m.

    Args:
        decimals: Number of decimals positions are rounded to
        time_resolution: Interval the time is truncated to, as a polars duration string, e.g. "1m"
    """

    def __init__(self, decimals: int = 4, time_resolution: str = "1m"):
        self.decimals = decimals
        self.time_resolution = time_resolution

    @describe("Round positions and truncate time")
    @input({"timestamp": {}, "lat": {"unit": "deg"}, "lon": {"unit": "deg"}})
    @output({"timestamp": {}, "lat": {}, "lon": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat = self.get_name("lat")
        lon = self.get_name("lon")
        df.lazyframe = df.lazyframe.with_columns(
            pl.col(lat).round(self.decimals),
            pl.col(lon).round(self.decimals),
            pl.col(self.get_name("timestamp")).dt.truncate(self.time_resolution),
        )
        return df


class DataMinimization(AnonymizationStrategy):
    """
    Keep fewer and coarser AIS data, keeping the MMSI - not anonymous, see the module.

    Example:

    ```python
    minimized = DataMinimization(decimals=3, time_resolution="5m").transform(df=adf)
    ```

    Args:
        rest_speed: Speed, in knots, below which a fix counts as at rest
        decimals: Number of decimals positions are rounded to
        time_resolution: Interval the time is truncated to, as a polars duration string
        drop_high_owner_risk: Whether to drop the vessels of high `OwnerRisk`
        columns: Names of the AIS columns in the data
        name: Name of the pipeline, see `AnonymizationStrategy`
        base_dir: Base directory of the pipeline, see `AnonymizationStrategy`
    """

    def __init__(self, *, rest_speed: float = 0.5, decimals: int = 4,
                 time_resolution: str = "1m", drop_high_owner_risk: bool = True,
                 columns: AISColumns = AISColumns(), name: str = "minimize",
                 base_dir: str | Path = tempfile.gettempdir()):
        self.rest_speed = rest_speed
        self.decimals = decimals
        self.time_resolution = time_resolution
        self.drop_high_owner_risk = drop_high_owner_risk
        super().__init__(columns=columns, name=name, base_dir=base_dir)

    def steps(self, columns: AISColumns) -> list[Step]:
        owner_risk = owner_risk_steps(columns) if self.drop_high_owner_risk else []
        return owner_risk + [
            Step("drop fixes at rest", DropFixesAtRest(rest_speed=self.rest_speed),
                 {"speed": columns.speed, "status": columns.status}),
            Step("coarsen position and time",
                 CoarsenPositionTime(decimals=self.decimals, time_resolution=self.time_resolution),
                 {"timestamp": columns.timestamp, "lat": columns.lat, "lon": columns.lon}),
        ]

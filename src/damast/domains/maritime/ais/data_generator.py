import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from logging import INFO, Logger, getLogger
from random import choice, randint, random
from typing import Any, List

import vaex
from pyais.ais_types import AISType
import pandas as pd
from damast.domains.maritime.ais import vessel_types
from damast.domains.maritime.ais.navigational_status import (
    AISNavigationalStatus
)
from damast.domains.maritime.data_specification import (
    MMSI,
    ColumnName,
    CourseOverGround,
    SpeedOverGround
)

_log: Logger = getLogger(__name__)
_log.setLevel(INFO)

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
AIS_DATA_COLUMNS = [
    "mmsi",
    "lon",
    "lat",
    "date_time_utc",
    "sog",
    "cog",
    "true_heading",
    "nav_status",
    "rot",
    "message_nr",
    "source"
]

TRAJECTORY_MIN_SIZE = 10
TRAJECTORY_MAX_SIZE = 200


class AISTestData:
    dataframe: vaex.DataFrame = None

    def __init__(self,
                 number_of_trajectories: int,
                 min_length: int = TRAJECTORY_MIN_SIZE,
                 max_length: int = TRAJECTORY_MAX_SIZE):

        self.number_of_trajectories = number_of_trajectories
        self.min_length = min_length
        self.max_length = max_length

        self.dataframe: vaex.DataFrame = self._generate_data()

    @staticmethod
    def generate_trajectory(min_size: int, max_size: int) -> List[List[Any]]:
        mmsi = randint(2*MMSI.min_value, 2*MMSI.max_value)  # Create some invalid messages as well

        lat_start = (random() * 180.0) - 90.0
        lon_start = (random() * 360.0) - 180.0

        last_day = datetime.datetime(year=2022, month=12, day=31)
        start_time = randint(0, last_day.timestamp())

        sog_start = random() * SpeedOverGround.min_value / SpeedOverGround.max_value
        cog_start = random() * CourseOverGround.min_value / CourseOverGround.max_value
        heading_start = cog_start

        rot_start = 0.0

        trajectory_length = randint(min_size, max_size)
        trajectory = []
        for _ in range(0, trajectory_length):
            lat_start += random() * 0.1 - 0.05
            lon_start += random() * 0.1 - 0.05

            # message within 2 min
            start_time += random() * 120.0

            sog_start += random() * 4 - 2
            cog_start += random() * 1 - 0.5
            heading_start = cog_start + random() / 10.0

            nav_status = choice([
                AISNavigationalStatus.EngagedInFishing.value,
                AISNavigationalStatus.UnderWayUsingEngine.value,
                AISNavigationalStatus.AtAnchor.value
            ])

            msg_id = AISType.POS_CLASS_A2.value

            trajectory.append(
                [mmsi,
                 lon_start,
                 lat_start,
                 # 2022-09-01 12:55:35
                 datetime.datetime.fromtimestamp(start_time).strftime(DATETIME_FORMAT),
                 sog_start,
                 cog_start,
                 heading_start,
                 nav_status,
                 rot_start,
                 msg_id,
                 choice(["s", "g"])  # Create some ground data that should be removed in processing
                 ])

        return trajectory

    def _generate_data(self) -> vaex.DataFrame:
        df = None
        for i in range(0, self.number_of_trajectories):
            trajectory = AISTestData.generate_trajectory(min_size=self.min_length, max_size=self.max_length)
            t_df = pd.DataFrame(trajectory, columns=AIS_DATA_COLUMNS)
            if df is None:
                df = t_df
            else:
                df = pd.concat([df, t_df], axis=0, ignore_index=True)
        return vaex.from_pandas(df)

    def generate_vessel_type_data(self) -> vaex.DataFrame:
        """Generate a :class:`vaex.DataFrame` with data imitating vessel-type info

        Returns:
            A dataframe with an `ColumnName.MMSI` column and a `ColumnName.VESSEL_TYPE` column.
        """
        vessel_type_data = []
        for mmsi in self.dataframe[ColumnName.MMSI.lower()].unique():
            vessel_type_data.append([mmsi, vessel_types.Fishing.to_id()])
        df = pd.DataFrame(vessel_type_data, columns=[ColumnName.MMSI, ColumnName.VESSEL_TYPE])
        return vaex.from_pandas(df)

    def generate_fishing_vessel_type_data(self) -> vaex.DataFrame:
        """Generate a `vaex.Dataframe` with data imitating vessel info from Global Fishing Watch

        Returns:
            A dataframe with an `ColumnName.MMSI` (lower-cased) column and a `ColumnName.VESSEL_TYPE` column.
        """
        fishing_vessel_type_data = []
        for mmsi in self.dataframe[ColumnName.MMSI.lower()].unique():
            fishing_vessel_type_data.append([mmsi, choice([
                vessel_types.DriftingLonglines.to_id(),
                vessel_types.PoleAndLine.to_id(),
                vessel_types.PotsAndTraps.to_id()
            ])])

        df = pd.DataFrame(fishing_vessel_type_data, columns=[ColumnName.MMSI.lower(), ColumnName.VESSEL_TYPE_GFW])

        return vaex.from_pandas(df)

    def generate_anchorage_type_data(self) -> vaex.DataFrame:
        # Generate anchorages close to starting points of vessel trajectories
        mmsi = ColumnName.MMSI.lower()
        lat_vaex = self.dataframe.groupby(mmsi).agg({"lat": vaex.agg.first})
        lat_vaex.rename("lat", "latitude")
        lat_vaex["latitude"] *= 1.05
        lon_vaex = self.dataframe.groupby(mmsi).agg({"lon": vaex.agg.first})
        lon_vaex.rename("lon", "longitude")
        lon_vaex["longitude"] *= -0.95
        df = lat_vaex.join(lon_vaex, on=mmsi)
        return df


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description="A simple generator for ais data that can be used as input to "
                            + "data processing stage")

    parser.add_argument("-t", "--number-of-trajectories", default=100, type=int)
    parser.add_argument("-o", "--output", default="test-data.hdf5", type=str)

    args = parser.parse_args()

    ais_test_data = AISTestData(number_of_trajectories=args.number_of_trajectories)
    ais_test_data.dataframe.export_hdf5(args.output)
    _log.info("Written: {args.output}")

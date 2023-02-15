import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from logging import INFO, Logger, getLogger
from random import choice, randint, random
from typing import Any, List

import numpy as np
import pandas as pd
from pyais.ais_types import AISType

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
    dataframe: pd.DataFrame = None

    def __init__(self,
                 number_of_trajectories: int,
                 min_length: int = TRAJECTORY_MIN_SIZE,
                 max_length: int = TRAJECTORY_MAX_SIZE):

        self.number_of_trajectories = number_of_trajectories
        self.min_length = min_length
        self.max_length = max_length

        self.dataframe: pd.DataFrame = self._generate_data()

    @staticmethod
    def generate_trajectory(min_size: int, max_size: int) -> List[List[Any]]:
        mmsi = randint(MMSI.min_value, MMSI.max_value)

        lat_start = (random() * 180.0) - 90.0
        lon_start = (random() * 360.0) - 180.0

        last_day = datetime.datetime(year=2022, month=12, day=31)
        start_time = randint(0, last_day.timestamp())

        sog_start = random() * SpeedOverGround.min_value / SpeedOverGround.max_value
        cog_start = random() * CourseOverGround.min_value / CourseOverGround.max_value
        heading_start = cog_start

        rot_start = 0.0
        source = "s"

        trajectory_length = randint(min_size, max_size)
        trajectory = []
        for msg_idx in range(0, trajectory_length):
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
                 source
                 ])

        return trajectory

    def _generate_data(self) -> pd.DataFrame:
        df = None
        for i in range(0, self.number_of_trajectories):
            trajectory = AISTestData.generate_trajectory(min_size=self.min_length, max_size=self.max_length)
            t_df = pd.DataFrame(trajectory, columns=AIS_DATA_COLUMNS)
            if df is None:
                df = t_df
            else:
                df = pd.concat([df, t_df], axis=0, ignore_index=True)
        return df

    def generate_vessel_type_data(self):
        vessel_type_data = []
        for mmsi in np.unique(self.dataframe[ColumnName.MMSI.lower()]):
            vessel_type_data.append([mmsi, vessel_types.Fishing.to_id()])
        df = pd.DataFrame(vessel_type_data, columns=[ColumnName.MMSI, ColumnName.VESSEL_TYPE])
        return df

    def generate_fishing_vessel_type_data(self):
        fishing_vessel_type_data = []

        for mmsi in np.unique(self.dataframe[ColumnName.MMSI.lower()]):
            fishing_vessel_type_data.append([mmsi, choice([
                vessel_types.DriftingLonglines.to_id(),
                vessel_types.PoleAndLine.to_id(),
                vessel_types.PotsAndTraps.to_id()
            ])])

        df = pd.DataFrame(fishing_vessel_type_data, columns=[ColumnName.MMSI.lower(), ColumnName.VESSEL_TYPE_GFW])
        return df

    def generate_anchorage_type_data(self):
        # Generate anchorages close to starting points of vessel trajectories
        lat = self.dataframe.groupby(ColumnName.MMSI.lower())["lat"].first() * 1.05
        lat_df = lat.reset_index()
        lat_df.rename(
            columns={"lat": "latitude"},
            inplace=True
        )

        lon = self.dataframe.groupby(ColumnName.MMSI.lower())["lon"].first() * -0.95
        lon_df = lon.reset_index()
        lon_df.rename(
            columns={"lon": "longitude"},
            inplace=True
        )
        anchorages_df = pd.merge(lat_df, lon_df, on=[ColumnName.MMSI.lower()])
        return anchorages_df


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description="A simple generator for ais data that can be used as input to "
                                        + "data processing stage")

    parser.add_argument("-t", "--number-of-trajectories", default=100, type=int)
    parser.add_argument("-o", "--output", default="test-data.csv", type=str)

    args = parser.parse_args()

    ais_test_data = AISTestData(number_of_trajectories=args.number_of_trajectories)
    ais_test_data.dataframe.to_csv(str(args.output), sep=";", index=False)
    _log.info("Written: {args.output}")

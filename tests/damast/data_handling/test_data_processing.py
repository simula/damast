from pathlib import Path

from random import choice
import numpy as np
import pandas as pd
import pytest

from damast.domains.maritime.ais import vessel_types
from damast.domains.maritime.ais.data_generator import generate_test_data
from damast.domains.maritime.data_specification import ColumnName

from damast.data_handling.data_processing import process_data, cleanse_and_sanitise

import logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture()
def workdir(tmp_path):
    return tmp_path


@pytest.fixture(name="default_config")
def default_config(workdir):
    params = {
        "workdir": workdir,
        "outputs": {"processed":
            {
                "dir": "outputs/data_processing",
                "h5_key": "data"
            }
        },
        "month": 1,
        "inputs": {
            "vessel_types": '',
            "fishing_vessel_types": ''
        },
        "columns": {
            "useless": [],
            "unused": [],
            "constraints": {}
        }
    }
    return params


def test_data_processing(default_config, workdir):
    test_data = generate_test_data(200)

    vessel_type_data = []
    fishing_type_data = []

    for mmsi in np.unique(test_data[ColumnName.MMSI.lower()]):
        vessel_type_data.append([mmsi, vessel_types.Fishing.to_id()])
        fishing_type_data.append([mmsi, choice([
            vessel_types.DriftingLonglines.to_id(),
            vessel_types.PoleAndLine.to_id(),
            vessel_types.PotsAndTraps.to_id()
        ])])

    # Generate anchorages close to starting points of vessel trajectories
    lat = test_data.groupby(ColumnName.MMSI.lower())["lat"].first()*1.05
    lat_df = lat.reset_index()
    lat_df.rename(
        columns={"lat": "latitude"},
        inplace=True
    )

    lon = test_data.groupby(ColumnName.MMSI.lower())["lon"].first()*-0.95
    lon_df = lon.reset_index()
    lon_df.rename(
        columns={"lon": "longitude"},
        inplace=True
    )
    anchorages_df = pd.merge(lat_df, lon_df, on=[ColumnName.MMSI.lower()])

    vessel_type_df = pd.DataFrame(vessel_type_data, columns=[ColumnName.MMSI, ColumnName.VESSEL_TYPE])
    fishing_type_df = pd.DataFrame(fishing_type_data, columns=[ColumnName.MMSI.lower(), ColumnName.VESSEL_TYPE_GFW])

    vessel_type_csv = workdir / "vessel_types.csv"
    fishing_vessel_type_csv = workdir / "fishing_vessel_types.csv"
    anchorages_csv = workdir / "anchorages.csv"

    vessel_type_df.to_csv(vessel_type_csv, sep=";", index=False)
    fishing_type_df.to_csv(fishing_vessel_type_csv, sep=",", index=False)
    anchorages_df.to_csv(anchorages_csv, sep=",", index=False)

    params = default_config.copy()
    params["inputs"]["vessel_types"] = vessel_type_csv
    params["inputs"]["fishing_vessel_types"] = fishing_vessel_type_csv
    params["inputs"]["anchorages"] = anchorages_csv
    params["MessageTypePosition"] = [2]
    params["columns"]["useless"] = ["rot", "BaseDateTime", "source"]

    # Data processing currently expects the following columns
    test_data.rename(
        columns={
            "mmsi": ColumnName.MMSI,
            "lon": ColumnName.LONGITUDE,
            "lat": ColumnName.LATITUDE,
            "date_time_utc": "BaseDateTime",
            "sog": ColumnName.SPEED_OVER_GROUND,
            "cog": ColumnName.COURSE_OVER_GROUND,
            "true_heading": ColumnName.HEADING,
            "nav_status": ColumnName.STATUS,
            "message_nr": "MessageType",
        },
        inplace=True
    )

    df = cleanse_and_sanitise(params=params, df=test_data)
    process_data(params=params,
                 df=df,
                 workdir=workdir)

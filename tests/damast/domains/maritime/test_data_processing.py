import logging
import pathlib
from typing import Tuple

import pytest
import vaex

from damast.domains.maritime.ais.data_generator import AISTestData
from damast.domains.maritime.data_processing import (
    cleanse_and_sanitise,
    process_data
)
from damast.domains.maritime.data_specification import ColumnName

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
            "constraints": {"LAT": {
                "min": -90,
                "max": 90,
                "type": 'float32',
                "default": 0}
            }}
    }
    return params


@pytest.fixture()
def ais_test_data() -> AISTestData:
    return AISTestData(number_of_trajectories=20)


@pytest.fixture()
def vessel_types_data(ais_test_data: AISTestData,
                      workdir: pathlib.Path) -> Tuple[vaex.DataFrame, pathlib.Path]:
    df = ais_test_data.generate_vessel_type_data()
    hdf5_path = workdir / "vessel_types.hdf5"
    df.export_hdf5(hdf5_path)
    return df, hdf5_path


@pytest.fixture()
def fishing_vessel_types_data(ais_test_data: AISTestData,
                              workdir: pathlib.Path) -> Tuple[vaex.DataFrame, pathlib.Path]:
    df = ais_test_data.generate_fishing_vessel_type_data()
    hdf5_path = workdir / "fishing_vessel_types.hdf5"
    df.export_hdf5(hdf5_path)
    return df, hdf5_path


@pytest.fixture()
def anchorages_data(ais_test_data: AISTestData,
                    workdir: pathlib.Path) -> Tuple[vaex.DataFrame, pathlib.Path]:
    df = ais_test_data.generate_anchorage_type_data()

    anchorages_csv = workdir / "anchorages.hdf5"
    df.export_hdf5(anchorages_csv)

    return df, anchorages_csv


def test_data_processing_with_plain_vaex(workdir,
                                         ais_test_data,
                                         fishing_vessel_types_data,
                                         vessel_types_data,
                                         anchorages_data):
    _ = vaex.from_pandas(ais_test_data.dataframe)


def test_data_processing(default_config, workdir,
                         ais_test_data,
                         fishing_vessel_types_data,
                         vessel_types_data,
                         anchorages_data):
    params = default_config.copy()
    params["inputs"]["vessel_types"] = vessel_types_data[1]
    params["inputs"]["fishing_vessel_types"] = fishing_vessel_types_data[1]
    params["inputs"]["anchorages"] = anchorages_data[1]
    params["MessageTypePosition"] = [2]
    params["columns"]["useless"] = ["rot", "BaseDateTime", "source"]

    # Data processing currently expects the following columns
    columns = {
        "mmsi": ColumnName.MMSI,
        "lon": ColumnName.LONGITUDE,
        "lat": ColumnName.LATITUDE,
        "date_time_utc": "BaseDateTime",
        "sog": ColumnName.SPEED_OVER_GROUND,
        "cog": ColumnName.COURSE_OVER_GROUND,
        "true_heading": ColumnName.HEADING,
        "nav_status": ColumnName.STATUS,
        "message_nr": "MessageType",
    }
    for (old_name, new_name) in columns.items():
        ais_test_data.dataframe.rename(old_name, new_name)

    df = cleanse_and_sanitise(params=params, df=ais_test_data.dataframe)
    process_data(params=params,
                 df=df,
                 workdir=workdir)

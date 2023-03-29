import logging
import pathlib
from typing import Tuple

import numpy as np
import pytest
import vaex

import damast.core
from damast.core import units
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


def test_data_processing(workdir,
                         ais_test_data,
                         fishing_vessel_types_data,
                         vessel_types_data,
                         anchorages_data):

    # Data processing currently expects the following columns
    columns = {
        "mmsi": ColumnName.MMSI,
        "lon": ColumnName.LONGITUDE,
        "lat": ColumnName.LATITUDE,
        "date_time_utc": ColumnName.DATE_TIME_UTC,
        "sog": ColumnName.SPEED_OVER_GROUND,
        "cog": ColumnName.COURSE_OVER_GROUND,
        "true_heading": ColumnName.HEADING,
        "nav_status": ColumnName.STATUS,
        "rot": "rot",
        "message_nr": ColumnName.MESSAGE_TYPE,
        "source": "source"
    }
    for (old_name, new_name) in columns.items():
        ais_test_data.dataframe.rename(old_name, new_name)

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.units.deg,
                                               representation_type=np.float64),
                 damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.units.deg,
                                               representation_type=np.float64),
                 damast.core.DataSpecification(ColumnName.DATE_TIME_UTC, representation_type=str),
                 damast.core.DataSpecification(ColumnName.SPEED_OVER_GROUND, representation_type=float),
                 damast.core.DataSpecification(ColumnName.COURSE_OVER_GROUND, representation_type=float),
                 damast.core.DataSpecification(ColumnName.HEADING, representation_type=float),
                 damast.core.DataSpecification(ColumnName.STATUS, representation_type=int),
                 damast.core.DataSpecification("rot", representation_type=float),
                 damast.core.DataSpecification("MessageType", representation_type=int),
                 damast.core.DataSpecification("source", representation_type=str),
                 ])

    adf = damast.core.AnnotatedDataFrame(ais_test_data.dataframe, metadata)
    print(adf._dataframe)
    adf = cleanse_and_sanitise(df=adf,
                              useless_columns=[ColumnName.DATE_TIME_UTC, "rot", "source"],
                              message_type_position=[2],
                              columns_default_values={ ColumnName.MMSI: 0, 
                                                       ColumnName.SPEED_OVER_GROUND: 1023,
                                                       ColumnName.COURSE_OVER_GROUND: 3600,

                                                       },
                              columns_compress_types={
                                                       ColumnName.SPEED_OVER_GROUND: "int16",
                                                       ColumnName.COURSE_OVER_GROUND: "int16",
                                                       },
                              workdir=workdir)
 
    adf = process_data(df=adf,
                 workdir=workdir,
                 vessel_type_hdf5=vessel_types_data[1],
                 fishing_vessel_type_hdf5=fishing_vessel_types_data[1],
                 anchorages_hdf5=anchorages_data[1])

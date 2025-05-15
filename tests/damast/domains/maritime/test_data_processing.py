import logging
import pathlib
from typing import Tuple

import numpy as np
import polars
import polars as pl
import pytest

import damast.core
from damast.core import units
from damast.core.types import DataFrame, XDataFrame
from damast.domains.maritime.ais.data_generator import AISTestData
from damast.domains.maritime.data_processing import CleanseAndSanitise, DataProcessing
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
                      workdir: pathlib.Path) -> Tuple[DataFrame, pathlib.Path]:
    df = ais_test_data.generate_vessel_type_data()
    hdf5_path = workdir / "vessel_types.hdf5"
    XDataFrame.export_hdf5(df, hdf5_path)
    return df, hdf5_path


@pytest.fixture()
def fishing_vessel_types_data(ais_test_data: AISTestData,
                              workdir: pathlib.Path) -> Tuple[DataFrame, pathlib.Path]:
    df = ais_test_data.generate_fishing_vessel_type_data()
    hdf5_path = workdir / "fishing_vessel_types.hdf5"
    XDataFrame.export_hdf5(df, hdf5_path)
    return df, hdf5_path


@pytest.fixture()
def anchorages_data(ais_test_data: AISTestData,
                    workdir: pathlib.Path) -> Tuple[DataFrame, pathlib.Path]:
    df = ais_test_data.generate_anchorage_type_data()

    anchorages_csv = workdir / "anchorages.hdf5"
    XDataFrame.export_hdf5(df, anchorages_csv)

    return df, anchorages_csv


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

    ais_test_data.dataframe = ais_test_data.dataframe.rename(columns)

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.units.deg,
                                               representation_type=float),
                 damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.units.deg,
                                               representation_type=float),
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
    pipeline = CleanseAndSanitise(message_types=[2],
                                  columns_default_values={},
                                  columns_compress_types={
        ColumnName.SPEED_OVER_GROUND: "int16",
        ColumnName.COURSE_OVER_GROUND: "int16",
    },
        workdir=workdir)


    adf_preprocess = pipeline.transform(adf)
    column_names = adf_preprocess.dataframe.column_names

    df = adf_preprocess.collect()
    assert df.filter(pl.col("source") != "s").count()[0,0] == 0

    assert df.select(ColumnName.DATE_TIME_UTC).null_count()[0,0] == 0
    assert df.filter(pl.col(ColumnName.MESSAGE_TYPE) != 2).count()[0,0] == 0

    assert ColumnName.TIMESTAMP in column_names
    assert ColumnName.SPEED_OVER_GROUND + "_int16" in column_names
    assert df.compat.dtype(ColumnName.SPEED_OVER_GROUND+"_int16") == pl.Int16
    assert ColumnName.COURSE_OVER_GROUND + "_int16" in column_names
    assert df.compat.dtype(ColumnName.COURSE_OVER_GROUND+"_int16") == pl.Int16

    pipeline2 = DataProcessing(workdir=workdir,
                               vessel_type_hdf5=vessel_types_data[1],
                               fishing_vessel_type_hdf5=fishing_vessel_types_data[1],
                               anchorages_hdf5=anchorages_data[1])

    adf_processed = pipeline2.transform(adf_preprocess)
    processed_column_names = adf_processed.dataframe.column_names

    assert ColumnName.VESSEL_TYPE in processed_column_names
    assert adf_processed[ColumnName.VESSEL_TYPE].collect().null_count()[0,0] == 0
    assert ColumnName.FISHING_TYPE in processed_column_names
    assert adf_processed[ColumnName.FISHING_TYPE].collect().null_count()[0,0] == 0
    assert adf_processed[ColumnName.FISHING_TYPE].collect().min()[0,0] >= -1

    assert ColumnName.DISTANCE_CLOSEST_ANCHORAGE in processed_column_names
    assert ColumnName.HISTORIC_SIZE in processed_column_names
    assert ColumnName.HISTORIC_SIZE_REVERSE in processed_column_names

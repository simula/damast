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
    CleanseAndSanitise,
    DataProcessing
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
    assert (adf_preprocess.dataframe["source"].evaluate().to_numpy(zero_copy_only=False) == "s").all()
    assert adf_preprocess.dataframe[ColumnName.DATE_TIME_UTC].countmissing() == 0
    assert (adf_preprocess.dataframe[ColumnName.MESSAGE_TYPE].evaluate() == 2).all()
    assert ColumnName.TIMESTAMP in column_names
    assert ColumnName.SPEED_OVER_GROUND + "_int16" in column_names
    assert adf_preprocess[ColumnName.SPEED_OVER_GROUND+"_int16"].dtype == np.int16
    assert ColumnName.COURSE_OVER_GROUND + "_int16" in column_names
    assert adf_preprocess[ColumnName.COURSE_OVER_GROUND+"_int16"].dtype == np.int16

    pipeline2 = DataProcessing(workdir=workdir,
                               vessel_type_hdf5=vessel_types_data[1],
                               fishing_vessel_type_hdf5=fishing_vessel_types_data[1],
                               anchorages_hdf5=anchorages_data[1])
    adf_processed = pipeline2.transform(adf_preprocess)
    processed_column_names = adf_processed.dataframe.column_names

    assert ColumnName.VESSEL_TYPE in processed_column_names
    assert adf_processed.dataframe[ColumnName.VESSEL_TYPE].countmissing() == 0
    assert ColumnName.FISHING_TYPE in processed_column_names
    assert adf_processed.dataframe[ColumnName.FISHING_TYPE].countmissing() == 0
    assert adf_processed.dataframe[ColumnName.FISHING_TYPE].min() >= -1
    assert ColumnName.DISTANCE_CLOSEST_ANCHORAGE in processed_column_names
    assert ColumnName.HISTORIC_SIZE in processed_column_names
    assert ColumnName.HISTORIC_SIZE_REVERSE in processed_column_names

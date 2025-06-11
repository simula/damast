import datetime as dt
from pathlib import Path

import numpy as np
import pandas
import polars
import polars as pl
import polars.testing
import pytest
from astropy import units

import damast.core
from damast.data_handling.transformers.augmenters import (
    AddLocalIndex,
    AddUndefinedValue,
    )
from damast.domains.maritime.ais import AISNavigationalStatus
from damast.domains.maritime.ais.vessel_types import VesselType
from damast.domains.maritime.data_specification import ColumnName
from damast.domains.maritime.math import great_circle_distance
from damast.domains.maritime.transformers import (
    AddMissingAISStatus,
    AddVesselType,
    ComputeClosestAnchorage,
    DeltaDistance,
    )


def test_add_missing_ais_status(tmp_path):
    """
    Test if we can replace NaNs with empty rows which in turn can be filled with the appropriate AIS status
    """
    # Create dataframe with NaNs
    mmsi_a = 400000000
    timestamp = dt.datetime.now(dt.timezone.utc)
    status = AISNavigationalStatus.EngagedInFishing
    data = [
        [mmsi_a, timestamp, status],
        [mmsi_a, timestamp + dt.timedelta(seconds=1), status],
        [mmsi_a, timestamp + dt.timedelta(seconds=2), np.nan],
        [mmsi_a, timestamp + dt.timedelta(seconds=3), AISNavigationalStatus.Undefined],
        [mmsi_a, timestamp + dt.timedelta(seconds=4), AISNavigationalStatus.Power_DrivenVesselTowingAstern]
    ]
    mmsi_b = 500000000
    mmsi_c = 600000000
    for i in range(1, 51):
        data.append([mmsi_b, timestamp + dt.timedelta(seconds=1), np.nan])
        data.append([mmsi_c, timestamp + dt.timedelta(seconds=5 * i), np.nan])
    df_pd = pandas.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP, ColumnName.STATUS])
    df = polars.from_pandas(df_pd).with_columns(
            pl.col(ColumnName.STATUS).cast(int).alias(ColumnName.STATUS)
         )

    assert df.select("Status").null_count()[0,0] > 0
    assert len(df.filter(pl.col("Status") == np.nan)) == 0

    # Create annotated dataframe
    adf = damast.core.AnnotatedDataFrame(
            df,
            damast.core.MetaData(columns=[damast.core.DataSpecification("Status", representation_type=int)])
          )

    # Create pipeline
    pipeline = damast.core.DataProcessingPipeline(name="AddMissingAISStatus",
                                                  base_dir=tmp_path)
    pipeline.add("Add missing AIS status", AddMissingAISStatus(),
                 name_mappings={"x": "Status"})

    # Run pipeline
    new_df = pipeline.transform(adf)

    assert new_df.select("Status").null_count().collect()[0,0] == 0
    assert len(new_df.filter(pl.col("Status") == np.nan).collect()) == 0


def test_delta_column(tmp_path):
    mmsi_a = 400000000
    mmsi_b = 500000000
    timestamp = dt.datetime.now(dt.timezone.utc)

    lat = 0.0
    lon = 0.0

    data = [[mmsi_a, timestamp + dt.timedelta(seconds=100), lat, lon]]
    for i in range(1, 90):
        data.append([mmsi_a, timestamp + dt.timedelta(seconds=i * 1), lat + i * 1.0, lon + i * 1.0])
        data.append([mmsi_b, timestamp + dt.timedelta(seconds=i * 1), lat - i * 0.5, lon - i * 0.5])

    df_pd = pandas.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP,
                             ColumnName.LATITUDE, ColumnName.LONGITUDE])
    df = polars.from_pandas(df_pd)
    df = df.with_columns(
        pl.col(ColumnName.TIMESTAMP).dt.timestamp("ns").alias(ColumnName.TIMESTAMP)
    )

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.TIMESTAMP, representation_type=pl.datatypes.Datetime('ms')),
                 damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.deg),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.deg)])
    adf = damast.core.AnnotatedDataFrame(df, metadata)

    # Create pipeline
    pipeline = damast.core.DataProcessingPipeline(name="DeltaDistance",
                                                  base_dir=tmp_path)

    pipeline.add("Great circle distance", DeltaDistance(True, True),
                 name_mappings={"group": ColumnName.MMSI,
                                "sort": ColumnName.TIMESTAMP,
                                "x": ColumnName.LATITUDE,
                                "y": ColumnName.LONGITUDE,
                                "out": ColumnName.DELTA_DISTANCE})
    new_adf = pipeline.transform(adf)

    assert ColumnName.DELTA_DISTANCE in new_adf.column_names

    df_sorted = df.sort(ColumnName.TIMESTAMP)
    df_grouped = df_sorted.group_by(by=ColumnName.MMSI)

    for mmsi, data in df_grouped:
        # group specific distances in the result
        distances = new_adf._dataframe.filter(pl.col(ColumnName.MMSI) == mmsi[0]).select(ColumnName.DELTA_DISTANCE).collect().to_numpy()

        # original data in df, so check against
        expected_dataframe = data.select(
            pl.col("LAT").shift(1).alias("LAT_prev"),
            pl.col("LAT").alias("LAT"),
            pl.col("LON").shift(1).alias("LON_prev"),
            pl.col("LON").alias("LON"),
        ).drop_nans().drop_nulls()

        lat = expected_dataframe.select(pl.col("LAT")).to_numpy()
        lat_prev = expected_dataframe.select(pl.col("LAT_prev")).to_numpy()
        lon = expected_dataframe.select(pl.col("LON")).to_numpy()
        lon_prev = expected_dataframe.select(pl.col("LON_prev")).to_numpy()

        pandas_distances = great_circle_distance(lat, lon, lat_prev, lon_prev)

        assert np.allclose(pandas_distances, distances)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("vessel_file_mode", ["polars"]) #, "file"])
@pytest.mark.parametrize("right_on", [ColumnName.VESSEL_TYPE, "test_column"])
def test_add_vessel_type(tmp_path, vessel_file_mode: str, inplace: bool,
                         right_on: str):

    # Create vessel-type dataframe
    vessel_type_data = [
        [200000210, "unspecified"],
        [200008591, "fishing"],
        [200014514, "fishing"],
        [200016821, "fishing"],
        [200016826, "fishing"],
        [200024943, "cargo"],
        [200041743, "cargo"]
    ]
    columns = [ColumnName.MMSI, ColumnName.VESSEL_TYPE]
    pd_vessel_types = pandas.DataFrame(vessel_type_data, columns=columns)

    df_vessel_types = polars.from_pandas(pd_vessel_types)
    df_vessel_types = df_vessel_types.with_columns(
        pl.col(ColumnName.VESSEL_TYPE).replace_strict(VesselType.get_mapping()).alias(f"{ColumnName.VESSEL_TYPE}_as_int")
    )

    if vessel_file_mode == "polars":
        vessel_data = df_vessel_types
    else:
        vessel_data = Path(tmp_path) / "vessel_data_types.h5"
        df_vessel_types.export_hdf5(vessel_data)

    # Create input dataframe
    input_data = [
        [200014514, 12],
        [200000210, 10],
        [200008591, 11],
        [200016821, 13],
        [200016826, 14],
        [200024943, 15],
        [200041743, 16],
        [100000000, 17],
        [200024943, 11],
        [100000000, 12],
    ]
    df_pd = pandas.DataFrame(input_data, columns=[ColumnName.MMSI, ColumnName.SPEED_OVER_GROUND])
    df = polars.from_pandas(df_pd)

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.SPEED_OVER_GROUND, unit=units.m/units.s)])
    adf = damast.core.AnnotatedDataFrame(df, metadata)

    transformer = AddVesselType(right_on=ColumnName.MMSI,
                                dataset_column=ColumnName.VESSEL_TYPE,
                                dataset=vessel_data)

    pipeline = damast.core.DataProcessingPipeline(name="Add vessel type",
                                                  base_dir=tmp_path,
                                                  inplace_transformation=inplace)

    pipeline.add("Add vessel-type", transformer,
                 name_mappings={"x": ColumnName.MMSI,
                                "out": ColumnName.VESSEL_TYPE})

    copy_adf = damast.core.AnnotatedDataFrame(adf.dataframe,
                                              damast.core.MetaData(columns=adf.metadata.columns.copy()))
    new_adf = pipeline.transform(adf)
    if inplace:
        assert len(new_adf.column_names) == len(adf.column_names)
    else:
        assert len(new_adf.column_names) == len(adf.column_names) + 1

    missing_data = []

    for mmsi in new_adf[ColumnName.MMSI].unique().collect()[:,0]:
        entries_per_mmsi = new_adf.filter(pl.col(ColumnName.MMSI) == mmsi)
        vessel_types = entries_per_mmsi.select(ColumnName.VESSEL_TYPE).collect()[:,0]
        exact_vessel_types = df_vessel_types.filter(pl.col(ColumnName.MMSI) == mmsi).select(f"{ColumnName.VESSEL_TYPE}_as_int")[:,0]

        if len(exact_vessel_types) == 0:
            # If no entry found in original input, this entry should be masked
            missing_data.append(mmsi)
        elif len(exact_vessel_types) == 1:
            # If one entry found in lookup dataframe, all entries should match this
            assert vessel_types.min() == vessel_types.max()
            assert vessel_types.min() == exact_vessel_types[0]
        else:
            raise RuntimeError("Input vessel types have more than one entry for a single vessel")

    # Test replacing missing values with vessel type
    pipeline.add("Replace missing", AddUndefinedValue(VesselType["unspecified"]), name_mappings={"x": ColumnName.VESSEL_TYPE})
    fixed_adf = pipeline.transform(copy_adf)

    for mmsi in missing_data:
        entries_per_mmsi = fixed_adf.filter(pl.col(ColumnName.MMSI) == mmsi)
        vessel_types = entries_per_mmsi.select(ColumnName.VESSEL_TYPE).collect()[:,0]

        assert vessel_types.min() == vessel_types.max()
        assert vessel_types.min() == VesselType["unspecified"]


def test_add_distance_closest_anchorage(tmp_path):
    columns = ["s2id", "latitude", "longitude", "label", "sublabel", "iso3"]
    anchorage_data = [
        ["356ec741", 34.839059469352883, 128.42069569869318, "TONGYEONG", None, "KOR"],
        ["3574e907", 34.691433022457183, 125.4429067750192, "HEUKSANDO", None, "KOR"],
        ["356f2c7b", 35.137774967951927, 128.6030742948769, "MASAN", None, "KOR"],
        ["356f2af7", 35.080054265204275, 128.61175951519579, "NANPO- RI", None, "KOR"]
    ]

    dataset_pd = pandas.DataFrame(anchorage_data, columns=columns)
    dataset = polars.from_pandas(dataset_pd)
    data = [
        [34.839059469352883, 128.42069569869318],
        [30.0, 125.0]
    ]
    df_pd = pandas.DataFrame(data, columns=[ColumnName.LATITUDE, ColumnName.LONGITUDE])
    df = polars.from_pandas(df_pd)
    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.deg,
                                               representation_type=float),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.deg,
                                               representation_type=float)])
    adf = damast.core.AnnotatedDataFrame(df, metadata)

    transformer = ComputeClosestAnchorage(dataset, [columns[1], columns[2]])
    pipeline = damast.core.DataProcessingPipeline(name="Compute closest anchorage",
                                                  base_dir=tmp_path)

    pipeline.add("Add distance to anchorage", transformer,
                 name_mappings={"x": ColumnName.LATITUDE,
                                "y": ColumnName.LONGITUDE,
                                "distance": ColumnName.DISTANCE_CLOSEST_ANCHORAGE})
    new_adf = pipeline.transform(adf)
    closest_anchorages = new_adf[ColumnName.DISTANCE_CLOSEST_ANCHORAGE].collect()

    distances = np.zeros(len(anchorage_data))
    for idx in range(len(new_adf.collect())):
        for i, anchorage in enumerate(anchorage_data):
            lat, lon = anchorage[1:3]
            lat_prev, lon_prev = data[idx]
            distances[i] = great_circle_distance(lat, lon, lat_prev, lon_prev)

        assert np.isclose(np.min(distances), closest_anchorages[idx])


def test_message_index(tmp_path):
    mmsi_a = 400000000
    mmsi_b = 500000000
    mmsi_c = 600000000
    timestamp = dt.datetime.now(dt.timezone.utc)

    lat = 0.0
    lon = 0.0

    data = []
    num_messages_A = 90
    num_messages_B = num_messages_A - 2
    num_messages_C = 1

    # Insert last message first to validate time-based sorting
    data.append([mmsi_a, timestamp + dt.timedelta(seconds=num_messages_A), lat, lon, num_messages_A-1, 0])
    # Next insert first message
    data.append([mmsi_a, timestamp - dt.timedelta(seconds=1), lat, lon, 0, num_messages_A-1])
    data.append([mmsi_c, timestamp - dt.timedelta(seconds=1), lat, lon, 0, 0])

    for i in range(num_messages_B):
        data.append([mmsi_a, timestamp + dt.timedelta(seconds=i * 1), lat + i * 1.0, lon + i * 1.0,
                     i+1, num_messages_A-(i+2)])
        data.append([mmsi_b, timestamp + dt.timedelta(seconds=i * 1), lat - i * 0.5, lon - i * 0.5,
                     i, num_messages_B-(i+1)])
    df_pd = pandas.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP,
                             ColumnName.LATITUDE, ColumnName.LONGITUDE, "REF INDEX", "INVERSE REF"])

    df = polars.from_pandas(df_pd)
    df = df.with_columns(
            pl.col(ColumnName.TIMESTAMP).dt.timestamp("ns").alias(ColumnName.TIMESTAMP)
        )

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.TIMESTAMP, representation_type=pl.datatypes.Datetime("ns")),
                 damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.deg, representation_type=pl.Float64),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.deg, representation_type=pl.Float64)])
    adf = damast.core.AnnotatedDataFrame(df, metadata)

    # Create pipeline
    pipeline = damast.core.DataProcessingPipeline(name="Compute message index",
                                                  base_dir=tmp_path)

    pipeline.add("Compute local message index",  AddLocalIndex(),
                 name_mappings={"group": ColumnName.MMSI,
                                "sort": ColumnName.TIMESTAMP,
                                "local_index": ColumnName.HISTORIC_SIZE,
                                "reverse_{{local_index}}": ColumnName.HISTORIC_SIZE_REVERSE})
    new_adf = pipeline.transform(adf)

    polars.testing.assert_series_equal(
        left=new_adf["REF INDEX"].collect()[:,0],
        right=new_adf[ColumnName.HISTORIC_SIZE].collect()[:,0],
        check_names=False
   )

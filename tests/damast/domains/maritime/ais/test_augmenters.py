import datetime as dt
from pathlib import Path

import numpy as np
import pandas
import pytest
import vaex
from astropy import units

import damast.core
from damast.data_handling.transformers.augmenters import (
    AddLocalIndex,
    AddUndefinedValue
)
from damast.domains.maritime.ais import AISNavigationalStatus
from damast.domains.maritime.ais.vessel_types import VesselType
from damast.domains.maritime.data_specification import ColumnName
from damast.domains.maritime.math import great_circle_distance
from damast.domains.maritime.transformers import (
    AddMissingAISStatus,
    AddVesselType,
    ComputeClosestAnchorage,
    DeltaDistance
)


def test_add_missing_ais_status(tmp_path):
    """
    Test if we can replace NaNs with empty rows which in turn can be filled with the appropriate AIS status
    """
    # Create dataframe with NaNs
    mmsi_a = 400000000
    timestamp = dt.datetime.utcnow()
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
    df = vaex.from_pandas(df_pd)

    # Replace nans with mask
    assert df["Status"].countnan() > 0
    damast.core.replace_na(df, "int", ["Status"])

    assert df["Status"].countnan() == 0
    assert df["Status"].countna() > 0
    assert df["Status"].countmissing() > 0
    assert df["Status"].is_masked

    # Create annotated dataframe
    adf = damast.core.AnnotatedDataFrame(df, damast.core.MetaData(
        columns=[damast.core.DataSpecification("Status", representation_type=int)]))
    assert adf._dataframe["Status"].is_masked

    # Create pipeline
    pipeline = damast.core.DataProcessingPipeline(name="AddMissingAISStatus",
                                                  base_dir=tmp_path)
    pipeline.add("Add missing AIS status", AddMissingAISStatus(),
                 name_mappings={"x": "Status"})

    # Run pipeline
    new_df = pipeline.transform(adf)
    assert new_df["Status"].countna() == 0
    assert new_df["Status"].countmissing() == 0
    assert new_df["Status"].countnan() == 0


def test_delta_column(tmp_path):
    mmsi_a = 400000000
    mmsi_b = 500000000
    timestamp = dt.datetime.utcnow()

    lat = 0.0
    lon = 0.0

    data = []
    data.append([mmsi_a, timestamp + dt.timedelta(seconds=100), lat, lon])
    for i in range(1, 90):
        data.append([mmsi_a, timestamp + dt.timedelta(seconds=i * 1), lat + i * 1.0, lon + i * 1.0])
        data.append([mmsi_b, timestamp + dt.timedelta(seconds=i * 1), lat - i * 0.5, lon - i * 0.5])

    df_pd = pandas.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP,
                             ColumnName.LATITUDE, ColumnName.LONGITUDE])
    df = vaex.from_pandas(df_pd)

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.TIMESTAMP, representation_type="datetime64[ns]"),
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

    assert ColumnName.DELTA_DISTANCE in new_adf._dataframe.column_names
    pd_sorted = df_pd.sort_values(ColumnName.TIMESTAMP)
    df_grouped = pd_sorted.groupby(by=ColumnName.MMSI)

    vaex_groups = new_adf._dataframe.groupby(by=ColumnName.MMSI)
    for mmsi, global_indices in df_grouped.groups.items():
        vg_unsorted = vaex_groups.get_group(mmsi)
        vg = vg_unsorted.sort(ColumnName.TIMESTAMP)
        distances = vg[ColumnName.DELTA_DISTANCE].evaluate()
        lat = np.ma.masked_invalid(df_pd["LAT"][global_indices].shift(1).array)
        lat_prev = np.ma.masked_invalid(df_pd["LAT"][global_indices].array)
        lon = np.ma.masked_invalid(df_pd["LON"][global_indices].shift(1).array)
        lon_prev = np.ma.masked_invalid(df_pd["LON"][global_indices].array)
        pandas_distances = great_circle_distance(lat, lon, lat_prev, lon_prev)
        assert np.allclose(pandas_distances, distances)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("vessel_file_mode", ["vaex", "file"])
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
    df_vessel_types = vaex.from_pandas(pd_vessel_types)
    df_vessel_types[f"{ColumnName.VESSEL_TYPE}_as_int"] = \
        df_vessel_types[ColumnName.VESSEL_TYPE].map(mapper=VesselType.get_mapping())
    if vessel_file_mode == "vaex":
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
    df = vaex.from_pandas(df_pd)
    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.SPEED_OVER_GROUND, unit=units.m/units.s)])
    adf = damast.core.AnnotatedDataFrame(df, metadata)

    transformer = AddVesselType(right_on=ColumnName.MMSI,
                                dataset_col=ColumnName.VESSEL_TYPE,
                                dataset=vessel_data)

    pipeline = damast.core.DataProcessingPipeline(name="Add vessel type",
                                                  base_dir=tmp_path,
                                                  inplace_transformation=inplace)

    pipeline.add("Add vessel-type", transformer,
                 name_mappings={"x": ColumnName.MMSI,
                                "out": ColumnName.VESSEL_TYPE})
    copy_adf = damast.core.AnnotatedDataFrame(adf.dataframe.copy(),
                                              damast.core.MetaData(columns=adf.metadata.columns.copy()))
    new_adf = pipeline.transform(adf)
    if inplace:
        assert len(new_adf.get_column_names()) == len(adf.get_column_names())
    else:
        assert len(new_adf.get_column_names()) == len(adf.get_column_names()) + 1

    missing_data = []

    for mmsi in new_adf._dataframe[ColumnName.MMSI].unique():
        entries_per_mmsi = new_adf[new_adf[ColumnName.MMSI] == mmsi]
        vessel_types = entries_per_mmsi[ColumnName.VESSEL_TYPE].evaluate()
        exact_vessel_types = df_vessel_types[df_vessel_types[ColumnName.MMSI] == mmsi][f"{ColumnName.VESSEL_TYPE}_as_int"].evaluate()
        if len(exact_vessel_types) == 0:
            # If no entry found in original input, this entry should be masked
            missing_data.append(mmsi)
            assert vessel_types.mask.all()
        elif len(exact_vessel_types) == 1:
            # If one entry found in lookup dataframe, all entries should match this
            assert (vessel_types == exact_vessel_types).all()
        else:
            raise RuntimeError("Input vessel types have more than one entry for a single vessel")

    # Test replacing missing values with vessel type
    pipeline.add("Replace missing", AddUndefinedValue(VesselType["unspecified"]), name_mappings={"x": ColumnName.VESSEL_TYPE})
    fixed_adf = pipeline.transform(copy_adf)

    for mmsi in missing_data:
        entries_per_mmsi = fixed_adf[fixed_adf[ColumnName.MMSI] == mmsi]
        vessel_types = entries_per_mmsi[ColumnName.VESSEL_TYPE].evaluate()
        assert (vessel_types == VesselType["unspecified"]).all()


def test_add_distance_closest_anchorage(tmp_path):
    columns = ["s2id", "latitude", "longitude", "label", "sublabel", "iso3"]
    anchorage_data = [
        ["356ec741", 34.839059469352883, 128.42069569869318, "TONGYEONG", None, "KOR"],
        ["3574e907", 34.691433022457183, 125.4429067750192, "HEUKSANDO", None, "KOR"],
        ["356f2c7b", 35.137774967951927, 128.6030742948769, "MASAN", None, "KOR"],
        ["356f2af7", 35.080054265204275, 128.61175951519579, "NANPO- RI", None, "KOR"]
    ]

    dataset_pd = pandas.DataFrame(anchorage_data, columns=columns)
    dataset = vaex.from_pandas(dataset_pd)
    data = [
        [34.839059469352883, 128.42069569869318],
        [30.0, 125.0]
    ]
    df_pd = pandas.DataFrame(data, columns=[ColumnName.LATITUDE, ColumnName.LONGITUDE])
    df = vaex.from_pandas(df_pd)
    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.deg,
                                               representation_type=np.float64),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.deg,
                                               representation_type=np.float64)])
    adf = damast.core.AnnotatedDataFrame(df, metadata)

    transformer = ComputeClosestAnchorage(dataset, [columns[1], columns[2]])
    pipeline = damast.core.DataProcessingPipeline(name="Compute closest anchorage",
                                                  base_dir=tmp_path)

    pipeline.add("Add distance to achorage", transformer,
                 name_mappings={"x": ColumnName.LATITUDE,
                                "y": ColumnName.LONGITUDE,
                                "distance": ColumnName.DISTANCE_CLOSEST_ANCHORAGE})
    new_adf = pipeline.transform(adf)
    closest_anchorages = new_adf._dataframe[ColumnName.DISTANCE_CLOSEST_ANCHORAGE].evaluate()
    distances = np.zeros(len(anchorage_data))
    for idx in range(len(new_adf._dataframe)):
        for i, anchorage in enumerate(anchorage_data):
            pos = anchorage[1:3]
            distances[i] = great_circle_distance(pos[0], pos[1], data[idx][0], data[idx][1])
        assert np.isclose(np.min(distances), closest_anchorages[idx])


def test_message_index(tmp_path):
    mmsi_a = 400000000
    mmsi_b = 500000000
    timestamp = dt.datetime.utcnow()

    lat = 0.0
    lon = 0.0

    data = []
    num_messages_A = 90
    num_messages_B = num_messages_A - 2

    # Insert last message first to validate time-based sorting
    data.append([mmsi_a, timestamp + dt.timedelta(seconds=num_messages_A), lat, lon, num_messages_A-1, 0])
    # Next insert first message
    data.append([mmsi_a, timestamp - dt.timedelta(seconds=1), lat, lon, 0, num_messages_A-1])

    for i in range(num_messages_B):
        data.append([mmsi_a, timestamp + dt.timedelta(seconds=i * 1), lat + i * 1.0, lon + i * 1.0,
                     i+1, num_messages_A-(i+2)])
        data.append([mmsi_b, timestamp + dt.timedelta(seconds=i * 1), lat - i * 0.5, lon - i * 0.5,
                     i, num_messages_B-(i+1)])
    df_pd = pandas.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP,
                             ColumnName.LATITUDE, ColumnName.LONGITUDE, "REF INDEX", "INVERSE REF"])
    df = vaex.from_pandas(df_pd)

    metadata = damast.core.MetaData(
        columns=[damast.core.DataSpecification(ColumnName.MMSI, representation_type=int),
                 damast.core.DataSpecification(ColumnName.TIMESTAMP, representation_type="datetime64[ns]"),
                 damast.core.DataSpecification(ColumnName.LATITUDE, unit=units.deg, representation_type=np.float64),
                 damast.core.DataSpecification(ColumnName.LONGITUDE, unit=units.deg, representation_type=np.float64)])
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
    assert np.allclose(new_adf["REF INDEX"].evaluate(), new_adf[ColumnName.HISTORIC_SIZE].evaluate())
    assert np.allclose(new_adf["INVERSE REF"].evaluate(), new_adf[ColumnName.HISTORIC_SIZE_REVERSE].evaluate())

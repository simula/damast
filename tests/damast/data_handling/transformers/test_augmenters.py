import datetime as dt

import dask.dataframe as dd
import dask.diagnostics
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import damast.domains.maritime.ais.vessel_types as vessel
from damast.data_handling.transformers.augmenters import AddMissingAISStatus, InvertedBinariser, \
    AddDistanceClosestAnchorage, AddCombinedLabel, AddVesselType
from damast.data_handling.transformers.features import DeltaDistance, Feature
from damast.domains.maritime.ais.navigational_status import AISNavigationalStatus
from damast.domains.maritime.data_specification import ColumnName, FieldValue
from damast.math.spatial import great_circle_distance


def test_add_missing_ais_status():
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

    df = pd.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP, ColumnName.STATUS])

    a = AddMissingAISStatus()
    df_transformed = a.transform(df)

    # If previous status is known, the assumption is to keep the same status
    assert df_transformed.iloc[3][ColumnName.STATUS] == status

    # NaN should not exist in the final data
    assert not df_transformed[ColumnName.STATUS].isnull().values.any()
    assert df_transformed.iloc[30][ColumnName.STATUS] == AISNavigationalStatus.Undefined


def test_delta_column():
    mmsi_a = 400000000
    mmsi_b = 500000000
    timestamp = dt.datetime.utcnow()

    lat = 0.0
    lon = 0.0

    data = []
    for i in range(1, 90):
        data.append([mmsi_a, timestamp + dt.timedelta(seconds=i * 1), lat + i * 1.0, lon + i * 1.0])
        data.append([mmsi_b, timestamp + dt.timedelta(seconds=i * 1), lat - i * 0.5, lon - i * 0.5])

    df = pd.DataFrame(data, columns=[ColumnName.MMSI, ColumnName.TIMESTAMP, ColumnName.LATITUDE, ColumnName.LONGITUDE])

    delta_distance = DeltaDistance()
    df_transformed = delta_distance.transform(df)

    assert Feature.DELTA_DISTANCE in df_transformed.columns
    df_grouped = df.groupby(ColumnName.MMSI)
    for g, indexes in df_grouped.groups.items():
        prev_lat = None
        prev_lon = None

        for idx in indexes:
            delta_distance = df.iloc[idx][Feature.DELTA_DISTANCE]
            if prev_lat is None:
                continue

            expected_distance = great_circle_distance(lat,
                                                      lon,
                                                      prev_lat,
                                                      prev_lon
                                                      )

            prev_lat = lat
            prev_lon = lon

            # First entries should be 0.0
            if idx < 2:
                assert delta_distance == 0.0
                continue

            assert delta_distance == expected_distance


def test_inverted_binariser():
    data = [-10, 10, 20, 101]
    df = pd.DataFrame(data, columns=[ColumnName.HEADING])

    inverted_binariser = InvertedBinariser(base_column_name=ColumnName.HEADING,
                                           threshold=100)
    df_transformed = inverted_binariser.transform(df)

    expected_column_name = f"{ColumnName.HEADING}_TRUE"
    assert expected_column_name in df_transformed.columns
    assert df.iloc[0][expected_column_name] == 1
    assert df.iloc[1][expected_column_name] == 1
    assert df.iloc[2][expected_column_name] == 1
    assert df.iloc[3][expected_column_name] == 0


def test_add_vessel_type():
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
    df_vessel_types = pd.DataFrame(vessel_type_data, columns=columns)

    input_data = [
        [200000210, 10],
        [200008591, 11],
        [200014514, 12],
        [200016821, 13],
        [200016826, 14],
        [200024943, 15],
        [200041743, 16],
        [100000000, 17]
    ]

    df = pd.DataFrame(input_data, columns=[ColumnName.MMSI, ColumnName.SPEED_OVER_GROUND])

    transformer = AddVesselType(vessel_type_data=df_vessel_types)
    df_transformed = transformer.fit_transform(df)

    assert ColumnName.VESSEL_TYPE in df_transformed
    for i in range(0, 7):
        vessel_type_name = vessel_type_data[i][1]
        vessel_type = FieldValue.UNDEFINED
        if vessel_type_name != "unspecified":
            vessel_type = vessel.VesselType[vessel_type_name]
        assert df_transformed.at[i, ColumnName.VESSEL_TYPE] == vessel_type

    assert df_transformed.at[7, ColumnName.VESSEL_TYPE] == FieldValue.UNDEFINED


def test_add_distance_closest_anchorage():
    columns = ["s2id", "latitude", "longitude", "label", "sublabel", "iso3"]
    anchorage_data = [
        ["356ec741", 34.839059469352883, 128.42069569869318, "TONGYEONG", None, "KOR"],
        ["3574e907", 34.691433022457183, 125.4429067750192, "HEUKSANDO", None, "KOR"],
        ["356f2c7b", 35.137774967951927, 128.6030742948769, "MASAN", None, "KOR"],
        ["356f2af7", 35.080054265204275, 128.61175951519579, "NANPO- RI", None, "KOR"]
    ]

    df_anchorages = pd.DataFrame(anchorage_data, columns=columns)
    transformer = AddDistanceClosestAnchorage(anchorages_data=df_anchorages)

    data = [
        [34.839059469352883, 128.42069569869318],
        [30.0, 125.0]
    ]
    df = pd.DataFrame(data, columns=[ColumnName.LATITUDE, ColumnName.LONGITUDE])

    df_transformed = transformer.transform(df)

    assert ColumnName.DISTANCE_CLOSEST_ANCHORAGE in df_transformed
    assert df_transformed.at[0, ColumnName.DISTANCE_CLOSEST_ANCHORAGE] == 0.0
    assert df_transformed.at[1, ColumnName.DISTANCE_CLOSEST_ANCHORAGE] > 0.0


def test_dask_partitioning():
    data = [
        [-0.549872956991289, 0.6344306851567723],
        [-0.5498659661934275, 0.6344568507144823],
        [-0.5498607397397882, 0.6344725633649138],
    ]
    lat_lon_rad_columns = ["lat_rad", "lon_rad"]
    df = pd.DataFrame(data, columns=lat_lon_rad_columns)

    pb = dask.diagnostics.ProgressBar()
    pb.register()

    def compute_distance(x):
        """
        Compute the Haversine distance between the closest anchorage and a set of points
        """
        # NOTE: We have to define the ball-tree on each process to gain any speedup
        # We also need to copy the input to avoid tkinter issues
        tree = BallTree(df[lat_lon_rad_columns].copy(), metric='haversine')
        output = tree.query(np.array(x[lat_lon_rad_columns]), k=1, return_distance=False)
        return output.reshape(-1)

    dask_dataframe = dd.from_pandas(df[lat_lon_rad_columns], chunksize=2)
    dist_output = dask_dataframe.map_partitions(lambda df_part:
                                                compute_distance(df_part))
    df["new_column"] = dist_output.compute()
    assert "new_column" in df
    assert np.allclose(df["new_column"], np.arange(df.shape[0]))


def test_add_combined_label():
    col_a_name = "a"
    col_a_permitted_values = {"0": "a0", "1": "a1", "2": "a2"}

    col_b_name = "b"
    col_b_permitted_values = {"0": "b0", "1": "b1", "2": "b2"}

    data = [["a0", "b0"],
            ["a0", "b1"]]
    df = pd.DataFrame(data, columns=[col_a_name, col_b_name])

    add_combined_label = AddCombinedLabel(
        column_permitted_values={
            col_a_name: col_a_permitted_values,
            col_b_name: col_b_permitted_values,
        },
        column_names=[col_a_name, col_b_name],
        combination_name=ColumnName.STATUS
    )

    df_transformed = add_combined_label.transform(df)
    labels = add_combined_label.label_mapping
    for index, row in df_transformed.iterrows():
        assert [row[col_a_name], row[col_b_name]] == labels[str(row[ColumnName.STATUS])]

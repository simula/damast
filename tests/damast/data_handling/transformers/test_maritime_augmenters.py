import datetime as dt

import dask.dataframe as dd
import dask.diagnostics
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


from damast.data_handling.transformers.augmenters import (
    AddCombinedLabel,
    AddDistanceClosestAnchorage,
    InvertedBinariser,
)

from damast.domains.maritime.data_specification import ColumnName


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

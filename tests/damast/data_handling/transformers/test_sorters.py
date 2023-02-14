import numpy as np
import pandas as pd

from damast.data_handling.transformers.features import DeltaTime
from damast.data_handling.transformers.sorters import GenericSorter, GroupBy


def test_generic_sorter():
    data = [["a", 0], ["a", 1], ["b", 1], ["b", 0]]
    column_names = ["a", "b"]
    df = pd.DataFrame(data, columns=column_names)

    gs = GenericSorter(column_names=column_names)
    df = gs.transform(df)

    assert (df.iloc[[0]] == ["a", 0]).all().all()
    assert (df.iloc[[1]] == ["a", 1]).all().all()
    assert (df.iloc[[2]] == ["b", 0]).all().all()
    assert (df.iloc[[3]] == ["b", 1]).all().all()

    gs = GenericSorter(column_names=column_names, ascending=False)
    df = gs.transform(df)

    assert (df.iloc[[3]] == ["a", 0]).all().all()
    assert (df.iloc[[2]] == ["a", 1]).all().all()
    assert (df.iloc[[1]] == ["b", 0]).all().all()
    assert (df.iloc[[0]] == ["b", 1]).all().all()


def test_group_by_delta_time():
    data = [
        ["0", 0],
        ["2", 0],
        ["2", 1],
        ["1", 1]
    ]

    column_names = ["id", "timestamp"]
    df = pd.DataFrame(data, columns=column_names)

    transformer = GroupBy(group_by="id",
                          transformer=DeltaTime(name="delta_time",
                                                timestamp_column="timestamp")
                          )

    transformed_df = transformer.fit_transform(df)

    assert "delta_time" in transformed_df
    assert np.isnan(transformed_df.loc[0, "delta_time"])
    assert np.isnan(transformed_df.loc[1, "delta_time"])
    assert transformed_df.loc[2, "delta_time"] == 1
    assert np.isnan(transformed_df.loc[3, "delta_time"])

import numpy as np
import pandas as pd
import pytest
import vaex

from damast.data_handling.accessors import GroupSequenceAccessor, SequenceIterator


@pytest.fixture()
def dataframe():
    data = []
    for group_id in range(0, 100):
        for i in range(1000, 3000):
            data.append([group_id, i, i, i * i])
    columns = ["id", "timestamp", "x", "y"]
    df_pandas = pd.DataFrame(data, columns=columns)
    df_pandas = df_pandas.sample(frac=1)

    return vaex.from_pandas(df_pandas)


@pytest.fixture()
def invalid_dataframe():
    """Dataframe with mixed dtype. Meant to cause ValueError
    """
    data = []
    for group_id in range(0, 100):
        for i in range(1000, 3000):
            data.append([group_id, np.float64(i), i, np.float64(i * i)])
    columns = ["id", "timestamp", "x", "y"]
    df_pandas = pd.DataFrame(data, columns=columns)
    df_pandas = df_pandas.sample(frac=1)

    return vaex.from_pandas(df_pandas)


def test_group_sequence_accessor(dataframe):
    gsa = GroupSequenceAccessor(df=dataframe,
                                group_column="id")

    sequence_length = 50
    batch_size = 20
    features = ["id", "x", "y"]

    data_gen = gsa.to_keras_generator(features,
                                      sequence_length=sequence_length,
                                      batch_size=batch_size,
                                      infinite=True)

    for i in range(0, 2):
        epoch = 20
        for batch in data_gen:
            epoch -= 1
            assert batch[0].shape == (batch_size, sequence_length, len(features))
            for group in batch[0]:
                ids_in_group = np.unique(group.T[0])
                assert len(ids_in_group) == 1
            if epoch < 0:
                break


@pytest.mark.parametrize("ratios", [[1.4, 0.2, 0.1, 0.3], [0.7, 0.1766, 1.1234], ])
def test_group_split_random(dataframe, ratios):
    gsa = GroupSequenceAccessor(df=dataframe,
                                group_column="id")
    groups = gsa.split_random(ratios=ratios)

    real_ratios = np.array(ratios)
    real_ratios /= 2

    num_total_groups = len(dataframe["id"].unique())
    num_groups = [np.round(ratio*num_total_groups).astype(int) for ratio in real_ratios]
    for group, num in zip(groups, num_groups):
        assert (len(group) == num)

    for i, group_A in enumerate(groups):
        for j, group_B in enumerate(groups):
            if i != j:
                assert np.isin(group_A, group_B, invert=True).all()


@pytest.mark.parametrize("sequence_length", [50, 1999])
@pytest.mark.parametrize("sequence_forecast", [0, 1, 2])
def test_sequence_accessor(dataframe, sequence_forecast, sequence_length):
    group_df = dataframe[dataframe["id"] == 84]
    iterator = SequenceIterator(group_df, ["timestamp"])
    target = None if sequence_forecast == 0 else ["x", "y"]
    try:
        generator = iterator.to_keras_generator(["x", "y"], target=target,
                                                sequence_length=sequence_length,
                                                sequence_forecast=sequence_forecast)
    except RuntimeError as e:
        if sequence_length + sequence_forecast > len(group_df):
            pytest.xfail(f"{e}")

    num_runs = 0
    for gen in generator:
        if sequence_forecast != 0:
            X, y = gen
            assert (X.shape == (sequence_length, 2))
            assert (y.shape == (sequence_forecast, 2))
        else:
            X = gen[0]
            assert (X.shape == (sequence_length, 2))
        num_runs += 1
    assert num_runs == len(group_df)-(sequence_forecast + sequence_length) + 1


def test_mixed_features_dtype(invalid_dataframe):
    group_df = invalid_dataframe[invalid_dataframe["id"] == 84]
    iterator = SequenceIterator(group_df, ["timestamp"])
    try:
        iterator.to_keras_generator(["x", "y"], sequence_length=50, sequence_forecast=1)
    except ValueError as e:
        pytest.xfail(f"{e}")


def test_mixed_targets_dtype(invalid_dataframe):
    group_df = invalid_dataframe[invalid_dataframe["id"] == 84]
    iterator = SequenceIterator(group_df, ["timestamp"])
    try:
        iterator.to_keras_generator(
            ["x"], target=["id", "timestamp"], sequence_length=50, sequence_forecast=1)
    except ValueError as e:
        pytest.xfail(f"{e}")

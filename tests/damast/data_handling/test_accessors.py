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
    """Dataframe with mixed dtype and short sequences to test error-handling.
    """
    data = []
    for group_id in range(0, 100):
        for i in range(10, 20):
            data.append([group_id, np.float64(i), i, np.float64(i * i)])
    columns = ["id", "timestamp", "x", "y"]
    df_pandas = pd.DataFrame(data, columns=columns)
    df_pandas = df_pandas.sample(frac=1)

    return vaex.from_pandas(df_pandas)


def test_yield_one_generator(dataframe):
    iterator = GroupSequenceAccessor(dataframe, group_column="id")
    sequence_length = 5
    features = ["id", "x", "y"]
    c = 0
    for i in iterator.to_keras_generator(features,
                                         sequence_length=sequence_length, infinite=False):
        c += 1
    assert c == 1


@pytest.mark.parametrize("sequence_length", [1998, 50])
@pytest.mark.parametrize("sort_column, shuffle", [(None, True), (["timestamp"], False)])
@pytest.mark.parametrize("target, sequence_forecast", [(["timestamp"], 2), (["timestamp"], 1), (None, 0)])
def test_group_sequence_accessor(dataframe, target, sequence_forecast, sort_column, shuffle, sequence_length):
    gsa = GroupSequenceAccessor(df=dataframe,
                                sort_columns=sort_column,
                                group_column="id", timeout_in_s=5)
    batch_size = 13
    epoch = batch_size
    features = ["id", "x", "y"]
    data_gen = gsa.to_keras_generator(features, target=target,
                                      sequence_forecast=sequence_forecast,
                                      sequence_length=sequence_length,
                                      batch_size=epoch,
                                      infinite=True, shuffle=shuffle)

    for i in range(0, 2):
        epoch = batch_size
        for batch in data_gen:
            epoch -= 1
            # NOTE: In the current implementation we then get a (x,y,1) array if we only
            # have one feature. I am not sure this is what we want.
            assert batch[0].shape == (batch_size, sequence_length, len(features))
            if target is not None:
                if sequence_forecast == 1:
                    assert batch[1].shape == (batch_size, len(target))
                else:
                    assert batch[1].shape == (batch_size, sequence_forecast, len(target))
            else:
                assert len(batch) == 1
            for group in batch[0]:
                ids_in_group = np.unique(group.T[0])
                assert len(ids_in_group) == 1
            if epoch < 0:
                break


def test_short_sequence(invalid_dataframe):
    gsa = GroupSequenceAccessor(df=invalid_dataframe,
                                group_column="id", timeout_in_s=1)
    sequence_length = 50
    batch_size = 1
    features = ["y"]
    data_gen = gsa.to_keras_generator(features,
                                      sequence_length=sequence_length,
                                      batch_size=batch_size,
                                      infinite=True)

    with pytest.raises(RuntimeError, match="could not identify a sufficiently long sequence"):
        for _ in data_gen:
            continue


@pytest.mark.parametrize("iterator_class", [SequenceIterator, GroupSequenceAccessor])
@pytest.mark.parametrize("target, sequence_forecast",
                         [(["timestamp"], 0),
                          (["timestamp"], -1),
                          (None, 2)])
def test_invalid_target(dataframe, target, sequence_forecast, iterator_class):
    if iterator_class == SequenceIterator:
        group_df = dataframe[dataframe["id"] == 84]
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")
    sequence_length = 5
    features = ["id", "x", "y"]
    with pytest.raises(ValueError, match="(?i)Targets|Sequence forecast"):
        iterator.to_keras_generator(features, target=target,
                                    sequence_forecast=sequence_forecast,
                                    sequence_length=sequence_length)


def test_sort_shuffle(dataframe):
    gsa = GroupSequenceAccessor(df=dataframe,
                                group_column="id", sort_columns=["timestamp"])
    sequence_length = 5
    batch_size = 10
    features = ["id", "x", "y"]
    with pytest.raises(RuntimeError, match="Cannot sort and shuffle sequence at the same time"):
        gsa.to_keras_generator(features,
                               sequence_length=sequence_length,
                               batch_size=batch_size, shuffle=True,
                               infinite=True)


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

    if sequence_length + sequence_forecast > len(group_df):
        with pytest.raises(RuntimeError, match="SequenceIterator: 'Sequence length' plus 'forecast length' (.*) larger than" +
                           " dataframe of size (.*)"):
            iterator.to_keras_generator(["x", "y"], target=target,
                                        sequence_length=sequence_length,
                                        sequence_forecast=sequence_forecast)
    else:
        generator = iterator.to_keras_generator(["x", "y"], target=target,
                                                sequence_length=sequence_length,
                                                sequence_forecast=sequence_forecast)
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


@pytest.mark.parametrize("iterator_class", [SequenceIterator, GroupSequenceAccessor])
def test_mixed_features_dtype(invalid_dataframe, iterator_class):
    if iterator_class == SequenceIterator:
        group_df = invalid_dataframe[invalid_dataframe["id"] == 84]
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(invalid_dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")

    with pytest.raises(ValueError, match=r".*: Features \['x', 'y'\] do not have a consistent \(single\) datatype, got \[int64, float64\]"):
        iterator.to_keras_generator(["x", "y"], sequence_length=50, sequence_forecast=1)


@pytest.mark.parametrize("iterator_class", [SequenceIterator, GroupSequenceAccessor])
def test_mixed_targets_dtype(invalid_dataframe, iterator_class):
    if iterator_class == SequenceIterator:
        group_df = invalid_dataframe[invalid_dataframe["id"] == 84]
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(invalid_dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")
    with pytest.raises(ValueError, match=r".*: Targets \['id', 'timestamp'\] do not have a consistent \(single\) datatype, got \[int64, float64\]"):
        iterator.to_keras_generator(
            ["x"], target=["id", "timestamp"], sequence_length=50, sequence_forecast=1)


@pytest.mark.parametrize("iterator_class", [SequenceIterator, GroupSequenceAccessor])
def test_negative_sequence_forecast(dataframe, iterator_class):
    if iterator_class == SequenceIterator:
        group_df = dataframe[dataframe["id"] == 84]
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")
    with pytest.raises(ValueError, match=r".*: Sequence forecast cannot be negative"):
        iterator.to_keras_generator(
            ["x"], target=["id", "timestamp"], sequence_length=50, sequence_forecast=-10)

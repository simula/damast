from datetime import timedelta

import numpy as np
import pandas as pd
import polars
import polars as pl
import pytest

from damast.data_handling.accessors import GroupSequenceAccessor, GroupWindowAccessor, SequenceIterator


@pytest.fixture()
def dataframe():
    data = []
    for group_id in range(100):
        for i in range(1000, 3000):
            data.append([group_id, i, i, i * i])
    columns = ["id", "timestamp", "x", "y"]
    df_pandas = pd.DataFrame(data, columns=columns)
    df_pandas = df_pandas.sample(frac=1)

    return polars.from_pandas(df_pandas)


@pytest.fixture()
def invalid_dataframe():
    """Dataframe with mixed dtype and short sequences to test error-handling.
    """
    data = []
    for group_id in range(100):
        for i in range(10, 20):
            data.append([group_id, np.float64(i), i, np.float64(i * i)])
    columns = ["id", "timestamp", "x", "y"]
    df_pandas = pd.DataFrame(data, columns=columns)
    df_pandas = df_pandas.sample(frac=1)

    return polars.from_pandas(df_pandas)


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

    for i in range(2):
        epoch = batch_size
        for batch in data_gen:
            epoch -= 1
            # NOTE: In the current implementation we then get a (x,y,1) array if we only
            # have one feature. I am not sure this is what we want.
            assert batch[0].shape == (batch_size, sequence_length, len(features))
            if target is not None:
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
        group_df = dataframe.filter(pl.col("id") == 84)
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
    group_df = dataframe.filter(pl.col("id") == 84)
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
        group_df = invalid_dataframe.filter(pl.col("id") == 84)
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(invalid_dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")

    with pytest.raises(ValueError, match=r".*: Features \['x', 'y'\] do not have a consistent \(single\) datatype, got \[Int64, Float64\]"):
        iterator.to_keras_generator(["x", "y"], sequence_length=50, sequence_forecast=1)


@pytest.mark.parametrize("iterator_class", [SequenceIterator, GroupSequenceAccessor])
def test_mixed_targets_dtype(invalid_dataframe, iterator_class):
    if iterator_class == SequenceIterator:
        group_df = invalid_dataframe.filter(pl.col("id") == 84)
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(invalid_dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")
    with pytest.raises(ValueError, match=r".*: Targets \['id', 'timestamp'\] do not have a consistent \(single\) datatype, got \[Int64, Float64\]"):
        iterator.to_keras_generator(
            ["x"], target=["id", "timestamp"], sequence_length=50, sequence_forecast=1)


@pytest.mark.parametrize("iterator_class", [SequenceIterator, GroupSequenceAccessor])
def test_negative_sequence_forecast(dataframe, iterator_class):
    if iterator_class == SequenceIterator:
        group_df = dataframe.filter(pl.col("id") == 84)
        iterator = iterator_class(group_df, ["timestamp"])
    elif iterator_class == GroupSequenceAccessor:
        iterator = iterator_class(dataframe, group_column="id")
    else:
        raise RuntimeError(f"Unknown {iterator_class=}")
    with pytest.raises(ValueError, match=r".*: Sequence forecast cannot be negative"):
        iterator.to_keras_generator(
            ["x"], target=["id", "timestamp"], sequence_length=50, sequence_forecast=-10)


# GroupWindowAccessor: timestamps are integer seconds, windows are given in seconds
WINDOW = 1800
HORIZON = 600
SPAN = WINDOW + HORIZON
MAX_GAP = 120
T_OFFSET = 1_000_000
GAPPY_GROUP = 100
SPARSE_GROUP = 200
WINDOW_FEATURES = ["gid", "tv", "z"]


def _group_frame(gid, gaps):
    t = T_OFFSET + np.concatenate([[0], np.cumsum(gaps)]).astype(np.int64)
    return pl.DataFrame({"id": np.full(len(t), gid),
                         "timestamp": t,
                         "gid": np.full(len(t), float(gid)),
                         # the timestamp as feature, to recover the window start of a sample
                         "tv": t.astype(float),
                         # non-linear, so that interpolation depends on which rows are fetched
                         "z": (t - T_OFFSET) ** 2 / 1e6})


@pytest.fixture()
def irregular_dataframe():
    """Irregularly reporting groups, one with a single oversized gap and one that is too sparse throughout."""
    rng = np.random.default_rng(42)
    frames = [_group_frame(gid, rng.integers(1, 60, 1999)) for gid in range(10)]
    gaps = rng.integers(1, 60, 999)
    gaps[500] = 10_000
    frames.append(_group_frame(GAPPY_GROUP, gaps))
    frames.append(_group_frame(SPARSE_GROUP, np.full(200, 6 * MAX_GAP)))
    return pl.concat(frames).sample(fraction=1, shuffle=True, seed=0)


def _reference(df, gid, t0, times, column):
    """Brute-force interpolation over all rows of the group."""
    group = df.filter(pl.col("id") == gid).sort("timestamp")
    return np.interp(t0 + times, group["timestamp"].to_numpy(), group[column].to_numpy())


def _gap_free_segments(df, gid):
    t = np.sort(df.filter(pl.col("id") == gid)["timestamp"].to_numpy())
    breaks = np.flatnonzero(np.diff(t) > MAX_GAP)
    return list(zip(t[np.r_[0, breaks + 1]], t[np.r_[breaks, len(t) - 1]]))


@pytest.mark.parametrize("target, forecast_length", [(None, 1), (["z"], 1), (["z", "tv"], 3)])
@pytest.mark.parametrize("infinite", [True, False])
def test_group_window_accessor_shape(irregular_dataframe, target, forecast_length, infinite):
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    batch_size = 16
    generator = gwa.to_keras_generator(WINDOW_FEATURES, target=target,
                                       window=WINDOW, sequence_length=20,
                                       forecast_horizon=HORIZON if target else None,
                                       forecast_length=forecast_length,
                                       max_gap=MAX_GAP, batch_size=batch_size, infinite=infinite)
    batch = next(generator)
    assert batch[0].shape == (batch_size, 20, len(WINDOW_FEATURES))
    if target is None:
        assert len(batch) == 1
    else:
        assert batch[1].shape == (batch_size, forecast_length, len(target))


@pytest.mark.parametrize("infinite", [True, False])
def test_group_window_accessor_values(irregular_dataframe, infinite):
    """Inputs and targets match a brute-force interpolation over the full group, which also
    verifies that the bucketed fetch returns every row needed."""
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    sequence_length, forecast_length = 13, 4
    X, y = next(gwa.to_keras_generator(WINDOW_FEATURES, target=["z"],
                                       window=WINDOW, sequence_length=sequence_length,
                                       forecast_horizon=HORIZON, forecast_length=forecast_length,
                                       max_gap=MAX_GAP, batch_size=64, infinite=infinite, shuffle=True))
    input_times = np.linspace(0, WINDOW, sequence_length)
    target_times = WINDOW + HORIZON * np.arange(1, forecast_length + 1) / forecast_length
    for sample, target in zip(X, y):
        # each sample stems from a single group
        assert len(np.unique(sample[:, 0])) == 1
        gid, t0 = sample[0, 0], sample[0, 1]
        np.testing.assert_allclose(sample[:, 1], t0 + input_times)
        np.testing.assert_allclose(sample[:, 2], _reference(irregular_dataframe, gid, t0, input_times, "z"))
        np.testing.assert_allclose(target[:, 0], _reference(irregular_dataframe, gid, t0, target_times, "z"))


def test_group_window_accessor_bucket_boundaries(irregular_dataframe):
    """Windows straddling a bucket boundary still fetch all their rows."""
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    grid = np.linspace(0, SPAN, 25)
    width = SPAN + 2 * MAX_GAP
    boundary = (T_OFFSET // width + 2) * width
    t0 = np.array([boundary - MAX_GAP - 1, boundary - MAX_GAP, boundary - MAX_GAP + 1,
                   boundary - SPAN // 2, boundary + 1], dtype=np.int64)
    groups = pl.Series("id", np.zeros(len(t0), dtype=np.int64))
    resampled = gwa._resample(columns=["z"], groups=groups, t0=t0, grid=grid, max_gap=MAX_GAP)
    for start, values in zip(t0, resampled):
        np.testing.assert_allclose(values[:, 0], _reference(irregular_dataframe, 0, start, grid, "z"))


def test_group_window_accessor_max_gap(irregular_dataframe):
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    X, = next(gwa.to_keras_generator(WINDOW_FEATURES, window=SPAN, max_gap=MAX_GAP,
                                     batch_size=2048, infinite=True, verbose=False))
    gids = X[:, 0, 0]
    assert SPARSE_GROUP not in gids

    # windows of the gappy group lie entirely within one of its two gap-free segments
    segments = _gap_free_segments(irregular_dataframe, GAPPY_GROUP)
    assert len(segments) == 2
    t0 = X[gids == GAPPY_GROUP, 0, 1]
    assert len(t0) > 0
    for start in t0:
        assert any(begin <= start and start + SPAN <= end for begin, end in segments)

    # starts are drawn in time, not per row
    assert not np.isin(X[:, 0, 1], irregular_dataframe["timestamp"].to_numpy()).all()

    with pytest.raises(RuntimeError, match="could not identify a window"):
        gwa.to_keras_generator(WINDOW_FEATURES, groups=[SPARSE_GROUP], window=WINDOW, max_gap=MAX_GAP)


def test_group_window_accessor_single_pass(irregular_dataframe):
    """A single pass tiles all gap-free segments with non-overlapping windows."""
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    batches = list(gwa.to_keras_generator(WINDOW_FEATURES, target=["z"], window=WINDOW,
                                          forecast_horizon=HORIZON, max_gap=MAX_GAP,
                                          batch_size=50, infinite=False))
    X = np.concatenate([batch[0] for batch in batches])

    expected = sum((end - start) // SPAN
                   for gid in irregular_dataframe["id"].unique()
                   for start, end in _gap_free_segments(irregular_dataframe, gid))
    assert len(X) == expected

    for gid in np.unique(X[:, 0, 0]):
        t0 = np.sort(X[X[:, 0, 0] == gid, 0, 1])
        assert (np.diff(t0) >= SPAN).all()


def test_group_window_accessor_groups(irregular_dataframe):
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    X, = next(gwa.to_keras_generator(WINDOW_FEATURES, groups=np.array([3, 7]), window=WINDOW,
                                     max_gap=MAX_GAP, batch_size=256, infinite=True))
    assert set(np.unique(X[:, 0, 0])) == {3.0, 7.0}


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_group_window_accessor_datetime(irregular_dataframe, time_unit):
    df = irregular_dataframe.with_columns(pl.from_epoch("timestamp", time_unit="s")
                                          .cast(pl.Datetime(time_unit)).alias("timestamp"))
    gwa = GroupWindowAccessor(df.lazy(), group_column="id", timestamp_column="timestamp")
    X, y = next(gwa.to_keras_generator(WINDOW_FEATURES, target=["z"], window="30m", sequence_length=7,
                                       forecast_horizon=timedelta(minutes=10), max_gap="2m",
                                       batch_size=32, infinite=True))
    input_times = np.linspace(0, WINDOW, 7)
    for sample, target in zip(X, y):
        gid, t0 = sample[0, 0], sample[0, 1]
        np.testing.assert_allclose(sample[:, 2], _reference(irregular_dataframe, gid, t0, input_times, "z"))
        np.testing.assert_allclose(target[:, 0], _reference(irregular_dataframe, gid, t0, np.array([SPAN]), "z"))


@pytest.mark.parametrize("timestamp_type, kwargs, match", [
    ("numeric", {"window": "30m"}, "window must be a number"),
    ("numeric", {"window": timedelta(minutes=30)}, "window must be a number"),
    ("numeric", {"window": 0}, "window must be positive"),
    ("datetime", {"window": 1800}, "window must be a timedelta"),
    ("datetime", {"window": "30 minutes"}, "Invalid duration"),
    ("numeric", {"sequence_length": 1}, "sequence_length must be at least 2"),
    ("numeric", {"target": ["z"]}, "Targets require a forecast_horizon"),
    ("numeric", {"forecast_horizon": HORIZON}, "Cannot do a forecast_horizon with no targets"),
    ("numeric", {"target": ["z"], "forecast_horizon": HORIZON, "forecast_length": 0},
     "forecast_length must be at least 1"),
])
def test_group_window_accessor_invalid(irregular_dataframe, timestamp_type, kwargs, match):
    df = irregular_dataframe
    if timestamp_type == "datetime":
        df = df.with_columns(pl.from_epoch("timestamp", time_unit="s"))
    gwa = GroupWindowAccessor(df, group_column="id", timestamp_column="timestamp")
    arguments = {"window": WINDOW, "max_gap": MAX_GAP} if timestamp_type == "numeric" else {"window": "30m"}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        gwa.to_keras_generator(WINDOW_FEATURES, **arguments)


def test_group_window_accessor_invalid_features(irregular_dataframe):
    gwa = GroupWindowAccessor(irregular_dataframe, group_column="id", timestamp_column="timestamp")
    with pytest.raises(ValueError, match=r"Features \['id', 'z'\] do not have a consistent"):
        gwa.to_keras_generator(["id", "z"], window=WINDOW, max_gap=MAX_GAP)

    df = irregular_dataframe.with_columns(label=pl.lit("a"))
    gwa = GroupWindowAccessor(df, group_column="id", timestamp_column="timestamp")
    with pytest.raises(ValueError, match="Column 'label' must be numeric"):
        gwa.to_keras_generator(["label"], window=WINDOW, max_gap=MAX_GAP)

"""
Module for creating generators for accessing sequences of data from a DataFrame
"""

import logging
import numbers
import random
import re
import sys
import time
from datetime import timedelta
from typing import Any

import keras.utils
import numpy as np
import pandas as pd
import polars as pl

from damast.core.types import DataFrame, XDataFrame
from damast.ml import keras

__all__ = [
    "GroupSequenceAccessor",
    "GroupWindowAccessor",
    "SequenceIterator",
    "partition_sizes"
]
logger = logging.getLogger("damast")


if sys.platform == "darwin":
    # Handle "Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead"
    def _mps_precision(data):
        if data.dtype == np.float64:
            return data.astype(np.float32)
        return data
else:
    def _mps_precision(data):
        return data


def _check_single_dtype(df: DataFrame, columns: list[str], kind: str, owner: str):
    """
    Ensure that all columns share one datatype, so that they can be stacked into a single array.

    :param df: The dataframe holding the columns
    :param columns: Names of the columns to check
    :param kind: Label of the columns for the error message, e.g. "Features"
    :param owner: Name of the calling class for the error message
    """
    datatypes = [XDataFrame(df).dtype(c) for c in columns]
    for dtype in datatypes:
        if dtype != datatypes[0]:
            raise ValueError(f"{owner}:"
                             f" {kind} {columns} do not have a consistent (single) datatype,"
                             f" got {datatypes}")


_DURATION_TOKEN = re.compile(r"(\d+)(us|ms|s|m|h|d|w)")
_DURATION_UNITS = {
    "us": timedelta(microseconds=1),
    "ms": timedelta(milliseconds=1),
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
}


def _parse_duration(value: str) -> timedelta:
    """
    Parse a polars-style duration string, e.g. ``"30m"`` or ``"1h30m"``.

    Supported units are ``us``, ``ms``, ``s``, ``m``, ``h``, ``d`` and ``w``.

    :param value: The duration string
    :return: The duration
    """
    tokens = _DURATION_TOKEN.findall(value)
    if not tokens or "".join(number + unit for number, unit in tokens) != value:
        raise ValueError(f"Invalid duration '{value}': expected e.g. '30m' or '1h30m',"
                         f" with units {list(_DURATION_UNITS)}")
    return sum((int(number) * _DURATION_UNITS[unit] for number, unit in tokens), timedelta())


def _log_steps_per_epoch(steps_per_epoch: int):
    """Log the recommended steps per epoch, independent of the current log level."""
    current_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    logger.info(f'Recommended {steps_per_epoch=}')
    logger.setLevel(current_level)


def partition_sizes(number_of_items: int, ratios: list[float]) -> np.ndarray:
    """
    Split a number of items into partitions of the given relative sizes.

    Uses largest remainder rounding, so that the sizes always add up to ``number_of_items`` - plain
    rounding may not, e.g. 5 items at ``[0.5, 0.5]`` would yield 2 + 2.

    :param number_of_items: Total number of items to distribute
    :param ratios: Relative partition sizes (will be normalized, so that all elements sum to 1)
    :return: Size per partition, each within 1 of its exact share
    """
    exact_sizes = number_of_items * np.asarray(ratios, dtype=float) / sum(ratios)
    sizes = np.floor(exact_sizes).astype(int)
    remainder = number_of_items - sizes.sum()
    # 'remainder' items are still unassigned: give one each to the partitions
    # that lost the largest fraction when flooring
    sizes[np.argsort(sizes - exact_sizes, kind="stable")[:remainder]] += 1
    return sizes


class _GroupAccessorBase:
    """
    Common base of the accessors that sample from groups of a dataframe.

    :param df: The dataframe from which the data (train, test, ...) shall be extracted
    :param group_column: the name of the column that identifies the group
    :param groups: Dataframe listing the available groups in ``group_column``
    """

    def __init__(self, df: DataFrame, group_column: str, groups: pl.DataFrame):
        self.df = df
        self.group_column = group_column
        self.groups = groups

    def _group_values(self, groups: Any) -> list[Any]:
        """
        Get the plain group ids from any of the accepted forms of ``groups``: a list or array of ids
        (as returned by :func:`split_random`), a :class:`polars.Series`, or a dataframe holding ``group_column``.

        :param groups: The groups
        :return: List of group ids
        """
        if isinstance(groups, (pl.DataFrame, pl.LazyFrame)):
            groups = groups.lazy().select(self.group_column).collect()[self.group_column]
        return pl.Series(groups).to_list()

    def split_random(self, ratios: list[float]) -> list[list[Any]]:
        """
        Create ``N=len(ratios)`` groups of the dataframe, with given ratios, return the corresponding groups.

        The groups are based on :attr:`group_column`.

        :param ratios: List of relative partition sizes (will be normalized, so that all elements sum to 1
        :return: Following the ratios, returns lists of randomly sampled values from the id/group column
        """
        groups = self.groups[self.group_column].to_numpy().copy()

        random.shuffle(groups)
        number_of_groups = len(groups)

        from_idx = 0
        partitions = []
        for ps in partition_sizes(number_of_groups, ratios):
            to_idx = min(from_idx + ps, number_of_groups)
            partitions.append(groups[from_idx:to_idx])
            from_idx = to_idx

        return partitions


# https://www.tensorflow.org/tutorials/structured_data/time_series
class GroupSequenceAccessor(_GroupAccessorBase):
    """
    A generator that allows access to a length-limited sequence of a particular group.

    The resulting dataset (X) will have a shape of:
    :code:`(<batch_size>, <sequence_length>, <number-of-features>)`.

    The resulting targets (y) if requires will lead to a label dataset of shape
    :code:`(<batch_size>, <sequence_length>, <number-of-targets>)`.

    Per default the dataset is assumed to be sorted, e.g., typically in time-based order.
    One can name however use ``sort_columns``, to require sorting so that a sequence becomes a valid timeline.
    From the overall group-based sequence a random subsequence of given length is sampled.

    :param df: The dataframe from which the data (train, test, ...) shall be extracted.

        .. note::
            This can be the combined dataframe for train, test, validate since the generator allows to limit the group
            ids from which will be selected

    :param group_column: the name of the column that identifies the group
    :param sort_columns: Names of the columns that shall be used for sorting - if not set, no sorting will be done
    :param timeout_in_s: Searching for a sequence of a given length might fail, since the dataset might not contain data
        of the given length.
    """
    DEFAULT_TIMEOUT_IN_S: float = 30.0

    def __init__(self,
                 df: DataFrame,
                 group_column: str,
                 sort_columns: list[str] = None,
                 timeout_in_s: int = DEFAULT_TIMEOUT_IN_S):
        super().__init__(df=df, group_column=group_column, groups=df.unique(group_column))

        if sort_columns is not None:
            self.sort_columns = sort_columns if type(sort_columns) is list else [sort_columns]
        else:
            self.sort_columns = sort_columns

        self.timeout_in_s = timeout_in_s

    def to_keras_generator(self, features: list[str],
                           target: list[str] = None,
                           groups: list[str] = None,
                           sequence_length: int = 50,
                           sequence_forecast: int = 0,
                           batch_size: int = 1024,
                           shuffle: bool = False,
                           infinite: bool = False,
                           verbose: bool = True) -> keras.utils.Sequence:
        """
        Create a batch generator suitable as a Keras datasource.

        By default, the generator is infinite, i.e. it loops continuously over the data.
        Thus, you need to specify the :code:`"steps_per_epoch"` arg when fitting a Keras model,
        the :code:`"validation_steps"` when using it for validation, and :code:`"steps"` when
        calling the :code:`"predict"` method of a keras model.

        :param features: A list of features.
        :param target: The dependent or target column or a list of columns, if any.
        :param groups: A list of group ids for which sequence generation will be done - this must be a subset
            of the existing group value in the dataframe, see :func:`split_random`
        :param sequence_length: Required length of the to-be-generated sequence
        :param sequence_forecast: If target is given, this is the length of a forecasted sequence -
            for a sequence-to-sequence generator
        :param batch_size: Number of samples per chunk of data. This can be thought of as the batch size.
        :param shuffle: If True, shuffle a sequence - if sort_columns is given, the setting will have no effect
        :param infinite: If True, the generator is infinite, i.e. it loops continuously over the data.
            If False, the generator does only one pass over the data.
        :param verbose: If True, show an info on the recommended :code:`"steps_per_epoch"`
            based on the total number of samples and :code:`"batch_size"`.

        Example:

            .. highlight:: python
            .. code-block:: python

                from damast.data_handling.accessors import GroupSequenceAccessor
                import tensorflow.keras as K

                df = ...
                features = ['lat', 'lon']
                target = ['nav_status']

                gsa = GroupSequenceAccessor(df=df, group_column="mmsi", sort_columns=["timestamp"])
                train_ids, validate_ids, test_ids = gsa.split_random(ratios=[0.8, 0.1, 0.1])

                # Create a training generator
                train_generator = gsa.to_keras_generator(features=features, target=target,
                                                        sequence_length=50, sequence_forecast=1, batch_size=10)

                # Build a recurrent neural network model to deal with the sequence, e.g.,
                # to forecast the next sequence element
                nn_model = K.Sequential()
                nn_model.add(K.layers.SimpleRNN(2, return_sequences=True, input_shape=[50, 2])
                nn_model.add(K.layers.SimpleRNN(2, return_sequences=True))
                nn_model.add(K.layers.SimpleRNN(2))
                nn_model.compile(optimizer='sgd', loss='mse')

                nn_model.fit(x=train_generator, epochs=3, steps_per_epoch=645)
        """
        if verbose:
            _log_steps_per_epoch(np.ceil(len(self.df) / batch_size))

        # Sanity checks before creating generator
        # Check that all features have the same data-type
        _check_single_dtype(self.df, features, "Features", self.__class__.__name__)

        use_target = target is not None
        if use_target:
            _check_single_dtype(self.df, target, "Targets", self.__class__.__name__)

        if use_target:
            target = target if type(target) is list else [target]

            if sequence_forecast < 0:
                raise ValueError(f"{self.__class__.__name__}: Sequence forecast cannot be negative")
            if sequence_forecast == 0:
                raise ValueError(f"{self.__class__.__name__}: Cannot do extract targets with no sequence forecast")
        else:
            if sequence_forecast > 0:
                raise ValueError(f"{self.__class__.__name__}: Cannot do sequence forecast with no targets")

        if self.sort_columns is not None and shuffle:
            raise RuntimeError(f"{self.__class__.__name__}: Cannot sort and shuffle sequence at the same time")

        def _generator(features: list[str], target: list[str] | None,
                       groups: list[Any],
                       sequence_length: int, sequence_forecast: int,
                       chunk_size: int, shuffle: bool, infinite: bool,
                       ):
            """
            A generator function to yield the next sequence.

            If `targets` is not supplied, the generator has unlabeled data.
            Otherwise, the generator will provide labels for forecasting, i.e.
            taking samples from the group (after sequence length).

            :param features: List of feature name
            :param target: List of targets
            :param groups: List of groups from which sequences can be generated
            :param sequence_length: Required length of a sequence
            :param sequence_forecast: Length of the target sequence
            :param chunk_size: Number of sequences in a chunk (aka batch)
            :param shuffle: Whether the data (rows) in a sequence should be shuffled
            :param infinite: Run the generator infinitely, i.e. requires functions that use this generator to define
                a stopping criteria, e.g., steps_in_epoch
            """
            # Gather all columns in one list
            all_columns = features.copy()
            if self.sort_columns is not None:
                all_columns += self.sort_columns
            use_target = target is not None
            if use_target:
                target = target if type(target) is list else target
                all_columns += target
            else:
                # If no target, we are not forecasting
                sequence_forecast = 0
            all_columns = list(set(all_columns))

            groups = self._group_values(self.groups if groups is None else groups)

            while True:
                chunk = []
                target_chunk = []
                for i in range(chunk_size):
                    sequence: DataFrame = None
                    start_time = time.perf_counter()
                    sample_count = 0
                    sample_length = 0

                    # Find a valid subsequence, i.e. one with the given length
                    while not (time.perf_counter() - start_time) > self.timeout_in_s:
                        group = random.choice(groups)
                        # Since we will need the timeline later - we further deal with pandas DataFrame
                        # directly - thus, we do not use a copy of the DataFrame (only used columns)
                        sequence = self.df\
                                    .filter(pl.col(self.group_column) == group)\
                                    .select(all_columns)

                        # If sort columns are set, then ensure that the sorting is done
                        if self.sort_columns is not None:
                            sequence = sequence.sort(by=self.sort_columns)
                        elif shuffle:
                            sequence = sequence.sample(fraction=1)

                        if isinstance(sequence, DataFrame):
                            sequence = sequence.collect()

                        len_sequence = sequence.shape[0]
                        if len_sequence >= (sequence_length + sequence_forecast):
                            break

                        sample_count += 1
                        sample_length += len_sequence
                    if sequence is None or len_sequence < (sequence_length + sequence_forecast):
                        raise RuntimeError(f"{self.__class__.__name__}: could not identify a sufficiently long sequence"
                                           f" within given timeout of {self.timeout_in_s}:"
                                           f" mean length was {sample_length / sample_count}")

                    max_start_idx = len(sequence) - (sequence_length + sequence_forecast)
                    if max_start_idx == 0:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, max_start_idx)

                    sequence_window = sequence[start_idx:start_idx + sequence_length]
                    chunk.append(sequence_window[features].to_numpy())
                    # Target is a list of output "labels" in the ML sense
                    if use_target:
                        # Target is taken from the message "after" those in the window.
                        target_start_idx = start_idx + sequence_length
                        target_end_idx = target_start_idx + sequence_forecast

                        target_window = sequence[target_start_idx:target_end_idx][target]
                        if sequence_forecast == 1:
                            target_window = target_window[0]

                        # target it the last step in the timeline, so the last
                        target_chunk.append(target_window.to_numpy())

                X = _mps_precision(np.array(chunk))
                if use_target:
                    if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
                        y = _mps_precision(np.array(target_chunk))
                    else:
                        y = _mps_precision(np.array(target_chunk, copy=False))
                    yield (X, y)
                else:
                    yield (X,)
                if not infinite:
                    break

        return _generator(features=features, target=target,
                          groups=groups, sequence_length=sequence_length, sequence_forecast=sequence_forecast,
                          chunk_size=batch_size, shuffle=shuffle, infinite=infinite)


class GroupWindowAccessor(_GroupAccessorBase):
    """
    A generator of fixed-duration windows of a group, resampled onto a regular time grid.

    Unlike :class:`GroupSequenceAccessor`, which takes a fixed *number* of rows, this accessor takes a
    fixed *duration*: the input covers ``window`` and the targets lie up to ``forecast_horizon`` beyond it,
    independent of how often a group reports. Each window is linearly interpolated onto equidistant
    time points, so that the resulting dataset (X) has a shape of
    :code:`(<batch_size>, <sequence_length>, <number-of-features>)`, and the targets (y) a shape of
    :code:`(<batch_size>, <forecast_length>, <number-of-targets>)`.

    A window is only used if no two consecutive rows of the group within it - including the forecast
    horizon - are further apart than ``max_gap``, so that interpolation never bridges more than ``max_gap``.
    Window starts are drawn uniformly in time, not per row, so that periods of a high reporting rate are
    not favoured.

    .. warning::
        All features and targets are interpolated linearly. Angular values such as course or heading
        wrap around at 360° and must be transformed beforehand, e.g. into their sine and cosine.

    .. note::
        Only the timestamps are scanned upfront; each batch then fetches just the rows of its windows.
        With a file-backed :class:`polars.LazyFrame` this means reading the files once per batch - pass
        ``df.collect().lazy()`` if the data fits into memory.

    :param df: The dataframe from which the data (train, test, ...) shall be extracted
    :param group_column: the name of the column that identifies the group
    :param timestamp_column: the name of the time column, either of type :class:`polars.Datetime` or numeric
    """

    def __init__(self,
                 df: DataFrame | pl.DataFrame | XDataFrame,
                 group_column: str,
                 timestamp_column: str):
        if isinstance(df, XDataFrame):
            df = df.lazyframe
        df = df.lazy()
        super().__init__(df=df, group_column=group_column,
                         groups=df.select(group_column).unique().collect())
        self.timestamp_column = timestamp_column

    def to_keras_generator(self, features: list[str],
                           target: list[str] = None,
                           groups: list[Any] = None,
                           window: str | timedelta | float = "30m",
                           sequence_length: int = 50,
                           forecast_horizon: str | timedelta | float = None,
                           forecast_length: int = 1,
                           max_gap: str | timedelta | float = "5m",
                           batch_size: int = 1024,
                           shuffle: bool = False,
                           infinite: bool = False,
                           verbose: bool = True):
        """
        Create a batch generator suitable as a Keras datasource.

        The input grid consists of ``sequence_length`` points spanning the closed interval
        :code:`[t0, t0 + window]`; the target grid of ``forecast_length`` points at
        :code:`t0 + window + k * forecast_horizon / forecast_length` for :code:`k = 1..forecast_length`.

        Durations are given as :class:`datetime.timedelta` or a duration string such as ``"1h30m"``
        if the timestamp column is a :class:`polars.Datetime`, otherwise as a number in the unit of the
        timestamp column.

        :param features: A list of (numeric) features.
        :param target: A list of (numeric) targets, if any - requires ``forecast_horizon``
        :param groups: A list of group ids for which windows will be generated - this must be a subset
            of the existing group values in the dataframe, see :func:`split_random`
        :param window: Duration of the input window
        :param sequence_length: Number of equidistant points the input window is resampled to (at least 2)
        :param forecast_horizon: Duration between the end of the input window and the last target point
        :param forecast_length: Number of equidistant target points within the forecast horizon
        :param max_gap: Maximum time between two consecutive rows that may be interpolated
        :param batch_size: Number of windows per batch
        :param shuffle: If True, randomise the order of windows in a single pass (``infinite=False``)
        :param infinite: If True, draw random windows endlessly, so the caller needs to define a stopping
            criterion, e.g., :code:`"steps_per_epoch"`.
            If False, do one pass over all non-overlapping windows of the data, e.g., for evaluation.
        :param verbose: If True, show an info on the recommended :code:`"steps_per_epoch"` based on the
            number of non-overlapping windows and :code:`"batch_size"`.

        Example:

            .. highlight:: python
            .. code-block:: python

                from damast.data_handling.accessors import GroupWindowAccessor

                df = ...
                features = ['lat_x', 'lat_y', 'lon_x', 'lon_y', 'cog_x', 'cog_y']

                gwa = GroupWindowAccessor(df=df, group_column="mmsi", timestamp_column="timestamp")
                train_ids, validate_ids, test_ids = gwa.split_random(ratios=[0.8, 0.1, 0.1])

                # 30 minutes resampled to 60 steps (every ~30 s); predict the position 15 minutes ahead
                train_generator = gwa.to_keras_generator(features=features, target=['lat_x', 'lat_y', 'lon_x', 'lon_y'],
                                                         groups=train_ids,
                                                         window="30m", sequence_length=60,
                                                         forecast_horizon="15m", max_gap="5m",
                                                         batch_size=32, infinite=True)
        """
        owner = self.__class__.__name__

        _check_single_dtype(self.df, features, "Features", owner)
        use_target = target is not None
        if use_target:
            _check_single_dtype(self.df, target, "Targets", owner)
        for column in features + (target if use_target else []):
            if not XDataFrame(self.df).is_numeric(column):
                raise ValueError(f"{owner}: Column '{column}' must be numeric to be interpolated")

        if use_target and forecast_horizon is None:
            raise ValueError(f"{owner}: Targets require a forecast_horizon")
        if not use_target and forecast_horizon is not None:
            raise ValueError(f"{owner}: Cannot do a forecast_horizon with no targets")
        if sequence_length < 2:
            raise ValueError(f"{owner}: sequence_length must be at least 2, got {sequence_length}")
        if forecast_length < 1:
            raise ValueError(f"{owner}: forecast_length must be at least 1, got {forecast_length}")

        window_units = self._to_time_units("window", window)
        max_gap_units = self._to_time_units("max_gap", max_gap)
        horizon_units = self._to_time_units("forecast_horizon", forecast_horizon) if use_target else 0
        span = window_units + horizon_units

        if groups is not None:
            groups = self._group_values(groups)
        segments = self._segments(groups=groups, max_gap=max_gap_units)
        valid_segments = segments.filter((pl.col("end") - pl.col("start")) >= span)
        if valid_segments.is_empty():
            longest = (segments["end"] - segments["start"]).max() if not segments.is_empty() else None
            raise RuntimeError(f"{owner}: could not identify a window of {window} plus a forecast horizon of"
                               f" {forecast_horizon} without gaps larger than {max_gap}"
                               f" in {segments[self.group_column].n_unique()} groups:"
                               f" longest gap-free span is {longest} (in units of '{self.timestamp_column}')")
        segments = valid_segments

        columns = list(dict.fromkeys(features + (target if use_target else [])))
        feature_idx = [columns.index(f) for f in features]
        target_idx = [columns.index(t) for t in target] if use_target else []

        grid = np.linspace(0, window_units, sequence_length)
        if use_target:
            grid = np.concatenate([grid,
                                   window_units + horizon_units * np.arange(1, forecast_length + 1) / forecast_length])

        starts = segments["start"].to_numpy()
        lengths = segments["end"].to_numpy() - starts
        tile_counts = (lengths // span).astype(np.int64)
        if verbose:
            _log_steps_per_epoch(np.ceil(tile_counts.sum() / batch_size))

        def _batch(segment_idx: np.ndarray, t0: np.ndarray):
            values = self._resample(columns=columns,
                                    groups=segments[self.group_column].gather(segment_idx),
                                    t0=t0, grid=grid, max_gap=max_gap_units)
            X = _mps_precision(values[:, :sequence_length, feature_idx])
            if use_target:
                return X, _mps_precision(values[:, sequence_length:, target_idx])
            return (X,)

        def _random_generator():
            while True:
                yield _batch(*self._draw_starts(segments=segments, span=span, size=batch_size))

        def _single_pass_generator():
            # Tile each segment with non-overlapping windows, so that each target is seen once
            segment_idx = np.repeat(np.arange(len(starts)), tile_counts)
            tile_idx = np.arange(len(segment_idx)) - np.repeat(np.cumsum(tile_counts) - tile_counts, tile_counts)
            t0 = starts[segment_idx] + tile_idx * span

            order = np.random.permutation(len(t0)) if shuffle else np.arange(len(t0))
            for from_idx in range(0, len(order), batch_size):
                selected = order[from_idx:from_idx + batch_size]
                yield _batch(segment_idx[selected], t0[selected])

        return _random_generator() if infinite else _single_pass_generator()

    def _to_time_units(self, name: str, value: str | timedelta | float) -> float:
        """
        Convert a duration into the physical unit of the timestamp column.

        :param name: Name of the parameter, for the error message
        :param value: The duration
        :return: The duration as a number in the unit of the timestamp column
        """
        owner = self.__class__.__name__
        dtype = XDataFrame(self.df).dtype(self.timestamp_column)
        if isinstance(dtype, pl.Datetime):
            if isinstance(value, str):
                value = _parse_duration(value)
            if not isinstance(value, timedelta):
                raise ValueError(f"{owner}: {name} must be a timedelta or duration string (e.g. '30m'),"
                                 f" since '{self.timestamp_column}' is {dtype} - got {value!r}")
            microseconds = value // timedelta(microseconds=1)
            converted = {"ns": microseconds * 1000, "us": microseconds, "ms": microseconds // 1000}[dtype.time_unit]
        elif dtype.is_numeric():
            if not isinstance(value, numbers.Real):
                raise ValueError(f"{owner}: {name} must be a number in the unit of '{self.timestamp_column}',"
                                 f" since it is {dtype} - got {value!r}")
            converted = value
        else:
            raise ValueError(f"{owner}: timestamp column '{self.timestamp_column}' must be a Datetime or numeric,"
                             f" got {dtype}")

        if converted <= 0:
            raise ValueError(f"{owner}: {name} must be positive, got {value!r}")
        return converted

    def _segments(self, groups: list[Any] | None, max_gap: float) -> pl.DataFrame:
        """
        Compute the gap-free segments of all groups, i.e. the maximal runs of rows without a
        gap larger than ``max_gap`` between consecutive rows.

        :param groups: The groups to consider, or None for all
        :param max_gap: Maximum gap in the physical unit of the timestamp column
        :return: Dataframe with the columns ``group_column``, ``start`` and ``end``, sorted by group and start
        """
        timestamps = self.df.select(pl.col(self.group_column),
                                    pl.col(self.timestamp_column).to_physical().alias("__t")).drop_nulls()
        if groups is not None:
            timestamps = timestamps.filter(pl.col(self.group_column).is_in(groups))

        return (timestamps
                .sort(self.group_column, "__t")
                .with_columns(__segment=(pl.col("__t").diff() > max_gap).fill_null(False).cum_sum()
                              .over(self.group_column))
                .group_by(self.group_column, "__segment")
                .agg(start=pl.col("__t").min(), end=pl.col("__t").max())
                .drop("__segment")
                .sort(self.group_column, "start")
                .collect())

    def _draw_starts(self, segments: pl.DataFrame, span: float, size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw random window starts: a group uniformly, then a start uniformly in time within the
        valid part of its segments.

        :param segments: Segments that can hold at least one window, sorted by group
        :param span: Duration of a window including the forecast horizon
        :param size: Number of starts to draw
        :return: Tuple of the segment index and start time per window
        """
        starts = segments["start"].to_numpy()
        valid_lengths = segments["end"].to_numpy() - starts - span

        # Segments of a group are contiguous; the cumulative valid length maps a uniform draw
        # within a group onto a segment and an offset in one step
        group_ids = segments.select(pl.col(self.group_column).rle_id())[:, 0].to_numpy()
        number_of_groups = group_ids[-1] + 1
        group_bounds = np.searchsorted(group_ids, np.arange(number_of_groups + 1))
        cumulative = np.concatenate([[0.0], np.cumsum(valid_lengths, dtype=np.float64)])

        group = np.random.randint(number_of_groups, size=size)
        lower, upper = group_bounds[group], group_bounds[group + 1]
        position = cumulative[lower] + np.random.random(size) * (cumulative[upper] - cumulative[lower])
        segment_idx = np.clip(np.searchsorted(cumulative, position, side="right") - 1, lower, upper - 1)
        offset = np.clip(position - cumulative[segment_idx], 0, valid_lengths[segment_idx])

        if np.issubdtype(starts.dtype, np.integer):
            offset = np.floor(offset).astype(starts.dtype)
        return segment_idx, starts[segment_idx] + offset

    def _resample(self, columns: list[str], groups: pl.Series, t0: np.ndarray,
                  grid: np.ndarray, max_gap: float) -> np.ndarray:
        """
        Fetch the rows of the given windows and interpolate them onto the grid.

        Rows are fetched in the range :code:`[t0 - max_gap, t0 + grid[-1] + max_gap]`, which ensures a
        row at or beyond both ends of the grid. To avoid joining each window with all rows of its group,
        time is split into buckets as wide as this range, so that a window touches at most two buckets.

        :param columns: The columns to interpolate
        :param groups: Group per window
        :param t0: Start time per window
        :param grid: Time points relative to ``t0``
        :param max_gap: Maximum gap in the physical unit of the timestamp column
        :return: Array of shape :code:`(len(t0), len(grid), len(columns))`
        """
        width = grid[-1] + 2 * max_gap
        lower = t0 - max_gap
        requests = (pl.DataFrame([pl.Series("__window", np.arange(len(t0))),
                                  groups.alias(self.group_column),
                                  pl.Series("__lower", lower),
                                  pl.Series("__upper", lower + width)])
                    .with_columns(__bucket=(pl.col("__lower") // width).cast(pl.Int64)))
        requests = pl.concat([requests, requests.with_columns(pl.col("__bucket") + 1)])

        value_columns = [f"__value_{i}" for i in range(len(columns))]
        rows = (self.df
                .select(pl.col(self.group_column),
                        pl.col(self.timestamp_column).to_physical().alias("__t"),
                        *[pl.col(c).cast(pl.Float64).alias(v) for c, v in zip(columns, value_columns)])
                .with_columns(__bucket=(pl.col("__t") // width).cast(pl.Int64))
                .join(requests.lazy(), on=[self.group_column, "__bucket"])
                .filter(pl.col("__t").is_between(pl.col("__lower"), pl.col("__upper")))
                .sort("__window", "__t")
                .collect())

        window_bounds = np.searchsorted(rows["__window"].to_numpy(), np.arange(len(t0) + 1))
        timestamps = rows["__t"].to_numpy()
        values = rows.select(value_columns).to_numpy()

        resampled = np.empty((len(t0), len(grid), len(columns)))
        for i in range(len(t0)):
            rows_in_window = slice(window_bounds[i], window_bounds[i + 1])
            relative_time = (timestamps[rows_in_window] - t0[i]).astype(np.float64)
            for j in range(len(columns)):
                resampled[i, :, j] = np.interp(grid, relative_time, values[rows_in_window, j])
        return resampled


class SequenceIterator:
    """
    A generator that allows iterate over the windows of a length-limited sequence.

    The resulting dataset (X) will have a shape of:
        (<sequence_length>, <number-of-features>).

    The resulting targets (y) if requires will lead to a label dataset of shape
        (<sequence_length>, <number-of-targets>)

    Per default the dataset is assumed to be sorted, e.g., typically in time-based order.
    One can name however use :attr:`"sort_columns"`, to require sorting so that a sequence becomes a valid timeline.

    :param df: The dataframe from which the data (train, test, ...) shall be extracted.
    :param sort_columns: Names of the columns that shall be used for sorting - if None, no sorting will be done
    """

    df: DataFrame | pd.DataFrame

    def __init__(self,
                 df: DataFrame | pd.DataFrame,
                 sort_columns: list[str] = None):
        if sort_columns is not None:
            if isinstance(df, DataFrame):
                df = df.collect()

            if isinstance(df, pl.dataframe.DataFrame):
                self.df = df.sort(by=sort_columns)
            elif isinstance(df, pd.DataFrame):
                self.df = df.sort_values(by=sort_columns)
            else:
                raise RuntimeError(f"{self.__class__.__name__}.__init__: df was {type(df)}, but must be "
                                   f"either polars.dataframe.DataFrame or pandas.DataFrame")
        else:
            self.df = df

    def to_keras_generator(self, features,
                           target=None,
                           sequence_length: int = 50,
                           sequence_forecast: int = 1) -> keras.utils.Sequence:
        """
        Create a batch generator suitable as a Keras datasource.

        By default, the generator is infinite, i.e. it loops continuously over the data.
        Thus, you need to specify the :attr:`"steps_per_epoch"` arg when fitting a Keras model,
        the :attr:`"validation_steps"` when using it for validation, and :attr:`"steps"` when
        calling the :attr:`"predict"` method of a keras model.

        :param features: A list of features.
        :param target: The dependent or target column or a list of columns, if any.
        :param sequence_length: Length of input sequence
        :param sequence_forecast: If target is given, this is the length of a forecasted sequence -
            for a sequence-to-sequence generator

        Example:

            .. highlight:: python
            .. code-block:: python

                from damast.data_handling.accessors import GroupSequenceAccessor
                import tensorflow.keras as K

                df = ...
                features = ['lat', 'lon']
                target = ['nav_status']

                it = SequenceIterator(df=df, sort_columns=["timestamp"])
                # Create a training generator
                it = it.to_keras_generator(features=features, target=target, sequence_length=50, sequence_forecast=1)

                # Build a recurrent neural network model to deal with the sequence, e.g.,
                # to forecast the next sequence element
                nn_model = K.Sequential()
                nn_model.add(K.layers.SimpleRNN(2, return_sequences=True, input_shape=[50, 2])
                nn_model.add(K.layers.SimpleRNN(2, return_sequences=True))
                nn_model.add(K.layers.SimpleRNN(2))
                nn_model.compile(optimizer='sgd', loss='mse')

                nn_model.fit(x=train_generator, epochs=3, steps_per_epoch=645)
        """

        # Sanity checks before creating generator
        # Check that all features have the same data-type
        _check_single_dtype(self.df, features, "Features", self.__class__.__name__)

        use_target = target is not None
        if use_target:
            _check_single_dtype(self.df, target, "Targets", self.__class__.__name__)

        if sequence_forecast < 0:
            raise ValueError(f"{self.__class__.__name__}: Sequence forecast cannot be negative")

        if use_target:
            target = target if type(target) is list else target
            if sequence_forecast == 0:
                raise ValueError(f"{self.__class__.__name__}: Cannot do extract targets with no sequence forecast")
        else:
            if sequence_forecast > 0:
                raise ValueError(f"{self.__class__.__name__}: Cannot do sequence forecast with no targets")

        sequence = self.df

        # Since we will need the timeline later - we further deal with
        # the actual data
        if not isinstance(sequence, pl.dataframe.DataFrame):
            sequence = sequence.collect()

        len_sequence = len(sequence)  # Equivalent to sequence.shape[0]
        if len_sequence < (sequence_length + sequence_forecast):
            raise RuntimeError(f"{self.__class__.__name__}:"
                               f" 'Sequence length' plus 'forecast length' ({sequence_length + sequence_forecast})"
                               f" larger than dataframe of size ({len_sequence})")

        def _generator(sequence: pl.dataframe.DataFrame,
                       features: list[str],
                       target: list[str] | None,
                       sequence_length: int,
                       sequence_forecast: int):
            """
            A generator function to yield the next sequence.

            If ``sequence_forecast>=1`` then this generator returns a tuple ``(X, y)`` where
            `X` is a dataframe of shape `(sequence_length, len(features)`,
            `y` is a dataframe of shape `(sequence_forecast, len(target)`
            selected randomly from

            :param sequence: The time-series
            :param features: List of feature name
            :param target: List of targets (labels)
            :param sequence_length: Required length of a sequence
            :param sequence_forecast: Length of the target sequence
            :param chunk_size: Number of sequences in a chunk (aka batch)
            :param shuffle: Whether the data (rows) in a sequence should be shuffled
            :param infinite: Run the generator infinitely, i.e. requires functions that use this generator to define
                             a stopping criteria, e.g., steps_in_epoch

            """
            max_start_idx = len_sequence - (sequence_length + sequence_forecast) + 1
            use_target = target is not None

            # (Num_batches=1, Length of Sequences, Number of features)
            X = np.empty((sequence_length, len(features)), dtype=XDataFrame(sequence).dtype(features[0]).to_python())
            if target is not None:
                y = np.empty((sequence_forecast, len(target)), dtype=XDataFrame(sequence).dtype(target[0]).to_python())

            # Iterate through the windowed sequences until the index is exhausted
            for start_idx in range(max_start_idx):

                # Extract features
                X[:, :] = sequence[start_idx:start_idx + sequence_length][features].to_numpy()

                if use_target:
                    target_start_idx = start_idx + sequence_length
                    target_end_idx = target_start_idx + sequence_forecast
                    # Extract targets
                    y[:, :] = sequence[target_start_idx:target_end_idx][target].to_numpy()

                if use_target:
                    yield (X, y)
                else:
                    yield (X,)

        return _generator(sequence=sequence, features=features, target=target,
                          sequence_length=sequence_length, sequence_forecast=sequence_forecast,
                          )

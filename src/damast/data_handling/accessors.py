"""
Module for creating generators for accessing sequences of data from a DataFrame
"""

import logging
import random
import time
from typing import Any, List, Optional, Union

import keras.utils
import numpy as np
import pandas as pd
import polars as pl

from damast.core.types import DataFrame, XDataFrame

__all__ = [
    "GroupSequenceAccessor",
    "SequenceIterator"
]
logger = logging.getLogger("damast")


# https://www.tensorflow.org/tutorials/structured_data/time_series
class GroupSequenceAccessor:
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
                 sort_columns: List[str] = None,
                 timeout_in_s: int = DEFAULT_TIMEOUT_IN_S):
        self.df = df

        self.group_column = group_column
        self.groups = df.unique(group_column)

        if sort_columns is not None:
            self.sort_columns = sort_columns if type(sort_columns) == list else [sort_columns]
        else:
            self.sort_columns = sort_columns

        self.timeout_in_s = timeout_in_s

    def split_random(self, ratios: List[float]) -> List[List[Any]]:
        """
        Create ``N=len(ratios)`` groups of the dataframe, with given ratios, return the corresponding groups.

        The groups are based on :attr:`GroupSequenceAccessor.group_column`.

        :param ratios: List of relative partition sizes (will be normalized, so that all elements sum to 1
        :return: Following the ratios, returns lists of randomly sampled values from the id/group column
        """
        scaled_ratios = np.asarray(ratios) / sum(ratios)
        groups = self.groups[self.group_column].to_numpy().copy()

        random.shuffle(groups)
        number_of_groups = len(groups)

        partition_sizes = np.asarray(np.round(number_of_groups*scaled_ratios), dtype=int)
        assert (len(groups) == sum(partition_sizes))
        from_idx = 0
        partitions = []
        for ps in partition_sizes:
            to_idx = min(from_idx + ps, number_of_groups)
            partitions.append(groups[from_idx:to_idx])
            from_idx = to_idx

        return partitions

    def to_keras_generator(self, features: List[str],
                           target: List[str] = None,
                           groups: List[str] = None,
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
            current_level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            steps_per_epoch = np.ceil(len(self.df) / batch_size)
            logger.info(f'Recommended {steps_per_epoch=}')
            logger.setLevel(current_level)

        # Sanity checks before creating generator
        # Check that all features have the same data-type
        datatypes = [XDataFrame(self.df).dtype(f) for f in features]
        for dtype in datatypes:
            if dtype != datatypes[0]:
                raise ValueError(f"{self.__class__.__name__}:"
                                 f" Features {features} do not have a consistent (single) datatype,"
                                 f" got {datatypes}")

        use_target = target is not None
        if use_target:
            datatypes = [XDataFrame(self.df).dtype(t) for t in target]
            for dtype in datatypes:
                if dtype != datatypes[0]:
                    raise ValueError(f"{self.__class__.__name__}:"
                                     f" Targets {target} do not have a consistent (single) datatype,"
                                     f" got {datatypes}")

        if use_target:
            target = target if type(target) == list else [target]

            if sequence_forecast < 0:
                raise ValueError(f"{self.__class__.__name__}: Sequence forecast cannot be negative")
            if sequence_forecast == 0:
                raise ValueError(f"{self.__class__.__name__}: Cannot do extract targets with no sequence forecast")
        else:
            if sequence_forecast > 0:
                raise ValueError(f"{self.__class__.__name__}: Cannot do sequence forecast with no targets")

        if self.sort_columns is not None and shuffle:
            raise RuntimeError(f"{self.__class__.__name__}: Cannot sort and shuffle sequence at the same time")

        def _generator(features: List[str], target: Optional[List[str]],
                       groups: List[Any],
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
                target = target if type(target) == list else target
                all_columns += target
            else:
                # If no target, we are not forecasting
                sequence_forecast = 0
            all_columns = list(set(all_columns))

            if groups is None:
                groups = self.groups

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
                                    .filter(pl.col(self.group_column) == group[self.group_column])\
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

                X = np.array(chunk)
                if use_target:
                    if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
                        y = np.array(target_chunk)
                    else:
                        y = np.array(target_chunk, copy=False)
                    yield (X, y)
                else:
                    yield (X,)
                if not infinite:
                    break

        return _generator(features=features, target=target,
                          groups=groups, sequence_length=sequence_length, sequence_forecast=sequence_forecast,
                          chunk_size=batch_size, shuffle=shuffle, infinite=infinite)


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

    df: Union[DataFrame, pd.DataFrame]

    def __init__(self,
                 df: Union[DataFrame, pd.DataFrame],
                 sort_columns: List[str] = None):
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
        datatypes = [XDataFrame(self.df).dtype(f) for f in features]
        for dtype in datatypes:
            if dtype != datatypes[0]:
                raise ValueError(f"{self.__class__.__name__}:"
                                 f" Features {features} do not have a consistent (single) datatype,"
                                 f" got {datatypes}")

        use_target = target is not None
        if use_target:
            datatypes = [XDataFrame(self.df).dtype(t) for t in target]
            for dtype in datatypes:
                if dtype != datatypes[0]:
                    raise ValueError(f"{self.__class__.__name__}:"
                                     f" Targets {target} do not have a consistent (single) datatype,"
                                     f" got {datatypes}")

        if sequence_forecast < 0:
            raise ValueError(f"{self.__class__.__name__}: Sequence forecast cannot be negative")

        if use_target:
            target = target if type(target) == list else target
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
                       features: List[str],
                       target: Optional[List[str]],
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

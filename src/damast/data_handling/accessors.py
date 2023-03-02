import random
import time
from typing import Any, List, Optional

import keras.utils
import numpy as np
import pandas
import vaex
import vaex.ml.generate
import vaex.ml.state
import vaex.serialize
from tensorflow import keras as K
from vaex import DataFrame

__all__ = ["GroupSequenceAccessor"]


# https://www.tensorflow.org/tutorials/structured_data/time_series
class GroupSequenceAccessor:
    """
    A generator that allows access to a length-limited sequence of a particular group.

    The resulting dataset (X) will have a shape of:
        (<batch_size>, <sequence_length>, <number-of-features>).

    The resulting targets (y) if requires will lead to a label dataset of shape
        (<batch_size>, <sequence_length>, <number-of-targets>)

    Per default the dataset is assumed to be sorted, e.g., typically in time-based order.
    One can name however use 'sort_columns', to require sorting so that a sequence becomes a valid timeline.
    From the overall group-based sequence a random subsequence of given length is sampled.

    :param df: the dataframe from which the data (train, test, ...) shall be taken - this can be the combine dataframe
               for train, test, validate since the generator allows to limit the group ids from which will be selected
    :param group_column: the name of the column that identifies the group
    :param sort_columns: Names of the columns that shall be used for sorting - if None, no sorting will be done
    :param timeout_in_s: Searching for a sequence of a given length might fail, since the dataset might not contain data
                    of the given length. After timeout (in seconds) elapsed a RuntimeError will be thrown
    :raise RuntimeError: forward search might not find sufficient messages to create a sequence with the exact length.
    """
    DEFAULT_TIMEOUT_IN_S: float = 30.0

    def __init__(self,
                 df: DataFrame,
                 group_column: str,
                 sort_columns: str = None,
                 timeout_in_s: int = DEFAULT_TIMEOUT_IN_S):
        self.df = df

        self.group_column = group_column
        self.groups = df.unique(getattr(df, group_column))

        self.sort_columns = sort_columns
        self.timeout_in_s = timeout_in_s

    def split_random(self, ratios: List[float]) -> List[List[Any]]:
        """
        Split the
        :param ratios: List of relative partition sizes (will be normalized, so that all elements sum to 1
        :return: Following the ratios, returns lists of randomly sampled values from the id/group column
        """
        groups = self.groups.copy()
        random.shuffle(groups)
        number_of_groups = len(groups)

        partition_sizes = [int(x * number_of_groups / sum(ratios)) for x in ratios]

        from_idx = 0
        partitions = []
        for ps in partition_sizes:
            to_idx = min(from_idx + ps, number_of_groups)
            partitions.append(groups[from_idx:to_idx])
            from_idx = to_idx

        return partitions

    def to_keras_generator(self, features,
                           target=None,
                           groups: List[Any] = None,
                           sequence_length=50,
                           sequence_forecast=1,
                           batch_size=1024,
                           shuffle=False,
                           infinite=False,
                           verbose=True) -> keras.utils.Sequence:
        """
        Create a batch generator suitable as a Keras datasource.

        By default, the generator is infinite, i.e. it loops continuously over the data.
        Thus, you need to specify the "steps_per_epoch" arg when fitting a Keras model,
        the "validation_steps" when using it for validation, and "steps" when
         calling the "predict" method of a keras model.

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
        :param verbose: If True, show an info on the recommended "steps_per_epoch"
                        based on the total number of samples and "batch_size".

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
             train_generator = gsa.to_keras_generator(features=features, target=target, sequence_length=50, sequence_forecast=1, batch_size=10)

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
            steps_per_epoch = np.ceil(len(self.df) / batch_size)
            print(f'Recommended "steps_per_epoch" arg: {steps_per_epoch}')

        def _generator(features: List[str], target: Optional[List[str]],
                       groups: List[Any],
                       sequence_length: int, sequence_forecast: int,
                       chunk_size: int, shuffle: bool, infinite: bool,
                       ):
            """
            A generator function to yield the next sequence:

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
            if target is not None:
                target = vaex.utils._ensure_list(target)
                target = vaex.utils._ensure_strings_from_expressions(target)

            if groups is None:
                groups = self.groups

            while True:
                chunk = []
                target_chunk = []
                for i in range(0, chunk_size):
                    sequence: pandas.DataFrame = None
                    start_time = time.time()
                    sample_count = 0
                    sample_length = 0

                    # Find a valid subsequence, i.e. one with the given length
                    while not (time.time() - start_time) > self.timeout_in_s:
                        group = random.choice(groups)
                        # Since we will need the timeline later - we further deal with pandas DataFrame
                        # directly - thus, we do not use a copy of the vaex DataFrame
                        sequence = self.df[getattr(self.df, self.group_column) == group].to_pandas_df()
                        # If sort columns are set, then ensure that the sorting is done
                        if self.sort_columns is not None:
                            sequence.sort_values(by=self.sort_columns, inplace=True, ignore_index=True)
                        elif shuffle:
                            sequence = sequence.sample(frac=1)

                        len_sequence = sequence.shape[0]

                        if len_sequence >= (sequence_length + sequence_forecast):
                            break

                        sample_count += 1
                        sample_length += len_sequence

                    if sequence is None or len_sequence < (sequence_length + sequence_forecast):
                        raise RuntimeError("GroupTimelineAccessor: could not identify a sufficiently long sequence"
                                           f"within given timeout of {self.timeout_in_s}:"
                                           f" mean length was {sample_length / sample_count}")

                    max_start_idx = len(sequence) - (sequence_length + sequence_forecast)
                    if max_start_idx == 0:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, max_start_idx)

                    sequence_window = sequence[start_idx:start_idx + sequence_length]
                    chunk.append(sequence_window[features].to_numpy())

                    if target is not None:
                        target_start_idx = start_idx + sequence_length
                        target_end_idx = target_start_idx + sequence_forecast

                        target_window = sequence[target_start_idx:target_end_idx][target]
                        if sequence_forecast == 1:
                            target_window = target_window.iloc[0]

                        # target it the last step in the timeline, so the last
                        target_chunk.append(target_window.to_numpy())

                X = np.array(chunk)
                if target is not None:
                    y = np.array(target_chunk, copy=False)
                    yield (X, y)
                else:
                    yield (X,)

                if not infinite:
                    break

        return _generator(features=features, target=target,
                          groups=groups, sequence_length=sequence_length, sequence_forecast=sequence_forecast,
                          chunk_size=batch_size, shuffle=shuffle, infinite=infinite)

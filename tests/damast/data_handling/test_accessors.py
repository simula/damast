import numpy as np
import pandas as pd
import pytest
import vaex

from damast.data_handling.accessors import GroupSequenceAccessor


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

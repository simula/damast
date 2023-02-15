import numpy as np
import pandas as pd

from damast.data_handling.transformers.features import DeltaTime


def test_delta_time():
    data = [
        ["0", 0],
        ["2", 2],
        ["1", 1],
    ]

    column_names = ["id", "timestamp"]
    df = pd.DataFrame(data, columns=column_names)

    transformer = DeltaTime(name="delta_time",
                            timestamp_column="timestamp")

    transformed_df = transformer.fit_transform(df)

    assert "delta_time" in transformed_df
    assert np.isnan(transformed_df.loc[0, "delta_time"])
    assert transformed_df.loc[1, "delta_time"] == 2.0
    assert transformed_df.loc[2, "delta_time"] == -1.0

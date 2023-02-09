import numpy as np
import numpy.typing as npt
import pandas
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from damast.data_handling.transformers import (CyclicDenormalizer,
                                               CyclicNormalizer,
                                               MinMaxNormalizer, LogNormalizer)
from damast.domains.maritime.math.normalization import normalize


def cyclic_normalisation(x: npt.NDArray[np.float64],
                         x_min: float,
                         x_max: float) -> npt.NDArray[np.float64]:
    """ Returns the data cycli-normalised between 0 and 1 """
    return np.vstack(
        [
            np.sin(2 * np.pi * normalize(x, x_min, x_max, 0, 1)),
            np.cos(2 * np.pi * normalize(x, x_min, x_max, 0, 1))
        ]
    ).T


@pytest.mark.parametrize("transform", ["default", "pandas"])
@pytest.mark.parametrize("i", [0, 1])
def test_cyclic_normalizer(transform, i):
    x = np.arange(60, dtype=np.float64).reshape(-1, 2)
    X = pandas.DataFrame(x, columns=["Column0", "Column1"])
    col = f"Column{i}"
    c_norm = ColumnTransformer([("CyclicColumn", CyclicNormalizer(0, 60), [col])],
                               verbose_feature_names_out=False).set_output(transform=transform)
    output_1 = c_norm.fit_transform(X)
    assert np.allclose(cyclic_normalisation(x[:, i], 0, 60), output_1)

    if transform == "pandas":
        c_denorm = ColumnTransformer([("DeCyclicColumn", CyclicDenormalizer(0, 60),
                                       ["cyclicnormalizer0", "cyclicnormalizer1"])],
                                     verbose_feature_names_out=False).set_output(transform=transform)
    else:
        c_denorm = ColumnTransformer([("DeCyclicColumn", CyclicDenormalizer(0, 60),
                                       [0, 1])],
                                     verbose_feature_names_out=False).set_output(transform=transform)

    output_2 = c_denorm.fit_transform(output_1)

    if transform == "pandas":
        assert np.allclose(output_2["cyclicdenormalizer0"].to_numpy(),
                           X[col].to_numpy())
    else:

        assert np.allclose(X[col].to_numpy(), output_2.reshape(-1))

    # test Pipeline
    pipeline = Pipeline(steps=[("normalize", c_norm), ("denormalize", c_denorm)])
    y = pipeline.fit_transform(X)
    if transform == "pandas":
        assert np.allclose(y["cyclicdenormalizer0"].to_numpy(),
                           X[col].to_numpy())
    else:

        assert np.allclose(X[col].to_numpy(), y.reshape(-1))


@pytest.mark.parametrize("transform", ["default", "pandas"])
@pytest.mark.parametrize("i", [0, 1])
def test_normalizer(i, transform):
    x = np.linspace(0, 250, 88, dtype=np.float64).reshape(22, 4)
    cols = ["x", "y", "z", "t"]
    X = pandas.DataFrame(x, columns=cols)
    cf = ColumnTransformer(
        [("normalized", MinMaxNormalizer(0, 300, -1, 1), [cols[i]])]).set_output(transform=transform)
    y = cf.fit_transform(X)

    y_ex = normalize(x[:, i], 0, 300, -1, 1)
    if transform == "pandas":
        assert np.allclose(y[f"normalized__{cols[i]}"].to_numpy(),
                           y_ex)
    else:

        assert np.allclose(y_ex, y.reshape(-1))


def test_log_normalisation():
    data = [
        [0, 20],
        [10, -20]
    ]
    column_names = ["a", "b"]
    df = pd.DataFrame(data, columns=column_names)

    log_normalisation = LogNormalizer(column_names=column_names)
    transformed_df = log_normalisation.fit_transform(X=df)

    def normalise_ge_zero(value):
        return np.log1p(value)

    def normalise_lt_zero(value):
        return np.negative(np.log1p(np.abs(value)))

    assert transformed_df.loc[0, "a"] == normalise_ge_zero(0)
    assert transformed_df.loc[1, "a"] == normalise_ge_zero(10)
    assert transformed_df.loc[0, "b"] == normalise_ge_zero(20)
    assert transformed_df.loc[1, "b"] == normalise_lt_zero(-20)

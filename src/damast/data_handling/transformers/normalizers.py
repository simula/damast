from typing import List

import numpy as np
import numpy.typing as npt
import pandas
import pandas as pd
from numba import njit
from sklearn.base import ClassNamePrefixFeaturesOutMixin, OneToOneFeatureMixin

from damast.data_handling.transformers.base import BaseTransformer
from .augmenters import BaseAugmenter

__all__ = ["Normalizer", "CyclicNormalizer", "CyclicDenormalizer", "normalize"]


@njit
def normalize(
        x: npt.NDArray[np.float64], x_min: float, x_max: float, a: float, b: float
) -> npt.NDArray[np.float64]:
    """
    Normalize data in array `x` with lower bound `x_min` and upper bound `x_max`
    to be in the range `[a, b]`

    .. math::

        x_n = (b-a)\\frac{x-x_{min}}{x_{max}-x_{min}} + a

    :param x: Input array
    :param x_min: Minimum bound of input data
    :param x_max: Maximum bound of input data
    :param a: Minimum bound of output data
    :param b: Maximum bound of output data

    :returns: Normalized data
    """
    return (b - a) * (x - x_min) / (x_max - x_min) + a


class Normalizer(OneToOneFeatureMixin, BaseTransformer):
    """
    Normalize data in range `[X_min, X_max]` to output range `[y_min, y_max]`.
    """

    X_min: float
    X_max: float
    y_min: float
    y_max: float

    def __init__(self, X_min: float, X_max: float, y_min: float, y_max: float):
        self.X_min = X_min
        self.X_max = X_max
        self.y_min = y_min
        self.y_max = y_max

        assert self.X_min < self.X_max
        super().__init__()

    def transform(self, X):
        if isinstance(X, pandas.DataFrame):
            X_np = X.to_numpy()
        else:
            X_np = X
        assert self.X_min <= np.min(X_np)
        assert np.max(X_np) <= self.X_max
        output = normalize(X_np, self.X_min, self.X_max, self.y_min, self.y_max)
        return output


class CyclicNormalizer(BaseAugmenter, ClassNamePrefixFeaturesOutMixin):
    """
    Normalize data `X` in range `[X_min, X_max]` to cyclic coordinates

    ..math::

        X_n = normalize(X, X_min, X_max, 0, 1)
        y = (\\sin(2 \\pi X_n), \\cos(2 \\pi X_n))


    Args:
        Normalizer (_type_): _description_
    """

    X_min: float
    X_max: float

    def __init__(self, X_min: float, X_max: float):
        self.X_min = X_min
        self.X_max = X_max
        self.y_min = 0
        self.y_max = 1

    def fit(self, X, y=None):
        self._n_features_out = 2
        return self

    def transform(self, X):
        X_norm = normalize(X.to_numpy(), self.X_min, self.X_max, self.y_min, self.y_max)
        X_sin = np.sin(2 * np.pi * X_norm)
        X_cos = np.cos(2 * np.pi * X_norm)
        return np.hstack([X_sin, X_cos])


class CyclicDenormalizer(BaseAugmenter, ClassNamePrefixFeaturesOutMixin):
    """
    Transform cyclic data (X_sin, X_cos) to non-cyclic data between `[y_min, y_max]`.
    """

    y_min: float
    y_max: float

    def __init__(self, y_min: float, y_max: float):
        self.y_min = y_min
        self.y_max = y_max

    def fit(self, X, y=None):
        self._n_features_out = 1
        return self

    def transform(self, X):
        if isinstance(X, pandas.DataFrame):
            X_np = X.to_numpy()
        else:
            X_np = X
        # Split input
        X_sin = X_np[:, 0]
        X_cos = X_np[:, 1]
        # Check validity of input
        assert -1 <= np.min(X_sin) and np.max(X_sin) <= 1
        assert -1 <= np.min(X_cos) and np.max(X_cos) <= 1

        # Convert input from [0, 2pi]  to -pi, pi
        X_sin_shft = np.sin(np.arcsin(X_sin) - np.pi)
        X_cos_shft = np.cos(np.arccos(X_cos) - np.pi)
        X_hat_normed = np.where(
            X_sin_shft > 0, np.arccos(X_cos_shft), -np.arccos(X_cos_shft)
        )
        data = normalize(
            normalize(X_hat_normed, -np.pi, np.pi, 0, 1), 0, 1, self.y_min, self.y_max
        )
        return data.reshape(-1, 1)


class LogNormalisation(Normalizer):
    #: Column that should be normalised
    column_names: List[str] = None

    def __init__(self,
                 column_names: List[str]):
        super().__init__()

        self.column_names = column_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().transform(df)

        # When condition holds, then yield x, otherwise y
        tf_df = df[self.column_names]
        conditioned_df = np.log1p(np.abs(tf_df))
        df[self.column_names] = conditioned_df.where(cond=tf_df >= 0.0,
                                                     other=np.negative(np.log1p(np.abs(tf_df))))
        return df

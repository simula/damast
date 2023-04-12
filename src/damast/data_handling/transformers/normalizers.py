"""
Module containing (de)normalisation functions
"""
import numpy as np
import numpy.typing as npt
from numba import njit

__all__ = [
    "cyclic_denormalisation",
    "cyclic_normalisation",
    "normalize"
]


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


def cyclic_normalisation(x: npt.NDArray[np.float64],
                         x_min: float,
                         x_max: float) -> npt.NDArray[np.float64]:
    """ Returns the data cycli-normalised between 0 and 1 """
    return np.array(
        [
            np.sin(2 * np.pi * normalize(x, x_min, x_max, 0, 1)),
            np.cos(2 * np.pi * normalize(x, x_min, x_max, 0, 1))
        ]
    )


def cyclic_denormalisation(x_sin: npt.NDArray[np.float64],
                           x_cos: npt.NDArray[np.float64],
                           min: float,
                           max: float) -> npt.NDArray[np.float64]:
    """ Returns the data cycli-denormalised"""
    # Flatten the input
    original_shape = x_sin.shape
    x_sin = x_sin.flatten()
    x_cos = x_cos.flatten()

    # Limit the min and max
    x_sin = np.clip(x_sin, -1, 1)
    x_cos = np.clip(x_cos, -1, 1)

    # Convert input between 0, 2PI -> -PI, PI
    x_sin = np.sin(np.arcsin(x_sin) - np.pi)
    x_cos = np.cos(np.arccos(x_cos) - np.pi)

    # Denormalization
    x_hat_normed = np.where(x_sin > 0, np.arccos(x_cos), - np.arccos(x_cos))
    hat = normalize(normalize(x_hat_normed, -np.pi, np.pi, 0, 1), 0, 1, min, max)

    # Reshape the output as the input
    return hat.reshape(original_shape)

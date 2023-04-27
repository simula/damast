import warnings

import numpy as np
import numpy.typing as npt

__all__ = [
    "N_sigma",
    "N_sigma_limited"
]


def N_sigma_limited(x: npt.NDArray[np.float64],
                    N: int,
                    p: float = 0.99) -> npt.NDArray[np.float64]:
    """ Compute and return the N_sigma given a list without considering the
    :math:`100\\cdot(1-p)\\%` highest values

    :param x: The input array
    :param N: The number of standard deviations to add to the median
    :param p: The threshold (0 to 1) of values to include from the sorted array
    :returns: The median of the truncated input plus `N` times the standard deviation

    """
    warnings.warn("It is uncertain if this function gives you any sensible quantity")
    limit = x.quantile(p)
    return x.loc[x < limit].median() + N * x.loc[x < limit].std()


def N_sigma(x: npt.NDArray[np.float64], N: int) -> npt.NDArray[np.float64]:
    """ Compute and return the N_sigma given a list
        :param x: The input array

    :param N: The number of standard deviations to add to the median
    :returns: The median of the input `x` plus `N` times the standard deviation
    """
    warnings.warn("It is uncertain if this function gives you any sensible quantity")
    return x.median() + N * x.std()

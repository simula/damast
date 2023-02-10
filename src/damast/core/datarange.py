from typing import Any

__all__ = [
    "DataRange",
    "MinMax",
    "CyclicMinMax"
]


class DataRange:
    pass


class MinMax(DataRange):
    min: Any = None
    max: Any = None

    def __init__(self,
                 min: Any,
                 max: Any):
        super().__init__()
        """
        Initialise the min and max range.

        :param min: minimum allowed value, data must be greater or equal
        :param max: maximum allowed value, data must be less or equal
        """
        assert min < max
        self.min = min
        self.max = max

    def is_in_range(self, value: Any) -> bool:
        """
        Check if a value is in the defined range.

        :param value: data value
        :return: True if value is in the set range, false otherwise
        """
        return self.min <= value <= self.max

    def __contains__(self, value):
        """
        Check if value lies in the given range via:

        >>> custom_range = MinMax(0.0, 10.0)
        >>> if 1.0 in custom_range:
        >>>    ...

        :param value: Value to test
        :return:
        """

        return self.is_in_range(value=value)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.min}, {self.max}]"


class CyclicMinMax(MinMax):
    """
    Define a cyclic min max range
    """

    def __init__(self,
                 min: Any,
                 max: Any):
        super().__init__(min=min, max=max)

import importlib
from abc import abstractmethod, ABC
from typing import Any, List, Dict

__all__ = [
    "CyclicMinMax",
    "DataRange",
    "ListOfValues",
    "MinMax"
]


class DataRange(ABC):
    """
    Representation of a data range.

    This class should be subclassed to implement a custom data range
    """
    @abstractmethod
    def is_in_range(self, value) -> bool:
        """
        Check if a value lies in the permitted range.

        :param value: Value to check
        :return: True if value lies in range, False otherwise
        """
        pass

    def __contains__(self, value):
        """
        Check if value lies in the given range via 'in':

        >>> custom_range = MinMax(0.0, 10.0)
        >>> if 1.0 in custom_range:
        >>>    ...

        :param value: Value to test
        :return:
        """

        return self.is_in_range(value=value)

    def to_dict(self) -> str:
        """
        Convert object to dictionary to allow plain type serialisation.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.to_dict not implemented")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dtype: Any = None):
        """
        Load the data range from a plain type description in a dictionary.

        >>> min_max_range = DataRange.from_dict({ "MinMax": { "min": 0, "max": 1 }}, dtype=float)

        :param data: dictionary data
        :param dtype: the value type to use for initialisation,
        :return:
        """
        for klass, data in data.items():
            try:
                datarange_m = importlib.import_module("damast.core.datarange")
                datarange_subclass = getattr(datarange_m, klass)
                return datarange_subclass.from_data(data=data, dtype=dtype)
            except ImportError as ie:
                raise ValueError(f"DataRange.from_dict: unknown range definition '{klass}'")


class ListOfValues:
    values: List[Any] = None

    def __init__(self, values: List[Any]):
        if not isinstance(values, list):
            raise ValueError(f"{self.__class__.__name__}.__init__: required list of values for initialisation")

        self.values = values

    def is_in_range(self, value) -> bool:
        return value in self.values

    def __eq__(self, other) -> bool:
        if self.__class__ != other.__class__:
            return False

        if self.values != other.values:
            return False

        return True

    @classmethod
    def from_data(cls,
                  data: List[Any],
                  dtype: Any):
        if len(data) > 0 and dtype is not None:
            actual_dtype = type(data[0])
            if actual_dtype != dtype:
                raise ValueError(f"{cls.__name__}.from_data: expected list of {dtype.__name__}, but received"
                                 f" {actual_dtype}")
        return cls(values=data)

    def to_dict(self) -> Dict[str, Any]:
        return {self.__class__.__name__: self.values}


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

    @classmethod
    def from_data(cls,
                  data: Dict[str, Any],
                  dtype: Any):
        """
        Load the MinMax range from the given dictionary specification

        >>> min_max_range = MinMax.from_data({"min": 0, "max": 1}, dtype=float)

        :param data: The dictionary to describe the range
        :param dtype: value type that should be used to initialse this min max range
        :return: MinMax instance
        """
        for required_key in ["min", "max"]:
            if required_key not in data:
                raise KeyError(f"{cls.__name__}.from_data: missing '{required_key}'")

        if dtype is not None:
            return cls(min=dtype(data["min"]), max=dtype(data["max"]))

        return cls(min=data["min"], max=data["max"])

    def __eq__(self, other) -> bool:
        if self.__class__ != other.__class__:
            return False

        if self.min != other.min:
            return False

        if self.max != other.max:
            return False

        return True

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.min}, {self.max}]"

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {self.__class__.__name__: {"min": self.min, "max": self.max}}


class CyclicMinMax(MinMax):
    """
    Define a cyclic min max range
    """

    def __init__(self,
                 min: Any,
                 max: Any):
        super().__init__(min=min, max=max)

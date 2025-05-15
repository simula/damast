"""
This module contains data range definitions.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

__all__ = ["CyclicMinMax", "DataElement", "DataRange", "ListOfValues", "MinMax"]


class DataElement:
    """
    Wrapper class to create an instance of an object given by a dtype
    """

    @classmethod
    def create(cls, value, dtype):
        """
        Create a DataElement from a given value and d(ata)type

        :param value: value of the DataElement
        :param dtype: datatype of the value
        """
        if isinstance(dtype, pl.datatypes.classes.DataTypeClass):
            dtype = dtype.to_python()
        elif isinstance(dtype, pl.datatypes.Datetime):
            if type(value) == str:
                value = dt.datetime.fromisoformat(value)
            return dtype.to_python().fromtimestamp(value.timestamp())

        return dtype(value)


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

    def is_not_in_range(self, value) -> bool:
        return not self.is_in_range(value)

    def __contains__(self, value) -> bool:
        """
        Check if value lies in the given range via 'in':

        .. highlight:: python
        .. code-block:: python

            custom_range = MinMax(0.0, 10.0)
            if 1.0 in custom_range:
                ...

        :param value: Value to test
        :return: True if value is in range, False otherwise
        """

        return self.is_in_range(value=value)

    @abstractmethod
    def to_dict(self) -> str:
        """
        Convert object to dictionary to allow plain type serialisation.

        :raise NotImplementedError: If method has not been implemented by a subclass
        """
        return dict(self)

    def __iter__(self):
        """
        Convert object to dictionary to allow plain type serialisation.

        :raise NotImplementedError: If method has not been implemented by a subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__iter__ not implemented")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dtype: Any = None) -> DataRange:
        """
        Load the data range from a plain type description in a dictionary.

        .. highlight:: python
        .. code-block:: python

            min_max_range = DataRange.from_dict({ "MinMax": { "min": 0, "max": 1 }}, dtype=float)

        :param data: dictionary data
        :param dtype: the value type to use for initialisation,
        :return: The newly created class
        :raise ValueError: Raises if the requested range type is not known
        """
        assert len(data.keys()) == 1
        klass, values = data.popitem()
        try:
            import damast
            datarange_subclass = getattr(damast.core.data_description, klass)
            return datarange_subclass.from_data(data=values, dtype=dtype)
        except AttributeError as e:
            msg = f"DataRange.from_dict: unknown range definition '{klass}' -- {e}"
            raise ValueError(msg) from e


class ListOfValues:
    """
    Represent a list of values.
    """

    #: Values in this list
    values: List[Any]

    def __init__(self, values: List[Any]):
        """
        Initialise ListOfValue

        :param values: values that define this list
        :raise ValueError: Raises if values is not a list.
        """
        if not isinstance(values, list):
            raise ValueError(
                f"{self.__class__.__name__}.__init__: required list of values for initialisation, got {type(values)}"
            )

        self.values = values

    def is_in_range(self, value) -> bool:
        """
        Check if value can be found in the list of values.

        :param value: Value to test
        :return: True if value is in the list, false otherwise
        """
        return value in self.values

    def is_not_in_range(self, value) -> bool:
        return not self.is_in_range(value)

    def __eq__(self, other) -> bool:
        """
        Check equality based on the 'values' property

        :param other: Other object
        :return: True if object are considered equal.
        """
        if self.__class__ != other.__class__:
            return False

        if self.values != other.values:
            return False

        return True

    @classmethod
    def from_data(cls, data: List[Any], dtype: Any) -> ListOfValues:
        """
        Create an instance from data and given datatype (dtype)

        :param data: The actual list of values
        :param dtype: Datatype of values in the list
        :return: instance
        :raise ValueError: If dtype is given, but does not match the value type in the given list of values
        """
        if data and dtype is not None:
            for sample in data:
                if sample is not None:
                    break

            actual_dtype = type(sample)
            if actual_dtype != dtype:
                raise ValueError(
                    f"{cls.__name__}.from_data: expected list of {dtype.__name__}, but received"
                    f" {actual_dtype}, {data=}"
                )
        return cls(values=data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representing this object

        :return: dictionary
        """
        return dict(self)

    def __iter__(self):
        yield self.__class__.__name__, self.values


class MinMax(DataRange):
    """
    A Minimum-Maximum Range Definition

    :param min: minimum allowed value, data must be greater or equal
    :param max: maximum allowed value, data must be less or equal
    """

    #: Minimum / Lower Bound of range
    min: Any
    #: Maximum / Upper Bound of range
    max: Any

    #: Allow missing values for this data range
    allow_missing: bool

    def __init__(self, min: Any, max: Any, allow_missing: bool = True):
        """
        Constructor
        """
        super().__init__()

        if not min <= max:
            raise RuntimeError(f"DataRange.__init__: invalid range - min: {min} max: {max}")

        self.min = min
        self.max = max

        self.allow_missing = allow_missing

    def is_in_range(self, value: Any) -> bool:
        """
        Check if a value is in the defined range.

        :param value: data value

        :return: `True` if value is in the set range, `False` otherwise
        """
        # To make masked/missing values work
        try:
            if value is None:
                return self.allow_missing

            if np.isnan(value) or np.isnat(value):
                return self.allow_missing
        except Exception:
            pass

        return self.min <= value <= self.max

    @classmethod
    def from_data(cls, data: Dict[str, Any], dtype: Any) -> MinMax:
        """
        Load the MinMax range from the given dictionary specification

        .. highlight:: python
        .. code-block:: python

            min_max_range = MinMax.from_data({"min": 0, "max": 1}, dtype=float)

        :param data: The dictionary to describe the range
        :param dtype: value type that should be used to initialise this min max range
        :return: A :class:`MinMax` instance
        :raise KeyError: Raises if min/max keys are missing in the dictionary
        """
        for required_key in ["min", "max"]:
            if required_key not in data:
                raise KeyError(f"{cls.__name__}.from_data: missing '{required_key}'")

        min_value = data["min"]
        max_value = data["max"]

        if dtype is not None:
            min_value = DataElement.create(data["min"], dtype)
            max_value = DataElement.create(data["max"], dtype)

        keys = {
            'min': min_value,
            'max': max_value
        }

        if "allow_missing" in data:
            keys['allow_missing'] = data['allow_missing']

        return cls(**keys)

    def __eq__(self, other) -> bool:
        """
        Check equality based on min/max properties

        :param other: Other object
        :return: `True` if objects are considered the same, `False` otherwise
        """
        if self.__class__ != other.__class__:
            return False

        if self.min != other.min:
            return False

        if self.max != other.max:
            return False

        return True

    def __repr__(self) -> str:
        """
        Create representation for this object:

        :return: Representation for this instance
        """
        return f"{self.__class__.__name__}[{self.min}, {self.max}]"

    def merge(self, other: MinMax):
        """
        Extend the range based on another range definition.

        If any of the DataRange instances does not allow missing value, the resulting instance
        will as not allow missing values.

        :param other: MinMax object to extend the bound of the current one
        """
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)

        # Since allow missing is by default true - explicitly disabling it should
        # be propagated when merging
        self.allow_missing = self.allow_missing and other.allow_missing

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Create a dictionary representing this object

        :return: dictionary
        """
        return dict(self)

    def __iter__(self):
        yield self.__class__.__name__, {"min": self.min, "max": self.max, "allow_missing": self.allow_missing}


class CyclicMinMax(MinMax):
    """
    Define a cyclic min max range
    """

class NumericValueStats(BaseModel):
    mean: float = 0.0
    stddev: float = 0.0
    total_count: int  = 0
    null_count: int

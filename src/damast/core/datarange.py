"""
This module contains data range definitions.
"""
import warnings
from .data_description import (
    CyclicMinMax,
    DataRange,
    DataElement,
    ListOfValues,
    MinMax
)

__all__ = ["CyclicMinMax", "DataElement", "DataRange", "ListOfValues", "MinMax"]

warnings.warn("damast.core.datarange has been deprecated and will be remove in future versions. Use damast.core.data_description instead")


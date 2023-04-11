"""
Module which collect all filters that filter the existing data.

"""

from typing import Any

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement

__all__ = [
    "RemoveValueRows",
    "FilterWithin"
]


class RemoveValueRows(PipelineElement):
    """
    Remove rows that do not have a defined value for a given column
    :param remove_value: remove rows with this value.
    """
    _remove_value: Any
    _inplace: bool

    def __init__(self, remove_value: Any, inplace: bool = False):
        super().__init__()
        self._inplace = inplace
        self._remove_value = remove_value

    @property
    def remove_value(self):
        return self._remove_value

    @damast.core.describe("Remove rows where a column has a specific value")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Delete rows with remove_values
        """
        mapped_name = self.get_name("x")
        new_dataframe = df._dataframe[(df._dataframe[mapped_name] != self._remove_value)]
        if self._inplace:
            df._dataframe = new_dataframe
            return df
        else:
            metadata = df._metadata.columns.copy()
            return AnnotatedDataFrame(new_dataframe, metadata=damast.core.MetaData(
                metadata))


class DropMissing(PipelineElement):
    """
    DropMissing rows that do not have a defined value for a given column

    :param inplace: If True drop changes in input dataframe, else create a
        new :class:`damast.core.dataframe.DataFrame`.
    """
    _inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self._inplace = inplace

    @damast.core.describe("Drop rows where a column has a missing value")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Drop rows with missing value
        """
        mapped_name = self.get_name("x")
        if self._inplace:
            dataframe = df._dataframe
        else:
            dataframe = df._dataframe.copy()
        new_dataframe = dataframe.dropmissing(column_names=[mapped_name])
        if self._inplace:
            df._dataframe = new_dataframe
            return df
        else:
            metadata = df._metadata.columns.copy()
            return AnnotatedDataFrame(new_dataframe, metadata=damast.core.MetaData(
                metadata))


class FilterWithin(PipelineElement):
    """
    Filter rows and keep those within given values
    :param within_values: list of values to keeps
    :param inplace: If True drop changes in input dataframe, else create a
        new :class:`damast.core.dataframe.DataFrame`.
    """
    _within_values: Any
    _inplace: bool

    def __init__(self, within_values: Any, inplace: bool = False):
        super().__init__()
        self._within_values = within_values
        self._inplace = inplace

    @property
    def within_values(self):
        return self._within_values

    @damast.core.describe("Filter rows and keep those within given values")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Filter rows and keep those within given values
        """
        mapped_name = self.get_name("x")
        if self._inplace:
            dataframe = df._dataframe
        else:
            dataframe = df._dataframe.copy()

        new_dataframe = dataframe[dataframe[mapped_name].isin(self._within_values)]
        if self._inplace:
            df._dataframe = new_dataframe
            return df
        else:
            metadata = df._metadata.columns.copy()
            return AnnotatedDataFrame(new_dataframe, metadata=damast.core.MetaData(
                metadata))

        return df

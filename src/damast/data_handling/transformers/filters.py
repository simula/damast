"""
Module which collect all filters that filter the existing data.
"""

from typing import Any

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement

__all__ = [
    "FilterWithin",
    "RemoveValueRows"
]


class RemoveValueRows(PipelineElement):
    """
    Remove rows that do not have a defined value for a given column.

    :param remove_value: remove rows with this value.
    """
    _remove_value: Any

    def __init__(self, remove_value: Any):
        super().__init__()
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
        df._dataframe = new_dataframe
        return df


class DropMissing(PipelineElement):
    """
    DropMissing rows that do not have a defined value or NaN for a given column.
    """

    @damast.core.describe("Drop rows where a column has a missing value")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Drop rows with missing value
        """
        mapped_name = self.get_name("x")
        dataframe = df._dataframe
        nonnan_dataframe = dataframe.dropnan(column_names=[mapped_name])
        new_dataframe = nonnan_dataframe.dropmissing(column_names=[mapped_name])
        df._dataframe = new_dataframe
        return df


class FilterWithin(PipelineElement):
    """
    Filter rows and keep those within given values.

    :param within_values: list of values to keep
    """
    _within_values: Any

    def __init__(self, within_values: Any):
        super().__init__()
        self._within_values = within_values

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
        dataframe = df._dataframe
        new_dataframe = dataframe[dataframe[mapped_name].isin(self._within_values)]
        df._dataframe = new_dataframe
        return df

from __future__ import annotations

from typing import ClassVar

import numpy as np
import polars
from polars import LazyFrame


class Meta(type):
    _base_impl: ClassVar["str"] = "polars"

    def __getattr__(cls, attr_name):
        if cls._base_impl == 'polars':
            return getattr(polars, attr_name)

        raise AttributeError(f"'{cls.__name__}' has not attribute '{attr_name}'")


class PolarsDataFrame(metaclass=Meta):
    _dataframe: LazyFrame

    def __init__(self, df: LazyFrame | polars.DataFrame):
        if type(df) == polars.DataFrame:
            self._dataframe = df.lazy()
        else:
            self._dataframe = df

    @property
    def dataframe(self) -> PolarsDataFrame:
        """
        Allows to access the underlying dataframe directly.

        .. note::
            AnnotatedDataFrame behaves like a ``polars.LazyDataFrame``, so typically you will not need to access the
            dataframe through this property.

        :return: The underlying dataframe
        """
        return PolarsDataFrame(self._dataframe)


    def __getitem__(self, column_name: str):
        """
        Make dataframe subscriptable and behave like the :class:`vaex.DataFrame`.

        :param item: Name of the key when using [] operators
        :return: item/column from the underlying vaex.dataframe
        """
        return self._dataframe.select(column_name)

    @property
    def column_names(self) -> list[str]:
        return self._dataframe.collect_schema().names()

    def dtype(self, column_name: str) -> polars.datatypes.DataType:
        idx = self.column_names.index(column_name)
        return self._dataframe.collect_schema().dtypes()[idx]

    def minmax(self, column_name: str):
        min_value = self._dataframe.select(column_name).min().collect()[0,0]
        max_value = self._dataframe.select(column_name).max().collect()[0,0]

        return min_value, max_value

    def set_dtype(self, column_name, representation_type) -> PolarsDataFrame:
        if representation_type == np.int64:
            representation_type = polars.Int64

        self._df = self._dataframe.with_columns(polars.col(column_name).cast(representation_type).alias(column_name))
        return self

    def __getattr__(self, attr_name):
        """
        Ensure that this object behaves like a :class:`polars.LazyFrame`.

        :param attr_name: Attribute / Name of column
        :return: The column data
        """
        # allow dataframe.col_one
        if attr_name in self.column_names:
            return self._dataframe.select(attr_name)

        if attr_name in ["__setstate__", "__getstate__"]:
            raise AttributeError(f"{self.__class__.__name__}.__getattr__: {attr_name} does not exist")

        """ Called for failed attribute accesses so forwarding to underlying polars frame """
        return getattr(self._dataframe, attr_name)

    def __setitem__(self, key, values):
        """
        Set the column for the annotated dataframe, and allow to behave like the polars.Dataframe.

        :param key: Column name
        :param value: Value to set the column to
        """
        if type(values) == polars.LazyFrame:
            values = values.collect().to_numpy()

        self._dataframe = self._dataframe.with_columns(
                    polars.Series(
                        name=key,
                        values=values
                    )
                )

    def __len__(self) -> int:
        """
        Get the length of the (underlying) dataframe.

        :return: Length of the dataframe
        """
        return len(self._dataframe.collect())

    def equals(self, other: PolarsDataFrame) -> bool:
        return self._dataframe.collect().equals(other._dataframe.collect())

    def open(filename: str | Path, sep = ',') -> DataFrame:
        if file_path.suffix == ".csv":
            return polars.scan_csv(file_path, sep=sep)

        raise ValueError(f"{cls.__name__}.load_data: Unsupported input file format {file_path.suffix}")




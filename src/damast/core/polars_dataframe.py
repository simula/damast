from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars
import polars.api
from polars import LazyFrame

VAEX_HDF5_ROOT: str = "/table"
VAEX_HDF5_COLUMNS: str = f"{VAEX_HDF5_ROOT}/columns"

class Meta(type):
    _base_impl: ClassVar["str"] = "polars"

    def __getattr__(cls, attr_name):
        if cls._base_impl == 'polars':
            return getattr(polars, attr_name)

        raise AttributeError(f"'{cls.__name__}' has not attribute '{attr_name}'")

@polars.api.register_dataframe_namespace("compat")
@polars.api.register_lazyframe_namespace("compat")
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
            AnnotatedDataFrame behaves like a ``polars.LazyFrame``, so typically you will not need to access the
            dataframe through this property.

        :return: The underlying dataframe
        """
        return PolarsDataFrame(self._dataframe)


    def __getitem__(self, column_name: str):
        """
        Make dataframe subscriptable and behave more like the :class:`pandas.DataFrame`.

        :param item: Name of the key when using [] operators
        :return: item/column from the underlying vaex.dataframe
        """
        return self._dataframe.select(column_name)

    @property
    def column_names(self) -> list[str]:
        """
        Get all column names (without collecting the full dataframe)
        """
        return self._dataframe.collect_schema().names()

    def dtype(self, column_name: str) -> polars.datatypes.DataType:
        """
        Get column dtype (without collecting the full dataframe)
        """
        idx = self.column_names.index(column_name)
        return self._dataframe.collect_schema().dtypes()[idx]

    def minmax(self, column_name: str) -> Tuple[any, any]:
        """
        Tuple of min and max values of the given column
        """
        min_value = self._dataframe.select(column_name).min().collect()[0,0]
        max_value = self._dataframe.select(column_name).max().collect()[0,0]

        return min_value, max_value

    def set_dtype(self, column_name, representation_type) -> PolarsDataFrame:
        """
        Set the dtype for a column to the given representation type.
        Using polars cast functionality
        :return: The updated object
        """
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

    def open(path: str | Path, sep = ',') -> DataFrame:
        path = Path(path)
        if path.suffix == ".csv":
            return polars.scan_csv(path, sep=sep)
        elif path.suffix in [".h5", ".hdf5"]:
            import pandas as pd

            from damast.core.metadata import DAMAST_HDF5_ROOT

            pandas_df = pd.read_hdf(path, key=DAMAST_HDF5_ROOT)
            return polars.from_pandas(pandas_df)

        raise ValueError(f"{cls.__name__}.load_data: Unsupported input file format {path.suffix}")

    @classmethod
    def from_vaex_hdf5(cls, path: str | Path) -> Tuple[DataFrame, MetaData]:
        # avoid circular dependencies
        from damast.core.annotations import Annotation
        from damast.core.metadata import DataSpecification, MetaData

        try:
            import tables
        except LoadError as e:
            print("Could not load pytables -- please install to use hdf5 functionality")

        annotations = []
        column_specifications = []
        with tables.open_file(str(path)) as hdf5file:
            if not VAEX_HDF5_ROOT in hdf5file:
                raise TypeError(f"This HDF5 file '{hdf5file}' has not been exported with vaex")

            table_attrs = hdf5file.get_node(VAEX_HDF5_ROOT)._v_attrs
            for key in table_attrs._f_list():
                value = table_attrs[key]
                annotations.append(
                        Annotation(name=key, value=value)
                )

            data = {}
            for column in hdf5file.get_node(VAEX_HDF5_COLUMNS):
                raw_data = column.data.read()
                if isinstance(raw_data, np.ndarray) and np.issubdtype(raw_data.dtype, np.bytes_):
                    raw_data = [x.decode('utf-8') for x in raw_data]
                data[column._v_name] = raw_data

                data_specification_dict = {}
                for key in column._v_attrs._f_list():
                    data_specification_dict[key] = column._v_attrs[key]

                if data_specification_dict:
                    column_spec = DataSpecification.from_dict(data=data_specification_dict)
                    column_specifications.append(column_spec)

        metadata = None
        if column_specifications or annotations:
            metadata = MetaData(columns=column_specifications, annotations=annotations)

        return polars.LazyFrame(data), metadata

    def import_hdf5(cls, path: str | Path) -> Tuple[DataFrame, MetaData]:
        try:
            import tables
        except LoadError as e:
            print("Could not load pytables -- please install to use hdf5 functionality")

        try:
            import pandas
        except LoadError as e:
            print("Could not load pandas -- please install to use hdf5 functionality")

        try:
            pandas_df = pandas.read_hdf(filename)
            df = polars.from_pandas(pandas_df)

            return df.lazy(), None
        except tables.exceptions.NoSuchNodeError as e:
            logger.warning(f"HDF5 {filename} cannot be imported with pandas")

        return cls.from_vaex_hdf5(filename)


    def export_hdf5(df: DataFrame, path: str | Path):
        import pandas

        from damast.core.metadata import DAMAST_HDF5_ROOT

        if isinstance(df, polars.dataframe.DataFrame):
            df = df.lazy()

        pandas_df = df.collect().to_pandas()
        pandas_df.to_hdf(path, key=DAMAST_HDF5_ROOT)




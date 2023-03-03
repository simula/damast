"""
Module to define a dataframe
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import vaex
from vaex import DataFrame
import numpy as np

import h5py

from .metadata import DataSpecification, MetaData
from .annotations import Annotation

__all__ = [
    "AnnotatedDataFrame", "replace_na"
]

VAEX_HDF5_ROOT: str = "/table"
VAEX_HDF5_COLUMNS: str = f"{VAEX_HDF5_ROOT}/columns"
DAMAST_SPEC_SUFFIX: str = ".spec.yaml"


def replace_na(df: DataFrame, dtype: str, column_names: List[str] = None):
    """
    Replace `Not Available` and `Not a Number` with a mask, and convert to given `dtype`.
    This means that one later call `df[column].fill_missing(...)` to replace values

    :param df: The dataframe to modify
    :param dtype: The datatype to convert the columns to
    :param column_names: The list of columns to change
    """
    if column_names is None:
        column_names = []
    for column in column_names:
        mask = df[column].isnan() or df[column].isna()
        df[column] = np.ma.masked_array(df[column].evaluate(), mask.evaluate(), dtype=dtype)


class AnnotatedDataFrame:
    """
    A dataframe that is associated with metadata.
    """

    #: Metadata associated with the dataframe
    _metadata: MetaData

    #: The actual dataframe
    _dataframe: DataFrame

    def __init__(self,
                 dataframe: DataFrame,
                 metadata: MetaData):
        if not isinstance(dataframe, DataFrame):
            raise ValueError(f"{self.__class__.__name__}.__init__: dataframe must be"
                             f" of type 'DataFrame', but was '{type(dataframe)}")

        if not isinstance(metadata, MetaData):
            raise ValueError(f"{self.__class__.__name__}.__init__: metadata must be"
                             f" of type 'MetaData', but was '{type(metadata)}")

        self._dataframe = dataframe
        self._metadata = metadata

        # Ensure conformity of the metadata with the dataframe
        self._metadata.apply(df=self._dataframe)

    def is_empty(self) -> bool:
        """
        Check if annotated dataframe has associated data.

        :return: False if there is an internal _dataframe set, True otherwise
        """
        if self._dataframe is None:
            return True

        return False

    def get_fulfillment(self, expected_specs: List[DataSpecification]) -> MetaData.Fulfillment:
        """
        Get the :class:`MetaData.Fulfillment` with respect to the given expected specification.

        :param expected_specs: The expected specification
        :return: Instance of :class:`MetaData.Fulfillment` to investigate the degree of fulfillment
        """
        return self._metadata.get_fulfillment(expected_specs=expected_specs)

    def update(self,
               expectations: List[DataSpecification]):
        """
        Update the metadata based on a set of validated expectations.

        :param expectations: List of :class:`DataSpecifications` as expectations that this data meets.
        """
        for expected_data_spec in expectations:
            column_name = expected_data_spec.name
            if column_name not in self._metadata:
                # Column name description is not yet part of the metadata
                # Verify that is it part of the data frame
                if column_name not in self._dataframe.column_names:
                    raise RuntimeError(f"{self.__class__.__name__}.update:"
                                       f" required output '{column_name}' is not"
                                       f" present in the result dataframe")
                else:
                    self._metadata.columns.append(DataSpecification.from_dict(data=expected_data_spec.to_dict()))

    def save(self, *, filename: Union[str, Path]) -> AnnotatedDataFrame:
        """
        Save this instance in an hdf5 file.
        The resulting file can be inspected using HDF5 tools and in particular h5dump.
        to get the HDF5 groups and their corresponding metdata (header only e.g. no data printed), use the command:

        h5dump -H data.hdf5

        See: https://portal.hdfgroup.org/display/HDF5/HDF5+Command-line+Tools
       

        :param filename: Filename to use for saving

        """
        if self._dataframe is not None:
            metadata_filename = Path(filename).with_suffix(DAMAST_SPEC_SUFFIX)
            self._metadata.save_yaml(filename=metadata_filename)
            if Path(filename).suffix == ".hdf5" or Path(filename).suffix == ".h5":
                self.export_hdf5(filename)
                ## Add metadata
                metadata = self._metadata.to_dict()
                annotations = self._metadata.Key.annotations.value
                columns = self._metadata.Key.columns.value
                list_attrs = metadata[columns]
                dict_annotations = metadata[annotations]
                list_columns = list(self.columns)
                h5f = h5py.File(filename, 'r+')
                # Add annotations to main group
                for key in dict_annotations.keys():
                    if key in h5f[VAEX_HDF5_ROOT].attrs.keys() and h5f[VAEX_HDF5_ROOT].attrs[key] != dict_annotations[key]:
                        raise RuntimeError(f"{self.__class__.__name__}.save:"
                                           f" attribute '{key}' present"
                                           f" in vaex dataframe but different from user-defined")
                    else:        
                        h5f[VAEX_HDF5_ROOT].attrs[key] = dict_annotations[key]    
                # Add attributes for columns
                for attrs in list_attrs:
                    if DataSpecification.Key.name.value in attrs.keys() and attrs[DataSpecification.Key.name.value] in list_columns:
                        group_name = f"/{VAEX_HDF5_COLUMNS}/{attrs[DataSpecification.Key.name.value]}"
                        if group_name in h5f:
                            for key in attrs.keys():
                                h5f[group_name].attrs[key] = attrs[key]
                h5f.close()
            else:
                self.export(filename)
        else:
            raise ValueError(f"{self.__class__.__name__}.save: no dataframe to save")

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> AnnotatedDataFrame:
        """
        Create an annotated dataframe from an hdf5 file.

        :param filename: Filename to use for importing and creating the annotated dataframe

        """
        df = vaex.open(filename)
        metadata = None
        annotations = {}
        list_attrs = []
        list_columns = list(df.columns)
        if Path(filename).suffix == ".hdf5" or Path(filename).suffix == ".h5":
            ## Read metadata
            h5f = h5py.File(filename, 'r')
            for key in h5f[VAEX_HDF5_ROOT].attrs.keys():
                annotations[key] = Annotation(name=key, value=h5f[VAEX_HDF5_ROOT].attrs[key])
            # Read attributes for columns
            for colname in list_columns:
                group_name = f"/{VAEX_HDF5_COLUMNS}/{colname}"
                if not group_name in h5f:
                    continue
                ds_dict = {}
                for key in h5f[group_name].attrs.keys():
                    ds_dict[key] = h5f[group_name].attrs[key]
                if ds_dict:
                    column_spec = DataSpecification.from_dict(data=ds_dict)
                    list_attrs.append(column_spec)
            h5f.close()

        if not list_attrs:
            metadata_filename = Path(filename).with_suffix(DAMAST_SPEC_SUFFIX)
            if metadata_filename.exists():
                metadata = MetaData.load_yaml(filename=metadata_filename)
            else:
                # metadata missing
                raise RuntimeError(f"{cls.__name__}.from_file:"
                                   f" metadata is missing in '{filename}'"
                                   f" and needs to be added")

        if not metadata:
            metadata = MetaData(columns=list_attrs, annotations=annotations)
        return cls(dataframe=df, metadata=metadata)

    def __getattr__(self, item):
        """
        Ensure that this object behaves like a `vaex.DataFrame`.

        :param item: Name of column
        :return: The column data
        """
        return getattr(self._dataframe, item)

    def __getitem__(self, item) -> Any:
        """
        Make dataframe subscriptable and behave like the vaex.dataframe.

        :param item: Name of the key when using [] operators
        :return: item/column from the underlying vaex.dataframe
        """
        return self._dataframe[item]

    def __setitem__(self, key, value):
        """
        Set the column for the annotated dataframe, and allow to behave like the vaex.dataframe.

        :param key: Column name
        :param value: Value to set the column to
        """
        self._dataframe[key] = value

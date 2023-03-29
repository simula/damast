"""
Module to define a dataframe
"""
from __future__ import annotations

import gc
import logging
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import List, Union, Any, Callable, Dict

import vaex
from vaex import DataFrame
import numpy as np

import h5py
from logging import getLogger, Logger, INFO

from .metadata import DataSpecification, MetaData, ValidationMode
from .annotations import Annotation

__all__ = ["AnnotatedDataFrame", "replace_na"]

VAEX_HDF5_ROOT: str = "/table"
VAEX_HDF5_COLUMNS: str = f"{VAEX_HDF5_ROOT}/columns"
DAMAST_SPEC_SUFFIX: str = ".spec.yaml"

logging.basicConfig()
_log: Logger = getLogger(__name__)
_log.setLevel(INFO)


def replace_na(df: DataFrame, dtype: str, column_names: List[str] = None):
    """
    Replace ``Not Available`` and ``Not a Number`` with a mask, and convert to given ``dtype``.
    This means that one later call ``df[column].fill_missing(...)`` to replace values

    :param df: The dataframe to modify
    :param dtype: The datatype to convert the columns to
    :param column_names: The list of columns to change
    """
    if column_names is None:
        column_names = []
    for column in column_names:
        mask = df[column].isnan() or df[column].isna()
        df[column] = np.ma.masked_array(
            df[column].evaluate(), mask.evaluate(), dtype=dtype
        )


class AnnotatedDataFrame:
    """
    A dataframe that is associated with metadata.

    :param dataframe: The vaex dataframe holding the data
    :param metadata: The metadata for the dataframe
    :param validation_mode: If :attr:`damast.core.ValidationMode.UPDATE_DATA` replace values outside of valid range (specified in :attr:`metadata`)
        with missing value. If :attr:`damast.core.ValidationMode.UPDATE_METADATA` update the metadata according to the encountered values.
        Else :attr:`damast.core.ValidationMode.READONLY` will throw when encountering inconsistencies
    """

    #: Metadata associated with the dataframe
    _metadata: MetaData

    #: The actual dataframe
    _dataframe: DataFrame

    def __init__(
        self,
        dataframe: DataFrame,
        metadata: MetaData,
        validation_mode: ValidationMode = ValidationMode.READONLY,
    ):
        if not isinstance(dataframe, DataFrame):
            raise ValueError(
                f"{self.__class__.__name__}.__init__: dataframe must be"
                f" of type 'DataFrame', but was '{type(dataframe)}"
            )

        if not isinstance(metadata, MetaData):
            raise ValueError(
                f"{self.__class__.__name__}.__init__: metadata must be"
                f" of type 'MetaData', but was '{type(metadata)}"
            )

        self._dataframe = dataframe
        self._metadata = metadata

        # Ensure conformity of the metadata with the dataframe
        self.validate_metadata(validation_mode=validation_mode)

    @property
    def dataframe(self) -> vaex.DataFrame:
        """
        Allows to access the underlying dataframe directly.

        .. note::
            AnnotatedDataFrame behaves like a ``vaex.DataFrame``, so typically you will not need to access the
            dataframe through this property.

        :return: The underlying dataframe
        """
        return self._dataframe

    @property
    def metadata(self):
        return self._metadata

    def validate_metadata(
        self, validation_mode: ValidationMode = ValidationMode.READONLY
    ) -> None:
        """
        Validate this annotated dataframe and ensure that data and spec match.

        :param validation_mode: Select the validation mode that should be used
        :raise RuntimeError: Dependending on the validation mode an exception will be raise to ensure the data spec
               conformance
        """
        self._metadata.apply(df=self._dataframe, validation_mode=validation_mode)

    def is_empty(self) -> bool:
        """
        Check if annotated dataframe has associated data.

        :return: False if there is an internal :code:`_dataframe` set, True otherwise
        """
        if self._dataframe is None:
            return True

        return False

    def get_fulfillment(
        self, expected_specs: List[DataSpecification]
    ) -> MetaData.Fulfillment:
        """
        Get the :class:`MetaData.Fulfillment` with respect to the given expected specification.

        :param expected_specs: The expected specification
        :return: Instance of :class:`MetaData.Fulfillment` to investigate the degree of fulfillment
        """
        return self._metadata.get_fulfillment(expected_specs=expected_specs)

    def update(self, expectations: List[DataSpecification]):
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
                    raise RuntimeError(
                        f"{self.__class__.__name__}.update:"
                        f" required output '{column_name}' is not"
                        f" present in the result dataframe"
                    )
                else:
                    self._metadata.columns.append(
                        DataSpecification.from_dict(data=dict(expected_data_spec))
                    )

    def save(self, *, filename: Union[str, Path]) -> AnnotatedDataFrame:
        """
        Save this instance in an hdf5 file.
        The resulting file can be inspected using HDF5 tools and in particular h5dump.
        To get the HDF5 groups and their corresponding metdata (header only e.g. no data printed), use the command:

        .. code-block:: console

            h5dump -H data.hdf5

        See: https://portal.hdfgroup.org/display/HDF5/HDF5+Command-line+Tools

        :param filename: Filename to use for saving

        """
        if self._dataframe is not None:
            metadata_filename = Path(filename).with_suffix(DAMAST_SPEC_SUFFIX)
            self._metadata.save_yaml(filename=metadata_filename)

            if Path(filename).suffix not in [".hdf5", ".h5"]:
                self.export(filename)
                return

            # First save the hdf5 file in order to add then the metadata to it
            self.export_hdf5(filename)

            # Add metadata
            metadata = dict(self._metadata)
            annotations = self._metadata.Key.annotations.value
            columns = self._metadata.Key.columns.value
            list_attrs = metadata[columns]
            dict_annotations = metadata[annotations]
            list_columns = list(self.columns)

            h5f = h5py.File(filename, "r+")
            # Add annotations to main group
            for key in dict_annotations.keys():
                if (
                    key in h5f[VAEX_HDF5_ROOT].attrs.keys()
                    and h5f[VAEX_HDF5_ROOT].attrs[key] != dict_annotations[key]
                ):
                    raise RuntimeError(
                        f"{self.__class__.__name__}.save:"
                        f" attribute '{key}' present"
                        f" in vaex dataframe but different from user-defined"
                    )
                else:
                    h5f[VAEX_HDF5_ROOT].attrs[key] = dict_annotations[key]

            # Add attributes for columns
            for attrs in list_attrs:
                if (
                    DataSpecification.Key.name.value in attrs.keys()
                    and attrs[DataSpecification.Key.name.value] in list_columns
                ):
                    group_name = f"/{VAEX_HDF5_COLUMNS}/{attrs[DataSpecification.Key.name.value]}"
                    if group_name in h5f:
                        for key in attrs.keys():
                            if isinstance(attrs[key], dict):
                                h5f[group_name].attrs[key] = str(attrs[key])
                            else:
                                h5f[group_name].attrs[key] = attrs[key]
            h5f.close()
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
        annotations = []
        list_attrs = []
        list_columns = list(df.columns)
        if Path(filename).suffix in [".hdf5", ".h5"]:
            # Read metadata
            h5f = h5py.File(filename, "r")
            for key in h5f[VAEX_HDF5_ROOT].attrs.keys():
                annotations.append(
                    Annotation(name=key, value=h5f[VAEX_HDF5_ROOT].attrs[key])
                )

            # Read attributes for columns
            for colname in list_columns:
                group_name = f"/{VAEX_HDF5_COLUMNS}/{colname}"
                if group_name not in h5f:
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
                raise RuntimeError(
                    f"{cls.__name__}.from_file:"
                    f" metadata is missing in '{filename}'"
                    f" and needs to be added"
                )

        if not metadata:
            metadata = MetaData(columns=list_attrs, annotations=annotations)
        return cls(dataframe=df, metadata=metadata)

    def __getattr__(self, item):
        """
        Ensure that this object behaves like a :class:`vaex.DataFrame`.

        :param item: Name of column
        :return: The column data
        """
        return getattr(self._dataframe, item)

    def __getitem__(self, item) -> Any:
        """
        Make dataframe subscriptable and behave like the :class:`vaex.DataFrame`.

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

    @classmethod
    def convert_csv_to_adf(
        cls,
        csv_filenames: List[Union[Path, str]],
        metadata_filename: Union[Path, str],
        output_filename: Union[Path, str],
        validation_mode: ValidationMode = ValidationMode.READONLY,
        progress: Callable[[float], None] = None,
        csv_sep: str = ";",
    ):
        """
        Convert a csv file toa an annotated dataframe

        :param csv_filenames: The input csv file that shall be converted
        :param metadata_filename: The metadata specification
        :param output_filename: The output file that will be generated
        """
        metadata = MetaData.load_yaml(filename=metadata_filename)
        with TemporaryDirectory() as tmpdirname:
            hdf_filenames = []
            for filename in sorted(csv_filenames):
                _log.info(f"Loading csv file: {filename}")
                # Ensure that temporary dataframe conversion
                # will not accidentally exhaust memory
                gc.collect()

                tmp_hdf5 = Path(tmpdirname) / f"{Path(filename).stem}.hdf5"
                df = vaex.from_csv(
                    filename, convert=str(tmp_hdf5), progress=progress, sep=csv_sep
                )

                # validate the data
                AnnotatedDataFrame(
                    metadata=metadata, dataframe=df, validation_mode=validation_mode
                )

                hdf_filenames.append(str(tmp_hdf5))

            _log.info(f"Concatenating files: {hdf_filenames}")
            df = vaex.concat([vaex.open(x) for x in hdf_filenames])

            adf = cls(dataframe=df, metadata=metadata)

            _log.info(f"Metadata: {dict(metadata)}")
            _log.info(f"Saving dataframe into {output_filename}")

            adf.save(filename=output_filename)

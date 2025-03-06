"""
Module to define an annotated dataframe, i.e. the combination of data and metadata.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
from logging import INFO, Logger, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, List, Optional, Union

import h5py
import numpy as np
import polars
import pyarrow
import pyarrow.parquet as pq

from .annotations import Annotation
from .metadata import DataSpecification, MetaData, ValidationMode
from .types import DataFrame, XDataFrame

__all__ = ["AnnotatedDataFrame"]

DAMAST_SPEC_SUFFIX: str = ".spec.yaml"

logging.basicConfig()
_log: Logger = getLogger(__name__)
_log.setLevel(INFO)


class AnnotatedDataFrame(XDataFrame):
    """
    A dataframe that is associated with metadata.

    :param dataframe: The polars dataframe holding the data
    :param metadata: The metadata for the dataframe
    :param validation_mode:
        - If :attr:`damast.core.ValidationMode.UPDATE_DATA` replace values outside of
          valid range (specified in :attr:`metadata`) with missing value.
        - If :attr:`damast.core.ValidationMode.UPDATE_METADATA` update the metadata according to the encountered values.
        - Else :attr:`damast.core.ValidationMode.READONLY` will throw when encountering inconsistencies
    """

    #: Metadata associated with the dataframe
    _metadata: MetaData

    #: The actual dataframe
    _dataframe: DataFrame

    def __init__(
        self,
        dataframe: polars.DataFrame | polars.LazyFrame | XDataFrame,
        metadata: MetaData,
        validation_mode: ValidationMode = ValidationMode.READONLY,
    ):
        if isinstance(dataframe, XDataFrame):
            dataframe = dataframe._dataframe

        if isinstance(dataframe, polars.DataFrame):
            dataframe = dataframe.lazy()

        if not isinstance(dataframe, polars.LazyFrame):
            raise ValueError(
                f"{self.__class__.__name__}.__init__: dataframe must be"
                f" of type 'DataFrame', but was '{type(dataframe)}"
            )

        if not isinstance(metadata, MetaData):
            raise ValueError(
                f"{self.__class__.__name__}.__init__: metadata must be"
                f" of type 'MetaData', but was '{type(metadata)}"
            )

        super().__init__(df=dataframe)

        self._metadata = metadata

        # Ensure conformity of the metadata with the dataframe
        self.validate_metadata(validation_mode=validation_mode)

    @property
    def metadata(self) -> MetaData:
        """Get the metadata for this dataframe"""
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
        self._dataframe = self._metadata.apply(df=self._dataframe, validation_mode=validation_mode)

    def is_empty(self) -> bool:
        """
        Check if annotated dataframe has associated data.

        :return: False if there is an internal :code:`_dataframe` set, True otherwise
        """
        return self._dataframe is None

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
            if column_name not in self.metadata:
                # Column name description is not yet part of the metadata
                # Verify that is it part of the data frame
                if column_name in self.dataframe.column_names:
                    self._metadata.columns.append(
                        DataSpecification.from_dict(data=dict(expected_data_spec))
                    )
                else:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.update:"
                        f" required output '{column_name}' is not"
                        f" present in the result dataframe - available columns are:"
                        f" {','.join(self.dataframe.column_names)}"
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
        if self._dataframe is None:
            raise ValueError(f"{self.__class__.__name__}.save: no dataframe to save")

        metadata_filename = Path(filename).with_suffix(DAMAST_SPEC_SUFFIX)
        self._metadata.save_yaml(filename=metadata_filename)

        if Path(filename).suffix not in [".hdf5", ".h5"]:
            self.export(filename)
            return self

        # First save the hdf5 file in order to add then the metadata to it
        self.export_hdf5(filename)
        self._metadata.append_to_hdf(filename)

        return self

    def export(self, filename: str | Path):
        arrow_table = self._dataframe.collect().to_arrow()
        new_schema = arrow_table.schema.with_metadata({b'annotated_dataframe': json.dumps(dict(self._metadata)).encode('UTF-8')})
        arrow_table = pyarrow.Table.from_arrays(arrow_table.columns, schema=new_schema)
        pq.write_table(arrow_table, filename)

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> AnnotatedDataFrame:
        """
        Create an annotated dataframe from an hdf5 file.

        :param filename: Filename to use for importing and creating the annotated dataframe

        """
        metadata = None
        if Path(filename).suffix in [ ".pq", ".parquet"]:
            df = polars.scan_parquet(filename)
            schema = pq.read_schema(filename)

            data = schema.metadata[b"annotated_dataframe"]
            metadata = MetaData.from_dict(json.loads(data.decode('UTF-8')))
        elif Path(filename).suffix in [ ".csv", ".parquet"]:
            df = polars.scan_csv(filename)
        elif Path(filename).suffix in [".h5", ".hdf5"]:
            import tables
            try:
                import pandas
                pandas_df = pandas.read_hdf(filename)
                df = polars.from_pandas(pandas_df)
            except tables.exceptions.NoSuchNodeError as e:
                df, metadata = XDataFrame.from_vaex_hdf5(filename)
        else:
            raise RuntimeError(f"Could not load {filename} - please use parquet files")

        if not metadata:
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

    @classmethod
    def convert_csv_to_adf(
        cls,
        csv_filenames: List[Union[Path, str]],
        metadata_filename: Union[Path, str],
        output_filename: Union[Path, str],
        validation_mode: ValidationMode = ValidationMode.READONLY,
        progress: Optional[Callable[[float], None]] = None,
        csv_sep: str = ";",
    ):
        """
        Convert a csv file to an annotated dataframe

        :param csv_filenames: The input csv file that shall be converted
        :param metadata_filename: The metadata specification
        :param output_filename: The output file that will be generated
        :param validation_mode: how to validate / enforce the conformity of the metadata and the data
        :param progress: Callable to set for the dataframe conversion
        :param csv_sep: Separator to use when loading csv files
        """
        metadata = MetaData.load_yaml(filename=metadata_filename)
        df = polars.scan_csv(sorted(csv_filenames), separator=csv_sep)
        adf = cls(dataframe=df, metadata=metadata)

        _log.info(f"Metadata: {dict(metadata)}")
        _log.info(f"Saving dataframe into {output_filename}")

        adf.save(filename=output_filename)

    def drop(self, columns, strict: bool = True) -> AnnotatedDataFrame:
        self._dataframe = self._dataframe.drop(columns, strict=strict)
        self._metadata.drop(columns)
        return self

    def copy(self):
        return copy.deepcopy(self)

    @property
    def shape(self):
        return self._dataframe.collect().shape

    def __deepcopy__(self, memo=None):
        return AnnotatedDataFrame(self._dataframe.clone(), copy.deepcopy(self._metadata))

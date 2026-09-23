"""
Module to define an annotated dataframe, i.e. the combination of data and metadata.
"""
from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from logging import INFO, Logger, getLogger
from pathlib import Path

import polars
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import Any

try:
    from typing import deprecated
except ImportError:
    from typing_extensions import deprecated

from .annotations import Annotation
from .constants import (
    DAMAST_SPEC_SUFFIX,
    DAMAST_SUPPORTED_FILE_FORMATS,
)
from .data_description import ListOfValues, MinMax
from .metadata import DataSpecification, MetaData, ValidationMode
from .partitioning import PartitionStrategy, warn_on_local_time_buckets
from .polars_dataframe import scan_csv
from .types import DataFrame, XDataFrame

__all__ = ["AnnotatedDataFrame"]


logging.basicConfig()
_log: Logger = getLogger(__name__)
_log.setLevel(INFO)

COMPRESSION_CODECS = ["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD", "BZ2"]

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
    :param metadata_inferred: Whether `metadata` was inferred from the data (e.g. by
        :func:`infer_annotation`, when no spec file was found) rather than loaded from an
        actual metadata spec - see :attr:`metadata_inferred`
    """

    #: Metadata associated with the dataframe
    _metadata: MetaData

    def __init__(
        self,
        dataframe: polars.DataFrame | polars.LazyFrame | XDataFrame,
        metadata: MetaData,
        validation_mode: ValidationMode = ValidationMode.READONLY,
        metadata_inferred: bool = False,
    ):
        if isinstance(dataframe, XDataFrame):
            dataframe = dataframe.lazyframe

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
        self._metadata_inferred = metadata_inferred

        # Ensure conformity of the metadata with the dataframe
        self.validate_metadata(validation_mode=validation_mode)

    @property
    def metadata(self) -> MetaData:
        """Get the metadata for this dataframe"""
        return self._metadata

    @property
    def metadata_inferred(self) -> bool:
        """Whether `metadata` was inferred from the data rather than loaded from a spec file."""
        return self._metadata_inferred

    @classmethod
    def ensure_type(cls, obj: Any):
        if not isinstance(obj, cls):
            raise ValueError("Object {obj} is not an AnnotatedDataFrame")

        if not isinstance(obj.lazyframe, DataFrame):
            raise ValueError(f"AnnotatedDataFrame.lazyframe is not of type {DataFrame}")

    def validate_metadata(
        self, validation_mode: ValidationMode = ValidationMode.READONLY
    ) -> None:
        """
        Validate this annotated dataframe and ensure that data and spec match.

        :param validation_mode: Select the validation mode that should be used
        :raise RuntimeError: Dependending on the validation mode an exception will be raise to ensure the data spec
               conformance
        """
        self.lazyframe = self._metadata.apply(df=self.lazyframe, validation_mode=validation_mode)

    def is_empty(self) -> bool:
        """
        Check if annotated dataframe has associated data.

        :return: False if there is an internal :code:`lazyframe` set, True otherwise
        """
        return self.lazyframe is None

    def get_fulfillment(
        self, expected_specs: list[DataSpecification]
    ) -> MetaData.Fulfillment:
        """
        Get the :class:`MetaData.Fulfillment` with respect to the given expected specification.

        :param expected_specs: The expected specification
        :return: Instance of :class:`MetaData.Fulfillment` to investigate the degree of fulfillment
        """
        return self._metadata.get_fulfillment(expected_specs=expected_specs)

    def update(self, expectations: list[DataSpecification]):
        """
        Update the metadata based on a set of validated expectations.

        :param expectations: List of :class:`DataSpecifications` as expectations that this data meets.
        """
        for expected_data_spec in expectations:
            column_name = expected_data_spec.name
            if column_name not in self.dataframe.column_names:
                raise RuntimeError(
                    f"{self.__class__.__name__}.update:"
                    f" required output '{column_name}' is not"
                    f" present in the result dataframe - available columns are:"
                    f" {','.join(self.dataframe.column_names)}"
                )

            new_spec = DataSpecification.from_dict(data=dict(expected_data_spec))
            if column_name not in self.metadata:
                # Column description is not yet part of the metadata - add it
                self._metadata.columns.append(new_spec)
            else:
                # Column is a declared output of this pipeline step, i.e. it was just
                # (re)computed. value_range/value_stats always come from the step's own
                # declared output (falling back to None if it doesn't declare any) - carrying
                # over a stale range/stats from before this step ran (e.g. inherited from a
                # previously exported file) would otherwise fail the READONLY
                # validate_metadata() check right after this call, even though the step does
                # not itself assert any particular range for this column.
                # Other fields (representation_type, description, unit, ...) describe the
                # column's structure rather than its current values, so - unless this step's
                # output declares them itself - they carry over unchanged from the existing
                # spec instead of being reset to None.
                existing_spec = self.metadata[column_name]
                for key in DataSpecification.Key:
                    if key in (DataSpecification.Key.name, DataSpecification.Key.value_range,
                               DataSpecification.Key.value_stats):
                        continue
                    if getattr(new_spec, key.value) is None:
                        setattr(new_spec, key.value, getattr(existing_spec, key.value))

                self._metadata.columns = [
                    new_spec if c.name == column_name else c for c in self._metadata.columns
                ]

    @deprecated("Use `export(..)` instead")
    def save(self, *, filename: str | Path) -> AnnotatedDataFrame:
        """
        Save this annotated dataframe in a file.

        Filetype can be .parquet (recommended) or *.hdf5.

        For hdf5 the resulting file can be inspected using HDF5 tools and in particular h5dump.
        To get the HDF5 groups and their corresponding metdata (header only e.g. no data printed), use the command:

        .. code-block:: console

            h5dump -H data.hdf5

        See: https://portal.hdfgroup.org/display/HDF5/HDF5+Command-line+Tools

        :param filename: Filename to use for saving

        """
        if self.lazyframe is None:
            raise ValueError(f"{self.__class__.__name__}.save: no dataframe to save")

        metadata_filename = Path(filename).with_suffix(DAMAST_SPEC_SUFFIX)
        self._metadata.save_yaml(filename=metadata_filename)

        if Path(filename).suffix not in [".hdf5", ".h5"]:
            self.export(filename)
            return self

        # First save the hdf5 file in order to add then the metadata to it
        XDataFrame.export_hdf5(self.lazyframe, filename)
        self._metadata.append_to_hdf(filename)

        return self

    def export(self, filename: str | Path, compression: str | None = None, compression_level: int | None = None):
        """
        Export the annotated dataframe to a file.
        By default the format is parquet.

        Compression is applied when possible, e.g., for parquet.
        """
        # this sets the default
        if compression is None:
            compression = "zstd"
            if not compression_level:
                compression_level = 5
        # this sets explicitly no compression
        elif compression == "NONE":
            compression = None

        arrow_table = self.lazyframe.compat.collected().to_arrow()
        new_schema = arrow_table.schema.with_metadata({b'annotated_dataframe': json.dumps(dict(self._metadata), default=str).encode('UTF-8')})
        arrow_table = pyarrow.Table.from_arrays(arrow_table.columns, schema=new_schema)
        pq.write_table(arrow_table, filename, compression=compression, compression_level=compression_level)

    def export_partitioned(
        self,
        directory: str | Path,
        strategy: PartitionStrategy,
        *,
        suffix: str = ".parquet",
        compression: str | None = None,
        compression_level: int | None = None,
    ) -> list[Path]:
        """
        Export this dataframe as one file per partition, as determined by `strategy`.

        Each partition is written via :meth:`save`/:meth:`export`, so the result round-trips
        through :meth:`from_files` exactly like any other multi-file dataset.

        .. note::
            This collects the full dataframe before splitting it (`polars.DataFrame.partition_by`
            requires an eager dataframe) - not suited for data too large to fit in memory.

        :param directory: Directory to write partition files into (created if missing)
        :param strategy: Determines the per-row partition key and its filename
        :param suffix: File extension appended to every :meth:`PartitionStrategy.filename`,
            with a leading ``.`` added if missing. Pass ``""`` when the strategy already
            returns complete filenames, e.g. when they come from an external naming scheme.
            This names the file only - the contents are parquet either way.
        :return: Paths of the written data files, one per partition

        Example:

        .. code-block:: python

            adf.export_partitioned("out/", ByColumn("mmsi"))
            adf.export_partitioned("out/", ByTime("timestamp", every="1d"))

            # filenames owned by the caller, e.g. to match an existing archive's convention
            adf.export_partitioned("out/", ByExpr(key, filename_fn=my_pattern), suffix="")
        """
        if self.lazyframe is None:
            raise ValueError(f"{self.__class__.__name__}.export_partitioned: no dataframe to export")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if suffix and not suffix.startswith("."):
            suffix = f".{suffix}"

        key_col = "__damast_partition_key__"
        collected = self.lazyframe.with_columns(strategy.key_expr().alias(key_col)).collect()
        warn_on_local_time_buckets(collected[key_col])

        written: list[Path] = []
        # as_dict=False + re-deriving the key from the partition itself (rather than
        # as_dict=True, which needs the key as a dict key) also works for a struct-valued
        # key_expr() (e.g. a composite of several columns) - a dict isn't hashable.
        partitions = collected.partition_by(key_col, as_dict=False, include_key=True, maintain_order=True)
        with tqdm(partitions, unit="partition") as pbar:
            for part in pbar:
                key = part[key_col][0]
                part = part.drop(key_col)
                filename = directory / f"{strategy.filename(key)}{suffix}"
                pbar.set_description(f"Exporting {filename}")
                filename.parent.mkdir(parents=True, exist_ok=True)
                # The file should be self-consistent, so update the metadata
                metadata = AnnotatedDataFrame.infer_annotation(df=part.lazy(), reference=self._metadata)
                part_adf = AnnotatedDataFrame(part, metadata, validation_mode=ValidationMode.IGNORE)
                part_adf.export(filename,
                                compression=compression,
                                compression_level=compression_level)
                written.append(filename)

        return written

    @classmethod
    def get_supported_format(cls, suffix: str) -> str | None:
        """
        Get the name of the supported format from the suffix
        """
        for format_name, suffixes in DAMAST_SUPPORTED_FILE_FORMATS.items():
            if suffix in suffixes:
                return format_name

        return None

    @classmethod
    def from_files(cls,
            files: list[str|Path],
            metadata_required: bool = True,
            validation_mode: ValidationMode = ValidationMode.READONLY,
            merge_strategy: DataSpecification.MergeStrategy = DataSpecification.MergeStrategy.THIS
        ) -> AnnotatedDataFrame:
        """
        Create an annotated dataframe by loading given files

        :param files: Files to use for importing and creating the annotated dataframe
        :param metadata_required metadata needs to be available either as spec.yaml file or embedded into the file
        :param validation_mode metadata will be validated, updated or ignored according to this mode
        :param merge_strategy for combining the metadata of multiple files, select the merge strategy

        """
        metadata = None
        if not files or type(files) is not list:
            raise ValueError(f"{cls.__name__}.from_files: file list required, but were {files=}")

        suffixes = set()
        for f in files:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"{cls.__name__}.from_files: could not find {f}")

            suffixes.add(path.suffix)

        if len(suffixes) != 1:
            raise RuntimeError(f"{cls.__name__}.from_files: one file type expected, but multiple suffixes found: {suffixes=}")

        suffix = list(suffixes)[0]
        load_fn = None
        # find load function by suffix
        for filetype, suffixes in DAMAST_SUPPORTED_FILE_FORMATS.items():
            if suffix in suffixes:
                if hasattr(cls, f"load_{filetype}"):
                    load_fn = getattr(cls, f"load_{filetype}")

        if load_fn:
            df, metadata = load_fn(files)
        else:
            raise RuntimeError(
                f"Could not load {files} - supported are currently {DAMAST_SUPPORTED_FILE_FORMATS}"
                " where .h5, .hdf5 generated by vaex or pandas"
            )

        if not metadata:
            _log.info("No metadata provided or found in files - searching now for an existing spec file")
            metadata, _ , metadata_file_candidates = MetaData.search(files)

            if metadata is None:
                _log.info("Found no candidate for a spec file")
                if metadata_required:
                    _log.info("Metadata is required but not available")
                    head = df.head(10).collect()
                    raise RuntimeError(
                        f"{cls.__name__}.from_files:"
                        f" metadata is missing for {files=}"
                        f" and needs to be added'\n"
                        f"{head} - {[str(x) for x in metadata_file_candidates]}"
                    )
                else:
                    _log.info("Metadata is not available and not required, so inferring annotation")
                    metadata = cls.infer_annotation(df)
                    _log.info("Metadata inferring completed")
                    return cls(dataframe=df, metadata=metadata, validation_mode=ValidationMode.IGNORE,
                               metadata_inferred=True)
            else:
                _log.info(f"Found metadata: {[str(x) for x in metadata_file_candidates]}")
        elif len(metadata) != len(files):
            _log.info("Metadata is not available for all files, so inferring annotation")
            metadata = cls.infer_annotation(df)
            _log.info("Metadata inferring completed")
            return cls(dataframe=df, metadata=metadata, validation_mode=ValidationMode.IGNORE,
                       metadata_inferred=True)
        elif len(metadata) == len(files):
            metadata_list = list(metadata.values())
            metadata = MetaData(
                        columns=metadata_list[0].columns,
                        annotations=list(metadata_list[0].annotations.values())
                    )
            for i in range(len(metadata_list)-1):
                metadata = metadata.merge(metadata_list[i+1], strategy=merge_strategy)
        else:
            for _, m in metadata.items():
                metadata = m
                break

        return cls(dataframe=df, metadata=metadata, validation_mode=validation_mode)

    @classmethod
    def load_parquet(cls, files) -> tuple[polars.LazyFrame, dict[str, MetaData]]:
            _log.info(f"Loading parquet: {files=}")
            metadata_per_file = {}

            pyarrow_schemas = []
            for file in files:
                schema = pq.read_schema(file)
                pyarrow_schemas.append(schema)
                if schema and hasattr(schema, "metadata"):
                    if schema.metadata is not None:
                        if b"annotated_dataframe" in schema.metadata:
                            data = schema.metadata[b"annotated_dataframe"]
                            m = MetaData.from_dict(json.loads(data.decode('UTF-8')))
                            m.set_annotation(Annotation(name=Annotation.Key.Source, value=Path(file).name))
                            metadata_per_file[file] = m

            # https://github.com/pola-rs/polars/issues/27280
            arrow_schema = pyarrow.unify_schemas(pyarrow_schemas)
            sorted_arrow_schema = pyarrow.schema(sorted(arrow_schema, key=lambda x: x.name))
            polars_schema = polars.from_arrow(sorted_arrow_schema.empty_table()).schema

            df = polars.scan_parquet(files, missing_columns='insert', schema=polars_schema)
            return df, metadata_per_file

    @classmethod
    def load_netcdf(cls, files) -> tuple[polars.LazyFrame, dict[str, MetaData]]:
        _log.info(f"Loading netcdf: {files=}")
        return XDataFrame.import_netcdf(files)

    @classmethod
    def load_hdf(cls, files) -> tuple[polars.LazyFrame, dict[str, MetaData]]:
        _log.info(f"Loading hdf: {files=}")
        return XDataFrame.import_hdf5(files)

    @classmethod
    def load_csv(cls, files) -> tuple[polars.LazyFrame, dict[str, MetaData]]:
        _log.info(f"Loading csv: {files=}")
        df = scan_csv(files, separator=";")
        if len(df.compat.column_names) <= 1:
            # unlikely that this frame has only one column, so trying with comma
            df = scan_csv(files, separator=",")
        return df, {}

    @classmethod
    def from_file(cls,
            filename: str | Path,
            metadata_required: bool = True,
            validation_mode: ValidationMode = ValidationMode.READONLY,
            merge_strategy: DataSpecification.MergeStrategy = DataSpecification.MergeStrategy.THIS
        ) -> AnnotatedDataFrame:
        """
        Create an annotated dataframe from an hdf5 file.

        :param filename: Filename to use for importing and creating the annotated dataframe

        """
        return cls.from_files([filename],
                              metadata_required=metadata_required,
                              validation_mode=validation_mode,
                              merge_strategy=merge_strategy)

    # declared fields that remain valid for a column when its data is filtered or reshaped -
    # unlike type and value range/stats, they cannot be observed from a subset of the data.
    _INHERITED_SPEC_FIELDS = ("description", "category", "abbreviation", "unit", "precision",
                              "missing_value", "is_optional")

    @classmethod
    def infer_annotation(cls, df: DataFrame, reference: MetaData | None = None) -> MetaData:
        """
        Infer the metadata of a dataframe from its data.

        :param df: The dataframe to annotate
        :param reference: Optional metadata, e.g. of the dataframe ``df`` was derived from: for
            columns it contains, the declared fields (unit, description, category, abbreviation,
            precision, missing value, is_optional) are taken over, while type and value
            range/stats are inferred. The reference's dataset-level annotations are carried over
            as well - they describe the dataset, not the rows that remain in ``df``
        :return: The inferred metadata
        """
        column_specs: list[DataSpecification] = []

        # Each collect() re-reads the input - so compute categories and min/max of all
        # non-numeric columns in (at most) two passes, which the loop below reads from cache
        string_columns = [c for c in df.compat.column_names if df.compat.is_string(c)]
        df.compat.precompute_categories(string_columns)
        minmax_columns = [c for c in df.compat.column_names
                          if not df.compat.is_numeric(c) and
                          not (c in string_columns and df.compat.categories(c))]
        try:
            df.compat.precompute_minmax(minmax_columns)
        except ValueError as e:
            # e.g. unsupported dtype - the loop below then tries each column individually
            _log.debug(f"AnnotatedDataFrame.infer_annotation: could not precompute value ranges -- {e}")

        numeric_columns: list[str] = []
        for column in df.compat.column_names:
            data = {'name': column,
                    'is_optional': False,
                    'representation_type': df.compat.dtype(column)
            }

            if df.compat.is_string(column):
                categories = df.compat.categories(column)
                if categories:
                    data['value_range'] = ListOfValues(categories)
            elif df.compat.is_numeric(column):
                numeric_columns.append(column)
                continue

            if 'value_range' not in data:
                try:
                    min_value, max_value = df.compat.minmax(column)
                    if min_value is not None and max_value is not None:
                        data['value_range'] = MinMax(min_value, max_value)
                except ValueError as e:
                    _log.debug(f"AnnotatedDataFrame.infer_annotation: could not compute value range for {column} -- {e}")

            ds = DataSpecification(**data)
            column_specs.append(ds)

        if numeric_columns:
            # To allow polars to optimize the query, process all numeric columns at once
            results = df.compat.minmax_stats(numeric_columns)
            for column in numeric_columns:
                data = {'name': column,
                        'is_optional': False,
                        'representation_type': df.compat.dtype(column)
                }

                min_value = results[column]["min_value"]
                max_value = results[column]["max_value"]
                if min_value is not None and max_value is not None:
                    data['value_range'] = MinMax(results[column]["min_value"], results[column]["max_value"])
                data['value_stats'] = results[column]["stats"]

                ds = DataSpecification(**data)
                column_specs.append(ds)

        annotations: list[Annotation] = []
        if reference is not None:
            for ds in column_specs:
                if ds.name in reference:
                    reference_spec = reference[ds.name]
                    for field in cls._INHERITED_SPEC_FIELDS:
                        value = getattr(reference_spec, field, None)
                        if value is not None:
                            setattr(ds, field, value)
            annotations = list(reference.annotations.values())

        return MetaData(columns=column_specs, annotations=annotations)

    @classmethod
    def convert_csv_to_adf(
        cls,
        csv_filenames: list[Path | str],
        metadata_filename: Path | str,
        output_filename: Path | str,
        validation_mode: ValidationMode = ValidationMode.READONLY,
        progress: Callable[[float], None] | None = None,
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
        df = scan_csv(sorted(csv_filenames), separator=csv_sep)
        adf = cls(dataframe=df, metadata=metadata)

        _log.info(f"Metadata: {dict(metadata)}")
        _log.info(f"Saving dataframe into {output_filename}")

        adf.save(filename=output_filename)

    def drop(self, columns, strict: bool = True) -> AnnotatedDataFrame:
        self.lazyframe = self.lazyframe.drop(columns, strict=strict)
        self._metadata = self._metadata.drop(columns)
        return self

    def copy(self):
        return copy.deepcopy(self)

    @property
    def shape(self):
        return self.lazyframe.compat.collected().shape

    def __deepcopy__(self, memo=None):
        # ignore validation, also erroneous frames should be copyable
        return AnnotatedDataFrame(self.lazyframe.clone(), copy.deepcopy(self._metadata), validation_mode=ValidationMode.IGNORE,
                                  metadata_inferred=self._metadata_inferred)

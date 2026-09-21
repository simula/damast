from __future__ import annotations

import ast
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars
import polars.api
from polars import LazyFrame
from polars.io.plugins import register_io_source
from pydantic import ValidationError

from damast.utils import ensure_packages

from .constants import DAMAST_CSV_DEFAULT_ARGS
from .data_description import NumericValueStats

logger = logging.getLogger(__name__)

VAEX_HDF5_ROOT: str = "/table"
VAEX_HDF5_COLUMNS: str = f"{VAEX_HDF5_ROOT}/columns"

# Prefer an engine affinity the embedding application/environment has already configured
# (via POLARS_ENGINE_AFFINITY or a prior polars.Config.set_engine_affinity call) over
# unconditionally overriding it - only apply damast's own default when nothing is set yet.
if "POLARS_ENGINE_AFFINITY" in os.environ:
    logger.warning(
        "damast.core.polars_dataframe: POLARS_ENGINE_AFFINITY is already set to"
        f" '{os.environ['POLARS_ENGINE_AFFINITY']}' - keeping it instead of damast's default 'streaming'"
    )
else:
    polars.Config.set_engine_affinity("streaming")

POLARS_TYPE_DICT = {
    key: value
    for key, value in vars(polars).items()
    if isinstance(value, type) and issubclass(value, polars.DataType)
}
POLARS_TYPE_DICT["DataType"] = polars.DataType

class Meta(type):
    _base_impl: ClassVar[str] = "polars"

    def __getattr__(cls, attr_name):
        if cls._base_impl == 'polars':
            return getattr(polars, attr_name)

        raise AttributeError(f"'{cls.__name__}' has not attribute '{attr_name}'")

@polars.api.register_dataframe_namespace("compat")
@polars.api.register_lazyframe_namespace("compat")
class PolarsDataFrame(metaclass=Meta):
    _polars_dataframe: PolarsDataFrame
    _dataframe_collected: polars.DataFrame
    _minmax_cache: dict[str, tuple]

    def __init__(self, df: LazyFrame | polars.DataFrame):
        self.lazyframe = df

    @property
    def lazyframe(self) -> LazyFrame:
        """
        The underlying ``polars.LazyFrame``.

        This is the sole point of mutation for the wrapped dataframe - assigning to it (rather
        than e.g. a plain, private instance attribute) is what lets us keep the ``collected()``
        cache and the ``dataframe`` accessor consistent with the data that is actually stored,
        instead of silently returning a stale snapshot after an update.
        """
        return self.__lazyframe

    @lazyframe.setter
    def lazyframe(self, df: LazyFrame | polars.DataFrame):
        if type(df) is polars.DataFrame:
            df = df.lazy()

        self.__lazyframe = df
        self._dataframe_collected = None
        self._polars_dataframe = None
        self._minmax_cache = {}

    @classmethod
    def types(cls) -> dict[str, Any]:
        return POLARS_TYPE_DICT

    @classmethod
    def resolve_type(cls, type_txt: str):
        if type_txt == "datetime":
            type_txt = "Datetime"
        elif type_txt == "str":
            type_txt = "String"
        elif type_txt == "int":
            type_txt = "Int64"
        elif type_txt == "float":
            type_txt = "Float64"

        try:
            return cls.types()[type_txt]
        except KeyError:
            pass

        # A parameterized dtype (e.g. "Datetime(time_unit='us', time_zone='UTC')", as produced
        # by str() on a DataType instance - see DataSpecification.__iter__) isn't a plain key in
        # types(). Parse it back into an instance instead of just the bare class, so such dtypes
        # round-trip through export/import instead of being silently downgraded/rejected.
        parameterized = cls._resolve_parameterized_type(type_txt)
        if parameterized is not None:
            return parameterized

        raise TypeError(f"{cls.__name__}.resolve_type: unknown polars type '{type_txt}'")

    @classmethod
    def _resolve_parameterized_type(cls, type_txt: str):
        """
        Parse a constructor-call-style dtype repr (e.g. "Datetime(time_unit='us')" or
        "List(Float64)") back into a `polars.datatypes.DataType` instance, or return None if
        `type_txt` isn't of that shape.

        Uses `ast` rather than `eval` to only ever construct a known polars dtype class, with
        arguments parsed as literals or (recursively) nested dtypes - never arbitrary code.
        """
        try:
            node = ast.parse(type_txt, mode="eval").body
        except SyntaxError:
            return None

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None

        dtype_class = cls.types().get(node.func.id)
        if dtype_class is None:
            return None

        try:
            args = [cls._resolve_arg(arg) for arg in node.args]
            kwargs = {kw.arg: cls._resolve_arg(kw.value) for kw in node.keywords}
            return dtype_class(*args, **kwargs)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _resolve_arg(cls, node: ast.expr):
        """
        Resolve a single argument node of a parameterized dtype repr: a bare dtype name (e.g.
        "Float64"), a nested parameterized dtype (e.g. "Datetime(time_unit='us')" or
        "List(Float64)"), a dict/list/tuple of such (e.g. "Struct({'a': List(String)})"), or a
        plain literal (str, int, ...).
        """
        if isinstance(node, ast.Name) and node.id in cls.types():
            return cls.types()[node.id]

        if isinstance(node, ast.Call):
            resolved = cls._resolve_parameterized_type(ast.unparse(node))
            if resolved is not None:
                return resolved

        if isinstance(node, ast.Dict):
            return {
                ast.literal_eval(key): cls._resolve_arg(value)
                for key, value in zip(node.keys, node.values)
            }

        if isinstance(node, (ast.List, ast.Tuple)):
            elements = [cls._resolve_arg(elt) for elt in node.elts]
            return elements if isinstance(node, ast.List) else tuple(elements)

        return ast.literal_eval(node)

    @property
    def dataframe(self) -> PolarsDataFrame:
        """
        Allows to access the underlying dataframe directly.

        .. note::
            AnnotatedDataFrame behaves like a ``polars.LazyFrame``, so typically you will not need to access the
            dataframe through this property.

        :return: The underlying dataframe
        """
        if self._polars_dataframe is None or self.lazyframe is not self._polars_dataframe.lazyframe:
            self._polars_dataframe = PolarsDataFrame(self.lazyframe)

        return self._polars_dataframe

    def collected(self):
        if self._dataframe_collected is None:
            self._dataframe_collected = self.lazyframe.collect()

        return self._dataframe_collected

    def is_string(self, column_name: str) -> bool:
        return str(self.dtype(column_name)).lower() in ["str", "string"]

    def is_numeric(self, column_name: str) -> bool:
        return self.dtype(column_name).is_numeric()

    def is_datetime(self, column_name: str) -> bool:
        return type(self.dtype(column_name)) is polars.Datetime

    def is_bool(self, column_name: str) -> bool:
        return type(self.dtype(column_name)) is polars.Boolean

    def is_date(self, column_name: str) -> bool:
        return type(self.dtype(column_name)) is polars.Date

    def __getitem__(self, column_name: str):
        """
        Make dataframe subscriptable and behave more like the :class:`pandas.DataFrame`.

        :param item: Name of the key when using [] operators
        :return: item/column from the underlying vaex.dataframe
        """
        return self.lazyframe.select(column_name)

    def ensure_column(self, column_name: str):
        """
        Ensure that a column exist, raise ValueError otherwise
        """
        if column_name not in self.column_names:
            raise ValueError("PolarsDataFrame.set_dtype: column '{column_name}' does not exist")

    @property
    def column_names(self) -> list[str]:
        """
        Get all column names (without collecting the full dataframe)
        """
        return self.lazyframe.collect_schema().names()

    def dtype(self, column_name: str) -> polars.datatypes.DataType:
        """
        Get column dtype (without collecting the full dataframe)
        """
        try:
            idx = self.column_names.index(column_name)
        except ValueError as e:
            if re.search("not in list", str(e)):
                raise ValueError(f"{e} -- known columns are {','.join(sorted(self.column_names))}")
            raise
        return self.lazyframe.collect_schema().dtypes()[idx]

    def set_dtype(self, column_name, representation_type) -> polars.datatype.DataType:
        """
        Set the dtype for a column to the given representation type.
        Using polars cast functionality
        :return: The updated object
        """
        self.ensure_column(column_name)

        if representation_type == np.int64:
            representation_type = polars.Int64
        elif type(representation_type) is str:
            if hasattr(polars, representation_type):
                representation_type = getattr(polars, representation_type)

        self.lazyframe = self.lazyframe.with_columns(polars.col(column_name).cast(representation_type).alias(column_name))
        return representation_type

    def rescale(self, column_name: str, factor: float) -> None:
        """
        Multiply a column's values by `factor` in place (e.g. to convert between two
        equivalent physical units), preserving the column's existing representation type.
        """
        self.ensure_column(column_name)

        original_dtype = self.dtype(column_name)
        self.lazyframe = self.lazyframe.with_columns(
            (polars.col(column_name) * factor).cast(original_dtype).alias(column_name)
        )

    def precompute_minmax(self, column_names: list[str]) -> None:
        """
        Compute min/max for several columns in a single collect() and cache the
        results, so that later `minmax(column_name)` calls for these columns are
        served from cache instead of each triggering their own collect().

        The cache is invalidated automatically whenever `lazyframe` is reassigned.
        """
        if not column_names:
            return

        for column_name in column_names:
            self.ensure_column(column_name)

        fields = []
        for column_name in column_names:
            fields.extend([
                polars.col(column_name).min().alias(f"{column_name}::min"),
                polars.col(column_name).max().alias(f"{column_name}::max"),
            ])

        try:
            result = self.lazyframe.select(fields).collect()
        except polars.exceptions.InvalidOperationError as e:
            raise ValueError(
                f"damast.core.polars_dataframe.precompute_minmax: cannot compute min/max for columns {column_names}"
            ) from e

        for column_name in column_names:
            self._minmax_cache[column_name] = (
                result[f"{column_name}::min"][0],
                result[f"{column_name}::max"][0],
            )

    def minmax(self, column_name: str) -> tuple[Any, Any]:
        """
        Tuple of min and max values of the given column
        """
        self.ensure_column(column_name)

        if column_name in self._minmax_cache:
            return self._minmax_cache[column_name]

        try:
            result = self.lazyframe.select([
                    polars.col(column_name).min().alias("min_value"),
                    polars.col(column_name).max().alias("max_value")
                ]).collect()
        except polars.exceptions.InvalidOperationError as e:
            raise ValueError(f"damast.core.polars_dataframe.minmax: cannot compute min/max for {column_name}") from e

        min_value = result["min_value"][0]
        max_value = result["max_value"][0]
        self._minmax_cache[column_name] = (min_value, max_value)

        return min_value, max_value

    def categories(self, column_name: str, max_count: int = 100) -> list[str]:
        self.ensure_column(column_name)

        try:
            categories = self.lazyframe.select(column_name).unique().sort(by=column_name).collect()[:,0].to_list()
        except Exception as e:
            raise RuntimeError(f"Failed to extract categories for column '{column_name}' -- {e}") from e

        if len(categories) <= max_count:
            # do not count every timepoint as category
            timepoint_like = 0
            for c in categories[:10]:
                if c is None:
                    continue

                if type(c) is not str:
                    raise ValueError(f"Column {column_name} with unexpected category: {c}")

                if c and re.search(r"[0-9]{2}:[0-9]{2}", c) is not None:
                    timepoint_like += 1

            if timepoint_like < 3:
                return categories

        return None


    def minmax_stats(self, column_names: list[str]) -> dict[str, dict[str, Any]]:
        """
        Tuple of min and max values of the given column
        """
        fields = []
        for column in column_names:
            fields.extend([
                polars.col(column).min().alias(f"{column}_min_value"),
                polars.col(column).max().alias(f"{column}_max_value"),
                polars.col(column).count().alias(f"{column}_total_count"),
                polars.col(column).null_count().alias(f"{column}_null_count")
            ])

            if self.is_datetime(column):
                fields.extend([
                    (polars.col(column).dt.timestamp("us").mean() / 1_000_000).alias(f"{column}_mean"),
                    (polars.col(column).dt.timestamp("us").std() / 1_000_000).alias(f"{column}_stddev")
                ])
            else:
                fields.extend([
                    polars.col(column).mean().alias(f"{column}_mean"),
                    polars.col(column).std().alias(f"{column}_stddev"),
                    polars.col(column).median().alias(f"{column}_median"),
                    polars.col(column).quantile(0.25, interpolation="linear").alias(f"{column}_lower_quantile"),
                    polars.col(column).quantile(0.75, interpolation="linear").alias(f"{column}_upper_quantile"),
                ])

        result = self.lazyframe.select(
                fields
        ).collect()

        results = {}
        for column in column_names:
            min_value = result[f"{column}_min_value"][0]
            max_value = result[f"{column}_max_value"][0]
            try:
                stats = NumericValueStats(
                    mean=result[f"{column}_mean"][0],
                    stddev=result[f"{column}_stddev"][0],
                    # Not computed for datetime columns (see the is_datetime() branch above) -
                    # default to None there, same as an undefined stddev
                    median=result[f"{column}_median"][0] if f"{column}_median" in result.columns else None,
                    lower_quantile=result[f"{column}_lower_quantile"][0] if f"{column}_lower_quantile" in result.columns else None,
                    upper_quantile=result[f"{column}_upper_quantile"][0] if f"{column}_upper_quantile" in result.columns else None,
                    total_count=result[f"{column}_total_count"][0],
                    null_count=result[f"{column}_null_count"][0],
                )
            except ValidationError as e:
                logger.warning(f"Unable to compute stats for {column=}")
                logger.debug(f"Validation error for {column=} -- {e}")
                stats = None

            results[column] = {
                    "min_value": min_value,
                    "max_value": max_value,
                    "stats": stats
            }

        return results


    def stats(self, column_name: str) -> NumericValueStats:
        self.ensure_column(column_name)

        result = self.lazyframe.select([
            polars.col(column_name).mean().alias("mean"),
            polars.col(column_name).std().alias("stddev"),
            polars.col(column_name).median().alias("median"),
            polars.col(column_name).quantile(0.25, interpolation="linear").alias("lower_quantile"),
            polars.col(column_name).quantile(0.75, interpolation="linear").alias("upper_quantile"),
            polars.col(column_name).count().alias("total_count"),
            polars.col(column_name).null_count().alias("null_count")
        ]).collect()

        return NumericValueStats(
                mean=result['mean'][0],
                stddev=result['stddev'][0],
                median=result['median'][0],
                lower_quantile=result['lower_quantile'][0],
                upper_quantile=result['upper_quantile'][0],
                total_count=result['total_count'][0],
                null_count=result['null_count'][0]
        )

    def __getattr__(self, attr_name):
        """
        Ensure that this object behaves like a :class:`polars.LazyFrame`.

        :param attr_name: Attribute / Name of column
        :return: The column data
        """
        # allow dataframe.col_one
        if attr_name in self.column_names:
            return self.lazyframe.select(attr_name)

        if attr_name in ["__setstate__", "__getstate__"]:
            raise AttributeError(f"{self.__class__.__name__}.__getattr__: {attr_name} does not exist")

        """ Called for failed attribute accesses so forwarding to underlying polars frame """
        return getattr(self.lazyframe, attr_name)

    def __setitem__(self, key, values):
        """
        Set the column for the annotated dataframe, and allow to behave like the polars.Dataframe.

        :param key: Column name
        :param value: Value to set the column to
        """
        if type(values) is polars.LazyFrame:
            values = values.collect().to_numpy()

        self.lazyframe = self.lazyframe.with_columns(
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
        return len(self.collected())

    def equals(self, other: PolarsDataFrame) -> bool:
        return self.collected().equals(other.collected())

    @classmethod
    def open(cls, path: str | Path, sep = ',') -> polars.LazyFrame:
        path = Path(path)
        if path.suffix == ".csv":
            return polars.scan_csv(path,
                                   sep=sep,
                                   **DAMAST_CSV_DEFAULT_ARGS
            )
        elif path.suffix in [".h5", ".hdf5"]:
            import pandas as pd

            from damast.core.metadata import DAMAST_HDF5_ROOT

            pandas_df = pd.read_hdf(path, key=DAMAST_HDF5_ROOT)
            return polars.from_pandas(pandas_df)
        elif path.suffix in [".pq", ".parquet"]:
            return polars.scan_parquet(path)

        raise ValueError(f"{cls.__name__}.load_data: Unsupported input file format {path.suffix}")

    @classmethod
    def from_vaex_hdf5(cls, path: str | Path) -> tuple[polars.LazyFrame, 'MetaData']: # noqa
        """
        Load hdf5 file and (damast) metadata if found in the file.
        """
        # avoid circular dependencies
        from damast.core.annotations import Annotation
        from damast.core.metadata import DataSpecification, MetaData
        try:
            import tables
        except ImportError as e:
            raise RuntimeError("Could not load pytables -- "
                    "please install 'tables' to use hdf5 functionality") from e

        annotations = []
        column_specifications = []
        with tables.open_file(str(path)) as hdf5file:
            if VAEX_HDF5_ROOT not in hdf5file:
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

    #: Maximum number of grid cells read per batch by the lazy NetCDF scan (before dropping padding)
    NETCDF_BATCH_SIZE: ClassVar[int] = 100_000

    @classmethod
    def import_netcdf(cls, path: list[str|Path]) -> tuple[polars.LazyFrame, dict[str, 'MetaData']]: #noqa
        """
        Lazily scan NetCDF files - see :func:`scan_netcdf` - and extract metadata from their CF
        attributes, see :func:`_metadata_from_cf_attributes`.
        """
        frames = []
        metadata = {}
        for f in path:
            lazyframe, variables = cls.scan_netcdf(f)
            frames.append(lazyframe)

            file_metadata = cls._metadata_from_cf_attributes(lazyframe.collect_schema(), variables,
                                                             source=Path(f).name)
            if file_metadata is not None:
                metadata[str(f)] = file_metadata

        return polars.concat(frames, how="diagonal_relaxed"), metadata

    @classmethod
    def scan_netcdf(cls, path: str | Path) -> tuple[polars.LazyFrame, dict[str, tuple[dict, dict]]]:
        """
        Lazily scan a NetCDF file as a table with one row per grid cell - the same layout as
        ``xarray.Dataset.to_dataframe()``, with the dimensions as leading columns.

        Nothing is read until the frame is collected. The grid is then read in slices along its
        first dimension, so memory is bounded by a slice rather than the whole grid, and rows are
        filtered/projected/limited per slice. Rows in which every data variable spanning the full
        grid is missing - e.g. the padding of a sparse (entity x time) grid - are dropped.

        :param path: The NetCDF file
        :return: The lazyframe, and variable name -> (CF attributes, xarray encoding)
        """
        ensure_packages(pkgs=["xarray"],
                        required_for="Loading netcdf files",
                        hint="additionally either netCDF4 or h5netcdf have to be installed")
        import xarray

        with xarray.open_dataset(path) as ds:
            variables = {name: (dict(variable.attrs), dict(variable.encoding))
                         for name, variable in ds.variables.items()}
            schema = cls._netcdf_schema(ds)

        def read_batches(with_columns: list[str] | None,
                         predicate: polars.Expr | None,
                         n_rows: int | None,
                         batch_size: int | None):
            with xarray.open_dataset(path) as ds:
                dims = list(ds.sizes)
                # A cell is padding if all variables spanning the full grid are missing there - lower
                # dimensional ones (e.g. static per-entity values) are just repeated into every cell
                data_vars = [name for name, var in ds.data_vars.items() if set(var.dims) == set(dims)]
                data_vars = data_vars or list(ds.data_vars)
                # cells per step along the first dimension - which to_dataframe() iterates slowest
                cells_per_step = math.prod(list(ds.sizes.values())[1:])
                # polars' batch_size is only a hint - cap it, so memory stays bounded per slice
                max_cells = min(batch_size or cls.NETCDF_BATCH_SIZE, cls.NETCDF_BATCH_SIZE)
                step = max(1, max_cells // max(1, cells_per_step))
                first_dim_size = ds.sizes[dims[0]] if dims else 1

                for start in range(0, first_dim_size, step):
                    if n_rows is not None and n_rows <= 0:
                        return

                    part = ds.isel({dims[0]: slice(start, start + step)}) if dims else ds
                    pandas_df = part.to_dataframe().reset_index()
                    if data_vars:
                        pandas_df = pandas_df.dropna(how="all", subset=data_vars)

                    # e.g. an all-missing string column would otherwise come back as Null
                    df = polars.from_pandas(pandas_df).cast(schema)
                    if predicate is not None:
                        df = df.filter(predicate)
                    if with_columns is not None:
                        df = df.select(with_columns)
                    if n_rows is not None:
                        df = df.head(n_rows)
                        n_rows -= df.height
                    yield df

        # According to polars documentation this functionality is considered unstable
        # https://docs.pola.rs/api/python/stable/reference/api/polars.io.plugins.register_io_source.html
        return register_io_source(read_batches, schema=schema), variables

    @staticmethod
    def _netcdf_schema(ds) -> polars.Schema:
        """
        Columns and dtypes of ``ds.to_dataframe()`` without reading data: taken from an empty
        slice, where object columns (strings) cannot be inferred and default to String.
        """
        empty = ds.isel({dim: slice(0, 0) for dim in ds.sizes}).to_dataframe().reset_index()
        return polars.Schema({
            column: polars.String if dtype.kind == "O" else polars.from_pandas(empty[column]).dtype
            for column, dtype in empty.dtypes.items()
        })

    @classmethod
    def _metadata_from_cf_attributes(cls,
                                     schema: polars.Schema,
                                     variables: dict[str, tuple[dict, dict]],
                                     source: str) -> 'MetaData' | None: # noqa
        """
        Create metadata for the columns of a loaded NetCDF file from the CF attributes of its
        variables: 'long_name' becomes the description, 'units' the unit - if it can be parsed -,
        and 'valid_range'/'valid_min'/'valid_max' the value range of a numeric column.

        '_FillValue'/'missing_value' are not mapped: xarray already decodes them to NaN (null in
        polars), while damast's missing_value is the value used to replace out-of-range values.

        :param variables: variable name -> (attributes, xarray encoding)
        :return: The metadata, or None if no variable carries any of these attributes - so that
            callers can fall back to searching for a spec file or inferring the metadata
        """
        # avoid circular dependencies
        from damast.core.annotations import Annotation
        from damast.core.data_description import MinMax
        from damast.core.metadata import DataSpecification, MetaData
        from damast.core.units import Unit

        column_specs = []
        has_cf_attributes = False
        for column, dtype in schema.items():
            attrs, encoding = variables.get(column, ({}, {}))
            spec = DataSpecification(name=column, representation_type=dtype)

            if "long_name" in attrs:
                spec.description = str(attrs["long_name"])
                has_cf_attributes = True

            if "units" in attrs:
                has_cf_attributes = True
                try:
                    spec.unit = Unit(str(attrs["units"]))
                except ValueError:
                    logger.info(f"NetCDF {source}: cannot interpret unit '{attrs['units']}' of '{column}' - ignoring it")

            valid_range = cls._cf_valid_range(attrs, encoding)
            if valid_range is not None:
                has_cf_attributes = True
                # e.g. a decoded time column cannot be compared with its (numeric) raw range
                if dtype.is_numeric():
                    spec.value_range = MinMax(*valid_range)
                else:
                    logger.info(f"NetCDF {source}: ignoring valid range of non-numeric '{column}'")

            column_specs.append(spec)

        if not has_cf_attributes:
            return None

        return MetaData(columns=column_specs,
                        annotations=[Annotation(name=Annotation.Key.Source, value=source)])

    @staticmethod
    def _cf_valid_range(attrs: dict, encoding: dict) -> tuple[Any, Any] | None:
        """
        (min, max) from the CF 'valid_range', or 'valid_min'/'valid_max' attributes - an open side
        becomes -inf/inf. CF defines them in packed units, so they are unpacked like the data via
        'scale_factor'/'add_offset', which xarray moves into the variable's encoding.

        :return: The range, or None if the variable declares none
        """
        if "valid_range" in attrs:
            low, high = np.asarray(attrs["valid_range"]).tolist()
        elif "valid_min" in attrs or "valid_max" in attrs:
            low = np.asarray(attrs.get("valid_min", -np.inf)).item()
            high = np.asarray(attrs.get("valid_max", np.inf)).item()
        else:
            return None

        if "scale_factor" in encoding or "add_offset" in encoding:
            scale = float(encoding.get("scale_factor", 1.0))
            offset = float(encoding.get("add_offset", 0.0))
            # a negative scale_factor swaps the bounds
            low, high = sorted([low * scale + offset, high * scale + offset])

        return low, high

    @classmethod
    def import_hdf5(cls, files: str | Path | list[str|Path]) -> tuple[polars.LazyFrame, dict[str, 'MetaData']]: # noqa
        """
        Import a dataframe stored as HDF5.

        This method tries to load using pandas first, then falls back to reading a vaex-based format
        using pytables.
        """
        ensure_packages(["tables", "pandas"],
                        required_for="Loading hdf5 files",
                        install={"tables": "pytables"})

        import pandas
        import tables

        if type(files) is not list:
            files = [files]

        try:
            import warnings

            # Avoid output pollution and ignore the following message
            #     tables/attributeset.py:295: DataTypeWarning: Unsupported type
            #     for attribute 'is_optional' in node 'height'. Offending HDF5
            #     class: 8
            warnings.filterwarnings("ignore", message="Unsupported type for attribute .*")

            data_frames = []
            for filename in files:
                pandas_df = pandas.read_hdf(str(filename))

                with tables.open_file(str(filename)) as hdf5file:
                    for column in pandas_df.columns:
                        column_attrs = hdf5file.get_node(f"/dataframe/columns/{column}")._v_attrs
                        if "representation_type" in column_attrs:
                            if column_attrs["representation_type"] == "int":
                                pandas_df[column] = pandas_df[column].astype("Int64")

                data_frames.append(pandas_df)
            pandas_df = pandas.concat(data_frames, ignore_index=True)

            df = polars.from_pandas(pandas_df)
            warnings.resetwarnings()

            return df.lazy(), {}
        except tables.exceptions.NoSuchNodeError:
            logger.debug(f"HDF5 {filename} cannot be imported with pandas")

        if len(files) > 1:
            raise RuntimeError("Loading from vaex is only supported with one file at a time")

        path = Path(files[0])
        df, metadata = cls.from_vaex_hdf5(path)

        return df, { path: metadata }

    @classmethod
    def export_hdf5(cls, df: polars.DataFrame | polars.LazyFrame, path: str | Path) -> Path:
        """
        Export the dataframe as hdf5. Please use only if really needed, otherwise, stick with the
        default format (parquet).
        """
        ensure_packages(pkgs=["pandas"],
                        required_for="Loading hdf5 files")


        from damast.core.metadata import DAMAST_HDF5_ROOT

        if isinstance(df, polars.dataframe.DataFrame):
            df = df.lazy()

        # An export of an int column with NaNs will automatically convert this to Float64
        # while Int64 could be used, the underlying exporter pytables does not (yet) support this
        pandas_df = df.collect().to_pandas()
        pandas_df.to_hdf(path, key=DAMAST_HDF5_ROOT)
        return path




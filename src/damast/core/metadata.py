"""
Module to collect the classes to define meta data
"""
from __future__ import annotations

import ast
import builtins
import inspect
import logging
import os
import re
import traceback
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import yaml

from .annotations import Annotation, History
from .constants import (
    DAMAST_HDF5_COLUMNS,
    DAMAST_HDF5_ROOT,
    DAMAST_SPEC_SUFFIX,
    DAMAST_SUPPORTED_FILE_FORMATS,
    )
from .data_description import DataElement, DataRange, MinMax, NumericValueStats
from .formatting import DEFAULT_INDENT
from .types import DataFrame, XDataFrame
from .units import Unit, unit_registry, units

__all__ = [
    "ArtifactSpecification",
    "DataCategory",
    "DataSpecification",
    "MetaData",
    "Status",
    "ValidationMode",
]

logger = logging.getLogger(__name__)

class DataCategory(str, Enum):
    """
    Specification of what state a data-set is in (static or dynamic)
    """

    DYNAMIC = "dynamic"
    """Data-category is dynamic"""

    STATIC = "static"
    """Data-category is static"""


class Status(str, Enum):
    """
    Denoting if :class:`DataSpecification` adheres to :class:`DataSpecification.Fulfillment`
    """

    OK = "OK"
    """Data-specification is fulfilled"""

    FAIL = "FAIL"
    """Data-specification is not adhered to"""

class ArtifactSpecification:
    """
    Specification of a (file) artifact that might be generated during data processing.

    :param requirements: Dictionary describing expected path artifacts, key is a descriptor,
        value an absolute path or a relative path (pattern)
    """

    #: Dictionary mapping an artifact description to a file glob pattern
    artifacts: Dict[str, str]

    def __init__(self, requirements: Dict[str, Any]) -> None:
        self.artifacts = requirements

    def validate(self, base_dir: Path):
        """
        Validate the list of artifacts.

        :param base_dir: Relative paths/patterns will be interpreted with ``base_dir`` as current working directory
        :raise RuntimeError: When a matching file could not be found
        """
        for name, path_pattern in self.artifacts.items():
            a_path = Path(path_pattern)
            if a_path.is_absolute():
                if not a_path.exists():
                    raise RuntimeError(
                        f"{self.__class__.__name__}.validate: artifact '{a_path}' is "
                        "missing"
                    )
            else:
                files = list(Path(base_dir).glob(path_pattern))
                if not files:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.validate: no artifact matching "
                        f" {path_pattern} found in '{base_dir}'"
                    )


class ValidationMode(str, Enum):
    """
    Specification of how to apply meta-data to :class:`DataFrame`
    """

    IGNORE = "IGNORE"
    """Metadata will not be validated"""

    READONLY = "READONLY"
    """Metadata cannot be changed"""

    UPDATE_DATA = "UPDATE_DATA"
    """Data should be updated to comply with the metadata"""

    UPDATE_METADATA = "UPDATE_METADATA"
    """Metadata should be updated to comply with the data"""


class DataSpecification:
    """
    Specification of a single column and/or dimension in a :class:`damast.core.dataframe.DataFrame`.

    :param name: Name of the element
    :param category: A :class:`DataCategory` element
    :param is_optional: If column is optional
    :param abbreviation: Abbreviation of name
    :param representation_type: The underlying representation type for this data element
    :param missing_value: The representation of a missing value for this data element
    :param unit: The (physical) unit of this data element
    :param precision: The precision of this data element
    :param value_range: The allowed data range, which remains as ``None`` if unrestricted
    :param value_stats: The statistics about the data
    :param value_meanings: A description of value meanings
    """

    class Key(str, Enum):
        """
        Valid inputs to a :class:`DataSpecification`
        """

        name = "name"
        description = "description"
        category = "category"
        is_optional = "is_optional"
        abbreviation = "abbreviation"

        representation_type = "representation_type"
        missing_value = "missing_value"
        precision = "precision"
        unit = "unit"
        value_range = "value_range"
        value_stats = "value_stats"
        value_meanings = "value_meanings"

    class Fulfillment:
        """
        Base-class used for specific fulfillments
        """

        # Would use Key here, but the concept for that is not yet ironed out in Python
        status: Dict[Enum, Dict[str, Union[Status, str]]]

        def __init__(self):
            self.status = {}

        def is_met(self) -> bool:
            """
            Check the status property to see if the fulfillment is considered as met.

            :raises KeyError: If `self.status` does not have the key ``"status"``
            :return: ``True`` if fulfillment is met, else ``False``.
            """
            for _, v in self.status.items():
                if "status" not in v:
                    raise KeyError(
                        f"{self.__class__.__name__}: internal error. Status misses key 'status'"
                    )

                if v["status"] == Status.FAIL:
                    return False
            return True

        def __repr__(self):
            status = {}
            for k, v in self.status.items():
                msg = v["status"].value
                if "message" in v:
                    msg += f" {v['message']}"
                status[k.value] = msg

            return f"{self.__class__.__name__}[{status}]"

    class MissingColumn(Fulfillment):
        """
        Represent a missing column fulfillment.

        :param missing_column: Name of the missing column
        :param known_columns: Names of the (by specification) known columns
        """

        missing_column: str
        known_columns: List[str]

        def __init__(self, missing_column: str, known_columns: List[str]):
            super().__init__()

            self.missing_column = missing_column
            self.known_columns = known_columns

        def is_met(self) -> bool:
            """
            Check if fulfillment is met.

            :return: Always returns False, since this represents a missing fulfillment
            """
            return False

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}: '{self.missing_column}', known: {','.join(self.known_columns)}"

    #: Name associated
    name: str
    #: Description of the data
    description: Optional[str] = None
    #: Category of data
    category: Optional[DataCategory] = None
    #: Whether this data element needs to be present
    is_optional: Optional[bool] = None
    abbreviation: Optional[str] = None

    #: The underlying representation type for this data element
    representation_type: Any = None
    #: The representation of a missing value for this data element
    missing_value: Any = None

    #: The unit of this data element
    unit: Optional[Unit] = None
    #: The precision of this data element
    # FIXME: The input to precision could be a `List[float]`, but this is not currently handled
    precision: Optional[float] = None

    #: The allowed data range, which remains None when being unrestricted
    value_range: Optional[DataRange] = None
    #: An explanation - str-based descriptor for the range, if this is a dictionary then it must provide
    #: a mapping from the value to a human-readable descriptor
    value_meanings: Optional[Dict[Any, str]] = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        category: Optional[Union[str, DataCategory]] = None,
        is_optional: bool = False,
        abbreviation: Optional[str] = None,
        representation_type: Any = None,
        missing_value: Any = None,
        unit: Optional[Unit] = None,
        precision: Any = None,
        value_range: Optional[DataRange] = None,
        value_stats: Optional[NumericValueStats] = None,
        value_meanings: Optional[Dict[Any, str]] = None,
    ):
        """
        Constructor
        """

        if name is None:
            raise ValueError(
                f"{self.__class__.__name__}.__init__: " f"name cannot be None"
            )

        self.name = name
        if isinstance(category, str):
            self.category = DataCategory(category)
        else:
            self.category = category

        self.is_optional = is_optional
        if abbreviation is not None and abbreviation != "":
            self.abbreviation = abbreviation

        self.representation_type = representation_type
        self.missing_value = missing_value

        self.unit = unit
        self.precision = precision
        self.value_range = value_range
        self.value_stats = value_stats
        self.value_meanings = value_meanings
        self.description = description
        self._validate()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        return all(
            getattr(self, m.value) == getattr(other, m.value)
            for m in DataSpecification.Key
        )

    def __repr__(self):
        return f"{self.__class__.__name__}[{dict(self)}]"

    def _validate(self):
        """
        Validate the data specification

        :raises ValueError: If the specification is invalid
        """
        if self.value_meanings and not self.value_range:
            raise ValueError(
                f"{self.__class__.__name__}.__init__: "
                f" value meanings requires value range to be set"
            )

        if self.value_meanings:
            for k, _ in self.value_meanings.items():
                if k not in self.value_range:
                    raise ValueError(
                        f"{self.__class__.__name__}.validate: "
                        " value {k} is not in the defined value range"
                    )

    @classmethod
    def resolve_representation_type(cls, type_name: str) -> Any:
        """
        Resolve the representation type from a given string.

        .. highlight:: python
        .. code-block:: python

            int_type = DataSpecification.resolve_representation_type("int")
            assert int_type == int

        :param type_name: Name of the type
        :return: Instance of the type object
        :raise ValueError: Raises if ``type_name`` cannot be resolved to a known type (in builtins or :mod:`polars`)
        """
        exceptions: List[Any] = []
        try:
            if type_name.lower() in ["string", "str"]:
                dtype = str
            else:
                dtype = getattr(builtins, type_name)
            return dtype
        except AttributeError as e:
            exceptions.append(e)

        try:
            m = re.match("(.*)\((.*)\)", type_name)
            call_args = None
            if m is not None:
                type_name = m.group(1)
                call_args = m.group(2)
            dtype = getattr(pl.datatypes, type_name)
            if call_args:
                dtype = eval(f"pl.datatypes.{dtype}({call_args})")
            return dtype
        except TypeError as e:
            exceptions.append(e)

        raise ValueError(
            f"{cls.__name__}.resolve_representation_type: "
            f" could not find type '{type_name}'."
            f"It does not exist in builtins, nor "
            f" is this a known polar datatype -- {exceptions}"
        )

    def __iter__(self):
        yield "name", self.name
        if self.is_optional is not None:
            yield "is_optional", self.is_optional

        if self.abbreviation is not None:
            yield "abbreviation", self.abbreviation

        if self.category is not None:
            yield "category", self.category.value

        if self.representation_type is not None:
            if inspect.isclass(self.representation_type):
                yield "representation_type", self.representation_type.__name__
            elif isinstance(self.representation_type, pl.datatypes.DataType):
                yield "representation_type", str(self.representation_type)
            else:
                raise TypeError(
                    f"{self.__class__.__name__}.__iter__ failed to identify representation_type from"
                    f" {self.representation_type}"
                )

        if self.missing_value is not None:
            yield "missing_value", self.missing_value

        if self.unit is not None:
            yield "unit", self.unit.to_string()

        if self.precision is not None:
            yield "precision", self.precision

        if self.value_range is not None:
            yield "value_range", dict(self.value_range)

        if self.value_stats is not None:
            yield "value_stats", self.value_stats.__dict__

        if self.value_meanings is not None:
            yield "value_meanings", self.value_meanings

        if self.description is not None:
            yield "description", self.description

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataSpecification:
        """
        Load the data specification from a given dictionary.

        This function in intended to deserialize specifications from a string-based
        representation

        :param data: data dictionary using primitive data types to represent the specification
        :return: the loaded data specification
        :raise KeyError: Raise if a required key is missing
        """
        for key in [cls.Key.name, cls.Key.is_optional]:
            if key.value not in data:
                raise KeyError(f"{cls.__name__}.from_dict: missing '{key.value}'")

        abbreviation = None
        if cls.Key.abbreviation.value in data:
            abbreviation = data[cls.Key.abbreviation.value]
            if not isinstance(abbreviation, str):
                raise TypeError(
                    f"{cls.__name__}.from_dict: " f"abbreviation must be of type 'str'"
                    f", but was {type(abbreviation)} - {abbreviation}"
                )

        data_category = None
        if cls.Key.category.value in data:
            data_category = data[cls.Key.category]
            if not isinstance(data_category, DataCategory):
                data_category = DataCategory(data_category)

        spec = cls(
            name=data[cls.Key.name],
            category=data_category,
            is_optional=bool(data[cls.Key.is_optional]),
            abbreviation=abbreviation,
        )

        if cls.Key.description in data:
            spec.description = data[cls.Key.description]

        if cls.Key.representation_type.value in data:
            spec.representation_type = cls.resolve_representation_type(
                data[cls.Key.representation_type.value]
            )
            if cls.Key.missing_value.value in data:
                spec.missing_value = DataElement.create(
                    data[cls.Key.missing_value.value], spec.representation_type
                )
            if cls.Key.precision.value in data:
                spec.precision = DataElement.create(
                    data[cls.Key.precision.value], spec.representation_type
                )
        # else: Missing representation type, missing value and precision will not be loaded

        if cls.Key.unit.value in data:
            # Check if unit is part of the default definitions
            unit_value = data[cls.Key.unit.value]
            unit = None
            if isinstance(unit_value, str):
                # This will include astropy and custom (damast) units
                unit = Unit(unit_value)
            elif isinstance(unit_value, Unit):
                unit = unit_value
            else:
                raise RuntimeError(
                    f"{cls.__name__}.from_dict: cannot interpret unit type "
                    f"'{type(unit_value)}"
                )

            spec.unit = unit

        if cls.Key.value_range.value in data:
            value_range = data[cls.Key.value_range.value]
            if isinstance(value_range, DataRange):
                spec.value_range = value_range
            else:
                if isinstance(value_range, str):
                    value_range = ast.literal_eval(value_range)
                spec.value_range = DataRange.from_dict(
                    value_range, dtype=spec.representation_type
                )

            if cls.Key.value_meanings.value in data:
                spec.value_meanings = data[cls.Key.value_meanings.value]

            if cls.Key.value_stats.value in data:
                if not str(spec.representation_type).lower().startswith("str"):
                    spec.value_stats = NumericValueStats(**data[cls.Key.value_meanings.value])

        return spec

    @classmethod
    def to_str(cls, specs: List[DataSpecification], indent_level=0):
        """
        Generate string representation for list of specs

        :param specs: List of data specifications
        :param indent_level: indentation levels
        :param spaces: number of space for one indentation level
        :return: String representation of specification
        """
        hspace = DEFAULT_INDENT * indent_level

        data = ""
        for spec in specs:
            data += hspace + spec.name
            if spec.unit is not None:
                data += f"[unit: {spec.unit.to_string()}]"
            data += "\n"
        return data

    def apply(
        self,
        df: DataFrame | XDataFrame,
        column_name: str,
        validation_mode: ValidationMode
        ) -> DataFrame:
        """
        Apply the metadata object to the dataframe

        :param df: Dataframe that should be associated with the data-specification
        :param column_name: Name of the column that need to be validated/checked
        :param validation_mode: Mode which shall apply to updating either data or
            metadata, when encountering inconsistencies
        """

        df_type = type(df)
        if df_type == pl.LazyFrame:
            pass
        elif df_type == pl.DataFrame:
            df = df.lazy()
        elif df_type == XDataFrame:
            df = df._dataframe
        else:
            raise TypeError("MetaData.apply: dataframe must be either "
                " polars.LazyFrame,"
                " polars.DataFrame,"
                " or damast.core.types.XDataFrame"
            )

        if validation_mode == ValidationMode.IGNORE:
            return df

        # Check if representation type is the same and apply known metadata
        if validation_mode == ValidationMode.READONLY:
            xdf = XDataFrame(df)
            if self.representation_type is not None:
                dtype = xdf.dtype(column_name)
                if dtype.from_python(self.representation_type) != self.representation_type and \
                    dtype.to_python() != self.representation_type:
                    raise ValueError(
                        f"{self.__class__.__name__}.apply: column '{column_name}':"
                        f" expected representation type: {self.representation_type},"
                        f" but got '{dtype}'"
                    )

            if self.value_range:
                min_value, max_value = xdf.minmax(column_name)
                if not self.value_range.is_in_range(min_value):
                    raise ValueError(
                        f"{self.__class__.__name__}.apply: minimum value '{min_value}'"
                        f" lies outside of range {self.value_range} for column '{column_name}'"
                    )
                if not self.value_range.is_in_range(max_value):
                    raise ValueError(
                        f"{self.__class__.__name__}.apply: maximum value '{max_value}'"
                        f" lies outside of range {self.value_range} for column '{column_name}'"
                    )
            return df

        if validation_mode == ValidationMode.UPDATE_DATA:
            xdf = XDataFrame(df)
            if self.representation_type is not None:
                dtype = xdf.dtype(column_name)
                if dtype != self.representation_type:
                    warnings.warn(
                        f"{self.__class__.__name__}.apply: column '{column_name}':"
                        f" expected representation type: {self.representation_type},"
                        f" but got '{dtype}'"
                    )
                    xdf.set_dtype(column_name, self.representation_type)

            if self.value_range:
                if self.missing_value is None:
                    warnings.warn(
                        f"Filtering out for column '{column_name}' values that are out of range."
                    )
                    xdf._dataframe = xdf._dataframe.filter(
                            (pl.col(column_name) >= self.value_range.min) &
                            (pl.col(column_name) <= self.value_range.max)
                        )
                else:
                    xdf._dataframe = xdf._dataframe.with_columns(
                                pl.when(
                                    (pl.col(column_name) < self.value_range.min) |
                                    (pl.col(column_name) > self.value_range.max)
                                ).then(self.missing_value)
                                .otherwise(pl.col(column_name))
                                .alias(column_name)
                              )
            return xdf._dataframe

        if validation_mode == ValidationMode.UPDATE_METADATA:
            xdf = XDataFrame(df)
            self.representation_type = xdf.dtype(column_name)

            try:
                min_value, max_value = xdf.minmax(column_name)
                if self.value_range:
                    if isinstance(self.value_range, MinMax):
                        self.value_range.merge(MinMax(min_value, max_value))
                else:
                     warnings.warn(
                         f"Setting MinMax range ({min_value}, {max_value}) for {column_name}"
                     )
                     self.value_range = MinMax(min_value, max_value)
            except ValueError:
                # Type might not be numeric
                pass
            return xdf._dataframe

    def get_fulfillment(self, data_spec: DataSpecification) -> Fulfillment:
        """
        Check the fulfillment of an existing and expected data specification against an available one.

        The current instance used for calling this method sets the expectations.

        :param data_spec: The available data specification
        :return: The degree of fulfillment
        """
        fulfillment = DataSpecification.Fulfillment()

        for key in DataSpecification.Key:
            expected_value = getattr(self, key.value)
            # Nothing expected so status is OK
            if expected_value is None:
                fulfillment.status[key] = {"status": Status.OK}
                continue

            # Value expected, but no value available in the tested spec so FAIL
            spec_value = getattr(data_spec, key.name)
            if spec_value is None:
                fulfillment.status[key] = {
                    "status": Status.FAIL,
                    "message": "column has no precision defined",
                }

            # some special handling is required for individual keys
            if key == DataSpecification.Key.precision:
                    if expected_value < spec_value:
                        fulfillment.status[key] = {
                            "status": Status.FAIL,
                            "message": "'data has insufficient precision: "
                            f" required '{expected_value}',"
                            f" available '{spec_value}'",
                        }
                    else:
                        fulfillment.status[key] = {"status": Status.OK}
            elif key == DataSpecification.Key.representation_type:
                if not issubclass(spec_value, expected_value):
                    fulfillment.status[key] = {
                        "status": Status.FAIL,
                        "message": "'representation type is not a subclass of the expected:"
                        f" required '{expected_value}',"
                        f" available '{spec_value}'",
                    }
                else:
                    fulfillment.status[key] = {"status": Status.OK}
            else:
                expected_value = getattr(self, key.value)
                if expected_value is not None:
                    spec_value = getattr(data_spec, key.name)
                    if spec_value is None:
                        fulfillment.status[key] = {
                            "status": Status.FAIL,
                            "message": f"Expected '{spec_value=}' == '{expected_value=}'",
                        }
                        continue

                    if expected_value != spec_value:
                        fulfillment.status[key] = {
                            "status": Status.FAIL,
                            "message": f"Expected '{spec_value=}' == '{expected_value=}'",
                        }
                        continue

                    fulfillment.status[key] = {"status": Status.OK}
        return fulfillment

    @classmethod
    def from_requirements(cls, requirements: Dict[str, Any]) -> List[DataSpecification]:
        """
        Get the list of DataSpecification from dictionary (keyword argument) based representation.

        DataSpecification.from_requirements(requirements={ "column_name": { "unit": units.m }})

        :param requirements: List of dictionaries describing the data specification
        :return: List of expectations
        """
        required_specs = []
        for k, v in requirements.items():
            kwargs = v
            # consider the key to be the column name
            kwargs[DataSpecification.Key.name.value] = k
            if DataSpecification.Key.category.value not in kwargs:
                kwargs[DataSpecification.Key.category.value] = None

            required_spec = cls(**kwargs)
            required_specs.append(required_spec)
        return required_specs

    def merge(self, other: DataSpecification) -> DataSpecification:
        """Merge the current data-specification and one input specification
        into a single :class:`DataSpecification` object.

        :param other: The other data-specification
        :raises ValueError: If :attr:`DataSpecification.name` differs between the two specs.
        :raises ValueError: If the data-specifications have overlapping attributes, that have distinct non-``None``
            values, the function throws an error.
        """
        if self.name != other.name:
            raise ValueError(
                f"{self.__class__.__name__}.merge: cannot merge specs with different name property"
            )

        ds = DataSpecification(name=self.name)
        for key in self.Key:
            if key == self.Key.name:
                continue

            this_value = getattr(self, key.value)
            other_value = getattr(other, key.value)

            if this_value is None:
                setattr(ds, key.value, other_value)
            elif other_value is None:
                setattr(ds, key.value, this_value)
            elif this_value == other_value:
                setattr(ds, key.value, this_value)
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.merge cannot merge specs: value for '{key.value}' differs: "
                    f" on self: '{this_value}' vs. other: '{other_value}'"
                )
        return ds

    @classmethod
    def merge_lists(
        cls, a_specs: List[DataSpecification], b_specs: List[DataSpecification]
    ) -> List[DataSpecification]:
        """
        Merge two lists of data-specifications into a single list.

        If a :class:`DataSpecification` in ``a_specs`` has the same :attr:`DataSpecification.name`
        as one in ``b_specs``, the specs are merged using :func:`merge`.

        :param a_specs: First list of specs
        :param b_specs: Second list of specs
        """
        result_specs: List[DataSpecification] = []

        b_column_dict = {x.name: x for x in b_specs}
        a_columns_names = []

        for a_spec in a_specs:
            column_name = a_spec.name
            if column_name in b_column_dict:
                # Need to check merge
                b_column_spec = b_column_dict[column_name]

                merged_spec = a_spec.merge(other=b_column_spec)
                result_specs.append(merged_spec)
            else:
                result_specs.append(a_spec)

            a_columns_names.append(a_spec.name)

        for b_spec in b_specs:
            if b_spec.name not in a_columns_names:
                result_specs.append(b_spec)

        return result_specs


class MetaData:
    """
    The representation for metadata that can be associated with a :class:`polars.LazyFrame`.

    :param columns: (Ordered) list of column specifications
    :param annotations:  List of annotations for this dataframe.

    .. note::
        Each annotation is assumed to have a unique :attr:`Annotation.name`.
    """

    class Key(str, Enum):
        """
        The two allowed objects in the :class:`MetaData` and their string representation.
        This is used for the dictionary representation dict(<MetaData>), :func:`MetaData.__iter__`,
        :func:`MetaData.from_dict`.
        """

        columns = "columns"
        annotations = "annotations"

    class Fulfillment:
        """
        A class containing all fulfillments (set of :class:`DataSpecification.Fulfillment`).

        A fulfillment describes whether a constraint on a column holds or not, added to :class:`MetaData`
        """

        column_fulfillments: Dict[str, DataSpecification.Fulfillment]

        def __init__(self):
            self.column_fulfillments = {}

        def is_met(self) -> bool:
            """Check if all fulfillments in the set of data-specifications are met."""
            return all(v.is_met() for k, v in self.column_fulfillments.items())

        def add_fulfillment(
            self, column_name: str, fulfillment: DataSpecification.Fulfillment
        ) -> None:
            """Add fulfillment to a given column.

            .. note::
                This function replaces any old fulfillment set on this column.
                Use :func:`DataSpecification.merge` to combine the existing spec
                and new spec prior to calling this function

            :param column_name: Name of column
            :param fulfillment: Fulfillment object
            """
            assert isinstance(fulfillment, DataSpecification.Fulfillment)
            self.column_fulfillments[column_name] = fulfillment

        def __repr__(self) -> str:
            txt = ""
            for column_name, fulfillment in self.column_fulfillments.items():
                txt += f"[{column_name}: {fulfillment}]"
            return txt

    #: Specification of columns in the
    columns: List[DataSpecification]

    # We store the annotations as a dictionary for easy lookup.
    # even if it means duplicating the name of the annotation as a key
    #: Dictionary containing all annotations
    _annotations: Dict[str, Annotation]

    def __init__(
        self,
        columns: List[DataSpecification],
        annotations: Optional[List[Annotation]] = None,
    ):
        assert isinstance(columns, list)
        self.columns = columns
        if annotations is None:
            self._annotations = {}
        else:
            assert isinstance(annotations, list)
            unique_annotations = {annotation.name for annotation in annotations}
            if len(unique_annotations) != len(annotations):
                raise ValueError(
                    f"{self.__class__.__name__}: Set of annotations in metadata has duplicate names "
                    + f"got {[dict(an) for an in annotations]}"
                )
            self._annotations = {an.name: an for an in annotations}

    def add_annotation(self, annotation: Annotation):
        if annotation.name in self._annotations:
            raise KeyError(f"Annotation {annotation.name} already exists")

        self._annotations[annotation.name] = annotation

    @property
    def annotations(self) -> Dict[str, Annotation]:
        """Get dictionary of annotations"""
        return self._annotations

    def __eq__(self, other) -> bool:
        """
        Check if the current meta-data object is equal to another.

        .. note::
            This function considers two :class:`MetaData` objects
            with the same columns and annotations (but in different order)
            as different objects
        """
        if self.__class__ != other.__class__:
            return False

        if self.columns != other.columns:
            return False

        if len(self.annotations) != len(other.annotations):
            return False

        for k,v in self.annotations.items():
            if k in other.annotations and other.annotations[k] == v:
                continue
            return False

        return True

    def __iter__(self):
        columns = [dict(ds) for ds in self.columns]
        yield self.Key.columns.value, columns
        annotations = {}
        for key, value in self.annotations.items():
            assert key == value.name
            annotations[key] = dict(value)[key]
        yield "annotations", annotations

    def to_str(self, columns: list[str] | None = None, indent: int = 0, default_indent: str = " " * 4) -> str:
        hspace = " " * indent
        txt_repr = [f"{hspace}Annotations:"]
        for name, annotation in self.annotations.items():
            txt_repr.append(hspace + default_indent + f"{name}: {annotation.value}")

        for spec in self.columns:
            spec_dict = dict(spec)
            if columns and spec_dict['name'] not in columns:
                continue

            txt_repr.append(hspace + default_indent + f"{spec_dict['name']}:")
            for field_name, value in spec_dict.items():
                if field_name == "name":
                    continue
                txt_repr.append(
                    hspace + default_indent + default_indent + f"{field_name}: {value}"
                )

        return "\n".join(txt_repr)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetaData:
        """
        Create a :class:`MetaData` object by parsing a dictionary

        .. note::
            This function ignores all keys but those defined in :attr:`MetaData.Key`.
            These keys are required to generate the object from a dictionary.

        The items corresponding to the key :attr:`MetaData.Key.columns`
        should be a list of :class:`DataSpecification` compatible dictionaries.
        See :func:`DataSpecification.from_dict` for more information about those requirements.

        The items corresponding to :attr:`MetaData.Key.annotations` should be a dictionary of
        either :class:`damast.core.Annotation` compatible dictionaries (see :func:`damast.core.Annotation.from_dict`)
        or :class:`damast.core.History` compatible dictionaries (see :func:`damast.core.History.from_dict`)

        :param data: Input dictionary
        :raises KeyError: If missing required keys
        :raises RuntimeError: If data-specification dictionaries not compatible
        """
        for key in [cls.Key.annotations.value, cls.Key.columns.value]:
            if key not in data:
                raise KeyError(
                    f"{cls.__name__}.from_dict: Missing {key} key in MetaData"
                )

        data_specs = []
        for data_spec_dict in data[cls.Key.columns.value]:
            try:
                data_specification = DataSpecification.from_dict(data=data_spec_dict)
                data_specs.append(data_specification)
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                raise RuntimeError(
                    f"{cls.__name__}.from_dict: could not"
                    f" process column specification: {data_spec_dict} -- {e}"
                ) from e

        annotations = []
        for annotation_key, annotation_value in data[cls.Key.annotations.value].items():
            annotation = None
            if annotation_key == Annotation.Key.History.value:
                annotation = History.from_dict(data={annotation_key: annotation_value})
            else:
                annotation = Annotation.from_dict(
                    data={annotation_key: annotation_value}
                )
            annotations.append(annotation)

        return cls(columns=data_specs, annotations=annotations)

    @classmethod
    def load_yaml(cls, filename: Union[str, Path]) -> MetaData:
        """
        Load history from a `yaml` file.

        .. note::
            It is assumed that the input file follows a structure that is read in as
            a dictionary compatible with :func:`from_dict`.

        :param filename: The input file
        :raises FileNotFoundError: If file does not exist
        :return: A MetaData object
        """
        yaml_path = Path(filename)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"{cls.__name__}.load_yaml: file " f" '{filename}' does not exist"
            )

        with open(yaml_path, "r") as f:
            md_dict = yaml.load(f, Loader=yaml.SafeLoader)

        return cls.from_dict(data=md_dict)

    def save_yaml(self, filename: Union[str, Path]):
        """
        Save the current object into a file.

        :param filename: Filename to use for saving
        """
        with open(filename, "w") as f:
            # Do not sort the keys, but keep the entry order the way
            # the dictionary has been constructed
            yaml.dump(dict(self), f, sort_keys=False)

    def append_to_hdf(self, path: str | Path, overwrite: bool = False):
        # Add metadata
        import tables
        with tables.open_file(path, "a") as f:
            main_node = f.get_node(DAMAST_HDF5_ROOT)

            # Add annotations to main group
            for key, annotation in self.annotations.items():
                if (
                    key in main_node._v_attrs._f_list()
                    and main_node._v_attrs[key] != annotation
                ) and not overwrite:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.save:"
                        f" attribute '{key}' present"
                        f" in file but different from user-defined"
                    )
                main_node._v_attrs[key] = annotation

            if DAMAST_HDF5_COLUMNS not in f:
                f.create_group(DAMAST_HDF5_ROOT, DAMAST_HDF5_COLUMNS.replace(DAMAST_HDF5_ROOT + "/",""))

            # Add attributes for columns
            for column_spec in self.columns:
                group_name = f"{DAMAST_HDF5_COLUMNS}/{column_spec.name}"
                if group_name not in f:
                    group = f.create_group(DAMAST_HDF5_COLUMNS, column_spec.name)
                else:
                    group = f.get_node(group_name)

                for key, value in dict(column_spec).items():
                    if isinstance(value, dict):
                        group._v_attrs[key] = str(value)
                    else:
                        group._v_attrs[key] = value


    def apply(
        self,
        df: pl.LazyFrame,
        validation_mode: ValidationMode = ValidationMode.READONLY,
    ) -> pl.LazyFrame:
        """Check that each column in the :class:`polars.LazyFrame` fulfills the
        data-specifications.

        :param df: The dataframe
        :param validation_mode: If a column does not comply to the spec, force the spec, e.g.,
            for a given range specification, map data within range.
        :raises: ValueError: If data-frame is missing a column in the Data-specification
        """
        assert isinstance(df, pl.LazyFrame)

        for column_spec in self.columns:
            columns = df.compat.column_names
            if column_spec.name in columns:
                df = column_spec.apply(
                    df=df, column_name=column_spec.name, validation_mode=validation_mode
                )
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.apply: missing column '{column_spec.name}' in dataframe."
                    f" Found {len(columns)} column(s): {','.join(columns)}"
                )
        return df

    def __contains__(self, column_name: str):
        """"""
        return any(colum_spec.name == column_name for colum_spec in self.columns)

    def drop(self, columns: str | list[str]):
        """
        Drop specification by column name
        """
        columns = [columns] if type(columns) == str else columns
        self.columns = [x for x in self.columns if x.name not in columns]

    def __getitem__(self, column_name: str) -> DataSpecification:
        """
        Get a data-specification by its column name
        """
        for column_spec in self.columns:
            if column_spec.name == column_name:
                return column_spec

        raise KeyError(
            f"{self.__class__.__name__}.__getitem__: failed to find column by name '{column_name}'"
        )

    def get_fulfillment(self, expected_specs: List[DataSpecification]) -> Fulfillment:
        """
        Get the fulfillment of the metadata with represent to the given data specification

        :param expected_specs: A list of data specifications
        :return: The fulfillment object
        """
        md_fulfillment = MetaData.Fulfillment()

        for expected_spec in expected_specs:
            if expected_spec.name in self:
                fulfillment = expected_spec.get_fulfillment(self[expected_spec.name])
                md_fulfillment.add_fulfillment(
                    column_name=expected_spec.name, fulfillment=fulfillment
                )
            else:
                md_fulfillment.add_fulfillment(
                    column_name=expected_spec.name,
                    fulfillment=DataSpecification.MissingColumn(
                        missing_column=expected_spec.name,
                        known_columns=[x.name for x in self.columns],
                    ),
                )
        return md_fulfillment


    @classmethod
    def search(cls, files: list[str | Path]) -> tuple[MetaData | None, str | None]:
        """
        Search for the metadata specfile for a given list of files
        """
        logger.info("Metadata is required, so searching now for an existing annotation file")
        commonpath = os.path.commonpath(files)
        if len(files) == 1:
            commonpath = Path(commonpath).parent

        commonprefix = os.path.commonprefix([Path(x).stem for x in files])
        metadata_file_candidates = Path(commonpath).glob(f"{commonprefix}*{DAMAST_SPEC_SUFFIX}")
        for f in metadata_file_candidates:
            try:
                return MetaData.load_yaml(filename=f), f, metadata_file_candidates
            except Exception as e:
                logger.debug(f"Loading {f} as metadata file failed -- {e}")

        return None, None, metadata_file_candidates

    @classmethod
    def specfile(cls, files: list[str | Path]) -> str:
        """
        Create the default filename for the spec file
        """
        if len(files) == 1:
            return Path(files[0]).with_suffix(DAMAST_SPEC_SUFFIX)

        commonpath = os.path.commonpath(files)
        commonprefix = os.path.commonprefix([Path(x).stem for x in files])
        return Path(commonpath) / f"{commonprefix}.collection{DAMAST_SPEC_SUFFIX}"

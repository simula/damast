"""
Module to collect the classes to define meta data
"""
from __future__ import annotations

import builtins
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import vaex
import yaml

from .annotations import Annotation, History
from .datarange import DataRange
from .formatting import DEFAULT_INDENT
from .units import Unit, unit_registry, units

__all__ = [
    "ArtifactSpecification",
    "DataCategory",
    "DataSpecification",
    "MetaData",
    "Status"
]


class DataCategory(str, Enum):
    """
    """
    DYNAMIC = "dynamic"
    STATIC = "static"


class Status(str, Enum):
    OK = "OK"
    FAIL = "FAIL"


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

        :param base_dir: Relative paths/patterns will be interpreted with base_dir as current working directory
        :raise RuntimeError: When a matching file could not be found
        """
        for name, path_pattern in self.artifacts.items():
            a_path = Path(path_pattern)
            if a_path.is_absolute():
                if not a_path.exists():
                    raise RuntimeError(f"{self.__class__.__name__}.validate: artifact '{a_path}' is "
                                       "missing")
            else:
                files = [x for x in Path(base_dir).glob(path_pattern)]
                if len(files) == 0:
                    raise RuntimeError(f"{self.__class__.__name__}.validate: no artifact matching "
                                       f" {path_pattern} found in '{base_dir}'")


class DataSpecification:
    """
    Specification of a single column and/or dimension in a dataframe.

    .. todo::

        Document inputs

    :param name: Name of the element
    :param category: Category
    :param is_optional:
    :param abbreviation:
    :param representation_type:
    :param missing_value:
    :param unit:
    :param precision:
    :param range:
    """

    class Key(str, Enum):
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
        value_meanings = "value_meanings"

    class Fulfillment:
        # Would use Key here, but the concept for that is not yet ironed out in Python
        status: Dict[Enum, Dict[str, Union[Status, str]]]

        def __init__(self):
            self.status = {}

        def is_met(self) -> bool:
            for k, v in self.status.items():
                if "status" not in v:
                    raise KeyError(f"{self.__class__.__name__}: internal error. Status misses key 'status'")

                if v["status"] == Status.FAIL:
                    return False
            return True

        def __repr__(self):
            status = {}
            for k, v in self.status.items():
                msg = v['status'].value
                if 'message' in v:
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

        def __init__(self, missing_column: str = None,
                     known_columns: List[str] = None):
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

    def __init__(self,
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
                 value_meanings: Optional[Dict[Any, str]] = None):
        """
        Constructor
        """

        if name is None:
            raise ValueError(f"{self.__class__.__name__}.__init__: "
                             f"name cannot be None")

        self.name = name
        if isinstance(category, str):
            self.category = DataCategory(category)
        else:
            self.category = category

        self.is_optional = is_optional
        if abbreviation is not None and abbreviation != '':
            self.abbreviation = abbreviation

        self.representation_type = representation_type
        self.missing_value = missing_value

        self.unit = unit
        self.precision = precision
        self.value_range = value_range
        self.value_meanings = value_meanings
        if description is None:
            self.description = ""
        self._validate()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        for m in DataSpecification.Key:
            if getattr(self, m.value) != getattr(other, m.value):
                return False

        return True

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}, {self.category.__class__.__name__}]"

    def _validate(self):
        """
        Validate the data specification

        :raises ValueError: If the spec is not valid, raises ValueError with details
        """
        if self.value_meanings and not self.value_range:
            raise ValueError(f"{self.__class__.__name__}.__init__: "
                             f" value meanings requires value range to be set")

        if self.value_meanings:
            for k, v in self.value_meanings.items():
                if k not in self.value_range:
                    raise ValueError(f"{self.__class__.__name__}.validate: "
                                     " value {k} is not in the defined value range")

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
        :raise ValueError: Raises if type_name cannot be resolved to a known type (in builtins or vaex)
        """
        exceptions: List[Any] = []
        try:
            dtype = getattr(builtins, type_name)
            return dtype
        except AttributeError as e:
            exceptions.append(e)

        try:
            dtype = vaex.dtype(type_name)
            return dtype
        except TypeError as e:
            exceptions.append(e)

        raise ValueError(f"{cls.__name__}.resolve_representation_type: "
                         f" could not find type '{type_name}'."
                         f"It does not exist in builtins, nor "
                         f" is this a known vaex.dtype -- {exceptions}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary to represent MetaData.

        :return: dictionary
        """
        data: Dict[str, Any] = {}
        data["name"] = self.name
        if self.is_optional is not None:
            data["is_optional"] = self.is_optional
        if self.abbreviation is not None:
            data["abbreviation"] = self.abbreviation

        if self.category is not None:
            data["category"] = self.category.value

        if self.representation_type is not None:
            data["representation_type"] = self.representation_type.__name__

        if self.missing_value is not None:
            data["missing_value"] = self.missing_value

        if self.unit is not None:
            data["unit"] = self.unit.to_string()

        if self.precision is not None:
            data["precision"] = self.precision

        if self.value_range is not None:
            data["value_range"] = self.value_range.to_dict()

        if self.value_meanings is not None:
            data["value_meanings"] = self.value_meanings

        return data

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
                raise TypeError(f"{cls.__name__}.from_dict: "
                                f"abbreviation must be of type 'str'")

        data_category = None
        if cls.Key.category.value in data:
            data_category = data[cls.Key.category]
            if not isinstance(data_category, DataCategory):
                data_category = DataCategory(data_category)

        spec = cls(name=data[cls.Key.name],
                   category=data_category,
                   is_optional=bool(data[cls.Key.is_optional]),
                   abbreviation=abbreviation)

        if cls.Key.representation_type.value in data:
            spec.representation_type = cls.resolve_representation_type(data[cls.Key.representation_type.value])
            if cls.Key.missing_value.value in data:
                spec.missing_value = spec.representation_type(data[cls.Key.missing_value.value])
            if cls.Key.precision.value in data:
                spec.precision = spec.representation_type(data[cls.Key.precision.value])
        else:
            # Missing representation type, missing value and precision will not be loaded
            pass

        if cls.Key.unit.value in data:
            # Check if unit is part of the default definitions
            unit_value = data[cls.Key.unit.value]
            if isinstance(unit_value, str):
                try:
                    unit = getattr(units, unit_value)
                except Exception:
                    # Check if unit part of definition in 'damast.core.units'
                    unit = unit_registry[unit_value]
            elif isinstance(unit_value, Unit):
                unit = unit_value
            else:
                raise RuntimeError(f"{cls.__name__}.from_dict: cannot interprete unit type "
                                   f"'{type(unit_value)}")

            spec.unit = unit

        if cls.Key.value_range.value in data:
            value_range = data[cls.Key.value_range.value]
            if isinstance(value_range, DataRange):
                spec.value_range = value_range
            else:
                spec.value_range = DataRange.from_dict(value_range, dtype=spec.representation_type)

            if cls.Key.value_meanings.value in data:
                spec.value_meanings = data[cls.Key.value_meanings.value]

        return spec

    @classmethod
    def to_str(cls, specs: List[DataSpecification],
               indent_level=0):
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
                data += "[unit: " + spec.unit.to_string() + "]"
            data += "\n"
        return data

    def apply(self,
              df: vaex.DataFrame,
              column_name: str):
        # Check if representation type is the same and apply known metadata
        if self.representation_type is not None:
            if df[column_name].dtype != self.representation_type:
                raise ValueError(f"{self.__class__.__name__}.apply: column '{column_name}':"
                                 f" expected representation type: {self.representation_type},"
                                 f" but got '{df[column_name].dtype}'")

        if self.unit is not None:
            if column_name not in df.units:
                df.units[column_name] = self.unit
            else:
                assert df.units[column_name] == self.unit

        if self.value_range:
            min_value, max_value = df.minmax(column_name)

            if not self.value_range.is_in_range(min_value):
                raise ValueError(f"{self.__class__.__name__}.apply: minimum value '{min_value}'"
                                 f" lies outside of range {self.value_range}")
            if not self.value_range.is_in_range(max_value):
                raise ValueError(f"{self.__class__.__name__}.apply: maximum value '{max_value}'"
                                 f" lies outside of range {self.value_range}")

    def get_fulfillment(self, data_spec: DataSpecification) -> Fulfillment:
        """
        Check the fulfillment of an existing and expected data specification against an available one.

        The current instance used for calling this method sets the expectations.

        :param data_spec: The available data specification
        :return: The degree of fulfillment
        """
        fulfillment = DataSpecification.Fulfillment()

        for key in DataSpecification.Key:
            # some special handling is required for individual keys
            if key == DataSpecification.Key.precision:
                if self.precision is not None:
                    if data_spec.precision is None:
                        fulfillment.status[key] = {'status': Status.FAIL,
                                                   'message': f"column has no precision defined"}
                    elif self.precision < data_spec.precision:
                        fulfillment.status[key] = {'status': Status.FAIL,
                                                   'message': f"'data has insufficient precision: "
                                                              f" required '{self.precision}',"
                                                              f" available '#{data_spec.precision}'"}
                    else:
                        fulfillment.status[key] = {'status': Status.OK}
            else:
                expected_value = getattr(self, key.value)
                if expected_value is not None:
                    spec_value = getattr(data_spec, key.name)
                    if spec_value is None:
                        fulfillment.status[key] = {'status': Status.FAIL,
                                                   'message': f"Expected '{spec_value=}' == '{expected_value=}'"
                                                   }
                        continue

                    if expected_value != spec_value:
                        fulfillment.status[key] = {'status': Status.FAIL,
                                                   'message': f"Expected '{spec_value=}' == '{expected_value=}'"
                                                   }
                        continue

                    fulfillment.status[key] = {'status': Status.OK}
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

    def merge(self,
              other: DataSpecification) -> DataSpecification:

        if self.name != other.name:
            raise ValueError(f"{self.__class__.__name__}.merge: cannot merge specs with different name property")

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
                    f" on self: '{this_value}' vs. other: '{other_value}'")
        return ds

    @classmethod
    def merge_lists(cls,
                    a_specs: List[DataSpecification],
                    b_specs: List[DataSpecification]) -> List[DataSpecification]:

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
    The representation for metadata associated which can be associated with a single dataset.
    """

    class Key(str, Enum):
        columns = 'columns'
        annotations = 'annotations'

    class Fulfillment:
        column_fulfillments: Dict[str, DataSpecification.Fulfillment]

        def __init__(self):
            self.column_fulfillments = {}

        def is_met(self) -> bool:
            for k, v in self.column_fulfillments.items():
                if not v.is_met():
                    return False
            return True

        def add_fulfillment(self,
                            column_name: str,
                            fulfillment: DataSpecification.Fulfillment) -> None:

            assert isinstance(fulfillment, DataSpecification.Fulfillment)
            self.column_fulfillments[column_name] = fulfillment

        def __repr__(self) -> str:
            txt = ""
            for column_name, fulfillment in self.column_fulfillments.items():
                txt += f"[{column_name}: {fulfillment}]"
            return txt

    #: Specification of columns in the
    columns: List[DataSpecification]

    #: Dictionary containing all annotations
    annotations: Dict[str, Annotation]

    def __init__(self,
                 columns: List[DataSpecification],
                 annotations: Optional[Dict[str, Annotation]] = None):
        """
        Initialise MetaData

        :param columns: (Ordered) list of column specifications
        :param annotations:  Annotation for this dataframe
        """
        assert isinstance(columns, list)
        self.columns = columns

        if annotations is None:
            self.annotations = {}
        else:
            assert isinstance(annotations, dict)
            self.annotations = annotations

    def __eq__(self, other) -> bool:
        if self.__class__ != other.__class__:
            return False

        if self.columns != other.columns:
            return False

        if self.annotations != other.annotations:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        data[self.Key.columns.value] = []
        for ds in self.columns:
            data[self.Key.columns.value].append(ds.to_dict())

        data[self.Key.annotations.value] = {}
        for key, annotation in self.annotations.items():
            a_dict = annotation.to_dict()
            annotation_key = list(a_dict.keys())[0]
            if annotation_key in data[self.Key.annotations.value]:
                raise KeyError(f"{self.__class__.__name__}.to_dict: '{annotation_key}' is already present")
            else:
                data[self.Key.annotations.value][annotation_key] = a_dict[annotation_key]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetaData:
        for key in [cls.Key.annotations.value, cls.Key.columns.value]:
            if key not in data:
                raise KeyError(f"{cls.__name__}.from_dict: Missing {key} key in MetaData")

        data_specs = []
        for data_spec_dict in data[cls.Key.columns.value]:
            try:
                data_specification = DataSpecification.from_dict(data=data_spec_dict)
                data_specs.append(data_specification)
            except Exception as e:
                raise RuntimeError(f"{cls.__name__}.from_dict: could not"
                                   f" process column specification: {data_spec_dict} -- {e}")

        annotations = {}
        for annotation_key, annotation_value in data[cls.Key.annotations.value].items():
            annotation = None
            if annotation_key == Annotation.Key.History.value:
                annotation = History.from_dict(data={annotation_key: annotation_value})
            else:
                annotation = Annotation.from_dict(data={annotation_key: annotation_value})
            annotations[annotation_key] = annotation

        metadata = cls(columns=data_specs,
                       annotations=annotations)
        return metadata

    @classmethod
    def load_yaml(cls, filename: Union[str, Path]) -> MetaData:
        yaml_path = Path(filename)
        if not yaml_path.exists():
            raise FileNotFoundError(f"{cls.__name__}.load_yaml: file "
                                    f" '{filename}' does not exist")

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
            yaml.dump(self.to_dict(), f, sort_keys=False)

    def apply(self, df: vaex.DataFrame):
        assert isinstance(df, vaex.DataFrame)

        for column_spec in self.columns:
            if column_spec.name in df.column_names:
                column_spec.apply(df=df, column_name=column_spec.name)
            else:
                raise ValueError(f"{self.__class__.__name__}.apply: missing column '{column_spec.name}' in dataframe."
                                 f" Found {len(df.column_names)} column(s): {','.join(df.column_names)}")

    def __contains__(self, column_name):
        for colum_spec in self.columns:
            if colum_spec.name == column_name:
                return True
        return False

    def __getitem__(self, column_name: str):
        for column_spec in self.columns:
            if column_spec.name == column_name:
                return column_spec

        raise KeyError(f"{self.__class__.__name__}.__getitem__: failed to find column by name '{column_name}'")

    def get_fulfillment(self, expected_specs: List[DataSpecification]) -> Fulfillment:
        """
        Get the fulfillment of the metadata with represent to the given data specification

        :param expected_specs: A list of data specifications
        :return:
        """
        md_fulfillment = MetaData.Fulfillment()

        for expected_spec in expected_specs:
            if expected_spec.name in self:
                fulfillment = expected_spec.get_fulfillment(self[expected_spec.name])
                md_fulfillment.add_fulfillment(column_name=expected_spec.name,
                                               fulfillment=fulfillment)
            else:
                md_fulfillment.add_fulfillment(column_name=expected_spec.name,
                                               fulfillment=DataSpecification.MissingColumn(
                                                   missing_column=expected_spec.name,
                                                   known_columns=[x.name for x in self.columns]
                                               ))
        return md_fulfillment

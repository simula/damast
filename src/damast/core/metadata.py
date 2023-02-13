import builtins
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Union, List, Dict

import vaex
import yaml
# Alternatively use pint
from astropy import units

from .annotations import Annotation, History
from .datarange import DataRange

__all__ = [
    "DataCategory",
    "DataSpecification",
    "MetaData"
]


class DataCategory(str, Enum):
    """
    """
    DYNAMIC = "dynamic"
    STATIC = "static"


class DataSpecification:
    """
    Specification of a single column and/or dimension in a dataframe.
    """
    #: Name associated
    name: str = None
    #: Category of data
    category: DataCategory = None
    #: Whether this data element needs to be present
    is_optional: bool = None
    abbreviation: str = None

    #: The underlying representation type for this data element
    representation_type: Any = None
    #: The representation of a missing value for this data element
    missing_value: Any = None

    #: The unit of this data element
    unit: Optional[units.Unit] = None
    #: The precision of this data element
    precision: Optional[Union[float, List[float]]] = None
    #: The allowed data range, which remains None when being unrestricted
    value_range: Union[DataRange] = None
    #: An explanation - str-based descriptor for the range, if this is a dictionary then it must provide
    #: a mapping from the value to a human-readable descriptor
    value_meanings: Dict[Any, str] = None

    def __init__(self,
                 name: str,
                 category: Union[str, DataCategory],
                 is_optional: bool = False,
                 abbreviation: str = None,
                 representation_type: Any = None,
                 missing_value: Any = None,
                 unit: Optional[units.Unit] = None,
                 precision: Any = None,
                 value_range: Union[DataRange] = None,
                 value_meanings: Dict[Any, str] = None):
        """
        Initialise the MetaData instance

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
        self.name = name
        if type(category) is str:
            self.category = DataCategory[category]
        else:
            self.category = category

        self.is_optional = is_optional
        if abbreviation is None:
            self.abbreviation = name
        else:
            self.abbreviation = abbreviation

        self.representation_type = representation_type
        self.missing_value = missing_value

        self.unit = unit
        self.precision = precision
        self.value_range = value_range
        self.value_meanings = value_meanings

        self._validate()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        for m in ["name", "category", "is_optional", "abbreviation", "representation_type",
                  "missing_value", "unit", "precision", "value_range", "value_meanings"]:
            if getattr(self, m) != getattr(other, m):
                return False

        return True

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name},{self.category.__class__.__name__}"

    def _validate(self):
        """
        Validate the data specification

        :raises ValueError: If the spec is not valid, raises ValueError with details
        """
        if self.value_range and not self.value_meanings:
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

        >>> int_type = DataSpecification.resolve_representation_type("int")
        >>> assert int_type == int

        :param str: Name of the type
        :return: the type object
        """
        exceptions = []
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
        data = {}
        data["name"] = self.name
        data["category"] = self.category.value
        data["is_optional"] = self.is_optional
        data["abbreviation"] = self.abbreviation

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
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSpecification':
        """
        Load the data specification from a given dictionary.

        This function in intended to deserialize specifications from a string-based
        representation

        :param data: data dictionary using primitive data types to represent the specificatoin
        :return: the loaded data specification
        """
        for key in ["name", "category", "is_optional", "abbreviation"]:
            if key not in data:
                raise KeyError(f"{cls.__name__}.from_dict: missing '{key}'")

        spec = cls(name=data["name"],
                   category=DataCategory(data["category"]),
                   is_optional=bool(data["is_optional"]),
                   abbreviation=data["abbreviation"])

        if "representation_type" in data:
            spec.representation_type = cls.resolve_representation_type(data["representation_type"])
            if "missing_value" in data:
                spec.missing_value = spec.representation_type(data["missing_value"])
            if "precision" in data:
                spec.precision = spec.representation_type(data["precision"])
        else:
            # Missing representation type, missing value and precisiong will not be loaded
            pass

        if "unit" in data:
            spec.unit = getattr(units, data["unit"])

        if "value_range" in data:
            spec.value_range = DataRange.from_dict(data["value_range"], dtype=spec.representation_type)
            if "value_meanings" in data:
                spec.value_meanings = data["value_meanings"]

        return spec

    def apply(self,
              df: vaex.DataFrame,
              column_name: str):
        # Check if representation type is the same and apply known metadata
        if self.representation_type is not None:
            assert df[column_name].dtype == self.representation_type

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



class MetaData:
    """
    The representation for metadata associated which can be associated with a single dataset.
    """

    class Key(str, Enum):
        COLUMNS = 'columns'
        ANNOTATIONS = 'annotations'

    #: Specification of columns in the
    columns: List[DataSpecification] = None

    annotations: Dict[str, Annotation] = None

    def __init__(self,
                 columns: List[DataSpecification],
                 annotations: Dict[str, Annotation] = None):
        self.columns = columns
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
        data = {}
        data[self.Key.COLUMNS.value] = []
        for ds in self.columns:
            data[self.Key.COLUMNS.value].append(ds.to_dict())

        data[self.Key.ANNOTATIONS.value] = {}
        for key, annotation in self.annotations.items():
            a_dict = annotation.to_dict()
            annotation_key = list(a_dict.keys())[0]
            if annotation_key in data[self.Key.ANNOTATIONS.value]:
                raise KeyError(f"{self.__class__.__name__}.to_dict: '{annotation_key}' is already present")
            else:
                data[self.Key.ANNOTATIONS.value][annotation_key] = a_dict[annotation_key]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaData':
        for key in [cls.Key.ANNOTATIONS.value, cls.Key.COLUMNS.value]:
            if key not in data:
                raise KeyError(f"{cls.__name__}.from_dict: Missing {key} key in MetaData")

        data_specs = []
        for data_spec_dict in data[cls.Key.COLUMNS.value]:
            data_specification = DataSpecification.from_dict(data=data_spec_dict)
            data_specs.append(data_specification)

        annotations = {}
        for annotation_key, annotation_value in data[cls.Key.ANNOTATIONS.value].items():
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
    def load_yaml(cls, filename: Union[str, Path]) -> 'MetaData':
        with open(filename, "r") as f:
            md_dict = yaml.load(f, Loader=yaml.SafeLoader)

        return cls.from_dict(data=md_dict)

    def apply(self, df: Union[vaex.DataFrame]):
        for column_spec in self.columns:
            if column_spec.name in df.column_names:
                column_spec.apply(df=df, column_name=column_spec.name)





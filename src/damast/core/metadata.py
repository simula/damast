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


class Status(str, Enum):
    OK = "OK"
    FAIL = "FAIL"


class DataSpecification:
    """
    Specification of a single column and/or dimension in a dataframe.
    """

    class Key(str, Enum):
        name = "name"
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
        status: Dict['Key', Status]

        def __init__(self):
            self.status = {}

        def is_met(self) -> bool:
            for k, v in self.status.items():
                if v == Status.FAIL:
                    return False
            return True

        def __repr__(self):
            return f"{self.__class__.__name__}[{self.status}]"

    class MissingColumn(Fulfillment):
        def is_met(self) -> bool:
            return False

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}"

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

        if name is None:
            raise ValueError(f"{self.__class__.__name__}.__init__: "
                             f"name cannot be None")

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

    def _validate(self) -> None:
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

        >>> int_type = DataSpecification.resolve_representation_type("int")
        >>> assert int_type == int

        :param type_name: Name of the type
        :return: Instance of the type object
        :raise ValueError: Raises if type_name cannot be resolved to a known type (in builtins or vaex)
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
        """
        Create a dictionary to represent MetaData.

        :return: dictionary
        """
        data = {}
        data["name"] = self.name
        data["is_optional"] = self.is_optional
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
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSpecification':
        """
        Load the data specification from a given dictionary.

        This function in intended to deserialize specifications from a string-based
        representation

        :param data: data dictionary using primitive data types to represent the specificatoin
        :return: the loaded data specification
        :raise KeyError: Raise if a required key is missing
        """
        for key in [cls.Key.name, cls.Key.is_optional, cls.Key.abbreviation]:
            if key.value not in data:
                raise KeyError(f"{cls.__name__}.from_dict: missing '{key.value}'")

        data_category = None
        if cls.Key.category.value in data:
            data_category = DataCategory(data[cls.Key.category])

        spec = cls(name=data[cls.Key.name],
                   category=data_category,
                   is_optional=bool(data[cls.Key.is_optional]),
                   abbreviation=data[cls.Key.abbreviation])

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
            spec.unit = getattr(units, data[cls.Key.unit.value])

        if cls.Key.value_range.value in data:
            spec.value_range = DataRange.from_dict(data[cls.Key.value_range.value], dtype=spec.representation_type)
            if cls.Key.value_meanings.value in data:
                spec.value_meanings = data[cls.Key.value_meanings.value]

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


class ExpectedDataSpecification(DataSpecification):
    def get_fulfillment(self, data_spec: DataSpecification) -> DataSpecification.Fulfillment:
        fulfillment = DataSpecification.Fulfillment()

        for key in [DataSpecification.Key.name,
                    DataSpecification.Key.category,
                    DataSpecification.Key.abbreviation,
                    DataSpecification.Key.unit,
                    DataSpecification.Key.representation_type]:
            expected_value = getattr(self, key.value)
            if expected_value is not None:
                spec_value = getattr(data_spec, key.name)
                if spec_value is None:
                    fulfillment.status[key] = Status.FAIL
                    continue

                if expected_value != spec_value:
                    fulfillment.status[key] = Status.FAIL
                    continue

                fulfillment.status[key] = Status.OK

        if self.precision is not None:
            if data_spec.precision is None:
                fulfillment.status[key] = Status.FAIL
            elif self.precision < data_spec.precision:
                fulfillment.status[key] = Status.FAIL
            else:
                fulfillment.status[key] = Status.OK

        return fulfillment


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
    columns: List[DataSpecification] = None

    #: Dictionary containing all annotations
    annotations: Dict[str, Annotation] = None

    def __init__(self,
                 columns: List[DataSpecification],
                 annotations: Dict[str, Annotation] = None):
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
        data = {}
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
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaData':
        for key in [cls.Key.annotations.value, cls.Key.columns.value]:
            if key not in data:
                raise KeyError(f"{cls.__name__}.from_dict: Missing {key} key in MetaData")

        data_specs = []
        for data_spec_dict in data[cls.Key.columns.value]:
            data_specification = DataSpecification.from_dict(data=data_spec_dict)
            data_specs.append(data_specification)

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
    def load_yaml(cls, filename: Union[str, Path]) -> 'MetaData':
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

    def apply(self, df: Union[vaex.DataFrame]):
        assert isinstance(df, vaex.DataFrame)

        for column_spec in self.columns:
            if column_spec.name in df.column_names:
                column_spec.apply(df=df, column_name=column_spec.name)

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


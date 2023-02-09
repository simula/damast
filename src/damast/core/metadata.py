from typing import Optional, Any, Union, List, Dict

# Alternatively use pint
from astropy import units

__all__ = [
    "DataCategory",
    "MinMax",
    "DataSpecification",
    "MetaData"
]


class DataCategory:
    """
    """
    DYNAMIC = 0
    STATIC = 1


class MinMax:
    min: Any = None
    max: Any = None

    def __init__(self, *,
                 min: Any,
                 max: Any):
        """
        Initialise the min and max range

        :param min: minimum allowed value, data must be greater or equal
        :param max: maximum allowed value, data must be less or equal
        """
        assert min < max
        self.min = min
        self.max = max

    def is_in_range(self, value: Any) -> bool:
        """
        Check if a value is in the defined range.

        :param value: data value
        :return: True if value is in the set range, false otherwise
        """
        return self.min <= value <= self.max

    def __repr__(self):
        return f"{self.__class__.__name__}[{min}, {max}]"


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
    range: Union[MinMax, List[Any]] = None

    def __init__(self,
                 name: str,
                 category: Union[str, DataCategory],
                 is_optional: bool = False,
                 abbreviation: str = None,
                 representation_type: Any = None,
                 missing_value: Any = None,
                 unit: Optional[units.Unit] = None,
                 precision: Any = None,
                 range: Union[List[Any], MinMax] = None):
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
        self.range = range

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name},{self.category.__class__.__name__}"


class Annotation:
    entries: Dict[str, Any] = None


class History:
    changes: List[str] = None


class MetaData:
    """
    The representation for metadata associated which can be associated with a single dataset.
    """

    #: Specification of columns in the
    columns: List[DataSpecification] = None

    annotation: Annotation = None
    history: History = None

    def __init__(self,
                 columns: List[DataSpecification],
                 annotation: Annotation = None,
                 history: History = None):
        self.columns = columns

        self.annotation = annotation
        self.history = history

"""
Module to define known constraints of the data and introduction of constants for column names and field values.
"""
__all__ = ["ColumnName",
           "FieldValue",
           "MMSI"
           ]

from typing import ClassVar


class CourseOverGround:
    min_value: ClassVar[float] = -360.0
    max_value: ClassVar[float] = 360.0

    unit: str = "deg"


class Heading:
    min_value: ClassVar[float] = 0.0
    max_value: ClassVar[float] = 359.0

    unit: str = "deg"


class Bearing:
    min_value: ClassVar[float] = 0.0
    max_value: ClassVar[float] = 359.0

    unit: str = "deg"


class SpeedOverGround:
    min_value: ClassVar[float] = 0.0
    max_value: ClassVar[float] = 102.0

    unit: str = "knots"


class RateOfTurn:
    min_value: ClassVar[float] = -720.0
    max_value: ClassVar[float] = 720.0

    unit: str = "deg"


class MMSI:
    """
    Class representing the MMSI data and corresponding (known) constraints
    """

    #: Minimum value for the MSSI which comes from a predefined allowed-range
    min_value: ClassVar[int] = 200000000

    #: Maximum value for the MSSI which comes from a predefined allowed-range
    max_value: ClassVar[int] = 799999999

    def __init__(self,
                 mmsi: int):
        self.value = mmsi

    @property
    def mid(self) -> int:
        return self.country_iso_code

    @property
    def country_iso_code(self) -> int:
        """
        Get the first three digits of the mmsi

        :return: Country code
        """
        return int(self.value / 1000000)

    @property
    def national_id(self) -> int:
        """
        Get the national id (i.e. 6 last digits of the mmsi)

        :return: National id
        """
        return self.value - self.country_iso_code * 1000000


class ColumnName:
    """
    Collection of existing column names to provide a single point of definition.
    """
    MMSI: str = "MMSI"
    COURSE_OVER_GROUND: str = "COG"
    SPEED_OVER_GROUND: str = "SOG"
    LATITUDE: str = "LAT"
    LONGITUDE: str = "LON"
    TIMESTAMP: str = "timestamp"
    VESSEL_TYPE: str = "vessel_type"
    VESSEL_TYPE_GFW: str = "vessel_class_gfw"
    STATUS: str = "Status"
    MESSAGE_TYPE: str = "MessageType"
    DATE_TIME_UTC: str = "date_time_utc"
    SOURCE: str = "source"

    # Computed Data
    DATA_MONTH: str = "data_month"
    FISHING_TYPE: str = "fishing_type"
    DISTANCE_CLOSEST_SATELLITE: str = "distance_closest_satellite"
    DISTANCE_CLOSEST_ANCHORAGE: str = "distance_closest_anchorages"
    HEADING: str = "Heading"

    TRAJECTORIES: str = "trajectories"
    COMBINATION: str = "combination"
    HISTORIC_SIZE: str = "HISTORIC_SIZE"
    HISTORIC_SIZE_REVERSE: str = "HISTORIC_SIZE_REVERSE"
    DELTA_DISTANCE: str = "DeltaDistance"
    LAST_POSITION: str = "last_position"


class FieldValue:
    VESSEL_TYPE_UNSPECIFIED: str = "unspecified"
    UNDEFINED: int = -1

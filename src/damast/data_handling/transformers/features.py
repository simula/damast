"""
Module to collect all features that can be added
"""
from enum import Enum
from typing import List, Tuple, Any

import numpy as np
from numpy import ndarray, dtype, float_
import pandas as pd

from damast.domains.maritime.data_specification import ColumnName
from damast.data_handling.transformers.base import BaseTransformer

__all__ = [
    "Angle",
    "DatetimeInEpochs",
    "DayOfYear",
    "Diff",
    "DeltaTime",
    "Feature",
    "FeatureExtractor",
    "MaxToBool",
    "ReformatTimestamp",
    "SecondOfDay",
]

from damast.math.spatial import great_circle_distance


class Feature(str, Enum):
    """
    Represent a list of features

    TODO: Check separation against ColumnNames
    """
    TRAJECTORY = "trajectory"
    LAST_POSITION = "last_position"

    ANGLE = "ANGLE"
    DATETIME = "DATETIME"
    DELTA_DISTANCE = "DELTA_DISTANCE"
    DELTA_DISTANCE_X = "DELTA_DISTANCE_X"
    DELTA_DISTANCE_Y = "DELTA_DISTANCE_Y"
    DELTA_TIME = "DELTA_TIME"

    HEADING = "Heading"

    SECOND_OF_DAY = "SECOND_OF_DAY"
    DAY_OF_YEAR = "DAY_OF_YEAR"

    COMBINATION__FishingVesselType_Status = "combination"

    @classmethod
    def get_feature_extractors(cls,
                               column_list: List[str],
                               ) -> Tuple[List['FeatureExtractor'], List[str], List[str]]:
        """
        Get the feature extractors for a given list of columns, which can include feature column and 'original' ones.


        :param column_list: A list of columns that should be in the data
        :return: Tuple of the feature extractors available, the final columns that shall be sued, and the required
                 input columns for the data
        """
        feature_extractors: List[FeatureExtractor] = []
        columns_to_use: List[str] = []
        required_input_columns: List[str] = []

        for x in column_list:
            try:
                f: FeatureExtractor = Feature.extractor_by_name(name=x)
                required_input_columns.extend(f.input_columns)
                feature_extractors.append(f)
            except KeyError as e:
                columns_to_use.append(x)

        required_input_columns = list(set(required_input_columns))

        for f in feature_extractors:
            if f.name not in columns_to_use:
                columns_to_use.append(f.name)

        columns_to_use = list(set(columns_to_use))

        return feature_extractors, columns_to_use, required_input_columns

    @classmethod
    def extractor_by_name(cls,
                          name: str,
                          **kwargs) -> 'FeatureExtractor':
        """
        Get a FeatureExtractor by its name

        :param name:  Name of the FeatureExtractor
        :param kwargs: arguments that can be forwarded to the feature generators constructor
        :return: an instance of the FeatureExtractor
        :raises KeyError: If not extractor goes by the given name, KeyError will be raised
        """
        for klass in FeatureExtractor.all_subclasses():
            try:
                instance = klass(**kwargs)
                if instance.name == name:
                    return instance
            except TypeError:
                # If the name is not set for this feature extractor,
                # then it is assumed to be abstract and not usable
                pass

        raise KeyError(f"Feature.extractor_by_name: could not find a FeatureExtractor for '{name}'")


class FeatureExtractor(BaseTransformer):
    """
    Base Class for all FeatureExtractor
    """
    name: str = None
    input_columns: List[str] = None

    @classmethod
    def all_subclasses(cls) -> List['FeatureExtractor']:
        """
        Get all subclasses of FeatureExtractor, i.e., available implementations

        :return:  List of subclasses
        """
        all = []
        subclasses = cls.__subclasses__()
        all.extend(subclasses)
        for sc in subclasses:
            all.extend(sc.all_subclasses())
        return all

    def __init__(self,
                 name: str,
                 input_columns: List[str]) -> None:
        """
        Initialize FeatureExtractor

        :param name: Name of the FeatureExtractor
        :param input_columns: List of required input columns for this extractor
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        if input_columns is None:
            raise ValueError("FeatureExtractor.__init__: input_columns cannot be None")

        self.input_columns = input_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().transform(df)
        """
        Base implementation of the feature extraction

        Check if all inputs are available.

        :param df: the Dataframe to operate on
        :return: the original, yet possibly updated dataframe
        """
        for col_name in self.input_columns:
            if col_name not in df.columns:
                raise RuntimeError(f"{self}.transform: required input column '{col_name}' does not exist."
                                   f" Input data has the following columns: {','.join(df.columns)}")
        return df


class ReformatTimestamp(FeatureExtractor):
    """
    Base class for reusing an existing timestamp column to generate a new
    column
    """
    #: Column from which to extract the timestamp
    timestamp_column: str = None

    def __init__(self, *,
                 name: str,
                 timestamp_column: str = None):
        """
        Initialize extractor

        :param name: Name of the extractor
        :param timestamp_column: The timestamp field/column name
        """
        if timestamp_column is None:
            self.timestamp_column = "timestamp"
        else:
            self.timestamp_column = timestamp_column

        super().__init__(
            name=name,
            input_columns=[self.timestamp_column]
        )


class DatetimeInEpochs(ReformatTimestamp):
    """
    FeatureExtractor to convert timestamp column to a datetime (Timestamp)
    """

    def __init__(self, *,
                 name: str = Feature.DATETIME.value,
                 timestamp_column: str = ColumnName.TIMESTAMP):
        """
        Initialize extractor

        :param name: Name of the extractor
        :param timestamp_column: The timestamp field/column name
        """
        super().__init__(name=name,
                         timestamp_column=timestamp_column
                         )

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the feature extraction.

        :param df: Dataframe to run the feature extraction on
        """
        df = super().transform(df)

        df[self.name] = pd.to_datetime(df[self.timestamp_column].astype(np.int),
                                       unit="s")
        return df


class SecondOfDay(ReformatTimestamp):
    """
    FeatureExtractor to convert a timestamp to the second of day
    """

    def __init__(self,
                 name: str = Feature.SECOND_OF_DAY.value,
                 timestamp_column: str = ColumnName.TIMESTAMP
                 ):
        super().__init__(name=name,
                         timestamp_column=timestamp_column
                         )

    def convert(self, row):
        """
        Convert existing value and return new values

        :param row:
        :return:
        """
        value = pd.to_datetime(row[self.timestamp_column].astype(np.int), unit="s")
        new_value = value.dt.hour * 60 + \
                    value.dt.minute * 60 + \
                    value.dt.second
        return new_value

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the feature extraction.

        :param df: Dataframe to run the feature extraction on
        """
        df = super().transform(df)

        kwargs = {self.name: self.convert}
        # Assign only returns a copy of the data, assign the new column to the original dataset
        df[self.name] = df.assign(**kwargs)[self.name]
        return df


class DayOfYear(ReformatTimestamp):
    """
    FeatureExtractor to convert a timestamp to the day of the year
    """

    def __init__(self,
                 name: str = Feature.DAY_OF_YEAR.value,
                 timestamp_column: str = None):
        super().__init__(name=name,
                         timestamp_column=timestamp_column
                         )

    def convert(self, row):
        value = pd.to_datetime(row[self.timestamp_column].astype(np.int), unit="s")
        return value.dt.dayofyear

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the feature extraction.

        :param df: Dataframe to run the feature extraction on
        """
        super().transform(df)

        # Assign only returns a copy of the data, assign the new column to the original dataset
        kwargs = {self.name: self.convert}
        df[self.name] = df.assign(**kwargs)[self.name]
        return df


class LatLonFeatureExtractor(FeatureExtractor):
    """
    FeatureExtractor as base for extractor using lat/lon input
    """
    mmsi_name: str = None
    lat_name: str = None
    lon_name: str = None

    def __init__(self, *,
                 name: str = None,
                 mmsi_name: str = ColumnName.MMSI,
                 lat_name: str = ColumnName.LATITUDE,
                 lon_name: str = ColumnName.LONGITUDE):
        self.mmsi_name = mmsi_name
        self.lat_name = lat_name
        self.lon_name = lon_name

        super().__init__(name=name,
                         input_columns=[lat_name, lon_name])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().transform(df)


class DeltaDistance(LatLonFeatureExtractor):
    """
    FeatureExtractor to compute the distance between location given by latitude/longitude
    """

    def __init__(self, *,
                 name: str = Feature.DELTA_DISTANCE.value,
                 lat_name: str = ColumnName.LATITUDE,
                 lon_name: str = ColumnName.LONGITUDE):
        super().__init__(name=name,
                         lat_name=lat_name,
                         lon_name=lon_name)

    def compute_distance(self, df) -> pd.DataFrame:
        dataframe = great_circle_distance(df[self.lat_name],
                                          df[self.lon_name],
                                          df[self.lat_name].shift(1),
                                          df[self.lat_name].shift(1))
        # default value for the first entry
        dataframe[dataframe.index[0]] = 0.0
        return dataframe

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the feature extraction.

        :param df: Dataframe to run the feature extraction on
        """
        df = super().transform(df)
        df[self.name] = self.compute_distance(df)
        return df


class DeltaDistanceX(DeltaDistance):
    """
    FeatureExtractor to compute the x-distance between location given by latitude/longitude
    """

    def __init__(self, *,
                 name: str = Feature.DELTA_DISTANCE_X.value,
                 lat_name: str = ColumnName.LATITUDE,
                 lon_name: str = ColumnName.LONGITUDE):
        super().__init__(name=name,
                         lat_name=lat_name,
                         lon_name=lon_name)

    def compute_distance(self, df):
        dataframe = great_circle_distance(df[self.lat_name].shift(1),
                                          df[self.lon_name],
                                          df[self.lat_name].shift(1),
                                          df[self.lat_name].shift(1))
        # default value for the first entry
        dataframe[dataframe.index[0]] = 0.0
        return dataframe


class DeltaDistanceY(DeltaDistance):
    """
    FeatureExtractor to compute the y-distance between location given by latitude/longitude
    """

    def __init__(self, *,
                 name: str = Feature.DELTA_DISTANCE_Y.value,
                 lat_name: str = ColumnName.LATITUDE,
                 lon_name: str = ColumnName.LONGITUDE):
        super().__init__(name=name,
                         lat_name=lat_name,
                         lon_name=lon_name)

    def compute_distance(self, df):
        dataframe = great_circle_distance(df[self.lat_name],
                                          df[self.lon_name].shift(1),
                                          df[self.lat_name].shift(1),
                                          df[self.lat_name].shift(1))
        # default value for the first entry
        dataframe[dataframe.index[0]] = 0.0
        return dataframe


class Angle(LatLonFeatureExtractor):
    def __init__(self, *,
                 name: str = Feature.ANGLE.value,
                 mmsi_name: str = ColumnName.MMSI,
                 lat_name: str = ColumnName.LATITUDE,
                 lon_name: str = ColumnName.LONGITUDE):
        super().__init__(name=name,
                         lat_name=lat_name,
                         lon_name=lon_name)

    def compute_bearing(self, df) -> ndarray[Any, dtype[float_]]:
        return bearing(df[self.lat_name].to_numpy(),
                       df[self.lon_name].to_numpy(),
                       df[self.lat_name].shift(1).to_numpy(),
                       df[self.lon_name].shift(1).to_numpy())

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the feature extraction.

        :param df: Dataframe to run the feature extraction on
        """
        df = super().transform(df)
        # We assume that the data is sorted
        df[self.name] = self.compute_bearing(df)
        return df


class Diff(FeatureExtractor):
    """
    Create an additional column that contains the delta between the current row's value and the predecessor's row.
    """

    def __init__(self, *,
                 input_column: str,
                 name: str = None
                 ):
        self.input_column = input_column

        if name is None:
            self.name = f"{input_column}_diff"
        else:
            self.name = name

        super().__init__(name=self.name,
                         input_columns=[self.input_column])

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the feature extraction.

        :param df: Dataframe to run the feature extraction on
        """
        dtype = df.dtypes[self.input_column]
        if not pd.api.types.is_numeric_dtype(dtype):
            raise ValueError(f"Diff.transform: column '{self.input_column}' is"
                             f" not a numeric data type , but {dtype}")

        df = super().transform(df)
        df[self.name] = df[self.input_column].diff()
        return df


class DeltaTime(Diff):
    def __init__(self, *,
                 name: str = Feature.DELTA_TIME.value,
                 timestamp_column: str = ColumnName.TIMESTAMP):
        super().__init__(name=name,
                         input_column=timestamp_column
                         )


class MaxToBool(FeatureExtractor):
    reference_column: str = None
    threshold_value: float = None

    def __init__(self, *,
                 name: str,
                 reference_column: str,
                 threshold_value: float):
        self.threshold_value = threshold_value
        self.reference_column = reference_column

        super().__init__(name,
                         input_columns=[self.reference_column])

    def transform(self,
                  df: pd.DataFrame) -> pd.DataFrame:
        df = super().transform(df)

        df[self.name] = (df[self.reference_column] < self.threshold_value) * 1
        df.loc[df[self.reference_column] > self.threshold_value, self.reference_column] = 0

        return df

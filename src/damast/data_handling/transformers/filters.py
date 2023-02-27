"""
Module which collect all filters that have been implemented as sklearn Transformers.

Note that sklearn uses generally Duck-Typing to allow the creation of valid transformers, which
require, so a class requires

class MyTransformer:

   def fit(self, X, y=None):
       # if there is nothing to do just return self
       return self

   def transform(self, X):
       # act on the data that can be sent to this transformer either via:
       #
       #     my_transformer.transform(X) or my_transformer.fit_transform(X)
       return X
"""

from typing import List

from sklearn.pipeline import Pipeline

from damast.domains.maritime.data_specification import MMSI, ColumnName

__all__ = ["AreaFilter",
           "BaseFilter",
           "DuplicateNeighboursFilter",
           "MinGroupSizeFilter",
           "MinMaxFilter",
           "MMSIFilter"
           ]

from damast.data_handling.transformers.base import BaseTransformer


class BaseFilter(BaseTransformer):
    pass


class MinMaxFilter(BaseFilter):
    """
    Filter Based on latitude and longitude
    """

    def __init__(self,
                 min: float,
                 max: float,
                 column_name: str):
        super().__init__()

        self.min = min
        self.max = max
        self.column_name = column_name

    def transform(self, df):
        df = super().transform(df)

        df = df[(df[self.column_name] >= self.min) &
                (df[self.column_name] <= self.max)]

        df.reset_index(drop=True, inplace=True)
        return df


class MinGroupSizeFilter(BaseFilter):
    """
    Group elements by column value and remove all that fall below a given min threshold
    """

    def __init__(self,
                 min: float,
                 column_name: str):
        super().__init__()

        self.min = min
        self.column_name = column_name

    def transform(self, df):
        df = super().transform(df)

        group_sizes = df.groupby([self.column_name]).size()
        df.drop(df[df[self.column_name].isin(group_sizes[group_sizes < self.min].index.values)].index)
        df.reset_index(drop=True, inplace=True)
        return df


class DuplicateNeighboursFilter(BaseFilter):
    """
    This filter assumes a somewhat sorted dataframe as input, so that neighbouring items can be filtered.

    The filter ensures that only one of two items of the same timestamp remain is the data
    """
    #: List of column combinations which should uniquely identify a row
    column_names: List[str] = None

    def __init__(self,
                 column_names: List[str]):
        self.column_names = column_names

    def transform(self, df):
        df = super().transform(df)

        condition = None
        for column_name in self.column_names:
            if condition is None:
                condition = getattr(df, column_name) == getattr(df, column_name).shift(1)
            else:
                condition = condition & (getattr(df, column_name) == getattr(df, column_name).shift(1))

        df = df.drop(df.loc[condition].index)
        return df


class AreaFilter(BaseFilter):
    """
    Filter Based on latitude and longitude
    """

    def __init__(self,
                 latitude_min=-90.0,
                 latitude_max=90,
                 latitude_name: str = ColumnName.LATITUDE,
                 longitude_min=-180.0,
                 longitude_max=180.0,
                 longitude_name: str = ColumnName.LONGITUDE):
        super().__init__()

        self.latitude_min = latitude_min
        self.latitude_max = latitude_max
        self.latitude_name = latitude_name

        self.longitude_min = longitude_min
        self.longitude_max = longitude_max
        self.longitude_name = longitude_name

        self._pipeline = Pipeline([
            ("latitude_filter", MinMaxFilter(min=self.latitude_min,
                                             max=self.latitude_max,
                                             column_name=self.latitude_name)),
            ("longitude_filter", MinMaxFilter(min=self.longitude_min,
                                              max=self.longitude_max,
                                              column_name=self.longitude_name))
        ])

    def transform(self, df):
        df = super().transform(df)
        return self._pipeline.transform(df)


class MMSIFilter(BaseFilter):
    #: Default minimum number of message samples per MMSI
    MMSI_DEFAULT_MIN_SAMPLES: int = 100

    def __init__(self,
                 mmsi_name: str = ColumnName.MMSI,
                 timestamp_name: str = ColumnName.TIMESTAMP,
                 min_value: int = MMSI.min_value,
                 max_value: int = MMSI.max_value,
                 min_samples: int = MMSI_DEFAULT_MIN_SAMPLES
                 ):
        """

        :param mmsi_name: Name of the column for MMSI
        :param min_value: Minimum required value for the MMSI
        :param max_value: Maximum required value for the MMSI
        :param min_samples: Minimum number of samples required for MMSI
        """
        super().__init__()

        self.mmsi_min_value = min_value
        self.mmsi_max_value = max_value
        self.mmsi_min_samples = min_samples
        self.mmsi_name = mmsi_name

        self.timestamp_name = timestamp_name

        # This is an internal pipeline for this filter -
        # this could of course also be performed
        self._pipeline = Pipeline([
            ("mmsi_range", MinMaxFilter(min=self.mmsi_min_value,
                                        max=self.mmsi_max_value,
                                        column_name=self.mmsi_name)
             ),
            ("mmsi_min_msg_count", MinGroupSizeFilter(min=self.mmsi_min_samples,
                                                      column_name=self.mmsi_name)
             ),
            ("mmsi_duplicates", DuplicateNeighboursFilter(column_names=[self.mmsi_name,
                                                                        self.timestamp_name])
             )
        ])

    def transform(self, df):
        df = super().transform(df)
        return self._pipeline.transform(df)

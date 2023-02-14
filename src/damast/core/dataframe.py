# code=utf-8
"""
Module to define a dataframe
"""

from pathlib import Path
from typing import Union

from vaex import DataFrame

from .metadata import MetaData

__all__ = [
    "AnnotatedDataFrame"
]


class AnnotatedDataFrame:
    """
    A dataframe that is associated with metadata.
    """

    #: Metadata associated with the dataframe
    _metadata: MetaData = None

    #: The actual dataframe
    _dataframe: DataFrame = None

    def __init__(self,
                 dataframe: DataFrame,
                 metadata: MetaData):
        self._dataframe = dataframe
        self._metadata = metadata

        # Ensure conformity of the metadata with the dataframe
        self._metadata.apply(df=self._dataframe)

    def save(self, *, filename: Union[str, Path]) -> 'AnnotatedDataFrame':
        raise NotImplementedError()

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'AnnotatedDataFrame':
        raise NotImplementedError()

    def __getattr__(self, item):
        """
        Ensure that this object behaves like a vaex.DataFrame
        :param item:
        :return:
        """
        return getattr(self._dataframe, item)

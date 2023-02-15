# code=utf-8
"""
Module to define a dataframe
"""

from pathlib import Path
from typing import List, Union

from vaex import DataFrame

from .metadata import DataSpecification, MetaData

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

    def get_fulfillment(self, expected_specs: List[DataSpecification]) -> MetaData.Fulfillment:
        return self._metadata.get_fulfillment(expected_specs=expected_specs)

    def update(self,
               expectations: List[DataSpecification]):
        for expected_data_spec in expectations:
            column_name = expected_data_spec.name
            if column_name not in self._metadata:
                # Column name description is not yet part of the metadata
                # Verify that is it part of the data frame
                if column_name not in self._dataframe.column_names:
                    raise RuntimeError(f"{self.__class__.__name__}.update:"
                                       f" required output '{column_name}' is not"
                                       f" present in the result dataframe")
                else:
                    self._metadata.columns.append(DataSpecification.from_dict(data=expected_data_spec.to_dict()))

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

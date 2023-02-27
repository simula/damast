"""
Module which collects transformers that change the order of data
"""

from typing import Any, Dict, List

import pandas as pd
from sklearn.base import TransformerMixin

__all__ = [
    "BaseSorter",
    "GenericSorter",
    "GroupBy"
]

from damast.data_handling.transformers.base import BaseTransformer


class BaseSorter(BaseTransformer):
    """
    Base Class for transformer that affect only the order of elements.
    """
    pass


class GenericSorter(BaseSorter):
    """
    A generic transformer that handles sorting of the data, based on the type of columns
    """

    #: Names of the columns (in order to which the sorting shall be done)
    column_names: List[str] = None

    #: Keyword argument for the sort_values function
    sort_values_args: Dict[str, Any] = None

    def __init__(self,
                 column_names: List[str],
                 **kwargs):
        """
        Initialize GenericSorter

        :param column_names: Column names in sort priority order
        :param kwargs: Arguments which will be forwarded the DataFrame.sort_values call
        """
        self.column_names = column_names
        self.sort_values_args = kwargs

    def transform(self, df):
        """
        Sort the data
        :param df: The input data
        :return: The sorted data
        """
        df = super().transform(df)
        return df.sort_values(by=self.column_names, **self.sort_values_args)


class GroupBy(BaseSorter):
    """
    Transformer that can act as decorator for other transformers with group_by requirements.
    """
    group_by: str = None
    group_keys: bool = None

    transformer: TransformerMixin = None

    def __init__(self, *,
                 group_by: List[str],
                 transformer: TransformerMixin,
                 group_keys: bool = False
                 ):
        self.group_by = group_by
        self.group_keys = group_keys
        self.transformer = transformer

    def fit_transform(self, df: pd.DataFrame, y=None, **fit_params) -> Any:
        grouped_df = df.groupby(by=self.group_by,
                                group_keys=self.group_keys)

        df = grouped_df.apply(lambda x, self=self, y=y, fit_params=fit_params:
                              self.transformer.fit_transform(x, y=y, **fit_params))

        return df

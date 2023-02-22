# code=utf-8
"""
Module containing the base functionality for custom transformer implementation in this project.
"""
from __future__ import annotations
from datetime import datetime
from logging import DEBUG, Logger, getLogger
from typing import Any, ClassVar, Dict, List, Union

import numpy as np
import pandas as pd
import vaex
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

_log: Logger = getLogger(__name__)
_log.setLevel(DEBUG)


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base-transformer class. Keeps track of input and output columns.
    """
    # Optional internal pipeline that might be used for the transformation
    _pipeline: Pipeline = None

    _stats: Dict[str, Any] = None

    RUNTIME_IN_S: ClassVar[float] = "performance_in_s"

    INPUT_COLUMNS: ClassVar[str] = "input_columns"
    INPUT_SHAPE: ClassVar[str] = "input_shape"

    OUTPUT_COLUMNS: ClassVar[str] = "output_columns"
    OUTPUT_SHAPE: ClassVar[str] = "output_shape"

    def fit(self, df, y=None) -> BaseTransformer:
        return self

    def transform(self, df: Union[pd.DataFrame, vaex.DataFrame]) -> Union[pd.DataFrame, vaex.DataFrame]:
        if type(df) not in [pd.DataFrame, vaex.DataFrame, np.ndarray]:
            raise RuntimeError(f"{self.__class__.__name__}.transform requires a pandas, vaex or numpy array as"
                               f" input, but got {type(df)} ")
        return df

    def _get_column_names(self, df):
        if hasattr(df, "columns"):
            return [x for x in df.columns]
        else:
            return [None for i in range(0, df.shape[0])]

    def fit_transform(self, df: pd.DataFrame, y=None, **fit_params) -> Any:
        """
        Custom decorator for running the actual fit_transform implementation.

        This decorator collects input/output spec and runtime performance data.

        :param df: Input data frame
        :param y: Labels (when running supervised)
        :param fit_params: parameters
        :return: The result of the implemented transformation
        """
        stats = dict()

        stats[self.INPUT_SHAPE] = df.shape
        stats[self.INPUT_COLUMNS] = self._get_column_names(df)

        start = datetime.utcnow()
        result = super().fit_transform(df, y, **fit_params)
        stats[self.RUNTIME_IN_S] = (datetime.utcnow() - start).total_seconds()

        stats[self.OUTPUT_SHAPE] = df.shape
        stats[self.OUTPUT_COLUMNS] = self._get_column_names(df)
        self._stats = stats

        _log.debug(f"{self.__class__.__name__}: {stats[self.RUNTIME_IN_S]} seconds")
        return result

    def get_feature_names_out(self) -> List[str]:
        """
        Get the feature / column names that result from this transformation

        This information is only available after(!) execution of the fit_transform function
        :return: list of feature names
        """
        if self._stats is not None:
            return self._stats[self.OUTPUT_COLUMNS]

        raise RuntimeError(f"{self.__class__.__name__}.get_feature_names_out: only available once"
                           f" the transformation has been run once")

    def get_stats(self) -> Dict[str, Any]:
        """
        Add runtime performance and input, output data specifications

        :return: Dictionary containing stats: RUNTIME_IN_S, INPUT_COLUMNS, INPUT_SHAPE
            OUTPUT_COLUMNS, OUTPUT_SHAPE
        """
        if self._stats is None:
            raise RuntimeError(f"{self.__class__.__name__}.get_stats: fit_transform has not yet been called")
        return self._stats

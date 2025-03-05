"""
Module which collects transformers that add / augment the existing data
"""
import datetime
import logging
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from sklearn.neighbors import BallTree

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement
from damast.core.types import DataFrame, XDataFrame

__all__ = [
    "AddLocalIndex",
    "AddTimestamp",
    "AddUndefinedValue",
    "BallTreeAugmenter",
    "JoinDataFrameByColumn",
    "MultiplyValue"
]

_log: Logger = getLogger(__name__)


class JoinDataFrameByColumn(PipelineElement):
    """
    Add a column to an input dataframe by merging it with another dataset.

    :param dataset: Path to `.csv/.hdf5`-file or a `polars.dataframe.LazyFrame`.
    :param right_on: Column from `dataset` to use for joining data
    :param dataset_column: Name of column in `dataset` to add
    :param col_name: Name of augmented column
    :param sep: Separator in CSV file

    .. note::
        :code:`right_on` will not be added as a new column in the transformed dataset
    """

    class JoinHowType(str, Enum):
        INNER = 'inner'
        LEFT = 'left'
        RIGHT = 'right'
        FULL = 'full'
        SEMI = 'semi'
        ANTI = 'anti'
        CROSS = 'cross'

    _right_on: str
    _dataset: DataFrame
    _dataset_column: str
    _join_how: JoinHowType

    def __init__(self,
                 dataset: Union[str, Path, XDataFrame],
                 right_on: str,
                 dataset_col: str,
                 how: JoinHowType = JoinHowType.LEFT,
                 sep: str = ";"):
        self._right_on: str = right_on
        self._dataset_column = dataset_col
        self._join_how = how

        # Load vessel type map
        if type(dataset) in [str, Path]:
            dataset = self.load_data(filename=dataset, sep=sep)

        # Check that the columns exist in dataset
        for col_name in [self._right_on, self._dataset_column]:
            if col_name not in dataset.columns:
                raise KeyError(f"Missing column: '{col_name}' in vessel type information: '{dataset.head()}'"
                               " - available are {','.join(vessel_type_data.columns)}")

        column_dtype = XDataFrame(dataset).dtype(self._dataset_column)
        if int not in [column_dtype, column_dtype.to_python()]:
            raise ValueError(f"{self.__class__.__name__}.__init__:"
                             f" column '{self._dataset_column}' must be of type int, "
                             f", but was: {column_dtype}")

        self._dataset = dataset

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path], sep: str) -> DataFrame:
        """
        Load dataset from file

        :param filename: The input file (or path)
        :param sep: Separator in csv
        :return: A `DataFrame` with the data
        """
        try:
            return XDataFrame.open(path=filename, sep=sep)
        except FileNotFoundError as e:
            raise RuntimeError(f"{cls}: Vessel type information not accessible -- {e}")

    @damast.core.input({"x": {}})
    @damast.core.output({"out": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Join datasets by column "x".
        Adds column "out".

        :returns: DataFrame with added column
        """
        other_df = self._dataset.select(
                pl.col(self._right_on),
                pl.col(self._dataset_column)
        ).lazy()

        dataframe = df._dataframe.join(
                other=other_df,
                left_on=self.get_name("x"),
                right_on=self._right_on,
                #validate='1:1',
                #https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html
                how=self._join_how.value
            )

        df._dataframe = dataframe.rename({self._dataset_column: self.get_name("out")})
        return df


class BallTreeAugmenter():
    """
    A class for computation  in distance computation using BallTree.

    Uses the `sklearn.neighbours.BallTree` to compute the distance for any n-dimensional
    feature. The BallTree is created prior to being passed in as the lambda function of a
    `DataFrame.add_virtual_column`. The object can later be depickled from the state,
    and one can retrieve any meta-data added to the class after construction.

    :param x: The points to use in the BallTree
    :param metric: The metric to use in the BallTree, for available metrics see:
        https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    """

    _tree: BallTree
    _metric: str
    _modified: datetime.datetime

    def __init__(self, x: npt.NDArray[np.float64], metric: str):
        self._tree = BallTree(x, metric=metric)
        self._metric = metric
        self.__name__ = f"Balltree_{self._metric}"
        self._modified = datetime.datetime.now()

    def update_balltree(self, x: npt.NDArray[np.float64]):
        """
        Replace points in the Balltree

        :param x: (npt.NDArray[np.float64]): The new points
        """
        logger = logging.getLogger("damast")
        if not np.allclose(self._tree.get_arrays()[0], x):
            logger.debug("Recreating balltree with new inputs")
            self._tree = BallTree(x, metric=self._metric)
            self._modified = datetime.datetime.now()
        else:
            logger.debug("No points to update in balltree")

    def __call__(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute distances between the Balltree and each entry in `x`"""
        return self._tree.query(np.vstack([x, y]).T, return_distance=True)[
            0].reshape(-1)

    @property
    def modified(self) -> datetime.datetime:
        """
        Last time the underlying BallTree was modified
        """
        return self._modified


class AddUndefinedValue(PipelineElement):
    """
    Replace missing and Not Available (NA) entries in a column with a given value.

    :param fill_value: The value replacing NA
    """
    _fill_value: Any

    def __init__(self, fill_value: Any):
        """Constructor"""
        self._fill_value = fill_value

    @property
    def fill_value(self):
        return self._fill_value

    @damast.core.describe("Fill undefined values")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Fill in values for NA and missing entries
        """
        mapped_name = self.get_name("x")
        df._dataframe = df._dataframe.with_columns(
                pl.col(mapped_name).fill_null(self.fill_value).alias(mapped_name),
            ).with_columns(
                pl.col(mapped_name).fill_nan(self.fill_value).alias(mapped_name)
            )
        return df


class AddLocalIndex(PipelineElement):
    """
    Compute the local index of an entry in a given group (sorted by a given column).

    Also compute the reverse index, i.e. how many entries in the group are after this message
    """

    @damast.core.describe("Compute the ")
    @damast.core.input({"group": {"representation_type": int},
                        "sort": {}})
    @damast.core.output({"local_index": {"representation_type": int},
                         "reverse_{{local_index}}": {"representation_type": int}})
    def transform(self, df: damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        dataframe = df._dataframe

        group_column = self.get_name("group")
        sort_column = self.get_name("sort")

        df._dataframe = dataframe\
                .sort(group_column, sort_column)\
                .with_columns(
                    pl.int_range(pl.len()).over(group_column).alias(self.get_name("local_index")),
                    pl.int_range(start=pl.len()-1, end=-1, step=-1).over(group_column).alias(self.get_name("reverse_{{local_index}}"))
                )

        return df


def convert_to_datetime(date_string: str) -> float:
    """
    Convert date-time to timestamp (float)

    :param date_string: String representation of date, expected format is
        ``YYYY-MM-DD HH:MM:SS``
    :return: Time-stamp as float. Returns `nan`
    """
    try:
        return datetime.datetime.timestamp(
            datetime.datetime.strptime(
                date_string, "%Y-%m-%d %H:%M:%S"))
    except TypeError:
        return float(np.nan)


class AddTimestamp(PipelineElement):
    """
    Add Timestamp from date Time UTC.

    If time-stamp is not supplied for a row add ``NaN``
    """

    @damast.core.describe("Add Timestamp")
    @damast.core.input({"from": {"representation_type": str}})
    @damast.core.output({"to": {"representation_type": float}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Add Timestamp from datetimeUTC
        """
        from_mapped_name = self.get_name("from")
        to_mapped_name = self.get_name("to")

        df._dataframe = df._dataframe.with_columns(
                pl.col(from_mapped_name).map_elements(convert_to_datetime, return_dtype=float).alias(to_mapped_name)
        )
        return df


class MultiplyValue(PipelineElement):
    """
    Multiply a column by a value.

    :param multiply_value: The value to use to multiply
    """
    _mul_value: Any

    def __init__(self, mul_value: Any):
        """Constructor"""
        self._mul_value = mul_value

    @property
    def mul_value(self):
        return self._mul_value

    @damast.core.describe("Multiply a column by a given values")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Multiply a column by a given value
        """
        mapped_name = self.get_name("x")
        df._dataframe = df._dataframe.with_columns(
            (pl.col(mapped_name)*self.mul_value).alias(mapped_name)
        )
        return df


class ChangeTypeColumn(PipelineElement):
    """
    Create a new column with the new type of a given column.

    The new column name can be defined by providing a name_mapping for a column 'y'.
    If no name_mapping is provided the column's new name will be 'y'

    :param new_type: The new type of the column
    """
    _new_type: Any

    def __init__(self, new_type: Any):
        """Constructor"""
        if type(new_type) == str:
            polar_type = new_type.capitalize()
            if not hasattr(pl.datatypes.classes, polar_type):
                raise TypeError(f"Type {new_type} has not correspondence in 'polars'")
            self._new_type = getattr(pl.datatypes.classes, polar_type)

    @property
    def new_type(self):
        return self._new_type

    @damast.core.describe("Create a new column from an existing column but with a new type")
    @damast.core.input({"x": {}})
    @damast.core.output({"y": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Change the default type of a column
        """
        input_mapped_name = self.get_name("x")
        output_mapped_name = self.get_name("y")

        df._dataframe = df._dataframe.with_columns(
            pl.col(input_mapped_name).cast(self._new_type).alias(output_mapped_name)
        )
        return df

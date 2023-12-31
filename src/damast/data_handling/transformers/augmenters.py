"""
Module which collects transformers that add / augment the existing data
"""
import datetime
import logging
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import vaex
from sklearn.neighbors import BallTree

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement

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

    :param dataset: Path to `.csv/.hdf5`-file or a `vaex.DataFrame`.
    :param right_on: Column from `dataset` to use for joining data
    :param dataset_column: Name of column in `dataset` to
    :param col_name: Name of augmented column
    :param sep: Separator in CSV file

    .. note::
        :code:`right_on` will not be added as a new column in the transformed dataset
    """
    _right_on: str
    _dataset: pd.DataFrame
    _dataset_column: str

    def __init__(self,
                 dataset: Union[str, Path, vaex.DataFrame],
                 right_on: str,
                 dataset_col: str,
                 sep: str = ";"):
        self._right_on: str = right_on
        self._dataset_column = dataset_col
        # Load vessel type map
        if not isinstance(dataset, vaex.DataFrame):
            dataset = self.load_data(filename=dataset, sep=sep)

        # Check that the columns exist in dataset
        for col_name in [self._right_on, self._dataset_column]:
            if col_name not in dataset.column_names:
                raise KeyError(f"Missing column: '{col_name}' in vessel type information: '{dataset.head()}'"
                               " - available are {','.join(vessel_type_data.columns)}")

        column_dtype = dataset[self._dataset_column].dtype
        if column_dtype != int:
            raise ValueError(f"{self.__class__.__name__}.__init__:"
                             f" column '{self._dataset_column}' must be of type int, "
                             f", but was: {column_dtype}")

        self._dataset = dataset

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path], sep: str) -> vaex.DataFrame:
        """
        Load dataset from file

        :param filename: The input file (or path)
        :param sep: Separator in csv
        :return: A `vaex.DataFrame` with the data
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise RuntimeError(f"Vessel type information not accessible. File {file_path} not found")

        if file_path.suffix == ".csv":
            return vaex.from_csv(file_path, sep=sep)
        elif file_path.suffix in {".hdf5", ".h5"}:
            return vaex.open(file_path)

        raise ValueError(f"{cls.__name__}.load_data: Unsupported input file format {file_path.suffix}")

    @damast.core.input({"x": {}})
    @damast.core.output({"out": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Join datasets by column "x".
        Adds column "out".

        :returns: DataFrame with added column
        """
        dataframe = df._dataframe
        dataframe.join(
            self._dataset[[self._right_on, self._dataset_column]],
            left_on=self.get_name("x"),
            right_on=self._right_on,
            inplace=True)
        # If left_on and right_on are not equal, join adds right_on as a new column, adding
        # duplicate data
        if self._right_on != self.get_name("x"):
            dataframe.drop(self._right_on, inplace=True)
        dataframe.rename(self._dataset_column, self.get_name("out"))
        return df


class BallTreeAugmenter():
    """
    A class for computation  in distance computation using BallTree.

    Uses the `sklearn.neighbours.BallTree` to compute the distance for any n-dimensional
    feature. The BallTree is created prior to being passed in as the lambda function of a
    `vaex.DataFrame.add_virtual_column`. The object can later be depickled from the state,
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
        df._dataframe[mapped_name] = df._dataframe[mapped_name].fillna(self._fill_value)
        df._dataframe[mapped_name] = df._dataframe[mapped_name].fillmissing(self._fill_value)
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

        local_index = np.empty(len(dataframe), dtype=int)
        reverse_local_index = np.empty(len(dataframe), dtype=int)

        # Identify the column index of the sort field
        def update_indexes(indices):
            group_size = len(indices)
            sorted_indices = sorted(indices, key=lambda x: x[1])

            for index, idx, in enumerate(sorted_indices):
                local_index[idx[0]] = index
                reverse_local_index[idx[0]] = group_size - 1 - index


        # Identify the min and max index of a group, as well as the total length
        groups_properties = {}
        # Identify groups by selecting the indices
        groups = {}
        group_column = self.get_name("group")
        sort_column = self.get_name("sort")

        _log.info("AddLocalIndex: getting group properties")
        # First identify first and last occurence of the items and keep count of the length of the trajectory
        # that will allow us to update the local index earyl, and keep the memory footprint smaller
        for i1, i2, chunk in dataframe.evaluate_iterator(group_column, chunk_size=500):
            for index in range(i1, i2):
                group_id = chunk[index - i1]
                if group_id not in groups_properties:
                    groups_properties[group_id] = [index, None, 1]
                else:
                    groups_properties[group_id][1] = index
                    groups_properties[group_id][2] += 1

        _log.info("AddLocalIndex: updating local index - no parallel execution")
        # Now reiterate over the chunks to identify the full index list per MMSI
        # update the local index as soon as the last item for a group has been encountered
        for i1, i2, chunks in dataframe.evaluate_iterator([group_column, sort_column], chunk_size=500, parallel=False):
            chunk_dict = dict(zip([group_column, sort_column], chunks))
            for index in range(i1, i2):
                group_id = chunk_dict[group_column][index - i1]
                sort_criteria = chunk_dict[sort_column][index - i1]

                if group_id not in groups:
                    groups[group_id] = [[index, sort_criteria]]
                else:
                    groups[group_id].append([index, sort_criteria])

                # check if the maximum index has been found, and if so, update the local index
                # and clean up
                max_index = groups_properties[group_id][1]
                length = groups_properties[group_id][2]
                if length == 1 or max_index == index:
                    # perform a local update of the index
                    update_indexes(groups[group_id])
                    del groups[group_id]

        if groups != {}:
            raise RuntimeError(f"{self.__class__.__name__}.transform: update index failed - remaining groups {groups}")

        # Assign global arrays to dataframe
        dataframe[self.get_name("local_index")] = local_index
        dataframe[self.get_name("reverse_{{local_index}}")] = reverse_local_index

        # Drop temporary index columns
        del local_index, reverse_local_index
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
        dataframe = df._dataframe

        from_mapped_name = self.get_name("from")
        to_mapped_name = self.get_name("to")

        dataframe[to_mapped_name] = dataframe[from_mapped_name].apply(convert_to_datetime)
        dataframe.materialize(to_mapped_name)
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
        df._dataframe[mapped_name] *= self._mul_value
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
        self._new_type = new_type

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
        df._dataframe[output_mapped_name] = df._dataframe[input_mapped_name].astype(self._new_type)
        return df

"""
Module which collects transformers that add / augment the existing data
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union
import datetime
import numpy as np
import pandas as pd
import vaex
from sklearn.neighbors import BallTree
from sklearn.preprocessing import Binarizer
import numpy.typing as npt
import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import DataSpecification, PipelineElement
from damast.data_handling.transformers.base import BaseTransformer
from damast.domains.maritime.data_specification import ColumnName, FieldValue

__all__ = [
    "AddCombinedLabel",
    "AddLocalMessageIndex",
    "JoinDataFrameByColumn",
    "BaseAugmenter",
    "InvertedBinariser",
    "BallTreeAugmenter",
    "AddUndefinedValue"
]


class BaseAugmenter(BaseTransformer):
    pass


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
    _inplace: bool

    def __init__(self,
                 dataset: Union[str, Path, vaex.DataFrame],
                 right_on: str,
                 dataset_col: str,
                 sep: str = ";",
                 inplace: bool = False):
        self._right_on: str = right_on
        self._inplace = inplace
        self._dataset_column = dataset_col
        # Load vessel type map
        if not isinstance(dataset, vaex.DataFrame):
            dataset = self.load_data(filename=dataset, sep=sep)

        # Check that the columns exist in dataset
        for col_name in [self._right_on, self._dataset_column]:
            if col_name not in dataset.column_names:
                raise KeyError(f"Missing column: '{col_name}' in vessel type information: '{dataset.head()}'"
                               " - available are {','.join(vessel_type_data.columns)}")

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
            dataframe = vaex.from_csv(file_path, sep=sep)
        elif file_path.suffix == ".hdf5":
            dataframe = vaex.open(file_path)
        return dataframe

    @damast.core.input({"x": {}})
    @damast.core.output({"out": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Join datasets by column "x".
        Adds column "out".

        :returns: DataFrame with added column
        """
        if not self._inplace:
            dataframe = df._dataframe.copy()
        else:
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
        new_spec = DataSpecification(self.get_name("out"))

        if self._inplace:
            df._metadata.columns.append(new_spec)
            return df
        else:
            metadata = df._metadata.columns.copy()
            metadata.append(new_spec)
            return AnnotatedDataFrame(dataframe, metadata=damast.core.MetaData(
                metadata))


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


class AddCombinedLabel(BaseAugmenter):
    """
    Create a single label from two existing ones

    :param column_names: List of columns to combine
    :param column_permitted_values: For each entry in `column names` the permitted values in each column
    :param combination name: Name of new label

    .. highlight:: python
    .. code-block:: python

        AddCombinedLabel(column_names=[col.FISHING_TYPE, col.STATUS],
                        column_permitted_values={col.FISHING_TYPE: {...},col.STATUS: { ... }) },
                        combination_name="combination")
    """
    #: List of columns that shall be combined
    column_names: List[str] = None

    # Mapping of id
    _label_mapping: Dict[str, List[str]] = None

    def __init__(self,
                 column_names: List[str],
                 column_permitted_values: Dict[str, List[str]],
                 combination_name: str = ColumnName.COMBINATION):
        number_of_columns = len(column_names)
        if number_of_columns != 2:
            raise ValueError("AddCombinedLabel: currently only the combination of exactly two columns is supported")

        self.column_names = column_names
        self.combination_name = combination_name
        self.permitted_values = column_permitted_values

    @property
    def label_mapping(self) -> Dict[str, Dict[str, str]]:
        if self._label_mapping is None:
            self._label_mapping = self.compute_label_mapping()
        return self._label_mapping

    def compute_label_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Compute the custom 2D label to id mapping.

        :return:
        """
        col_a, col_b = self.column_names
        """
        Create a dictionary containing the number and the name of all the combination of status and fishing type.
        """
        col_a_value_count = len(self.permitted_values[col_a].keys())

        combination_mapping: Dict[str, List[str]] = {}
        for a_index, (a_key, a_value) in enumerate(self.permitted_values[col_a].items()):
            for b_index, (b_key, b_value) in enumerate(self.permitted_values[col_b].items()):
                id = b_index * col_a_value_count + a_index
                combination_mapping[str(id)] = [a_value, b_value]
        return combination_mapping

    def transform(self, df):
        """
        Combine two existing labels to create a new one
        :param df: Input dataframe
        :return: Output dataframe
        """
        df = super().transform(df)

        conditions: List[List[bool]]
        choices: List[int]

        # Currently only the combination of two columns is supported
        column_a, column_b = self.column_names
        column_a_permitted_values = self.permitted_values[column_a]
        column_b_permitted_values = self.permitted_values[column_b]

        conditions, choices = zip(
            *[(((df[column_a] == v2) & (df[column_b] == v1)), i1 * len(column_a_permitted_values) + i2)
              for i1, (k1, v1) in enumerate(column_b_permitted_values.items())
              for i2, (k2, v2) in enumerate(column_a_permitted_values.items())
              ])

        df[self.combination_name] = np.select(conditions, choices, default=FieldValue.UNDEFINED)
        return df


class AddUndefinedValue(PipelineElement):
    """Replace missing and Not Available (NA) entries in a column with a given value.

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


class AddLocalMessageIndex(PipelineElement):
    """
    Compute the local index of an entry in a given group (sorted by a given column).
    Also compute the reverse index, i.e. how many entries in the group are after this message

    :param inplace: Copy input dataframe in transform if False
    """
    _inplace: bool

    def __init__(self, inplace: bool = True):
        self._inplace = inplace

    @damast.core.describe("Compute the ")
    @damast.core.input({"group": {"representation_type": int},
                        "sort": {}})
    @damast.core.output({"msg_index": {"representation_type": int},
                         "reverse_{{msg_index}}": {"representation_type": int}})
    def transform(self, df: damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        if not self._inplace:
            dataframe = df._dataframe.copy()
        else:
            dataframe = df._dataframe

        dataframe["INDEX"] = vaex.vrange(0, len(dataframe), dtype=int)

        historic_position = np.empty(len(dataframe), dtype=int)
        reverse_historic_position = np.empty(len(dataframe), dtype=int)
        # Sort each group
        groups = dataframe.groupby(by=self.get_name("group"))
        for _, group in groups:
            if len(group) == 0:
                continue
            sorted_group = group.sort(self.get_name("sort"))
            # For each group compute the local position
            position = np.arange(0, len(sorted_group), dtype=int)
            # Assign local position and reverse position to global arrays
            global_indices = sorted_group["INDEX"].evaluate()
            historic_position[global_indices] = position
            reverse_historic_position[global_indices] = len(group)-1-position
            del global_indices

        # Assign global arrays to dataframe
        dataframe[self.get_name("msg_index")] = historic_position
        dataframe[self.get_name("reverse_{{msg_index}}")] = reverse_historic_position

        # Drop global index column
        dataframe.drop("INDEX", inplace=True)
        del historic_position, reverse_historic_position
        new_specs = [damast.core.DataSpecification(self.get_name("msg_index"), representation_type=int),
                     damast.core.DataSpecification(self.get_name("reverse_{{msg_index}}"), representation_type=int)]
        if self._inplace:
            [df._metadata.columns.append(new_spec) for new_spec in new_specs]
            return df
        else:
            metadata = df._metadata.columns.copy()
            [metadata.append(new_spec) for new_spec in new_specs]
            return damast.core.AnnotatedDataFrame(dataframe, metadata=damast.core.MetaData(
                metadata))


class InvertedBinariser(BaseAugmenter):
    """
    Set all values below(!) a threshold to 1, and above the threshold to 0

    This complements sklearns binariser with an 'inverted' assignment
    """
    base_column_name: str
    threshold: float = None

    column_name: str = None

    #:
    _binarizer: Binarizer = None

    def __init__(self,
                 base_column_name: str,
                 threshold: float,
                 column_name: str = None):

        self.base_column_name = base_column_name
        self.threshold = threshold

        self._binarizer = Binarizer(threshold=threshold)

        if column_name is None:
            self.column_name = f"{base_column_name}_TRUE"
        else:
            self.column_name = column_name

    def transform(self, df) -> pd.DataFrame:
        df = super().transform(df)
        self._binarizer.fit(df)

        df[self.column_name] = self._binarizer.transform(df)
        # Invert the boolean and convert back to int
        df[self.column_name] = (-(df[self.column_name].astype(bool))).astype(int)
        return df

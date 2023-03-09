"""
Module which collects transformers that add / augment the existing data
"""
import re
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import vaex
from pyorbital.orbital import Orbital
from sklearn.preprocessing import Binarizer

import damast.core
import damast.data_handling.transformers.augmenters as augmenters
from damast.core.dataprocessing import PipelineElement
from damast.data_handling.transformers.augmenters import (
    BallTreeAugmenter,
    BaseAugmenter
)
from damast.domains.maritime.ais.navigational_status import (
    AISNavigationalStatus
)
from damast.domains.maritime.data_specification import ColumnName, FieldValue
from damast.domains.maritime.math.spatial import EARTH_RADIUS

__all__ = [
    "AddCombinedLabel",
    "AddMissingAISStatus",
    "ComputeClosestAnchorage"
]


class ComputeClosestAnchorage(PipelineElement):
    _function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    _inplace: bool

    def __init__(self,
                 dataset: Union[str, Path, vaex.DataFrame],
                 columns: List[str],
                 sep: str = ";", inplace: bool = True):
        """
        Compute the closest anchorage given a data-set with all closest anchorages

        :param dataset: Path to data-set with closest anchorages
        :param columns: Names of columns used to define the distance to anchorage (The data should be in degrees)
        :param sep: Separator used in dataset if dataset is a csv file
        :param inplace: If False copy dataset during transform
        """
        if isinstance(dataset, vaex.DataFrame):
            _dataset = dataset
        else:
            _dataset = self.load_data(dataset, sep)
        radian_dataset = [_dataset[column].deg2rad().evaluate() for column in columns]
        self._function = BallTreeAugmenter(np.vstack(radian_dataset).T, "haversine")
        self._inplace = inplace

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path], sep: str) -> vaex.DataFrame:
        """
        Load dataset from file

        :param filename: The input file (or path)
        :param sep: Separator in csv
        :return: A `pandas.DataFrame` where each row has a column MMSI and vessel_type
        """
        vessel_type_csv = Path(filename)
        if not vessel_type_csv.exists():
            raise RuntimeError(f"Vessel type information not accessible. File {vessel_type_csv} not found")

        if vessel_type_csv.suffix == ".csv":
            vessel_types = vaex.from_csv(vessel_type_csv, sep=sep)
        elif vessel_type_csv.suffix == ".hdf5":
            vessel_types = vaex.open(vessel_type_csv)
        vessel_types = pd.read_csv(vessel_type_csv, sep=sep)
        return vessel_types

    @damast.core.describe("Compute distance from dataset to closest anchorage")
    @damast.core.input({"x": {"representation_type": np.float64, "unit": damast.core.units.units.deg},
                        "y": {"representation_type": np.float64, "unit": damast.core.units.units.deg}})
    @damast.core.output({"distance": {"representation_type": np.float64}})
    def transform(self, df:  damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        if not self._inplace:
            dataframe = df._dataframe.copy()
        else:
            dataframe = df._dataframe

        # Transform latitude and longitude to radians
        dataframe.add_virtual_column(f"{self.get_name('x')}_rad",
                                     dataframe.apply(np.deg2rad, [self.get_name('x')], vectorize=True))
        dataframe.add_virtual_column(f"{self.get_name('y')}_rad",
                                     dataframe.apply(np.deg2rad, [self.get_name('y')], vectorize=True))
        dataframe.add_virtual_column(self.get_name("distance"),
                                     dataframe.apply(self._function,
                                                     [f"{self.get_name('x')}_rad", f"{self.get_name('y')}_rad"],
                                                     vectorize=True))

        # Drop/hide conversion columns
        dataframe.drop(columns=[f"{self.get_name('x')}_rad", f"{self.get_name('y')}_rad"], inplace=True)

        # Multiply distance column by earth radius
        dataframe[self.get_name("distance")] *= EARTH_RADIUS
        dataframe.units[self.get_name("distance")] = damast.core.units.units.km
        new_spec = damast.core.DataSpecification(self.get_name("out"), unit=damast.core.units.units.km)
        if self._inplace:
            df._metadata.columns.append(new_spec)
            return df
        else:
            metadata = df._metadata.columns.copy()
            metadata.append(new_spec)
            return damast.core.AnnotatedDataFrame(dataframe, metadata=damast.core.MetaData(
                metadata))


class AddMissingAISStatus(augmenters.AddUndefinedValue):
    """For a given column replace rows with missing entries with
    :attr:`damast.domains.maritime.ais.AISNavigationalStatus.Undefined`.
    """

    def __init__(self):
        self._fill_value = int(AISNavigationalStatus.Undefined)

    @damast.core.describe("Fill missing AIS status")
    @damast.core.input({"x": {"representation_type": int}})
    @damast.core.output({"x": {"representation_type": int}})
    def transform(self, df: damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        return super().transform(df)


class AddVesselType(augmenters.JoinDataFrameByColumn):
    def __init__(self, right_on: str,
                 dataset_col: str,
                 dataset: Union[str, Path, vaex.DataFrame],
                 inplace: bool = False
                 ):
        """Add in vessel type based on external data-set

        :param right_on: Name in data-set column to use for merging datasets
        :param dataset_col: Column to add to input dataframe
        :param dataset: Dataset or path to dataset
        :param inplace (bool, optional): If inplace do not copy input dataframe
        """
        super().__init__(dataset=dataset, right_on=right_on, dataset_col=dataset_col,
                         inplace=inplace)

    @damast.core.describe("Add vessel-type from other dataset to current dataset")
    @damast.core.input({"x": {}})
    @damast.core.output({"out": {}})
    def transform(self, df: damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        return super().transform(df)


class AddCombinedLabel(BaseAugmenter):
    """
    Create a single label from two existing ones
    """
    #: List of columns that shall be combined
    column_names: List[str] = None

    # Mapping of id
    _label_mapping: Dict[str, List[str]] = None

    def __init__(self,
                 column_names: List[str],
                 column_permitted_values: Dict[str, List[str]],
                 combination_name: str = ColumnName.COMBINATION) -> None:
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

        .. highlight:: python
        .. code-block:: python

            AddCombinedLabel(column_names=[col.FISHING_TYPE, col.STATUS],
                                column_permitted_values={col.FISHING_TYPE: {...},col.STATUS: { ... }) },
                                combination_name="combination")

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

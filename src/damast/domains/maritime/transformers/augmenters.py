"""
Module which collects transformers that add / augment the existing data
"""
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

import damast.core
import damast.data_handling.transformers.augmenters as augmenters
from damast.core.dataprocessing import PipelineElement
from damast.core.types import DataFrame, XDataFrame
from damast.data_handling.transformers.augmenters import BallTreeAugmenter
from damast.domains.maritime.ais.navigational_status import AISNavigationalStatus
from damast.domains.maritime.ais.vessel_types import VesselType
from damast.domains.maritime.math.spatial import EARTH_RADIUS

__all__ = [
    "AddMissingAISStatus",
    "ComputeClosestAnchorage"
]


class ComputeClosestAnchorage(PipelineElement):
    """
    Compute the closest anchorage given a data-set with all closest anchorages

    :param dataset: Path to data-set with closest anchorages
    :param columns: Names of columns used to define the distance to anchorage (The data should be in degrees)
    :param sep: Separator used in dataset if dataset is a csv file
    """
    _function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]

    def __init__(self,
                 dataset: Union[str, Path, DataFrame],
                 columns: List[str],
                 sep: str = ";"):
        if isinstance(dataset, DataFrame):
            _dataset = dataset
        else:
            _dataset = self.load_data(dataset, sep)
        radian_dataset = [_dataset[column].deg2rad().evaluate() for column in columns]
        self._function = BallTreeAugmenter(np.vstack(radian_dataset).T, "haversine")

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path], sep: str) -> DataFrame:
        """
        Load dataset from file

        :param filename: The input file (or path)
        :param sep: Separator in csv
        :return: A `pandas.DataFrame` where each row has a column MMSI and vessel_type
        """
        try:
            return XDataFrame.open(filename, sep=sep)
        except FileNotFoundError as e:
            raise RuntimeError(f"{cls}: Vessel type information not accessible. File {vessel_type_csv} not found")

    @damast.core.describe("Compute distance from dataset to closest anchorage")
    @damast.core.input({"x": {"representation_type": float, "unit": damast.core.units.units.deg},
                        "y": {"representation_type": float, "unit": damast.core.units.units.deg}})
    @damast.core.output({"distance": {"representation_type": float, "unit": damast.core.units.units.km}})
    def transform(self, df:  damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        dataframe = df._dataframe

        x_name = self.get_name('x')
        y_name = self.get_name('y')

        # Transform latitude and longitude to radians
        dataframe = dataframe.with_columns(
            (pl.col(x_name).map_elements(np.deg2rad)).alias(f"{x_name}_rad")
        )
        dataframe = dataframe.with_columns(
            (pl.col(y_name).map_elements(np.deg2rad)).alias(f"{y_name}_rad")
        )

        dataframe = dataframe.with_columns(
            pl.struct(x_name, y_name).map_elements(lambda x: self._function(x[x_name], x[y_name])).alias("distance")
        )

        # Multiply distance column by earth radius
        dataframe = dataframe.with_columns(
            (pl.col(distance)*EARTH_RADIUS).alias("distance")
        )
        dataframe[self.get_name("distance")] *= EARTH_RADIUS
        #dataframe.units[self.get_name("distance")] = damast.core.units.units.km

        # Drop/hide conversion columns
        dataframe = dataframe.drop(columns=[f"{self.get_name('x')}_rad", f"{self.get_name('y')}_rad"])
        return df


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
    """
    Add the vessel type (as integer value) based on external data-set and corresponding to ::class::`VesselType`.

    The vessel type might for instance come from the [global fishing watch database](https://globalfishingwatch.org/).
    It must be either a string corresponding to the snake_case class name, or the corresponding integer

    :param right_on: Name in data-set column to use for merging datasets
    :param dataset_col: Column to add to input dataframe
    :param dataset: Dataset or path to dataset
    :raise ValueError: when the input column that shall extend the dataset is neither int nor str
    """

    def __init__(self, right_on: str,
                 dataset_col: str,
                 dataset: Union[str, Path, DataFrame]
                 ):

        if not isinstance(dataset, DataFrame):
            dataset = XDataFrame.open(path=dataset)

        column_dtype = XDataFrame(dataset).dtype(dataset_col)
        name = f"{dataset_col}_mapped"
        if column_dtype == str:
            # VesselTypes should be mapped to integers
            mapping = VesselType.get_mapping()
            dataset = dataset.with_columns(
                pl.col(dataset_col).map_elements(
                    lambda x: mapping[x], return_dtype=pl.Int64i
                ).alias(name)
            )
        elif column_dtype == int:
            dataset = dataset.with_columns(
                    pl.col(dataset_col).alias(name)
            )
        else:
            raise ValueError(f"{self.__class__.__name__}.__init__: dtype of column '{dataset_col}',"
                             " must be either int or str, but was '{column_dtype}'")

        super().__init__(dataset=dataset, right_on=right_on, dataset_col=name)

    @damast.core.describe("Add vessel-type from other dataset to current dataset, "
                          "where the input x is the linking identifier")
    @damast.core.input({"x": {}})
    @damast.core.output({"out": {"representation_type": int}})
    def transform(self, df: damast.core.AnnotatedDataFrame) -> damast.core.AnnotatedDataFrame:
        return super().transform(df)

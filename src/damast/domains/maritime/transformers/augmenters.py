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
from damast.core.dataprocessing import PipelineElement
import damast.data_handling.transformers.augmenters as augmenters
from damast.data_handling.transformers.augmenters import (
    BallTreeAugmenter,
    BaseAugmenter
)
from damast.domains.maritime.ais.navigational_status import (
    AISNavigationalStatus
)
from damast.domains.maritime.data_specification import ColumnName, FieldValue
from damast.domains.maritime.math.spatial import (
    EARTH_RADIUS,
    angle_sat_c,
    chord_distance,
    distance_sat_vessel,
    great_circle_distance
)

__all__ = [
    "AddCombinedLabel",
    "AddDistanceClosestAnchorage",
    "AddDistanceClosestSatellite",
    "AddLocalMessageIndex",
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
        :param columns: Names of columns used to define the distance to anchorage (The data should be in radians)
        :param sep: Separator used in dataset if dataset is a csv file
        :param inplace: If False copy dataset during transform_
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


class AddDistanceClosestSatellite(BaseAugmenter):
    """
    Compute the distance uses the coordinate of the satellite computed by
    the tle file for a given timestamp.
    Create `SAT_DISTANCE` column on the data set.

    :param params: Input parameter dictionary. Requires the following keys:

    .. highlight:: python
    .. code-block:: python

        {"do_compute_distance_satellite": True/False,
         "list_sat": ["sat0", "sat1", ...],
         "tle_file": "path_to_tle_file"
        }

    The `tle_file` has the columns `SAT_LON`, `SAT_LAT`, `SAT_ALT`.
    From these entries one can compute the chord distance `CORDE`.

    :param df: Dataframe to write distances to.

    :return: The augmented dataframe
    """
    #: The TLE filename
    tle_filename: Path = None

    #: List of satellites
    satellites: List[str] = None

    def __init__(self,
                 satellite_tle_filename: Union[str, Path],
                 column_name: str = ColumnName.DISTANCE_CLOSEST_SATELLITE,
                 timestamp_name: str = ColumnName.TIMESTAMP,
                 latitude_name: str = "latitude",
                 longitude_name: str = "longitude"):
        """

        :param satellites: list of satellite (names), which shall be considered for this computation
        :param satellite_tle_filename: The two-line element set file, for the satellite positions,
            see also https://en.wikipedia.org/wiki/Two-line_element_set
        :param column_name: Name of the resulting column representing the distance to the closest satellite
        :param timestamp_name: Name of the column containing the timestamp data
        """
        # Fetch satellite data
        self.tle_filename = Path(satellite_tle_filename)
        self.satellites = []

        # Collect the satellite names for the existing tle file
        if self.tle_filename.exists():
            self.satellites = AddDistanceClosestSatellite.get_satellites(filename=self.tle_filename)
        else:
            raise RuntimeError("Satellites not found")
        self.column_name = column_name
        self.timestamp_name = timestamp_name

        self.longitude_name = longitude_name
        self.latitude_name = latitude_name

    @staticmethod
    def get_satellites(filename: Union[str, Path]) -> List[str]:
        """
        Get the list of satellites from the TLE file
        :param filename: The TLE file (Two-line element set) -- https://en.wikipedia.org/wiki/Two-line_element_set
        :return: List of satellite names
        """
        satellites = []
        # Collect the satellite names for the existing tle file
        filename = Path(filename)
        if filename.exists():
            with open(filename, "r") as f:
                contents = f.readlines()
                for line in contents:
                    if not re.match("^[0-9]", line):
                        satellites.append((line.strip()))

        return satellites

    def transform(self, X):
        X = super().transform(X)
        first_satellite = True
        if len(self.satellites) == 0:
            return X
        for sat in self.satellites:
            sat_specific_orbital = Orbital(sat, tle_file=str(self.tle_filename))

            # Compute intersection point between earth surface and a vector from earths center to the satellite
            # results will be stored in temporary columns
            # Compute the current satellite position (based on the given timestamp
            X["SAT_LON"], X["SAT_LAT"], X["SAT_ALT"] = sat_specific_orbital.get_lonlatalt(X[self.timestamp_name])

            # Shortest distance between satellite intersection and vessel
            X["CHORD_DISTANCE"] = chord_distance(
                great_circle_distance(X[self.latitude_name].values,
                                      X[self.longitude_name].values,
                                      X.SAT_LAT.values,
                                      X.SAT_LON.values))

            # Compute distance from satellite to vessels
            distances_to_current_sat = distance_sat_vessel(X.SAT_ALT.values,
                                                           X.CHORD_DISTANCE.values,
                                                           angle_sat_c(X.CHORD_DISTANCE.values))
            if first_satellite is True:
                X[self.column_name] = distances_to_current_sat
                first_satellite = False
            else:
                # Find closest satellite
                X[self.column_name] = np.minimum(X[self.column_name].values, distances_to_current_sat)

        # Drop temporary columns
        X.drop(["SAT_LON", "SAT_LAT", "SAT_ALT", "CHORD_DISTANCE"], axis=1, inplace=True)
        # Compress the new columns
        X[self.column_name] = (X[self.column_name] * 1000).astype(np.int32)
        return X


class AddLocalMessageIndex(BaseAugmenter):
    """
    Compute the number of messages sent by the same vessel before and after the current message.

    Each message will contain then an entry HISTORIC_SIZE telling how many message were sent BEFORE this one,
    and HISTORIC_SIZE_REVERSE how many were sent AFTER this one.

    The count are respectively stored in HISTORIC_SIZE and HISTORIC_SIZE_REVERSE
    """

    # Compute the historic size
    # - group data by MMSI -> then use range list of MMSI
    # -> returns: array([list([0,...,n_mmi0]), list([0,...,n_mmi1]), ..])
    # df = pd.DataFrame(data={"MMSI": [10, 10, 100, 100, 1000, 1000], "A": [11, 11, 111, 111, 1111, 1111],
    #                        "B": [12, 12, 122, 122, 1112, 11112]})
    # historic_size = df.groupby("MMSI")["MMSI"].apply(lambda df: df.reset_index().index.to_list()).values
    # np.concatenate(historic_size)
    # Out[72]: array([0, 1, 0, 1, 0, 1])
    def __init__(self,
                 mmsi_name: str = ColumnName.MMSI.lower(),
                 index_name: str = ColumnName.HISTORIC_SIZE,
                 reverse_index_name: str = ColumnName.HISTORIC_SIZE_REVERSE):
        self.mmsi_name = mmsi_name
        self.index_name = index_name
        self.reverse_index_name = reverse_index_name

    def transform(self, df):
        df = super().transform(df)

        historic_size = df.groupby(self.mmsi_name)[self.mmsi_name]. \
            apply(lambda df: np.array(df.reset_index().index.tolist())). \
            values
        df[self.index_name] = np.concatenate(historic_size)
        # Compute the reverse historic size
        df[self.reverse_index_name] = np.concatenate(
            (np.expand_dims(np.array(list(map(len, historic_size))), axis=0) - historic_size).
            reshape(historic_size.shape))
        df[self.index_name] = df[self.index_name].astype('int32')
        df[self.reverse_index_name] = df[self.reverse_index_name].astype('int32')
        return df


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

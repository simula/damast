# code=utf-8
"""
Module which collects transformers that add / augment the existing data
"""
import re
from pathlib import Path
from typing import Dict, List, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from pyorbital.orbital import Orbital
from sklearn.neighbors import BallTree
from sklearn.preprocessing import Binarizer

from damast.data_handling.transformers.base import BaseTransformer
from damast.domains.maritime.ais.navigational_status import (
    AISNavigationalStatus
)
from damast.domains.maritime.ais.vessel_types import Unspecified, VesselType
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
    "AddFishingVesselType",
    "AddLocalMessageIndex",
    "AddMissingAISStatus",
    "AddVesselType",
    "BaseAugmenter",
    "InvertedBinariser"
]


class BaseAugmenter(BaseTransformer):
    pass


class AddVesselType(BaseAugmenter):
    """
    Add a column `vessel_type` to the `pandas.DataFrame`.

    :param vessel_type_data: Path to map between MMSI identifier and vessel_type (.csv) file.
        Could also be a `pandas.DataFrame`.
    :param mmsi_name: Name of MMSI column in input data
    :param vessel_type_name: Name of column with vessel types
    """
    mmsi_name: str = None
    vessel_type_name: str = None
    vessel_types: pd.DataFrame = None

    def __init__(self,
                 vessel_type_data: Union[str, Path, pd.DataFrame],
                 mmsi_name: str = ColumnName.MMSI,
                 vessel_type_name: str = ColumnName.VESSEL_TYPE):
        self.mmsi_name = mmsi_name
        self.vessel_type_name: str = vessel_type_name

        # Load vessel type map
        if type(vessel_type_data) is not pd.DataFrame:
            vessel_type_data = AddVesselType.load_data(filename=vessel_type_data)

        # Check that the columns exist in input data
        for col_name in [mmsi_name, vessel_type_name]:
            if col_name not in vessel_type_data.columns:
                raise KeyError(f"Missing column: '{col_name}' in vessel type information: '{vessel_type_data.head()}'"
                               " - available are {','.join(vessel_type_data.columns)}")

        self.vessel_types = vessel_type_data

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path]) -> pd.DataFrame:
        """
        Load the vessel type map (MMSI->vessel_type).

        :param filename: The input file (or path)
        :return: A `pandas.DataFrame` where each row has a column MMSI and vessel_type
        """
        vessel_type_csv = Path(filename)
        if not vessel_type_csv.exists():
            raise RuntimeError(f"Vessel type information not accessible. File {vessel_type_csv} not found")
        vessel_types = pd.read_csv(vessel_type_csv, sep=";")
        return vessel_types

    def transform(self, df):
        """
        Add vessel type to the `pandas.DataFrame`.

        If the vessel type is not in look-up table,

        :param df: Input dataframe

        :returns: Dataframe with vessel-type added as a column.
        """
        df0 = super().transform(df)
        # Merge the existing dataset and the known labels (per MMSI)
        df = pd.merge(df0, self.vessel_types,
                      on=self.mmsi_name,
                      how="left")

        known_vessel_types = VesselType.get_types_as_str()
        df[self.vessel_type_name].replace(Unspecified.typename(), np.nan, inplace=True)
        df[self.vessel_type_name] = pd.Categorical(df[self.vessel_type_name], categories=known_vessel_types)
        df[self.vessel_type_name] = df[self.vessel_type_name].cat.codes
        df.reset_index(drop=True, inplace=True)
        df[self.vessel_type_name].fillna(-1, inplace=True)
        known_vessel_types = VesselType.get_types_as_str()

        # Set unspecified to nan, to later set all nan to -1
        # X[self.vessel_type_name].replace(self.vessel_unspecified, np.nan, inplace=True)

        # X[self.vessel_type_name] = pd.Categorical(X[self.vessel_type_name])
        # vessel_type_categories = dict(enumerate(df[col.VESSEL_TYPE].cat.categories))
        # _log.info(f"Collected vessel types: {json.dumps(vessel_type_categories, indent=4)}")
        # predefined_vessel_types = VesselType.get_types_as_str()
        # for x in df[col.VESSEL_TYPE].cat.categories:
        #    if x not in predefined_vessel_types:
        #        raise KeyError(f"Encountered a new vessel type: {x} - pls update vessel_type.py")

        # df[col.VESSEL_TYPE] = df[col.VESSEL_TYPE].cat.codes
        # _log.info(f"Category value count: {df[col.VESSEL_TYPE].value_counts()}")
        # df.reset_index(drop=True, inplace=True)

        # Set undefined values to -1 (as expected in later data processing)
        # df[self.vessel_type_name].fillna(-1, inplace=True)
        return df


class AddFishingVesselType(BaseAugmenter):
    """Add a column `fishing_type` to the `pandas.DataFrame`.

    Stores data from the Global Fishing Watch in transformer, making it possible to look up this for new entries.

    :param vessel_type_data: The MMSI->vessel_type map, either as Dataframe or as path to a file
    :param mmsi_in_name: Name of column with MMSI in `vessel_type_data`.
    :param mmsi_out_name: Name of column with MMSI Dataframe that will be augmented.
    :param column_name: Name of new column in augmented Dataframe.
    :param gfw_vessel_type_name: Name of `vessel_type` column in Global Fishing Watch data.

    .. todo:
        This should be just a merge into vessel_type - where fishing_type is detailed by
        the information from global fishing watch.
    """
    mmsi_name: str = None
    column_name: str = None

    #: Global Fishing Watch global vessel type name
    # Fishing type according to http://globalfishingwatch.org
    gfw_vessel_type_name: str = None
    vessel_types: pd.DataFrame = None

    def __init__(self,
                 vessel_type_data: Union[str, Path, pd.DataFrame],
                 mmsi_in_name: str = ColumnName.MMSI.lower(),
                 mmsi_out_name: str = ColumnName.MMSI,
                 column_name: str = ColumnName.FISHING_TYPE,
                 gfw_vessel_type_name: str = ColumnName.VESSEL_TYPE_GFW):
        self.mmsi_in_name = mmsi_in_name
        self.mmsi_out_name = mmsi_out_name
        self.column_name: str = column_name

        if type(vessel_type_data) is not pd.DataFrame:
            vessel_type_data = AddFishingVesselType.load_data(filename=vessel_type_data)

        for col_name in [mmsi_in_name, gfw_vessel_type_name]:
            if col_name not in vessel_type_data.columns:
                raise KeyError(
                    f"Missing column: '{col_name}' in fishing vessel type information: '{vessel_type_data.head()}'"
                    " - available are {','.join(vessel_type_data.columns)}")
        vessel_type_data.rename(columns={mmsi_in_name: mmsi_out_name,
                                         gfw_vessel_type_name: column_name}, inplace=True)
        self.vessel_types = vessel_type_data

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path]) -> pd.DataFrame:
        """
        Load the vessel type data into a `pandas.DataFrame` with (MMSI, VesselType) as columns.

        :param filename: Path to file
        :return: The `pandas.DataFrame`
        """
        vessel_type_csv = Path(filename)
        if not vessel_type_csv.exists():
            raise RuntimeError(f"Fishing vessel type information not accessible. File {vessel_type_csv} not found")
        vessel_types = pd.read_csv(vessel_type_csv)
        return vessel_types

    def transform(self, df):
        df = super().transform(df)

        # Merge fishing type information into the existing dataset
        df = pd.merge(left=df,
                      right=self.vessel_types[[self.mmsi_out_name, self.column_name]],
                      on=self.mmsi_out_name,
                      how="left")
        # Set 'unspecified' to nan, to later set all nan to -1
        df[self.column_name].replace(Unspecified.typename(), np.nan, inplace=True)

        known_vessel_types = VesselType.get_types_as_str()
        df[self.column_name].replace(Unspecified.typename(), np.nan, inplace=True)
        df[self.column_name] = pd.Categorical(df[self.column_name], categories=known_vessel_types)
        df[self.column_name] = df[self.column_name].cat.codes
        df.reset_index(drop=True, inplace=True)

        # Set undefined values to -1 (as expected in later data processing)
        df[self.column_name].fillna(-1, inplace=True)
        return df


class AddDistanceClosestAnchorage(BaseAugmenter):
    """
    Compute the distance to the closest anchorage.

    Using `DASK <https://www.dask.org/>`_ (`dd`)
    The dataset is split in 32 partitions, one for each core
    Then each core compute for each message the distance with all
    the anchorage present in the `anchorage.csv` file using the great circle distance.
    """

    # Read anchorages file
    def __init__(self,
                 anchorages_data: Union[str, Path, pd.DataFrame],
                 latitude_name: str = ColumnName.LATITUDE,
                 longitude_name: str = ColumnName.LONGITUDE,
                 anchorage_latitude_name: str = "latitude",
                 anchorage_longitude_name: str = "longitude",
                 column_name: str = ColumnName.DISTANCE_CLOSEST_ANCHORAGE,
                 ):

        if type(anchorages_data) is pd.DataFrame:
            self.anchorages = anchorages_data
        else:
            self.anchorages = AddDistanceClosestAnchorage.load_data(filename=anchorages_data,
                                                                    latitude_name=anchorage_latitude_name,
                                                                    longitude_name=anchorage_longitude_name)

        self.latitude_name = latitude_name
        self.longitude_name = longitude_name
        self.anchorage_latitude_name = anchorage_latitude_name
        self.anchorage_longitude_name = anchorage_longitude_name
        self.column_name = column_name

    @classmethod
    def load_data(cls,
                  filename: Union[str, Path],
                  latitude_name: str,
                  longitude_name: str) -> pd.DataFrame:
        """
        Load the vessel type data.

        :param filename: Filename for that anchorages data csv file
        :param latitude_name:  name of the latitude column
        :param longitude_name: name of the longitude column
        :return: the loaded data as DataFrame containing (only) the latitude and longitude column (if there are more)
        """
        anchorages_csv = Path(filename)
        if not anchorages_csv.exists():
            raise RuntimeError(f"Anchorages data not accessible. File {anchorages_csv} not found")
        anchorages = pd.read_csv(filepath_or_buffer=filename,
                                 usecols=[latitude_name, longitude_name])
        return anchorages

    def transform(self, df):
        df = super().transform(df)

        # Ensure that radians can be used for LAT/LON
        for column in [self.anchorage_latitude_name, self.anchorage_longitude_name]:
            self.anchorages[f'{column}_in_rad'] = self.anchorages[column].map(np.deg2rad)

        # Adding temporary columns
        lat_lon_rad_columns = []
        for data_column, anchorage_column in [(self.latitude_name, self.anchorage_latitude_name),
                                              (self.longitude_name, self.anchorage_longitude_name)]:
            col_in_rad = f'{anchorage_column}_in_rad'
            lat_lon_rad_columns.append(col_in_rad)
            df[col_in_rad] = df[data_column].map(np.deg2rad)

        # Provide a dask progress bar
        pbar = ProgressBar()
        # Performs global registration, see
        # https://www.coiled.io/blog/how-to-check-the-progress-of-dask-computations
        pbar.register()

        def compute_distance(x):
            """
            Compute the Haversine distance between the closest anchorage and a set of points
            """
            # NOTE: We have to define the ball-tree on each process to gain any speedup
            # We also need to copy the input to avoid tkinter issues
            tree = BallTree(self.anchorages[lat_lon_rad_columns].copy(), metric='haversine')
            output = tree.query(np.array(x[lat_lon_rad_columns]), k=1, return_distance=True)
            return output[0].reshape(-1)

        dask_dataframe = dd.from_pandas(df[lat_lon_rad_columns], chunksize=1000000)
        dist_output = dask_dataframe.map_partitions(lambda df_part:
                                                    compute_distance(df_part))
        z = dist_output.compute()
        df[self.column_name] = z * EARTH_RADIUS * 1000
        df[self.column_name] = df[self.column_name].astype(np.int32)
        # Drop the temporary columns
        df.drop(columns=lat_lon_rad_columns, inplace=True)

        return df


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
        :param satellite_tle_filename: The two-line element set file, for the satellite positions, see
            also https://en.wikipedia.org/wiki/Two-line_element_set
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
        Combine two existing labels to create a new one .
        >>> AddCombinedLabel(column_names=[col.FISHING_TYPE, col.STATUS],
        >>>                  column_permitted_values={col.FISHING_TYPE: {...},col.STATUS: { ... }) },
        >>>                  combination_name="combination")

        :param df:
        :return:
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


class AddMissingAISStatus(BaseAugmenter):
    """
    Fill missing AIS Status with the latest known previous status, otherwise "undefined".
    """
    column_name: str = None
    mmsi_name: str = None
    window_size: int = None

    def __init__(self,
                 column_name: str = ColumnName.STATUS,
                 mmsi_name: str = ColumnName.MMSI,
                 window_size: int = 50):
        self.column_name = column_name
        self.window_size = window_size
        self.mmsi_name = mmsi_name

    def transform(self, df):
        """
        Fill the missing value using ffill.

        AIS comes with a navigational status defined in the range of 0 to 15, where 15 means
        'undefined'.

        ffill means 'forward fill' and will propagate last valid observation
        """
        # Replace undefined status by NaN
        df[self.column_name] = df[self.column_name].replace([-99, AISNavigationalStatus.Undefined], np.NaN)

        # Fill NaN using ffill (forward fill) to propagate last valid observation
        # for each vessel (therefore groupby MMSI)
        df[self.column_name] = df.groupby(self.mmsi_name)[self.column_name].fillna(method="ffill")

        # Fill remaining NaN by {AISNavigationalStatus.Undefined}, i.e. (AIS Navigational) Status 'undefined'")
        df[self.column_name] = df[self.column_name].fillna(AISNavigationalStatus.Undefined)

        # # Remove non continuous status
        # df.loc[df[self.column_name] != df[self.column_name].shift(
        #     self.window_size), self.column_name] = AISNavigationalStatus.Undefined

        # # Set reserved and regional used flags to - 1
        # log(f"[DATASET_CREATION] Replace regional use and future flags: (AIS Navigational) Status >"
        #     {AISNavigationalStatus.UnderWaySailing}(under way sailing) by {val.UNDEFINED}")
        # df[self.column_name] = df[self.column_name].mask(df[self.column_name] > AISNavigationalStatus.UnderWaySailing,
        #                                                  FieldValue.UNDEFINED)
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

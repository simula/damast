"""
Data Processing Module

Contains AIS-specific pipelines for filtering and augmenting data
"""


from pathlib import Path
from typing import Any, Dict, List, Union

from damast.core import DataProcessingPipeline
from damast.core.types import XDataFrame
from damast.data_handling.exploration import plot_lat_lon
from damast.data_handling.transformers import (
    AddTimestamp,
    AddUndefinedValue,
    ChangeTypeColumn,
    DropMissingOrNan,
    FilterWithin,
    RemoveValueRows,
    )
from damast.data_handling.transformers.augmenters import AddLocalIndex
from damast.data_handling.transformers.visualisers import PlotLatLon
from damast.domains.maritime.ais import vessel_types
from damast.domains.maritime.transformers import AddVesselType, ComputeClosestAnchorage

from .data_specification import ColumnName

__all__ = ["CleanseAndSanitise", "DataProcessing"]


ParamsType = Dict[str, Any]


def get_outputs_dir(workdir: Union[str, Path]) -> Path:
    return Path(workdir) / "processed_data"


def get_plots_dir(workdir: Union[str, Path]) -> Path:
    return Path(workdir) / "plots"


class CleanseAndSanitise(DataProcessingPipeline):
    """
    A :class:`damast.core.DataProcessingPipeline` for cleaning and sanitising AIS data.

    :param message_types: List of allowable message types (all other message types will be removed)
    :param columns_default_values: Dictionary mapping column names to default values. All columns
        in this dict will have all empty rows replaced with this value
    :param columns_compress_type: Dictionary mapping column names to compressed data-types. All columns
        in this dict will have a new column with the compressed data
    :param workdir: Base directory towards which transformer output which be relative
    :param name: Name of pipeline.

    The expected input columns are:

    * :attr:`damast.domains.maritime.data_specification.ColumnName.MESSAGE_TYPE`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.DATE_TIME_UTC`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.SPEED_OVER_GROUND`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.COURSE_OVER_GROUND`
    * Any column in ``columns_default_values`` and in ``columns_compress_types``.

    The pipeline performs the following actions

    1. Keep only data from satellite, i.e. :code:`source != 'g'`
    2. Remove row which do not have a timestamp
    3. Keep only message-types in input ``message_types``
    4. Create a :code:`"timestamp"` field from date time UTC
    5. Replace NaN/Na with default values as defined in  :code:`"columns_default_values"`
    6. Create new columns with new types using :code:`columns_compress_types`.
       The new column is name from the original column name with "_newtype" as a suffix.

    """

    def __init__(self, message_types: List[int],
                 columns_default_values: Dict[str, Any],
                 columns_compress_types: Dict[str, str],
                 workdir: Union[str, Path],
                 name: str = "Cleanse and sanitise data",
                 name_mappings: Dict[str, str] = {}):
        super().__init__(name=name,
                         base_dir=workdir,
                         name_mappings=name_mappings)

        self.add("Remove rows with ground as source", RemoveValueRows("g"),
                 name_mappings={"x": ColumnName.SOURCE})
        self.add("Remove rows with null dates", DropMissingOrNan(),
                 name_mappings={"x": ColumnName.DATE_TIME_UTC})
        self.add("Filter rows within message types", FilterWithin(message_types),
                 name_mappings={"x": ColumnName.MESSAGE_TYPE})
        self.add("Add Timestamp column to each row", AddTimestamp(),
                 name_mappings={"from": ColumnName.DATE_TIME_UTC, "to": ColumnName.TIMESTAMP})

        for column in columns_default_values:
            default_value = columns_default_values[column]
            self.add(f"Set default values to {column}", AddUndefinedValue(default_value),
                     name_mappings={"x": column})

        for column in columns_compress_types:
            new_type = columns_compress_types[column]
            self.add(f"Compress column {column}", ChangeTypeColumn(new_type),
                     name_mappings={"x": column, "y": f"{column}_{new_type}"})


class DataProcessing(DataProcessingPipeline):
    """
    Create an AIS-specific pipeline to an annotated dataframe.

    :param vessel_type_hdf5: Path to HDF5 file containing vessel types
    :param fishing_vessel_type_hdf5: Path to HDF5 file containing fishing vessel types
    :param anchorages_hdf5: Path to HDF5 file containing anchorages' coordinates
    :param workdir: Base directory towards which transformer output which be relative
    :param name: Name of pipeline.

    .. note::
        Initializing this class plots the input of ``anchorages_hdf5`` which should probably
        be moved elsewhere

    The pipeline requires the following input columns:

    * :attr:`damast.domains.maritime.data_specification.ColumnName.MMSI`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.LATITUDE`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.LONGITUDE`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.MMSI`
    * :attr:`damast.domains.maritime.data_specification.ColumnName.TIMESTAMP`

    The pipeline does the following actions:

    * Plots longitude and latitude of all data
    * Adds vessel-type information from ``vessel_type_hdf5`` to column
      :attr:`damast.domains.maritime.data_specification.ColumnName.VESSEL_TYPE`.
    * Replace missing vessel-types with integer representation of
      :class:`damast.domains.maritime.ais.vessel_types.Unspecified`
    * Add global fishing-watch vessel type from ``fishing_vessel_type_hdf5`` to column
      :attr:`damast.domains.maritime.data_specification.ColumnName.FISHING_TYPE`
    * Replace missing values in gfw vessel type with `-1`
    * Compute closest distance to anchorages from ``anchorages_hdf5`` and att to column
      :attr:`damast.domains.maritime.data_specification.ColumnName.DISTANCE_CLOSEST_ANCHORAGE`
    * Group messages by :attr:`damast.domains.maritime.data_specification.ColumnName.MMSI` and
      local group index based on :attr:`damast.domains.maritime.data_specification.ColumnName.TIMESTAMP`.
      Column name for group index and reverse index is
      :attr:`damast.domains.maritime.data_specification.ColumnName.HISTORIC_SIZE` and
      :attr:`damast.domains.maritime.data_specification.ColumnName.HISTORIC_SIZE_REVERSE`, respectively.

    """

    def __init__(self, workdir: Union[str, Path],
                 vessel_type_hdf5: Union[str, Path],
                 fishing_vessel_type_hdf5: Union[str, Path],
                 anchorages_hdf5: Union[str, Path],
                 name: str = "AIS-processor",
                 name_mappings: Dict[str, str] = {}):

        super().__init__(name=name,
                         base_dir=workdir,
                         name_mappings=name_mappings)

        plots_dir = get_plots_dir(workdir=workdir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Output an overview over the existing anchorages
        anchorages_data = XDataFrame.open(anchorages_hdf5)

        plot_lat_lon(df=anchorages_data,
                     latitude_name="latitude",
                     longitude_name="longitude",
                     output_dir=plots_dir,
                     filename_prefix="anchorages-lat-lon")

        # Temporary converters to csv to be compatible
        if isinstance(vessel_type_hdf5, Path):
            vessel_type_csv = XDataFrame.open(vessel_type_hdf5)
        else:
            vessel_type_csv = vessel_type_hdf5

        if isinstance(anchorages_hdf5, Path):
            anchorages_csv = XDataFrame.open(anchorages_hdf5)
        else:
            anchorages_csv = anchorages_hdf5
        if isinstance(fishing_vessel_type_hdf5, Path):
            fishing_vessel_type_csv = XDataFrame.open(fishing_vessel_type_hdf5)
        else:
            fishing_vessel_type_csv = fishing_vessel_type_hdf5

        # FIXME: Name mapping does not apply here
        # self.add("plot_input-lat_lon",
        #         PlotLatLon(output_dir=plots_dir,
        #                    filename_prefix="lat-lon-input"),
        #         name_mappings={"LAT": ColumnName.LATITUDE,
        #                        "LON": ColumnName.LONGITUDE})

        self.add("augment_vessel_type",
                 AddVesselType(dataset=vessel_type_csv,
                               right_on=ColumnName.MMSI,
                               dataset_col=ColumnName.VESSEL_TYPE),
                 name_mappings={"x": ColumnName.MMSI,
                                "out": ColumnName.VESSEL_TYPE})
        self.add("Replace missing", AddUndefinedValue(vessel_types.Unspecified.to_id()),
                 name_mappings={"x": ColumnName.VESSEL_TYPE})

        self.add("augment_fishing_vessel_type",
                 AddVesselType(dataset=fishing_vessel_type_csv,
                               right_on=ColumnName.MMSI.lower(),
                               dataset_col=ColumnName.VESSEL_TYPE_GFW),
                 name_mappings={"x": ColumnName.MMSI,
                                "out": ColumnName.FISHING_TYPE})
        self.add("Replace missing", AddUndefinedValue(-1),
                 name_mappings={"x": ColumnName.FISHING_TYPE})
        self.add("augment_distance_to_closest_anchorage",
                 ComputeClosestAnchorage(dataset=anchorages_csv,
                                         columns=["latitude", "longitude"]),
                 name_mappings={"x": ColumnName.LATITUDE,
                                "y": ColumnName.LONGITUDE,
                                "distance": ColumnName.DISTANCE_CLOSEST_ANCHORAGE})
        self.add("Compute local message index",  AddLocalIndex(),
                 name_mappings={"group": ColumnName.MMSI,
                                "sort": ColumnName.TIMESTAMP,
                                "local_index": ColumnName.HISTORIC_SIZE,
                                "reverse_{{local_index}}": ColumnName.HISTORIC_SIZE_REVERSE})

"""
Data Processing Module

Contains modules for adding features and filtering data
"""
# -----------------------------------------------------------
# (C) 2020 Pierre Bernabe, Oslo, Norway
# email pierbernabe@simula.no
# -----------------------------------------------------------

import datetime

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import vaex

from damast.data_handling.exploration import plot_lat_lon
from damast.data_handling.transformers.augmenters import (
    AddLocalMessageIndex,
)
from damast.domains.maritime.ais import vessel_types
from damast.domains.maritime.transformers import ComputeClosestAnchorage
from damast.data_handling.transformers import (
    AddUndefinedValue, 
    RemoveValueRows, 
    FilterWithin, 
    DropMissing, 
    AddTimestamp,
    MultiplyValue,
    ChangeTypeColumn
)
from damast.domains.maritime.transformers import AddVesselType

from damast.data_handling.transformers.visualisers import PlotLatLon
from damast.core import DataProcessingPipeline, AnnotatedDataFrame
from damast.domains.maritime.data_specification import ColumnName

__all__ = ["cleanse_and_sanitise", "process_data"]

_log = getLogger("damast")

ParamsType = Dict[str, Any]


def get_outputs_dir(workdir: Union[str, Path]) -> Path:
    return workdir / f"processed_data"


def get_plots_dir(workdir: Union[str, Path]) -> Path:
    return workdir / f"plots"


def cleanse_and_sanitise(df: AnnotatedDataFrame,
                         useless_columns: list,
                         message_type_position: list,
                         columns_default_values: dict,
                         columns_compress_types: dict,
                         workdir: Union[str, Path]) -> AnnotatedDataFrame:
    """
    Cleanse and sanitise the dataframe by adjusting the type and performing a cleanup.

    1. Keep only data from satellite, i.e. :code:`source != 'g'`
    2. Remove row which do not have a timestamp
    3. Keep only Message Position Messages e.g. Message type = 2
    4. Create a :code:`"timestamp"` field from date time UTC
    5. Multiply SOG and COG by 10 to eliminate need for comma
    6. Replace NaN/Na with default values as defined in  :code:`"columns_default_values"` 
    7. Create new columns with new types using :code:`columns_compress_types`.
       The new column is name from the original column name with "_newtype" as a suffix.

    :param df: The input annotated dataframe
    :param useless_columns: The list of column names to remove from the original annotated dataframe 
    :param message_type_position: index (as a list) where the message type position can be found
    :param columns_constraints: The list of column names and associated constraints
    :returns: The cleaned and santised data
    """
    
    pipeline = DataProcessingPipeline("Cleanse and sanitise data", workdir)
 
    pipeline.add("Remove rows with ground as source", RemoveValueRows("g"),
                 name_mappings={"x": ColumnName.MESSAGE_TYPE})
    pipeline.add("Remove rows with null dates", DropMissing(),
                 name_mappings={"x": ColumnName.DATE_TIME_UTC})
    pipeline.add("Filter rows within message types", FilterWithin(message_type_position),
                 name_mappings={"x": ColumnName.MESSAGE_TYPE})
    pipeline.add("Add Timestamp column to each row", AddTimestamp(),
                 name_mappings={"from": ColumnName.DATE_TIME_UTC, "to": ColumnName.TIMESTAMP})

    pipeline.add("Multiply col with a float value", MultiplyValue(10.),
                 name_mappings={"x": ColumnName.SPEED_OVER_GROUND})
    
    pipeline.add("Multiply col with a float value", MultiplyValue(10.),
                 name_mappings={"x": ColumnName.COURSE_OVER_GROUND})

    for column in columns_default_values:
        default_value = columns_default_values[column]
        pipeline.add("Set default values", AddUndefinedValue(default_value),
                 name_mappings={"x": column})

    for column in columns_compress_types:
        new_type = columns_compress_types[column]
        pipeline.add("Set default values", ChangeTypeColumn(new_type),
                 name_mappings={"x": column, "y": f"{column}_{new_type}"})

    df = pipeline.transform(df)

    return df


def process_data(df: AnnotatedDataFrame,
                 workdir: Union[str, Path],
                 vessel_type_hdf5: Union[str, Path],
                 fishing_vessel_type_hdf5: Union[str, Path],
                 anchorages_hdf5: Union[str, Path]) -> AnnotatedDataFrame:
    """
    Applies a specific AIS pipeline to an annotated dataframe.
    Saves the data to an `h5` and `yaml` file.

    :param df: The dataframe
    :param workdir: Path to directory of execution
    :param vessel_type_hdf5: Path to HDF5 file containing vessel types
    :param fishing_vessel_type_hdf5: Path to HDF5 file containing fishing vessel types
    :param anchorages_hdf5: Path to HDF5 file containing anchorages' coordinates
    :returns:  The processed dataframe, the file containing the pipeline state and HDF5 file contianing the saved dataframe.
    """
    plots_dir = get_plots_dir(workdir=workdir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Output an overview over the existing anchorages
    anchorages_data = vaex.open(anchorages_hdf5)
    plot_lat_lon(df=anchorages_data,
                 latitude_name="latitude",
                 longitude_name="longitude",
                 output_dir=plots_dir,
                 filename_prefix="anchorages-lat-lon")

    # Temporary converters to csv to be compatible
    if isinstance(vessel_type_hdf5, Path):
        vessel_type_csv = vaex.open(vessel_type_hdf5)
    else:
        vessel_type_csv = vessel_type_hdf5

    if isinstance(anchorages_hdf5, Path):
        anchorages_csv = vaex.open(anchorages_hdf5)
    else:
        anchorages_csv = anchorages_hdf5
    if isinstance(fishing_vessel_type_hdf5, Path):
        fishing_vessel_type_csv = vaex.open(fishing_vessel_type_hdf5)
    else:
        fishing_vessel_type_csv = fishing_vessel_type_hdf5

    pipeline = DataProcessingPipeline("Compute message index", workdir)
    pipeline.add("plot_input-lat_lon",
                 PlotLatLon(output_dir=plots_dir,
                            filename_prefix="lat-lon-input"),
                 name_mappings={"LAT": ColumnName.LATITUDE,
                                "LON": ColumnName.LONGITUDE})

    pipeline.add("augment_vessel_type",
                 AddVesselType(dataset=vessel_type_csv,
                               right_on=ColumnName.MMSI,
                               dataset_col=ColumnName.VESSEL_TYPE),
                 name_mappings={"x": ColumnName.MMSI,
                                "out": ColumnName.VESSEL_TYPE})
    pipeline.add("Replace missing", AddUndefinedValue(vessel_types.Unspecified.to_id()),
                 name_mappings={"x": ColumnName.VESSEL_TYPE})

    pipeline.add("augment_fishing_vessel_type",
                 AddVesselType(dataset=fishing_vessel_type_csv,
                               right_on=ColumnName.MMSI.lower(),
                               dataset_col=ColumnName.VESSEL_TYPE_GFW),
                 name_mappings={"x": ColumnName.MMSI,
                                "out": ColumnName.FISHING_TYPE})
    pipeline.add("Replace missing", AddUndefinedValue(-1),
                 name_mappings={"x": ColumnName.FISHING_TYPE})
    pipeline.add("augment_distance_to_closest_anchorage",
                 ComputeClosestAnchorage(dataset=anchorages_csv,
                                         columns=["latitude", "longitude"]),
                 name_mappings={"x": ColumnName.LATITUDE,
                                "y": ColumnName.LONGITUDE,
                                "distance": ColumnName.DISTANCE_CLOSEST_ANCHORAGE})
    pipeline.add("Compute local message index",  AddLocalMessageIndex(),
                 name_mappings={"group": ColumnName.MMSI,
                                "sort": ColumnName.TIMESTAMP,
                                "msg_index": ColumnName.HISTORIC_SIZE,
                                "reverse_{{msg_index}}": ColumnName.HISTORIC_SIZE_REVERSE})

    df = pipeline.transform(df)

    # Get the setup for storing the processed data
    output_dir = get_outputs_dir(workdir=workdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.state_write(output_dir / "process_data_pipeline.yaml")
    df.save(filename=output_dir / "process_data.h5")

    return df

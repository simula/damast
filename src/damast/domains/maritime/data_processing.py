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
from typing import Any, Dict

import vaex

from damast.data_handling.exploration import plot_lat_lon
from damast.data_handling.transformers.augmenters import (
    AddLocalMessageIndex,
)
from damast.domains.maritime.ais import vessel_types
from damast.domains.maritime.transformers import ComputeClosestAnchorage
from damast.data_handling.transformers import AddUndefinedValue
from damast.domains.maritime.transformers import AddVesselType

from damast.data_handling.transformers.visualisers import PlotLatLon, PlotHistograms
from damast.core import DataProcessingPipeline, AnnotatedDataFrame
from damast.domains.maritime.data_specification import ColumnName

__all__ = ["cleanse_and_sanitise", "process_data"]

_log = getLogger("damast")

ParamsType = Dict[str, Any]


def get_processed_data_spec(params: ParamsType) -> Dict[str, Any]:
    processed_data_spec = params["outputs"]["processed"]
    return processed_data_spec


def get_outputs_dir(params: ParamsType) -> Path:
    workdir = Path(params["workdir"])
    processed_data_spec = get_processed_data_spec(params=params)
    return workdir / processed_data_spec["dir"]


def get_plots_dir(params: ParamsType) -> Path:
    outputs_dir = get_outputs_dir(params)
    plots_dir = outputs_dir / f"{int(params['month']):02}"
    return plots_dir


def cleanse_and_sanitise(params: ParamsType, df: vaex.DataFrame) -> vaex.DataFrame:
    """
    Cleanse and sanitise the dataframe by adjusting the type and performing a cleanup.

    1. Keep only data from satellite, i.e. :code:`source != 'g'`
    2. Remove row which do not have a timestamp
    3. Keep only Message Position Messages

        .. todo::

            are there not distress signals part of the other messages?)

    4. Create a :code:`"timestamp"` field
    5. Remove 'useless_columns' as defined in params
    6. Drop :code:`["columns"]["unused"]` as defined in params

        .. todo::

            What is the difference to :code:`useless_columns`
    7. Multiply SOG and COG by 10 to eliminate need for comma

        .. todo::

            Why do we need to multiply by 10?
    8. Replace NaN/Na with default values as defined in
       :code:`["columns"]["constraints"][<col_name>]["default"]` in params
    9. Set types via :code:`["columns"]["constraints"][<col_name>]["type"]` in params


    .. todo::

        * This should really be a Pipeline

    .. note::

        Very little computation is actually done inside this module, as we are using :py:mod:`vaex` for
        lazy evaluation.

    .. todo::

        * Consider other message types
        * Mark rows which contain invalid (out-of-spec data)

    :param params: The parameters used for cleaning the data, giving parameter types and ranges
    :param df: The input data
    :returns: The cleaned and santised data
    """

    # All logging headers
    header = "[COMPRESSION][{action}]"
    drop_hdr = header.format(action='DROP')
    add_hdr = header.format(action="ADD")
    fill_hdr = header.format(action="FILLNA")
    type_hdr = header.format(action="TYPE")

    # 1. Keep only satellite data
    df_1 = df[df.source == "g"]

    _log.info(f"{drop_hdr} The files contain {df.shape[0]} messages in total."
              f"Only {df_1.shape[0]} come from satellite(s) and are kept")

    # 2. Remove line where date is null
    df_2 = df_1.dropmissing(column_names=[ColumnName.DATE_TIME_UTC])
    _log.info(f"{drop_hdr} {df_1.shape[0] - df_2.shape[0]} lines have been dropped."
              " They do not have a time stamp")

    # 3. Keep only the position message
    df_3 = df_2[df_2.MessageType.isin(params["MessageTypePosition"])]
    _log.info(f"{drop_hdr} {df_2.shape[0] - df_3.shape[0]} lines have been dropped."
              " They are not of type position report")

    # 4. Add "timestamp" field
    def convert_to_datetime(date_string):
        return int(datetime.datetime.strptime(
            date_string, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S"))

    df_3[ColumnName.TIMESTAMP] = df_3[ColumnName.DATE_TIME_UTC].apply(convert_to_datetime)
    _log.info(f"{add_hdr} Column timestamps added")
    df_3.drop(ColumnName.DATE_TIME_UTC, inplace=True)
    # 5. Drop useless columns
    useless_columns = params["columns"]["useless"]
    df_3.drop(useless_columns, inplace=True)
    _log.info(f"{drop_hdr} These columns have been removed: " + ', '.join(
        useless_columns))

    # 6. Drop unused columns
    # TODO: what is the difference to useless_columns (?)
    unused_columns = params["columns"]["unused"]
    df_3.drop(unused_columns, inplace=True)
    _log.info(f"{drop_hdr} These columns have been removed: " + ', '.join(unused_columns))

    # 7. multiply columns in order to remove comma
    # FIXME: Unclear why SOG and COG was multiplied by 10
    df_3["SOG"] *= 10
    df_3["COG"] *= 10

    # 8. Fill empty value for then reduce the size thanks to the type
    columns_constraints = params["columns"]["constraints"]
    for column in columns_constraints:
        constraints = columns_constraints[column]

        if column not in df_3.column_names:
            continue
        if df_3[column].countnan() > 0:
            if "default" in constraints.keys():
                default_value = constraints["default"]
                df_3[column].fillnan(default_value)
                df_3[column].fillna(default_value)

                _log.debug(f"{fill_hdr} The NaN values for {column} have been filled" +
                           f" with {default_value} based on settings in params.yaml")
            else:
                _log.debug(f"{fill_hdr} {column} has NaN values,"
                           f" but not default value is specified in params.yaml")
        else:
            _log.debug(f"{fill_hdr} No NaN values in {column} - nothing to do")

    # 9. Reduction maximum des types
    _log.info(f"{type_hdr} Updating columns type for: {columns_constraints.keys()}")

    column_types = {}
    for column in columns_constraints:
        constraints = columns_constraints[column]
        if column not in df_3.column_names:
            continue
        else:
            if "type" in constraints.keys():
                df_3[column] = df_3[column].astype(constraints["type"])
                column_types[column] = constraints["type"]
            else:
                raise KeyError(f"Missing type information in params.yaml for column '{column}'")

    _log.info(f"{type_hdr} The type of the columns have been modified"
              f" to {column_types} as defined in params.yaml")
    return df_3


def process_data(params: Dict[str, Any],
                 df: AnnotatedDataFrame,
                 workdir: Path):
    """
    Applies a specific AIS pipeline to an annotated dataframe.
    Saves the data to an `h5` and `yaml` file.

    .. todo::

        * Document what input params are actually needed

    :param params: Document what we need from these
    :param df: The dataframe
    :param workdir: Path to directory of execution
    """
    plots_dir = get_plots_dir(params=params)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Data Feature extraction and augmentation
    vessel_type_hdf5 = params["inputs"]["vessel_types"]
    fishing_vessel_type_hdf5 = params["inputs"]["fishing_vessel_types"]
    anchorages_hdf5 = params["inputs"]["anchorages"]

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
    pipeline.add("plot_input-histograms",
                 PlotHistograms(output_dir=plots_dir,
                                filename_prefix="histogram-input-data-"))

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
    output_dir = get_outputs_dir(params=params)
    pipeline.state_write(output_dir / "pipeline.yaml")
    df.save(filename=output_dir / f"{int(params['month']):02}.h5")

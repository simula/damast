"""
Data Processing Module


1. Read all files from `data/<month>/*.zip`.
2. Concatenates the files into a single dataframe.
3. Perform feature extraction

  1. Filter areas (`lon_min`/`lon_max`, `lat_min`/`lat_max`).
  2. Filter vessels by MMSI range.
  3. Add labels: `"vessel_type"` and `"fishing_type"` (`data/fishing-vessels-v2.csv`).
  4. Add distance to closest anchorage.
  5. Add distance to satellite.
  6. Add historic message size (whatever that means).

7. Write data to `data/processed/<month>.h5`.
"""
# -----------------------------------------------------------
# (C) 2020 Pierre Bernabe, Oslo, Norway
# email pierbernabe@simula.no
# -----------------------------------------------------------

import argparse
import datetime
import glob
import time
from logging import Formatter, StreamHandler, basicConfig, getLogger
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import vaex
import yaml

from damast.data_handling.exploration import plot_lat_lon
from damast.data_handling.transformers.augmenters import (
    AddLocalMessageIndex,
    JoinDataFrameByColumn
)
from damast.domains.maritime.ais import vessel_types
from damast.domains.maritime.transformers import ComputeClosestAnchorage
from damast.data_handling.transformers.filters import (
    AreaFilter,
    DuplicateNeighboursFilter,
    MMSIFilter
)
from damast.data_handling.transformers import AddUndefinedValue
from damast.domains.maritime.transformers import AddVesselType

from damast.data_handling.transformers.visualisers import PlotLatLon, PlotHistograms
from damast.core import DataProcessingPipeline, AnnotatedDataFrame
from damast.data_handling.transformers.sorters import GenericSorter
from damast.domains.maritime.data_specification import ColumnName

__all__ = ["cleanse_and_sanitise", "process_files", "process_data"]

_log = getLogger(__name__)
LOG_FORMAT = '%(asctime)-5s [%(filename)s:%(lineno)d] %(message)s'

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

    .. todo::
        * This should really be a Pipeline

    Cleanse and sanitise the dataframe by adjusting the type and performing a cleanup.

    :param params: The parameters used for cleaning the data, giving parameter types and ranges
    :param df: The input data
    :returns: The cleaned and santised data

    .. note::

        Very little computation is actually done inside this module, as we are using `vaex` for
        lazy evaluation.

    .. todo::
        * Consider other message types
        * Mark rows which contain invalid (out-of-spec data)

    1. Keep only data from satellite, i.e. `source != 'g'`
    2. Remove row which do not have a timestamp
    3. Keep only Message Position Messages (TODO: are there not distress signals part of the other messages?)
    4. Create a "timestamp" field
    5. Remove 'useless_columns' as defined in params
    6. Drop `["columns"]["unused"]` as defined in params (TODO: what is the difference to useless_columns)
    7. Multiply SOG and COG by 10 to eliminate need for comma (TODO: Why do we need to multiply by 10)
    8. Replace NaN/Na with default values as defined in `["columns"]["constraints"][<col_name>]["default"]` in params
    9. Set types via `["columns"]["constraints"][<col_name>]["type"]` in params

    :return: The cleaned dataset
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
    df_2 = df_1.dropmissing(column_names=["BaseDateTime"])
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

    df_3[ColumnName.TIMESTAMP] = df_3["BaseDateTime"].apply(convert_to_datetime)
    _log.info(f"{add_hdr} Column timestamps added")
    df_3.drop("BaseDateTime", inplace=True)
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


def process_files(params: Dict[str, Any]) -> None:
    """
    Process input data. TODO: ADD MORE INFO

    :param  params: Options for processing


    .. highlight:: python
    .. code-block:: python

        params = {"workdir": "path_to_directory_of_execution",
                  "inputs": {"data":{"dir": "path_to_input_data"}},
                  "month": "integer",
                  "columns":{"brut": ["list","of","columns","to","use]}}


    :raises KeyError: If any of the required keys are missing from the input parameters
    :raises RuntimeError: No data-files are found in input data directory.
    """
    # Get the list of the file to read
    # files: List[str] = glob.glob(f"../data/{params['month']}/*.zip")

    workdir: Path
    try:
        workdir = Path(params["workdir"])
    except KeyError:
        raise KeyError('Missing "workdir" in params')

    in_path = params["inputs"]["data"]["dir"]
    month = int(params["month"])
    data_in_path = workdir / in_path / f"{month:02}" / "*.zip"
    files: List[str] = glob.glob(f"{data_in_path}")

    if len(files) == 0:
        raise RuntimeError(f"Found no data files to process in {data_in_path}")

    # Compress and Concatenate all the files
    df = pd.DataFrame()
    columns = params["columns"]["brut"]
    for file in files:
        _log.info(f" File: {file}")
        df = df.append(pd.read_csv(file, names=columns, sep=";", header=0, low_memory=False))

    df = cleanse_and_sanitise(params=params,
                              df=df)

    process_data(params=params,
                 df=df,
                 workdir=workdir)


def process_data(params: Dict[str, Any],
                 df: AnnotatedDataFrame,
                 workdir: Path):

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

    #     ("plot_processed-lat_lon", PlotLatLon(output_dir=plots_dir,
    #                                           filename_prefix="lat-lon-processed")),
    #     ("plot_processed-histograms", PlotHistograms(output_dir=plots_dir,
    #                                                  filename_prefix="histogram-processed-data-"))
    # ])

    df = pipeline.transform(df)

    # Get the setup for storing the processed data
    processed_data_spec = get_processed_data_spec(params=params)
    output_dir = get_outputs_dir(params=params)
    h5_key = processed_data_spec["h5_key"]
    pipeline.state_write(output_dir / "pipeline.yaml")
    df.save(filename=output_dir / f"{int(params['month']):02}.h5")


if __name__ == "__main__":

    default_params_yaml = Path(__file__).parent.parent / "params.yaml"

    # Management of arguments and options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--workdir", default=str(Path(".").resolve()))
    parser.add_argument("-p", "--parameters", default=str(default_params_yaml), type=str)
    parser.add_argument("-m", "--month", dest="month", default=1, help="month to process", type=int,
                        choices=range(1, 13))
    parser.add_argument("-F", "--filter_area", dest="filter_area", action="store_true", default=False,
                        help="Filter by area restricted through --lat_XXX options")
    parser.add_argument("--filter_mmsi", dest="filter_MMSI", action="store_true", default=False,
                        help="Filter by MMSI range as specified in parameters")
    parser.add_argument("--lat_min", dest="lat_min", default=-90.0, help="Latitude minimal", type=float)
    parser.add_argument("--lat_max", dest="lat_max", default=90.0, help="Latitude maximal", type=float)
    parser.add_argument("--lon_min", dest="lon_min", default=-180.0, help="Longitude minimal", type=float)
    parser.add_argument("--lon_max", dest="lon_max", default=180.0, help="Longitude maximal", type=float)
    parser.add_argument("-A", "--add_distance_closest_anchorage", dest="add_distance_closest_anchorage",
                        action="store_true", default=False,
                        help="Add a new columns that contain the distance to the closest anchorage")
    parser.add_argument("-S", "--add_distance_satellite", dest="add_distance_satellite",
                        action="store_true", default=False,
                        help="Add a new columns that contain the distance to the closest anchorage")
    parser.add_argument("--loglevel", dest="loglevel", type=int, default=10, help="Set loglevel to display")
    parser.add_argument("--logfile", dest="logfile", type=str, default=None,
                        help="Set file for saving log (default prints to terminal)")
    args = parser.parse_args()

    if args.logfile is None:
        basicConfig(format=LOG_FORMAT)
    else:
        outfile = open(args.logfile, "w")
        ch = StreamHandler(outfile)
        ch.setFormatter(Formatter(LOG_FORMAT, datefmt='%H:%M:%S'))
        _log.addHandler(ch)

    _log.setLevel(args.loglevel)
    start = time.perf_counter()
    # Load parameters
    params_yaml: Path = Path(args.parameters)
    yaml_file = open(f"{params_yaml}")
    params: ParamsType = yaml.load(yaml_file, Loader=yaml.FullLoader)["data_processing"]
    # Set bas path
    params["workdir"]: Path = Path(args.workdir)

    options = vars(args)
    params.update(options)

    _log.info("[START]")
    process_files(params)
    end = time.perf_counter()
    _log.info(f"[END] Execution Time = {end-start:.0f}s")

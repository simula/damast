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


An example of input data is shown below

.. csv-table:: Snapshot of data from `data/1/20220101.zip`
   :file: example.csv
   :align: center
   :header-rows: 1

"""
# -----------------------------------------------------------
# (C) 2020 Pierre Bernabe, Oslo, Norway
# email pierbernabe@simula.no
# -----------------------------------------------------------

import argparse
import datetime
import functools
import glob
from logging import Formatter, StreamHandler, basicConfig, getLogger
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from damast.data_handling.exploration import plot_lat_lon
from damast.data_handling.pipeline import Pipeline
from damast.data_handling.transformers.augmenters import (
    AddDistanceClosestAnchorage,
    AddFishingVesselType,
    AddLocalMessageIndex,
    AddVesselType
    )
from damast.data_handling.transformers.filters import (
    AreaFilter,
    DuplicateNeighboursFilter,
    MMSIFilter
    )
from damast.data_handling.transformers.sorters import GenericSorter
from damast.data_handling.transformers.visualisers import (
    PlotHistograms,
    PlotLatLon
    )
from damast.domains.maritime.data_specification import ColumnName

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


def cleanse_and_sanitise(params: ParamsType, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanse and sanitise the dataframe by adjusting the type and performing a cleanup.

    This also reduces the amount of data.

    TODO: Consider other message types
    TODO: Mark rows which contain invalid (out-of-spec data)

    1. Keep only data from satellite, i.e. source != 'g'
    2. Remove row which do not have a timestamp
    3. Keep only Message Position Messages (TODO: are there not distress signals part of the other messages?)
    4. Create a "timestamp" field
    5. Remove 'useless_columns' as defined in params
    6. Multiply SOG and COG by 10 to eliminate need for comma
    7. Replace NaN with default values as defined in '["columns"]["constraints"][<col_name>]["default"]' in params
    8. Set types via '["columns"]["constraints"][<col_name>]["type"]' in params
    9. Drop '["columns"]["unused"]' as defined in params

    :return: The cleaned dataset
    """

    # All logging headers
    header = "[COMPRESSION][{action}]"
    drop_hdr = header.format(action='DROP')
    add_hdr = header.format(action="ADD")
    fill_hdr = header.format(action="FILLNA")
    type_hdr = header.format(action="TYPE")

    # Keep only satellite data
    before = df.shape[0]

    df.drop(df[df.source == "g"].index, inplace=True)

    _log.info(f"{drop_hdr} The files contain {before} messages in total."
              f"Only {df.shape[0]} come from satellite(s) and are kept")

    # Remove line where date is null
    before = df.shape[0]
    df.drop(df[df.BaseDateTime.isnull()].index, inplace=True)

    # Keep only the position message
    df.drop(df[~df.MessageType.isin(params["MessageTypePosition"])].index, inplace=True)
    _log.info(f"{drop_hdr} {before - df.shape[0]} lines have been dropped."
              " They are not of type position report")

    # Date parsing
    df.loc[:, 'BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S")

    # Date to Timestamp
    df['timestamp'] = df.BaseDateTime.values.astype(np.int64) // 10 ** 9
    _log.info(f"{add_hdr} Column timestamps added")

    # Drop useless columns
    useless_columns = params["columns"]["useless"]
    df.drop(columns=useless_columns, inplace=True)
    _log.info(f"{drop_hdr} These columns have been removed: " + ', '.join(
        useless_columns))

    # Drop unused columns
    # TODO: what is the difference to useless_columns (?)
    unused_columns = params["columns"]["unused"]
    df.drop(columns=unused_columns, inplace=True)
    _log.info(f"{drop_hdr} These columns have been removed: " + ', '.join(unused_columns))

    # multiply columns in order to remove comma
    df.loc[:, 'SOG'] *= 10
    df.loc[:, 'COG'] *= 10

    # Fill empty value for then reduce the size thanks to the type
    columns_constraints = params["columns"]["constraints"]

    for column in columns_constraints:
        constraints = columns_constraints[column]
        if column not in df.columns:
            continue

        if df[column].hasnans:
            if "default" in constraints:
                default_value = constraints["default"]
                _log.debug(f"{fill_hdr} The NaN values for {column} have been filled"
                           f" with {default_value} based on settings in params.yaml")
                df[column].fillna(default_value, inplace=True)
            else:
                _log.debug(f"{fill_hdr} {column} has NaN values,"
                           f" but not default value is specified in params.yaml")
        else:
            _log.debug(f"{fill_hdr} No NaN values in {column} - nothing to do")

    # Reduction maximum des types
    _log.info(f"{type_hdr} Updating columns type for: {df.columns}")

    columns_type = {}
    for col in df.columns:
        if col in columns_constraints:
            if "type" in columns_constraints[col]:
                columns_type[col] = columns_constraints[col]["type"]
            else:
                raise KeyError(f"Missing type information in params.yaml for column '{col}'")

    df = df.astype(columns_type)
    _log.info(f"{type_hdr} The type of the columns have been modified"
              f" to {columns_type} as defined in params.yaml")

    return df


def process_files(params: Dict[str, Any]) -> None:
    """
    Process input data. TODO: ADD MORE INFO

    Args:
        params (Dict[str, Any]): Options for processing


    .. highlight:: python
    .. code-block:: python 

      params = {"workdir": "path_to_directory_of_execution",
                "inputs": {"data":{"dir": "path_to_input_data"}},
                "month": "integer",
                "columns":{"brut": ["list","of","columns","to","use]}}


    Raises:
        KeyError: If any of the required keys are missing from the input parameters
        RuntimeError: No data-files are found in input data directory.
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
                 df: pd.DataFrame,
                 workdir: Path):
    df.reset_index(drop=True, inplace=True)
    # region PROCESSING PIPELINE
    plots_dir = get_plots_dir(params=params)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Data Feature extraction and augmentation
    vessel_type_csv = params["inputs"]["vessel_types"]
    fishing_vessel_type_csv = params["inputs"]["fishing_vessel_types"]
    anchorages_csv = params["inputs"]["anchorages"]
    # https://en.wikipedia.org/wiki/Two-line_element_set
    # tle_filename = params["inputs"]["tle_file"]

    # Output an overview over the existing anchorages
    anchorages_data = pd.read_csv(anchorages_csv)
    plot_lat_lon(df=anchorages_data,
                 latitude_name="latitude",
                 longitude_name="longitude",
                 output_dir=plots_dir,
                 filename_prefix="anchorages-lat-lon")

    pipeline = Pipeline([
        ("plot_input-histograms", PlotHistograms(output_dir=plots_dir,
                                                 filename_prefix="histogram-input-data-")),
        ("plot_input-lat_lon", PlotLatLon(output_dir=plots_dir,
                                          filename_prefix="lat-lon-input")),
        # Filtering and dropping of data
        ("filter_areas", AreaFilter()),
        ("filter_mmsi", MMSIFilter()),
        ("sort_mmsi_timestamp", GenericSorter(column_names=[ColumnName.MMSI, ColumnName.TIMESTAMP])),
        ("filter_duplicate_timestamp", DuplicateNeighboursFilter(column_names=[ColumnName.MMSI, ColumnName.TIMESTAMP])),
        # Add and update data
        ("augment_vessel_type", AddVesselType(vessel_type_data=vessel_type_csv)),
        ("augment_fishing_vessel_type", AddFishingVesselType(vessel_type_data=fishing_vessel_type_csv)),
        ("augment_distance_to_closest_anchorage", AddDistanceClosestAnchorage(anchorages_data=anchorages_csv)),
        # ("augment_distance_to_closest_satellite", AddDistanceClosestSatellite(satellite_tle_filename=tle_filename,
        #                                                                      latitude_name=ColumnName.LATITUDE,
        #                                                                      longitude_name=ColumnName.LONGITUDE)),
        ("augment_local_message_index", AddLocalMessageIndex(mmsi_name=ColumnName.MMSI)),
        ("plot_processed-lat_lon", PlotLatLon(output_dir=plots_dir,
                                              filename_prefix="lat-lon-processed")),
        ("plot_processed-histograms", PlotHistograms(output_dir=plots_dir,
                                                     filename_prefix="histogram-processed-data-"))
    ])

    df = pipeline.fit_transform(df)

    # Get the setup for storing the processed data
    processed_data_spec = get_processed_data_spec(params=params)
    output_dir = get_outputs_dir(params=params)
    h5_key = processed_data_spec["h5_key"]

    pipeline.save_stats(filename=output_dir / f"{int(params['month']):02}" / "processing-stats.json")
    data_out_path = output_dir / f"{int(params['month']):02}.h5"
    data_out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_hdf(f"{data_out_path}", h5_key, format='table', mode="w")
    _log.info(f"[DATA][WRITE] Write {df.shape[0]} lines inside {data_out_path}")


if __name__ == "__main__":
    with elapsed_time() as timer:

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
        _log.info(f"[END] Execution Time = {timer():.0f}s")

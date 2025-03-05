"""
Module containing the functionality to explore and visualise data
"""
from pathlib import Path
from typing import List, Optional

from matplotlib import pyplot as plt

from damast.core.types import DataFrame

__all__ = ["plot_histograms",
           "plot_lat_lon",
           "PLOT_DPI"
           ]

#: Default DPI when plotting figures
PLOT_DPI: int = 300


def plot_lat_lon(*,
                 df: DataFrame,
                 output_dir: Path,
                 latitude_name: str = "LAT",
                 longitude_name: str = "LON",
                 filename_prefix: str = "lat-lon-scatter",
                 dpi: int = PLOT_DPI) -> Path:
    """
    Save a scatter plot of latitude longitude
    :param df:
    :param output_dir:
    :param longitude_name: Name of the longitude column in the dataframe
    :param latitude_name: Name of the latitude column in the dataframe
    :param filename_prefix: filename prefix for the output scatter plot
    :param dpi: DPI to save the final image with

    :return: Path to the file
    """
    if not isinstance(df, DataFrame):
        df = df.lazy()

    longitude = df.select(longitude_name).collect()
    latitude = df.select(latitude_name).collect()

    # .copy() to address ValueError: assignment destination is read-only
    # raised in numpy/ma/core.py:3528: in mask
    #      self.__setmask__(value)
    plt.scatter(x=longitude.to_numpy(),
                y=latitude.to_numpy(),
                alpha=1)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    filename = output_dir / f"{filename_prefix}.png"
    plt.savefig(filename, dpi=dpi)
    plt.close()

    return filename


def plot_histograms(*,
                    df: DataFrame,
                    output_dir: Path,
                    filename_prefix: str,
                    columns: Optional[List[str]] = None,
                    dpi: int = PLOT_DPI) -> Path:
    """
    Plot histograms for the current data frame in to directory defined through get_plot_dir()

    :param df: dataframe to generate the histograms for
    :param output_dir: output directory
    :param filename_prefix: filename prefix for the histograms
    :param dpi: plot the figure with this DPI (Dots Per Inch) setting
    :param columns: columns of the dataframe to use
    :return: Path to the plots' directory
    """
    if not isinstance(df, DataFrame):
        df = df.lazy()

    if columns is None:
        columns = df.columns

    # For each column plot the histogram
    for col_name in columns:
        if col_name not in df.columns:
            raise KeyError(f"plot_histogram: {col_name} is not an existing column,"
                           f" available are {','.join(df.columns)}")
        # Vaex uses the prefix "__" for hidden columns, i.e. columns that have either been removed, or
        # is a dependent of another column (say after converting radians to degrees)
        if col_name.startswith("__"):
            continue
        fig = plt.figure()

        df.select(col_name).collect().plot.hist()

        plt.tight_layout()
        path = output_dir / f"{filename_prefix}{col_name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return output_dir

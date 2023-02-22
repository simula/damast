"""
Module containing the functionality to explore and visualise data
"""
from pathlib import Path
from typing import List
import pandas as pd
import vaex
from matplotlib import pyplot as plt

__all__ = ["plot_histograms",
           "plot_lat_lon",
           "PLOT_DPI"
           ]

#: Default DPI when plotting figures
PLOT_DPI: int = 300


def plot_lat_lon(*,
                 df: vaex.DataFrame,
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

    if isinstance(df, pd.DataFrame):
        plt.scatter(x=df[longitude_name],
                    y=df[latitude_name],
                    alpha=1)
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        filename = output_dir / f"{filename_prefix}.png"
        plt.savefig(filename, dpi=dpi)
        plt.close()

    else:
        plt.scatter(x=df[longitude_name].evaluate().data,
                    y=df[latitude_name].evaluate().data,
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
                    df: vaex.DataFrame,
                    output_dir: Path,
                    filename_prefix: str,
                    columns: List[str] = None,
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
    if columns is None:
        columns = df.columns
    if isinstance(df, pd.DataFrame):
        # For each column plot the histogram
        for col_name in columns:
            if col_name not in df.columns:
                raise KeyError(f"plot_histogram: {col_name} is not an existing column,"
                               f" available are {','.join(df.columns_names)}")

            fig_hist, ax = plt.subplots(1, 1, figsize=(10, 10))
            df.hist(col_name, ax=ax)
            plt.tight_layout()
            path = output_dir / f"{filename_prefix}{col_name}.png"
            fig_hist.savefig(path, dpi=dpi)
            plt.close(fig_hist)

    else:
        # For each column plot the histogram
        for col_name in columns:
            if col_name not in df.columns_names:
                raise KeyError(f"plot_histogram: {col_name} is not an existing column,"
                               f" available are {','.join(df.columns_names)}")

            fig_hist, ax = plt.subplots(1, 1, figsize=(10, 10))
            df.viz.histogram(df[col_name], ax=ax)
            plt.tight_layout()
            path = output_dir / f"{filename_prefix}{col_name}.png"
            fig_hist.savefig(path, dpi=dpi)
            plt.close(fig_hist)
    return output_dir

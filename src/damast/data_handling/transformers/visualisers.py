"""
Module that contains transformers which do not modify data, but serve only data exploration.
"""
from pathlib import Path
from typing import Union

from damast.core.units import units
from damast.data_handling.exploration import PLOT_DPI, plot_histograms, plot_lat_lon

__all__ = [
    "BaseVisualiser",
    "PlotHistograms",
    "PlotLatLon",
]

import damast.core
from damast.core.dataprocessing import AnnotatedDataFrame, PipelineElement


class BaseVisualiser(PipelineElement):

    #: The output directory for generated artifacts
    output_dir: Path
    filename_prefix: str
    dpi: int

    def __init__(self,
                 output_dir: Union[str, Path],
                 filename_prefix: str = '',
                 dpi: int = PLOT_DPI):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.dpi = dpi


class PlotHistograms(BaseVisualiser):
    def __init__(self,
                 output_dir: Union[str, Path],
                 filename_prefix: str = "histogram"):
        super().__init__(output_dir=output_dir,
                         filename_prefix=filename_prefix)

    @damast.core.input({})
    @damast.core.output({})
    @damast.core.describe("Plot histograms of all columns in dataframe")
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        plot_histograms(df=df._dataframe,
                        output_dir=self.output_dir,
                        filename_prefix=self.filename_prefix)
        return df


class PlotLatLon(BaseVisualiser):
    def __init__(self,
                 output_dir: Union[str, Path],
                 filename_prefix: str = "lat-lon"):
        super().__init__(output_dir=output_dir,
                         filename_prefix=filename_prefix)

    @damast.core.input({"LAT": {"unit": units.deg}, "LON": {"unit": units.deg}})
    @damast.core.output({})
    @damast.core.describe("Plot Latitude and longitude ")
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:

        plot_lat_lon(df=df._dataframe,
                     output_dir=self.output_dir,
                     filename_prefix=self.filename_prefix)
        return df

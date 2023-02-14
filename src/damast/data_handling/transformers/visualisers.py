"""
Module that contains transformer which do not modify data, but serve only data exploration.
"""
from pathlib import Path
from typing import Union

from sklearn.pipeline import Pipeline

from damast.data_handling.exploration import (
    PLOT_DPI,
    plot_histograms,
    plot_lat_lon
    )

__all__ = [
    "BaseVisualiser",
    "PlotHistograms",
    "PlotLatLon",
]

from damast.data_handling.transformers.base import BaseTransformer


class BaseVisualiser(BaseTransformer):
    # Optional internal pipeline that might be used for the transformation
    _pipeline: Pipeline = None

    #: The output directory for generated artifacts
    output_dir: Path = None
    filename_prefix: str = None
    dpi: int = None

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

    def transform(self, df):
        super().transform(df)

        plot_histograms(df=df,
                        output_dir=self.output_dir,
                        filename_prefix=self.filename_prefix)

        return df


class PlotLatLon(BaseVisualiser):
    def __init__(self,
                 output_dir: Union[str, Path],
                 filename_prefix: str = "lat-lon"):
        super().__init__(output_dir=output_dir,
                         filename_prefix=filename_prefix)

    def transform(self, df):
        super().transform(df)

        plot_lat_lon(df=df,
                     output_dir=self.output_dir,
                     filename_prefix=self.filename_prefix)

        return df

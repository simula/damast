import numpy as np
import polars

from damast.core.data_description import MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement


class CycleTransformer(PipelineElement):
    def __init__(self, n: int):
        self.n = n

    @describe("Cycle transformation")
    @input({'x': {}})
    @output({
        "{{x}}_x": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_y": {"value_range": MinMax(-1.0, 1.0)},
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        feature = self.get_name('x')
        df._dataframe = df._dataframe.with_columns(
                (np.sin(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_x"),
                (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_y")
        )
        return df

class TimestampCycleTransformer(PipelineElement):
    @describe("Cycle transformation")
    @input({'x': {}})
    @output({
        "{{x}}_quarter_x": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_quarter_y": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_week_x": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_week_y": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_weekday_x": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_weekday_y": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_hour_x": {"value_range": MinMax(-1.0, 1.0)},
        "{{x}}_hour_y": {"value_range": MinMax(-1.0, 1.0)},
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        feature = self.get_name('x')

        df._dataframe = df._dataframe.with_columns(
                (np.sin(polars.col(feature).dt.quarter()*2*np.pi) / 4).alias(f"{feature}_quarter_x"),
                (np.cos(polars.col(feature).dt.quarter()*2*np.pi) / 4).alias(f"{feature}_quarter_y"),
                (np.sin(polars.col(feature).dt.week()*2*np.pi) / 53).alias(f"{feature}_week_x"),
                (np.cos(polars.col(feature).dt.week()*2*np.pi) / 53).alias(f"{feature}_week_y"),
                (np.sin(polars.col(feature).dt.weekday()*2*np.pi) / 7).alias(f"{feature}_weekday_x"),
                (np.cos(polars.col(feature).dt.weekday()*2*np.pi) / 7).alias(f"{feature}_weekday_y"),
                (np.sin(polars.col(feature).dt.hour()*2*np.pi) / 24).alias(f"{feature}_hour_x"),
                (np.cos(polars.col(feature).dt.hour()*2*np.pi) / 24).alias(f"{feature}_hour_y"),
        )
        return df

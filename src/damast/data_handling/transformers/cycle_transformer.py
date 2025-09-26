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
                (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_x"),
                (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_y")
        )
        return df

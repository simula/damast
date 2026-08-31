import polars

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement


class MyTripler(PipelineElement):
    """Multiplies a column by 3 - a minimal example of a local plugin transformer."""

    @describe("Triples a column")
    @input({"x": {}})
    @output({"{{x}}_tripled": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        feature = self.get_name("x")
        result = self.get_name("{{x}}_tripled")
        df.lazyframe = df.lazyframe.with_columns(
            (polars.col(feature) * 3).alias(result)
        )
        return df

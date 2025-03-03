from damast.core.transformations import Transformer
from damast.core.dataframe import AnnotatedDataFrame

class Pipeline:
    steps: list[Transformer]

    def __init__(self, steps: list[Transformer]):
        self.steps = steps

    def transform(self, df: AnnotatedDataFrame):
        return self.run(df=df)

    def run(self, df: AnnotatedDataFrame):
        for step in self.steps:
            df = step.fit_transform(df)
        return df

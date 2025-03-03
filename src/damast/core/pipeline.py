from damast.core.dataframe import AnnotatedDataFrame
from damast.core.transformations import Transformer


class Pipeline:
    steps: list[Transformer]

    def __init__(self, steps: list[Transformer]):
        self.steps = steps

    def transform(self, df: AnnotatedDataFrame):
        return self.run(df=df)

    def run(self, df: AnnotatedDataFrame):
        for idx, step in enumerate(self.steps):
            try:
                df = step.fit_transform(df)
            except Exception as e:
                raise RuntimeError(f"Step #{idx} in pipeline ({step.__class__.__name__}) failed: -- {e}")
        return df

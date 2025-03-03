import numpy as np
import polars

from damast.core.dataframe import AnnotatedDataFrame


class Transformer:
    def transform(self, df: AnnotatedDataFrame):
        return df

    def fit(self, df):
        pass

    def transform(self, df):
        return df

    def fit_transform(self, df: AnnotatedDataFrame):
        self.fit(df=df)
        return self.transform(df=df)

class CycleTransformer(Transformer):
    def __init__(self, features: list[str], n: int):
        self.features = features
        self.n = n

    def transform(self, df: AnnotatedDataFrame):
        if type(df) != AnnotatedDataFrame:
            raise ValueError(f"Transformer requires 'AnnotatedDataFrame',"
                    f" but got '{type(df)}")
        clone = df.copy()

        for feature in self.features:
            clone._dataframe = clone._dataframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_x")
                )
            clone._dataframe = clone._dataframe.with_columns(
                    (np.cos(polars.col(feature)*2*np.pi) / self.n).alias(f"{feature}_y")
                )
        return clone

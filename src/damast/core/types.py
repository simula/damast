from polars import LazyFrame as DataFrame

from .polars_dataframe import PolarsDataFrame as XDataFrame

__all__ = [
    "DataFrame",
    "XDataFrame"
]

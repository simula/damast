# code=utf-8
"""
This namespace contains the core modules that are required to define metada.
"""

from .annotations import Annotation
from .dataframe import AnnotatedDataFrame
from .dataprocessing import (
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    describe,
    input,
    output,
)
from .datarange import DataRange, MinMax
from .metadata import MetaData, DataSpecification
from .dataprocessing import DataProcessingPipeline

__all__ = [
    "AnnotatedDataFrame",
    "describe",
    "input",
    "output",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS",
    "Annotation",
    "DataRange",
    "MetaData",
    "DataSpecification",
    "DataProcessingPipeline",
    "MinMax",
]

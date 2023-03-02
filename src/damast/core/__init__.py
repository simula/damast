"""
This namespace contains the core modules that are required to define metadata.
"""

from .annotations import Annotation
from .dataframe import AnnotatedDataFrame, replace_na
from .dataprocessing import (
    artifacts,
    DECORATED_ARTIFACT_SPECS,
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    describe,
    input,
    output,
)
from .datarange import DataRange, MinMax
from .metadata import ArtifactSpecification, MetaData, DataSpecification
from .dataprocessing import DataProcessingPipeline
__all__ = [
    "AnnotatedDataFrame",
    "artifacts",
    "describe",
    "input",
    "output",
    "DECORATED_ARTIFACT_SPECS",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS",
    "Annotation",
    "DataRange",
    "MetaData",
    "DataSpecification",
    "DataProcessingPipeline",
    "MinMax",
    "replace_na",
    "ArtifactSpecification"
]

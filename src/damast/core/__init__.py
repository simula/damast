"""
This namespace contains the core modules that are required to define metadata.
"""

from .annotations import Annotation
from .dataframe import AnnotatedDataFrame
from .dataprocessing import (
    DECORATED_ARTIFACT_SPECS,
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    DataProcessingPipeline,
    artifacts,
    describe,
    input,
    output,
    )
from .data_description import DataRange, MinMax
from .metadata import (
    ArtifactSpecification,
    DataSpecification,
    History,
    MetaData,
    ValidationMode,
    )

__all__ = [
    "AnnotatedDataFrame",
    "Annotation",
    "ArtifactSpecification",
    "DECORATED_ARTIFACT_SPECS",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS",
    "DataProcessingPipeline",
    "DataRange",
    "DataSpecification",
    "History",
    "MetaData",
    "MinMax",
    "ValidationMode",
    "artifacts",
    "describe",
    "input",
    "output",
]

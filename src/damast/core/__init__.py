"""
This namespace contains the core modules that are required to define metadata.
"""

from .annotations import Annotation
from .data_description import DataRange, MinMax
from .dataframe import AnnotatedDataFrame
from .dataprocessing import DataProcessingPipeline
from .decorators import (
    DECORATED_ARTIFACT_SPECS,
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    artifacts,
    describe,
    input,
    output,
)
from .metadata import (
    ArtifactSpecification,
    DataSpecification,
    History,
    MetaData,
    ValidationMode,
)
from .partitioning import ByColumn, ByExpr, ByTime, PartitionStrategy

__all__ = [
    "AnnotatedDataFrame",
    "Annotation",
    "ArtifactSpecification",
    "ByColumn",
    "ByExpr",
    "ByTime",
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
    "PartitionStrategy",
    "ValidationMode",
    "artifacts",
    "describe",
    "input",
    "output",
]

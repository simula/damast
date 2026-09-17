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
from .partitioning import ByColumn, ByExpr, ByTime, PartitionStrategy, SaveAs

__all__ = [
    "DECORATED_ARTIFACT_SPECS",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS",
    "AnnotatedDataFrame",
    "Annotation",
    "ArtifactSpecification",
    "ByColumn",
    "ByExpr",
    "ByTime",
    "DataProcessingPipeline",
    "DataRange",
    "DataSpecification",
    "History",
    "MetaData",
    "MinMax",
    "PartitionStrategy",
    "SaveAs",
    "ValidationMode",
    "artifacts",
    "describe",
    "input",
    "output",
]

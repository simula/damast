"""
Shared pipeline fixtures for `damast.viz` tests - a small chained pipeline (two steps, one of
which also needs a column straight from the datasource) and a join pipeline, used by
`test_graph_model.py`, `test_svg_export.py`, and `test_mermaid_export.py` so every renderer's
tests exercise the same pipeline shapes.
"""
from astropy import units
import pytest

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement


class _StepOne(PipelineElement):
    @damast.core.describe("doubles alpha")
    @damast.core.input({"alpha": {"representation_type": int, "unit": units.m, "description": "raw reading"}})
    @damast.core.output({"alpha_doubled": {"representation_type": int}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class _StepTwo(PipelineElement):
    @damast.core.describe("adds beta")
    @damast.core.input({
        "alpha_doubled": {"representation_type": int},
        "beta": {"representation_type": int},
    })
    @damast.core.output({"gamma": {"representation_type": int}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class _JoinStep(PipelineElement):
    @damast.core.input({"key": {"representation_type": int}})
    @damast.core.input({"key": {"representation_type": int}}, label="other")
    @damast.core.output({"joined": {"representation_type": int}})
    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class _WideStep(PipelineElement):
    """A step with more than 4 input columns, to exercise row-wrapping in the renderers."""
    @damast.core.input({f"c{i}": {"representation_type": int} for i in range(9)})
    @damast.core.output({"result": {"representation_type": int}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


@pytest.fixture
def chained_pipeline(tmp_path) -> DataProcessingPipeline:
    return DataProcessingPipeline(name="chained", base_dir=tmp_path) \
        .add("step-one", _StepOne()) \
        .add("step-two", _StepTwo())


@pytest.fixture
def join_pipeline(tmp_path) -> DataProcessingPipeline:
    return DataProcessingPipeline(name="joined", base_dir=tmp_path) \
        .join("other", _JoinStep(), name_mappings={"df": {}, "other": {}})


@pytest.fixture
def wide_pipeline(tmp_path) -> DataProcessingPipeline:
    return DataProcessingPipeline(name="wide", base_dir=tmp_path) \
        .add("wide-step", _WideStep())

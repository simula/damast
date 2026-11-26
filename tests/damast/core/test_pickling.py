"""
Pickling of pipeline elements, as polars does for every UDF when a plan is serialized.

A UDF reading an attribute of its transformer - ``lambda s: s * self.factor`` - captures the
transformer. Through its parent pipeline that once took in the run's pending LazyFrames, whose
plans contain the same UDF, and serializing the plan failed with
``PicklingError: ... excessively deep recursion``.
"""
import copy
import pickle

import polars
import pytest

import damast.core
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.deferred import Deferred, PendingQueries
from damast.core.metadata import DataSpecification, MetaData
from damast.core.transformations import PipelineElement


class SelfCapturingScaler(PipelineElement):
    """Module level, so that pickle can find the class by name."""

    def __init__(self, factor: int = 3):
        self.factor = factor

    @damast.core.describe("scales x, with a UDF that captures the transformer")
    @damast.core.input({"x": {}})
    @damast.core.output({"x_scaled": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        feature = self.get_name("x")
        df.lazyframe = df.lazyframe.with_columns(
            polars.col(feature)
            .map_batches(lambda s: s * self.factor, return_dtype=polars.Int64)
            .alias(f"{feature}_scaled")
        )
        return df


def adf_of(**columns) -> AnnotatedDataFrame:
    return AnnotatedDataFrame(
        polars.LazyFrame(columns),
        metadata=MetaData([DataSpecification(name=name) for name in columns]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )


def scaler_of(pipeline: DataProcessingPipeline) -> SelfCapturingScaler:
    return next(node.transformer for node in pipeline.processing_graph.nodes()
                if isinstance(node.transformer, SelfCapturingScaler))


@pytest.fixture()
def pipeline(tmp_path) -> DataProcessingPipeline:
    return DataProcessingPipeline(name="self-capturing", base_dir=tmp_path) \
        .add("scale", SelfCapturingScaler(), name_mappings={"x": "x"})


@pytest.mark.parametrize("execute", [False, True], ids=["pending", "executed"])
def test_plan_with_a_self_capturing_udf_serializes(pipeline, tmp_path, execute):
    """The shape of damast.ml.Experiment.compute_features: transform, execute, save_state."""
    adf = adf_of(x=[1, 2, 3])
    result = pipeline.transform(adf)
    if execute:
        result.execute()

    state_file = pipeline.save_state(df=result, dir=tmp_path)

    restored = polars.LazyFrame.deserialize(state_file)
    assert restored.collect()["x_scaled"].to_list() == [3, 6, 9]


def test_pickling_an_element_leaves_out_its_pipeline(pipeline):
    element = scaler_of(pipeline)
    assert element.parent_pipeline is pipeline

    restored = pickle.loads(pickle.dumps(element))

    assert restored.factor == 3
    assert restored.name_mappings == element.name_mappings
    assert restored.parent_pipeline is None


def test_shallow_copy_keeps_the_pipeline(pipeline):
    """__deepcopy__ builds on copy.copy, which must not go through __getstate__."""
    element = scaler_of(pipeline)

    assert copy.copy(element).parent_pipeline is pipeline
    assert copy.copy(pipeline)._pending is pipeline._pending


def test_pickling_a_pipeline_leaves_out_its_run(pipeline):
    pipeline.transform(adf_of(x=[1, 2, 3]))
    assert pipeline._pending is not None

    restored = pickle.loads(pickle.dumps(pipeline))

    assert restored._pending is None
    assert restored._steps_in_flight == set()
    assert restored.name == pipeline.name


def test_pending_queries_are_released_once_run():
    pending = PendingQueries()
    seen = []
    pending.add(Deferred([polars.LazyFrame({"x": [1, 2]}).select(polars.len())],
                         lambda results: results[0].item()),
                seen.append)

    pending.run()

    assert seen == [2]
    assert len(pending) == 0
    assert pending.done


def test_pending_queries_are_released_when_a_check_fails():
    pending = PendingQueries()

    def fail(results):
        raise ValueError("out of range")

    pending.add(Deferred([polars.LazyFrame({"x": [1]}).select(polars.len())], fail))

    with pytest.raises(RuntimeError, match="out of range"):
        pending.run()
    assert len(pending) == 0

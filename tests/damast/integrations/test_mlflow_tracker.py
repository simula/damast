import uuid

import polars
import pytest
from astropy import units

from damast.core.data_description import MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.metadata import DataSpecification, MetaData, ValidationMode

mlflow = pytest.importorskip("mlflow")

from damast.integrations.mlflow_tracker import track_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def tracking_uri(tmp_path_factory):
    """
    One database for the whole session - creating it runs mlflow's alembic migrations, which
    are slow enough to have stalled past the CI test timeout on Windows. Tests stay isolated
    through their own experiment (see the `experiment` fixture) rather than their own file.
    """
    database = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    # .as_posix(), not str(): a Windows path's native "\" has no business inside a sqlite URI
    return f"sqlite:///{database.as_posix()}"


@pytest.fixture
def experiment(tracking_uri, tmp_path):
    """A fresh experiment per test, so runs remain isolated within the shared database."""
    name = f"damast-tests-{uuid.uuid4().hex[:8]}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(name, artifact_location=(tmp_path / "artifacts").as_uri())
    return name


@pytest.fixture
def adf():
    df = polars.DataFrame({"speed": [1.0, 2.0, 3.0]})
    metadata = MetaData(
        columns=[
            DataSpecification(
                name="speed", unit=units.m / units.s, value_range=MinMax(0.0, 10.0)
            ),
        ]
    )
    return AnnotatedDataFrame(
        df, metadata, validation_mode=ValidationMode.UPDATE_METADATA
    )


def _get_run(tracking_uri, experiment_name):
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    return runs[0]


def test_track_pipeline_logs_result_and_ends_run(tracking_uri, experiment, adf):
    pipeline = DataProcessingPipeline(name="test-pipeline")

    with track_pipeline(
        pipeline, run_name="a-run", tracking_uri=tracking_uri, experiment_name=experiment
    ) as tracker:
        tracker.log_result(adf)

    run = _get_run(tracking_uri, experiment)
    assert run.info.status == "FINISHED"
    assert run.data.params["col.speed.unit"] == adf.metadata["speed"].unit.to_string()
    assert run.data.tags["damast.pipeline"] == "test-pipeline"

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert "metadata.damast.yaml" in artifact_names


def test_track_pipeline_logs_step_stats(tracking_uri, experiment, adf):
    pipeline = DataProcessingPipeline(name="test-pipeline")
    # Simulate what on_transform_start/on_transform_end record during a real transform().
    pipeline._processing_stats["step_a"] = {
        "processing_time_in_s": 0.5,
        "output_dataframe_length": 3,
        "input_dataframe_length": {"df": 3},
    }

    with track_pipeline(pipeline, run_name="a-run", tracking_uri=tracking_uri,
                        experiment_name=experiment):
        pass

    run = _get_run(tracking_uri, experiment)
    assert run.data.metrics["step.step_a.processing_time_in_s"] == 0.5
    assert run.data.metrics["step.step_a.output_dataframe_length"] == 3.0


def test_track_pipeline_marks_run_failed_on_exception(tracking_uri, experiment):
    pipeline = DataProcessingPipeline(name="test-pipeline")

    with pytest.raises(RuntimeError):
        with track_pipeline(pipeline, run_name="a-run", tracking_uri=tracking_uri,
                            experiment_name=experiment):
            raise RuntimeError("boom")

    run = _get_run(tracking_uri, experiment)
    assert run.info.status == "FAILED"

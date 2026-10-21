import copy
import datetime
import os
from collections import OrderedDict
from pathlib import Path

import polars as pl
import pytest

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement
from damast.core.datarange import MinMax
from damast.core.metadata import MetaData
from damast.core.transformations import MultiCycleTransformer
from damast.core.units import units
from damast.domains.maritime.ais.data_generator import AISTestData, AISTestDataSpec
from damast.ml import keras
from damast.ml.experiments import (
    Experiment,
    ForecastTask,
    LearningTask,
    ModelInstanceDescription,
    TemporalForecastTask,
    TrainingParameters,
)
from damast.ml.models.base import BaseModel

os.environ["COLUMNS"] = '120'


class TransformerA(PipelineElement):
    @damast.input({"x": {}})
    @damast.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df

class LatLonTransformer(PipelineElement):
    @damast.core.describe("Lat/Lon cyclic transformation")
    @damast.core.input({
        "lat": {"unit": units.deg},
        "lon": {"unit": units.deg}
    })
    @damast.core.output({
        "lat_x": {"value_range": MinMax(-1.0, 1.0)},
        "lat_y": {"value_range": MinMax(-1.0, 1.0)},
        "lon_x": {"value_range": MinMax(-1.0, 1.0)},
        "lon_y": {"value_range": MinMax(-1.0, 1.0)}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat_cyclic_transformer = MultiCycleTransformer(features=["lat"], n=180.0)
        lon_cyclic_transformer = MultiCycleTransformer(features=["lon"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        return _df


class SimpleModel(BaseModel):
    input_specs = OrderedDict({
        "a": {"length": 1},
        "b": {"length": 1},
        "c": {"length": 1},
        "d": {"length": 1}
    })

    output_specs = OrderedDict({
        "a": {"length": 1},
        "b": {"length": 1},
        "c": {"length": 1},
        "d": {"length": 1}
    })

    def __init__(self, output_dir: str | Path):
        super().__init__(features=["a", "b", "c", "d"],
                         targets=["a", "b", "c", "d"],
                         output_dir=output_dir)

    def _init_model(self):
        inputs = keras.Input(shape=(4,),
                             name="input",
                             dtype=float)
        outputs = keras.layers.Dense(4)(inputs)

        self.model = keras.Model(inputs=inputs,
                                 outputs=outputs,
                                 name=self.__class__.__name__)


class ModelA(SimpleModel):
    pass


class ModelB(SimpleModel):
    pass


class Baseline(BaseModel):
    input_specs = OrderedDict({
        "lat_x": {"length": 1},
        "lat_y": {"length": 1},
        "lon_x": {"length": 1},
        "lon_y": {"length": 1}
    })

    output_specs = OrderedDict({
        "lat_x": {"length": 1},
        "lat_y": {"length": 1},
        "lon_x": {"length": 1},
        "lon_y": {"length": 1}
    })

    def __init__(self,
                 features: list[str],
                 timeline_length: int,
                 output_dir: Path,
                 name: str = "Baseline",
                 targets: list[str] | None = None):
        self.timeline_length = timeline_length

        super().__init__(name=name,
                         output_dir=output_dir,
                         features=features,
                         targets=targets)

    def _init_model(self):
        features_width = len(self.features)
        targets_width = len(self.targets)

        self.model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[self.timeline_length, features_width]),
            keras.layers.Dense(targets_width)
        ])


class BaselineA(Baseline):
    pass


class BaselineB(Baseline):
    pass


@pytest.fixture()
def experiment_dir(tmp_path):
    Experiment.touch_marker(tmp_path)

    model_a = ModelA(output_dir=tmp_path)
    model_a.save()

    model_b = ModelB(output_dir=tmp_path)
    model_b.save()

    return tmp_path


def test_learning_task_init():
    with pytest.raises(TypeError, match="could not instantiate"):
        LearningTask(label="test",
                     pipeline=10,
                     features=["a"],
                     targets=["b"],
                     models=[],
                     )

    # ModelInstanceDescription is incorrect
    with pytest.raises(TypeError, match="could not instantiate"):
        LearningTask(label="test",
                     pipeline=DataProcessingPipeline(name="test-pipeline"),
                     features=["a"],
                     targets=["b"],
                     models=[10],
                     )

    with pytest.raises(TypeError, match="training_parameters"):
        LearningTask(label="test",
                     pipeline=DataProcessingPipeline(name="test-pipeline"),
                     features=["a"],
                     targets=["b"],
                     models=[ModelInstanceDescription(BaseModel, {})],
                     training_parameters=10
                     )


def test_learning_task_io():
    with pytest.raises(KeyError, match="missing 'module_name'"):
        LearningTask.from_dict(data={})

    with pytest.raises(KeyError, match="missing 'class_name'"):
        LearningTask.from_dict(data={'module_name': 'damast.ml.experiments'})

    with pytest.raises(ValueError, match="could not find"):
        LearningTask.from_dict({'module_name': 'damast.ml.experiments',
                                'class_name': 'NotAnExistingClass'})


def test_validate_experiment_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Experiment.validate_experiment_dir(dir="UNKNOWN_DIR")

    with pytest.raises(NotADirectoryError):
        filename = tmp_path / "dummy-file"
        with open(filename, "w"):
            pass

        Experiment.validate_experiment_dir(dir=filename)

    with pytest.raises(NotADirectoryError, match="is not an experiment"):
        Experiment.validate_experiment_dir(dir=tmp_path)

    # Validate touch
    with pytest.raises(FileNotFoundError):
        Experiment.touch_marker(dir=f"{tmp_path}-does-not-exist")
    with pytest.raises(NotADirectoryError):
        file_in_dir = tmp_path / "test_file"
        with open(file_in_dir, "a"):
            pass
        Experiment.touch_marker(dir=file_in_dir)

    Experiment.touch_marker(dir=tmp_path)
    Experiment.validate_experiment_dir(dir=tmp_path)


def test_validate_from_directory(experiment_dir):
    models = Experiment.from_directory(experiment_dir)

    assert "ModelA" in models
    assert "ModelB" in models


def test_learning_task(tmp_path):
    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA())

    models = [
        ModelInstanceDescription(model=ModelA,
                                 parameters={})
    ]

    lt = LearningTask(
        label="test learning task",
        pipeline=pipeline,
        features=["a", "b", "c", "d"],
        models=models)

    data_dict = dict(lt)
    loaded_t = LearningTask.from_dict(data=data_dict)
    assert loaded_t == lt

    loaded_t.pipeline.name = "new-pipeline-name"
    assert loaded_t != lt

    assert loaded_t != 10


def test_forecast_task(tmp_path):
    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA())

    models = [
        ModelInstanceDescription(model=ModelA,
                                 parameters={})
    ]

    lt = ForecastTask(
        label="test forecast task",
        pipeline=pipeline,
        features=["a", "b", "c", "d"],
        models=models,
        sequence_length=20,
        forecast_length=1,
        group_column="key-column"
    )

    data_dict = dict(lt)
    loaded_t = LearningTask.from_dict(data=data_dict)
    assert isinstance(loaded_t, ForecastTask)

    assert loaded_t == lt

    loaded_t.pipeline.name = "new-pipeline-name"
    assert loaded_t != lt

    assert loaded_t != 10


def test_to_and_from_file(tmp_path):
    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA())

    models = [
        ModelInstanceDescription(model=ModelA,
                                 parameters={})
    ]

    task = ForecastTask(
        label="forecast-abcd",
        pipeline=pipeline,
        features=["a", "b", "c", "d"],
        models=models,
        group_column="a",
        sequence_length=50,
        forecast_length=1
    )

    with pytest.raises(TypeError, match="learning_task"):
        Experiment(learning_task="This is not a learning task",
                   batch_size=10,
                   input_data=Path(__file__).parent.parent / "data" / "test_dataframe.hdf5")

    experiment = Experiment(learning_task=task,
                            batch_size=10,
                            input_data=Path(__file__).parent.parent / "data" / "test_dataframe.hdf5")
    # The report records the default split as given - it should read as (train, test, validate) fractions
    assert dict(experiment)["split_data_ratios"] == [0.8, 0.1, 0.1]

    filename = tmp_path / "test-experiment.yaml"
    experiment.save(filename=filename)

    with pytest.raises(FileNotFoundError):
        Experiment.from_file("this-is-not-the-right-file")

    loaded_e = Experiment.from_file(filename)
    assert loaded_e == experiment

def test_experiment_run(tmp_path):
    print(f"Keras running with backend: {keras.backend.backend()}")

    pipeline = DataProcessingPipeline(name="ais_preparation",
                                      base_dir=tmp_path) \
        .add("cyclic", LatLonTransformer())
    features = ["lat_x", "lat_y", "lon_x", "lon_y"]

    data = AISTestData(1000)
    adf = AnnotatedDataFrame(dataframe=data.dataframe,
                             metadata=MetaData.from_dict(data=AISTestDataSpec.copy()))
    dataset_filename = tmp_path / "test.hdf5"
    adf.save(filename=dataset_filename)

    forecast_task = ForecastTask(
        label="forecast-ais-short-sequence",
        pipeline=pipeline, features=features,
        models=[ModelInstanceDescription(BaselineA, {}),
                ModelInstanceDescription(BaselineB, {}),
                ],
        group_column="mmsi",
        sequence_length=5,
        forecast_length=1,
        training_parameters=TrainingParameters(epochs=1,
                                               validation_steps=1)
    )

    experiment = Experiment(learning_task=forecast_task,
                            input_data=dataset_filename,
                            output_directory=tmp_path)
    report = experiment.run()
    print(report)

    # Wrong key column
    forecast_task = ForecastTask(
        label="forecast-ais-short-sequence",
        pipeline=pipeline, features=features,
        models=[ModelInstanceDescription(BaselineA, {}),
                ModelInstanceDescription(BaselineB, {}),
                ],
        group_column="no-valid-key-column",
        sequence_length=5,
        forecast_length=1,
        training_parameters=TrainingParameters(epochs=1,
                                               validation_steps=1)
    )

    experiment = Experiment(learning_task=forecast_task,
                            input_data=dataset_filename,
                            output_directory=tmp_path)
    with pytest.raises(RuntimeError, match="no column 'no-valid-key-column'"):
        experiment.run()

    # Wrong Task
    learning_task = LearningTask(
        label="a learning task",
        pipeline=pipeline, features=features,
        models=[ModelInstanceDescription(BaselineA, {}),
                ModelInstanceDescription(BaselineB, {}),
                ],
        training_parameters=TrainingParameters(epochs=1,
                                               validation_steps=1)
    )

    experiment = Experiment(learning_task=learning_task,
                            input_data=dataset_filename,
                            output_directory=tmp_path)
    with pytest.raises(NotImplementedError):
        experiment.run()


@pytest.mark.parametrize("number_of_groups, ratios", [(5, [0.5, 0.5]), (10, [1, 1, 1, 1]), (3, [0.8, 0.1, 0.1])])
def test_compute_train_test_validate_groups_rounding(number_of_groups, ratios):
    """Partition sizes add up to the number of groups, even if rounding each ratio does not."""
    df = pl.DataFrame({"id": list(range(number_of_groups))})
    adf = AnnotatedDataFrame(df, metadata=AnnotatedDataFrame.infer_annotation(df.lazy()))
    partitions = Experiment.compute_train_test_validate_groups(adf, group="id", ratios=ratios)

    assert sorted(pl.concat(partitions)["id"].to_list()) == list(range(number_of_groups))
    exact_sizes = [number_of_groups * r / sum(ratios) for r in ratios]
    assert all(abs(len(p) - size) < 1 for p, size in zip(partitions, exact_sizes))


def _temporal_task(pipeline, **kwargs) -> TemporalForecastTask:
    arguments = dict(label="forecast-ais-temporal",
                     pipeline=pipeline, features=["lat_x", "lat_y", "lon_x", "lon_y"],
                     models=[ModelInstanceDescription(BaselineA, {}),
                             ModelInstanceDescription(BaselineB, {})],
                     group_column="mmsi",
                     timestamp_column="date_time_utc",
                     window="10m", sequence_length=5,
                     forecast_horizon="2m", forecast_length=1,
                     max_gap="5m",
                     training_parameters=TrainingParameters(epochs=1, validation_steps=1))
    arguments.update(kwargs)
    return TemporalForecastTask(**arguments)


def test_temporal_forecast_task_io(tmp_path):
    pipeline = DataProcessingPipeline(name="ais_preparation", base_dir=tmp_path) \
        .add("cyclic", LatLonTransformer())
    task = _temporal_task(pipeline)

    loaded_task = LearningTask.from_dict(data=dict(task))
    assert isinstance(loaded_task, TemporalForecastTask)
    assert loaded_task == task
    assert loaded_task != _temporal_task(pipeline, window="20m")

    experiment = Experiment(learning_task=task, batch_size=10, input_data=tmp_path / "unused.parquet")
    filename = tmp_path / "temporal-experiment.yaml"
    experiment.save(filename=filename)
    assert Experiment.from_file(filename) == experiment

    with pytest.raises(TypeError, match="window must be a duration string or number"):
        _temporal_task(pipeline, window=datetime.timedelta(minutes=10))


def test_temporal_experiment_run(tmp_path):
    pipeline = DataProcessingPipeline(name="ais_preparation", base_dir=tmp_path) \
        .add("cyclic", LatLonTransformer())

    spec = copy.deepcopy(AISTestDataSpec)
    for column in spec["columns"]:
        if column["name"] == "date_time_utc":
            column["representation_type"] = "datetime"
    data = AISTestData(200)
    adf = AnnotatedDataFrame(dataframe=data.dataframe.with_columns(pl.col("date_time_utc").str.to_datetime()),
                             metadata=MetaData.from_dict(data=spec))
    dataset_filename = tmp_path / "test.parquet"
    adf.export(filename=dataset_filename)

    experiment = Experiment(learning_task=_temporal_task(pipeline),
                            input_data=dataset_filename,
                            output_directory=tmp_path)
    report = experiment.run()
    assert Path(report).exists()
    assert set(experiment._evaluation_report) == {"BaselineA-forecast-ais-temporal",
                                                  "BaselineB-forecast-ais-temporal"}


# --- classification, fixed-window evaluation, evaluation on other data

CLASSES = ["even", "odd"]


class ClassifierBaseline(Baseline):
    """One probability per class."""

    def _init_model(self):
        self.model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[self.timeline_length, len(self.features)]),
            keras.layers.Dense(len(self.targets), activation="softmax")
        ])


class StepBaseline(Baseline):
    """A forecast of ``forecast_length`` steps of all targets."""

    def __init__(self, forecast_length: int, **kwargs):
        self.forecast_length = forecast_length
        super().__init__(**kwargs)

    def _init_model(self):
        self.model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[self.timeline_length, len(self.features)]),
            keras.layers.Dense(self.forecast_length * len(self.targets)),
            keras.layers.Reshape((self.forecast_length, len(self.targets)))
        ])


def _temporal_dataset(path: Path, size: int = 200) -> Path:
    """AIS test data with a datetime column, and a label per vessel: the parity of its mmsi."""
    spec = copy.deepcopy(AISTestDataSpec)
    for column in spec["columns"]:
        if column["name"] == "date_time_utc":
            column["representation_type"] = "datetime"
    spec["columns"].append({"name": "parity", "is_optional": False, "representation_type": "str"})
    data = AISTestData(size).dataframe.with_columns(
        pl.col("date_time_utc").str.to_datetime(),
        parity=pl.when(pl.col("mmsi") % 2 == 0).then(pl.lit("even")).otherwise(pl.lit("odd")))
    adf = AnnotatedDataFrame(dataframe=data, metadata=MetaData.from_dict(data=spec))
    adf.export(filename=path)
    return path


def _classification_task(pipeline, **kwargs):
    from damast.ml.experiments import ClassificationTask

    arguments = dict(label="classify-parity", pipeline=pipeline,
                     features=["lat_x", "lat_y", "lon_x", "lon_y"],
                     models=[ModelInstanceDescription(ClassifierBaseline, {})],
                     label_column="parity", classes=CLASSES,
                     group_column="mmsi", timestamp_column="date_time_utc",
                     window="10m", sequence_length=5, max_gap="5m",
                     training_parameters=TrainingParameters(epochs=1, validation_steps=1, optimizer="adam",
                                                            learning_rate=0.01))
    arguments.update(kwargs)
    return ClassificationTask(**arguments)


def _pipeline(tmp_path):
    return DataProcessingPipeline(name="ais_preparation", base_dir=tmp_path).add("cyclic", LatLonTransformer())


def test_classification_task_io(tmp_path):
    from damast.ml.experiments import ClassificationTask

    task = _classification_task(_pipeline(tmp_path))

    assert task.targets == CLASSES
    assert task.default_loss == "categorical_crossentropy"
    loaded = LearningTask.from_dict(data=dict(task))
    assert isinstance(loaded, ClassificationTask) and loaded == task
    assert loaded != _classification_task(_pipeline(tmp_path), classes=["odd", "even"])

    experiment = Experiment(learning_task=task, input_data=tmp_path / "unused.parquet", seed=3,
                            max_evaluation_windows=10, evaluation_steps=None)
    experiment.save(filename=tmp_path / "experiment.yaml")
    assert Experiment.from_file(tmp_path / "experiment.yaml") == experiment

    with pytest.raises(ValueError, match="differ from the classes"):
        _classification_task(_pipeline(tmp_path), targets=["odd", "even"])


def test_training_parameters_of_an_older_file_still_load():
    parameters = TrainingParameters(**{"epochs": 1, "steps_per_epoch": 2, "validation_steps": 3,
                                       "learning_rate": 0.5, "loss_function": "mse"})

    assert (parameters.loss_function, parameters.optimizer, parameters.metrics) == ("mse", "sgd", None)
    # without a loss, the task's default applies
    assert TrainingParameters().loss_function is None and LearningTask.default_loss == "mse"


def test_split_repeats_with_the_seed_and_keeps_each_stratum_in_proportion():
    from damast.utils import set_random_seed

    df = pl.DataFrame({"id": list(range(100))})
    adf = AnnotatedDataFrame(df, metadata=AnnotatedDataFrame.infer_annotation(df.lazy()))
    strata = {i: "rare" if i < 20 else "common" for i in range(100)}

    def split():
        set_random_seed(5)
        return Experiment.compute_train_test_validate_groups(adf, group="id", ratios=[0.6, 0.2, 0.2],
                                                             strata=strata)

    first, second = split(), split()
    assert all(a.equals(b) for a, b in zip(first, second))
    rare_per_partition = [sum(strata[i] == "rare" for i in p["id"].to_list()) for p in first]
    assert rare_per_partition == [12, 4, 4]
    assert sorted(pl.concat(first)["id"].to_list()) == list(range(100))


def test_classification_experiment_run(tmp_path):
    dataset = _temporal_dataset(tmp_path / "test.parquet")

    def run(output):
        experiment = Experiment(learning_task=_classification_task(_pipeline(tmp_path)), input_data=dataset,
                                output_directory=output, batch_size=16, evaluation_steps=None, seed=11,
                                max_evaluation_windows=200)
        return Path(experiment.run()), experiment

    report, experiment = run(tmp_path / "first")
    run_dir = report.parent

    result = experiment._evaluation_report["ClassifierBaseline-classify-parity"]
    metrics = result["metrics"]
    assert metrics["classes"] == CLASSES
    assert sum(map(sum, metrics["confusion_matrix"])) == result["windows"] == metrics["windows"] > 0
    predictions = pl.read_parquet(run_dir / "ClassifierBaseline-classify-parity" / "predictions-test.parquet")
    assert predictions.height == result["windows"]
    assert {"mmsi", "window_start", "true_class", "predicted_class", "probability_even", "last_lat_x"} \
        <= set(predictions.columns)
    # every window of the test vessels, labelled with its vessel's parity
    groups = pl.read_parquet(run_dir / "groups.parquet")
    test_vessels = groups.filter(pl.col("partition") == "test")["mmsi"]
    assert predictions["mmsi"].is_in(test_vessels.implode()).all()
    expected = ["even" if mmsi % 2 == 0 else "odd" for mmsi in predictions["mmsi"].to_list()]
    assert predictions["true_class"].to_list() == expected
    assert (run_dir / "ais_preparation.damast.ppl").exists() or list(run_dir.glob("*.damast.ppl"))

    # the same seed, the same split
    second_report, _ = run(tmp_path / "second")
    assert pl.read_parquet(second_report.parent / "groups.parquet").equals(groups)


def test_temporal_forecast_evaluated_on_fixed_windows(tmp_path):
    dataset = _temporal_dataset(tmp_path / "test.parquet")
    task = _temporal_task(_pipeline(tmp_path), targets=["lat", "lon"], forecast_length=2,
                          target_reference="last_input", step_features=["lat_x", "lat_y", "lon_x", "lon_y"],
                          models=[ModelInstanceDescription(StepBaseline, {"forecast_length": 2})])
    experiment = Experiment(learning_task=task, input_data=dataset, output_directory=tmp_path,
                            batch_size=16, evaluation_steps=None, seed=2)

    run_dir = Path(experiment.run()).parent

    metrics = experiment._evaluation_report["StepBaseline-forecast-ais-temporal"]["metrics"]
    assert [step["step"] for step in metrics["steps"]] == [1, 2]
    predictions = pl.read_parquet(run_dir / "StepBaseline-forecast-ais-temporal" / "predictions-test.parquet")
    assert {"true_lat_1", "predicted_lat_1", "true_lon_2", "predicted_lon_2"} <= set(predictions.columns)


def test_fixed_windows_need_a_window_task(tmp_path):
    task = ForecastTask(label="rows", pipeline=_pipeline(tmp_path), features=["lat_x", "lat_y", "lon_x", "lon_y"],
                        models=[ModelInstanceDescription(BaselineA, {})], group_column="mmsi",
                        sequence_length=5, forecast_length=1)
    experiment = Experiment(learning_task=task, input_data=_temporal_dataset(tmp_path / "test.parquet"),
                            output_directory=tmp_path, evaluation_steps=None)

    with pytest.raises(ValueError, match="requires a window task"):
        experiment.run()


def test_evaluate_experiment_on_other_data(tmp_path):
    dataset = _temporal_dataset(tmp_path / "train.parquet")
    experiment = Experiment(learning_task=_classification_task(_pipeline(tmp_path)), input_data=dataset,
                            output_directory=tmp_path / "out", batch_size=16, evaluation_steps=None, seed=4)
    run_dir = Path(experiment.run()).parent
    trained = pl.read_parquet(run_dir / "groups.parquet").filter(pl.col("partition") != "test")["mmsi"]

    # other data: the same vessels again (a source reporting them too) and some new ones
    other = _temporal_dataset(tmp_path / "other.parquet", size=50)
    filename = Experiment.evaluate_experiment(run_dir, input_data=[other, dataset], dataset_label="other",
                                              exclude_groups=trained.to_list(), seed=1)

    import yaml
    report = yaml.safe_load(Path(filename).read_text())
    assert filename == run_dir / "evaluations" / "other" / "evaluation.yaml"
    assert report["dataset"] == "other" and report["excluded_groups"] == trained.len()
    result = report["evaluation"]["ClassifierBaseline-classify-parity"]
    assert result["windows"] == report["windows"] > 0
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == report["windows"]
    predictions = pl.read_parquet(Path(filename).parent / "ClassifierBaseline-classify-parity" / "predictions-other.parquet")
    # no window of a vessel the models were trained or validated on
    assert not predictions["mmsi"].is_in(trained.implode()).any()


def test_evaluation_csv_collects_every_evaluation(tmp_path):
    dataset = _temporal_dataset(tmp_path / "test.parquet")
    experiment = Experiment(learning_task=_temporal_task(_pipeline(tmp_path), models=[ModelInstanceDescription(BaselineA, {})]),
                            input_data=dataset, output_directory=tmp_path)
    experiment.run()
    model = experiment._trained_models[0]
    assert len(model.get_evaluations()) == 1

    features = experiment.learning_task.pipeline.transform(AnnotatedDataFrame.from_file(dataset)).execute()
    generator = experiment.create_generator(features, group="mmsi", group_ids=None)
    model.evaluate(label="again", evaluation_data=generator, steps=1)

    evaluations = model.get_evaluations()
    assert len(evaluations) == 2 and list(evaluations["dataset"]) == [model.name, "again"]

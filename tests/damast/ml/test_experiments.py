import keras
import tensorflow
import pytest
from collections import OrderedDict
from pathlib import Path

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.ml.models.base import BaseModel
from damast.ml.experiments import Experiment, LearningTask, ForecastTask, ModelInstanceDescription
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement


class ModelA(BaseModel):
    def __init_model(self):
        pass


class TransformerA(PipelineElement):
    @damast.input({"x": {}})
    @damast.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


@pytest.fixture()
def experiment_dir(tmp_path):
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

        def __init__(self):
            super().__init__(features=["a", "b", "c", "d"],
                             targets=["a", "b", "c", "d"],
                             output_dir=tmp_path)

        def _init_model(self):
            inputs = keras.Input(shape=(4,),
                                 name="input",
                                 dtype=tensorflow.float32)
            outputs = keras.layers.Dense(4)(inputs)

            self.model = keras.Model(inputs=inputs,
                                     outputs=outputs,
                                     name=self.__class__.__name__)

    class ModelA(SimpleModel):
        pass

    class ModelB(SimpleModel):
        pass

    Experiment.touch_marker(tmp_path)

    model_a = ModelA()
    model_a.save()

    model_b = ModelB()
    model_b.save()

    return tmp_path


def test_validate_experiment_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Experiment.validate_experiment_dir(dir="UNKNOWN_DIR")

    with pytest.raises(NotADirectoryError):
        filename = tmp_path / "dummy-file"
        with open(filename, "w") as f:
            pass

        Experiment.validate_experiment_dir(dir=filename)

    with pytest.raises(NotADirectoryError, match="is not an experiment"):
        Experiment.validate_experiment_dir(dir=tmp_path)

    # Validate touch
    with pytest.raises(FileNotFoundError):
        Experiment.touch_marker(dir=f"{tmp_path}-does-not-exist")
    with pytest.raises(NotADirectoryError):
        file_in_dir = tmp_path / "test_file"
        with open(file_in_dir, "a") as f:
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

    lt = LearningTask(pipeline=pipeline,
                      features=["a", "b", "c", "d"],
                      models=models)

    data_dict = dict(lt)
    loaded_t = LearningTask.from_dict(data=data_dict)
    assert loaded_t == lt


def test_to_and_from_file(tmp_path):
    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA())

    models = [
        ModelInstanceDescription(model=ModelA,
                                 parameters={})
    ]

    task = ForecastTask(
        pipeline=pipeline,
        features=["a", "b", "c", "d"],
        models=models,
        group_column="a",
        sequence_length=50,
        forecast_length=1
    )

    experiment = Experiment(learning_task=task,
                            batch_size=10,
                            input_data=Path(__file__).parent.parent / "data" / "test_dataframe.hdf5")

    filename = tmp_path / "test-experiment.yaml"
    experiment.save(filename=filename)
    loaded_e = Experiment.from_file(filename)

    assert loaded_e == experiment


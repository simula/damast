import keras
import tensorflow
import pytest
from collections import OrderedDict

from damast.ml.models.base import BaseModel
from damast.ml.experiments import Experiment


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
    Experiment.from_directory(experiment_dir)

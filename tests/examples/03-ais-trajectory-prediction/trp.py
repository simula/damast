import os
from collections import OrderedDict
from pathlib import Path
import numpy as np

import keras
import pytest

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement
from damast.core.datarange import MinMax
from damast.core.metadata import MetaData
from damast.core.transformations import MultiCycleTransformer
from damast.core.units import units
from damast.domains.maritime.ais.data_generator import AISTestData, AISTestDataSpec
from damast.ml.experiments import (
    Experiment,
    ForecastTask,
    LearningTask,
    ModelInstanceDescription,
    TrainingParameters,
    )
from damast.ml.models.base import BaseModel
import tempfile

os.environ["COLUMNS"] = '120'


tmp_path = Path(tempfile.gettempdir()) / "damast-example-03"
tmp_path.mkdir(parents=True, exist_ok=True)


class ModelA(BaseModel):
    def __init_model(self):
        pass


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


pipeline = DataProcessingPipeline(name="ais_preparation",
                                  base_dir=tmp_path) \
    .add("cyclic", LatLonTransformer())

features = ["lat_x", "lat_y", "lon_x", "lon_y"]

data = AISTestData(1000)
adf = AnnotatedDataFrame(dataframe=data.dataframe,
                         metadata=MetaData.from_dict(data=AISTestDataSpec.copy()))

dataset_filename = tmp_path / "test.parquet"
adf.save(filename=dataset_filename)


# TRAINING (including data preprocessing)
forecast_task = ForecastTask(
    label="forecast-ais-short-sequence",
    pipeline=pipeline, features=features,
    models=[ModelInstanceDescription(BaselineA, {})],
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


# RELOAD MODEL (one of the trained ones)
models = Experiment.from_directory(report.parent)
current_model = models["BaselineA-forecast-ais-short-sequence"]

# FORECAST
# .. reusing initial dataset here for simplicity
processed_data = AnnotatedDataFrame.from_file(dataset_filename)
input_adf = pipeline.transform(processed_data)

# creating input features
from damast.data_handling.accessors import SequenceIterator
sta = SequenceIterator(df=input_adf)
gen_predict = sta.to_keras_generator(features=features, target=features, sequence_length=5)

data = next(gen_predict)
X, y = data
input_data = X[np.newaxis, :, :]
predicted_sequence = current_model.predict(input_data, steps=1, verbose=0)

print(f"Input Sequence: {input_data}")
print(f"Predicted Sequence: {predicted_sequence}")

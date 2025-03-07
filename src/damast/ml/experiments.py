"""
Module containing functionality to setup up an Machine-Learning Experiment
"""
from __future__ import annotations

import datetime
import importlib
import os
import random
import tempfile
from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

import keras
import numpy as np
import polars as pl
import yaml

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.types import DataFrame
from damast.data_handling.accessors import GroupSequenceAccessor
from damast.ml.models.base import BaseModel, ModelInstanceDescription

basicConfig()
_log: Logger = getLogger(__name__)

__all__ = [
    "Experiment",
    "ForecastTask",
    "LearningTask",
    "ModelInstanceDescription",
    "TrainingParameters"
]


class TrainingParameters(NamedTuple):
    epochs: int = 10
    steps_per_epoch: int = 10
    validation_steps: int = 12
    learning_rate: float = 0.1
    loss_function: str = "mse"


class LearningTask:
    """
    Description of a trainable Learning Task.

    This class can associate a processing pipeline with a set of model to train and evaluation.

    :param label: Label of this learning task (e.g. to uniquely mark the combination of task and learning parameters)
           This label will be used to suffix the model-specific subfolder
    :param pipeline: The processing / feature extraction pipeline
    :param features: The features that this machine-learning model requires as input
    :param targets: The output features aka targets that this machine-learning model provides
    :param models: The actual model instances, i.e. model + parameters, that this learning
           task shall comprise
    """
    label: str
    pipeline: DataProcessingPipeline
    features: List[str]
    targets: List[str]

    models: List[ModelInstanceDescription]
    training_parameters: Union[Dict[str, Any], TrainingParameters] = TrainingParameters(),

    def __init__(self, *,
                 label: str,
                 pipeline: Union[Dict[str, Any], DataProcessingPipeline],
                 features: List[str],
                 models: List[Union[Dict[str, Any], ModelInstanceDescription]],
                 targets: Optional[List[str]] = None,
                 training_parameters: Union[Dict[str, Any], TrainingParameters] = TrainingParameters(),
                 ):

        self.label = label

        if isinstance(pipeline, DataProcessingPipeline):
            self.pipeline = pipeline
        elif isinstance(pipeline, dict):
            self.pipeline = DataProcessingPipeline(**pipeline)
        else:
            raise ValueError(f"{self.__class__.__name__}.__init__: could not instantiate DataProcessingPipeline"
                             f" from {type(pipeline)}")

        self.features = features
        if targets is None:
            self.targets = self.features
        else:
            self.targets = targets

        self.models = []
        for m in models:
            if isinstance(m, ModelInstanceDescription):
                self.models.append(m)
            elif isinstance(m, dict):
                self.models.append(ModelInstanceDescription.from_dict(data=m))
            else:
                raise ValueError(f"{self.__class__.__name__}.__init__: could not instantiate ModelInstanceDescription"
                                 f" from {type(m)}")

        if isinstance(training_parameters, TrainingParameters):
            self.training_parameters = training_parameters
        elif isinstance(training_parameters, dict):
            self.training_parameters = TrainingParameters(**training_parameters)
        else:
            raise ValueError(f"{self.__class__.__name__}.__init__: training_parameters must be either "
                             f"dict or TrainingParameters object")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if "module_name" not in data:
            raise KeyError(f"{cls.__name__}.create: missing 'module_name'")

        if "class_name" not in data:
            raise KeyError(f"{cls.__name__}.create: missing 'class_name'")

        module_name = data["module_name"]
        class_name = data["class_name"]

        m_module = importlib.import_module(module_name)
        if hasattr(m_module, class_name):
            del data["module_name"]
            del data["class_name"]

            klass = getattr(m_module, class_name)
            return klass(**data)

        raise ValueError(f"{cls.__name__}.from_dict: could not find '{class_name}' in '{module_name}'")

    def __iter__(self):
        yield "module_name", self.__class__.__module__
        yield "class_name", self.__class__.__qualname__

        yield "label", self.label
        yield "pipeline", dict(self.pipeline)
        yield "features", self.features
        yield "targets", self.targets
        yield "models", [dict(x) for x in self.models]

        yield "training_parameters", self.training_parameters._asdict() # type: ignore

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False

        for attr in ["pipeline", "features", "targets", "models"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class ForecastTask(LearningTask):
    sequence_length: int
    forecast_length: int
    group_column: str

    def __init__(self, *,
                 label: str,
                 pipeline: Union[Dict[str, Any], DataProcessingPipeline],
                 group_column: str,
                 features: List[str],
                 sequence_length: int,
                 forecast_length: int,
                 models: List[Union[Dict[str, Any], ModelInstanceDescription]],
                 targets: Optional[List[str]] = None,
                 training_parameters: Union[Dict[str, Any], TrainingParameters] = TrainingParameters(),
                 ):
        super().__init__(label=label,
                         pipeline=pipeline,
                         features=features,
                         models=models,
                         targets=targets,
                         training_parameters=training_parameters)

        self.group_column = group_column
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        for attr in ["pipeline", "features", "targets", "models", "group_column", "sequence_length",
                     "forecast_length", "training_parameters"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __iter__(self):
        for x in super().__iter__():
            yield x

        yield "group_column", self.group_column
        yield "sequence_length", self.sequence_length
        yield "forecast_length", self.forecast_length


class Experiment:
    MARKER_FILE: str = ".damast_experiment"
    TIMESTAMP_FORMAT: str = "%Y%m%d-%H%M%S"

    learning_task: LearningTask

    input_data: Path
    output_directory: Path
    label: str

    _batch_size: int
    _split_data_ratios: List[float]

    _timestamp: datetime.datetime
    _evaluation_steps: int
    _evaluation_report: Dict[str, Dict[str, Any]]

    _trained_models: List[BaseModel]

    def __init__(self,
                 learning_task: Union[Dict[str, Any], LearningTask],
                 input_data: Union[str, Path],
                 output_directory: Union[str, Path] = tempfile.gettempdir(),
                 batch_size: int = 2,
                 evaluation_steps=1,
                 split_data_ratios: List[float] = [1.6, 0.2, 0.2],
                 label: str = "damast-ml-experiment",
                 timestamp: Union[str, datetime.datetime] = datetime.datetime.utcnow(),
                 evaluation={}
                 ):
        """
        The ratios of how to split (test, training, validation) data from the input dataset

        :param learning_task:  A description of the machine-learning model setup
        :param input_data:
        :param evaluation_steps:
        :param split_data_ratios:
        :param label:
        :param timestamp:
        :param evaluation:
        :param batch_size: The batch-size
        :param output_directory: Path to direct output to
        """
        if isinstance(learning_task, LearningTask):
            self.learning_task = learning_task
        elif isinstance(learning_task, dict):
            self.learning_task = LearningTask.from_dict(data=learning_task)
        else:
            raise ValueError(f"{self.__class__.__name__}.__init__: learning_task must be either"
                             f"dict or LearningTask object")

        self.input_data = Path(input_data)
        self.output_directory = Path(output_directory)
        self.label = label

        self._batch_size = batch_size
        self._evaluation_steps = evaluation_steps
        self._split_data_ratios = split_data_ratios
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.strptime(timestamp, Experiment.TIMESTAMP_FORMAT)

        self._timestamp = timestamp
        self._evaluation_report = evaluation
        self._trained_models = []

    @classmethod
    def from_file(cls,
                  filename: Union[str, Path]) -> Experiment:
        if not Path(filename).exists():
            raise FileNotFoundError(f"{cls.__name__}.from_file: the given experiment description file"
                                    f" does not exist: '{filename}'")

        with open(filename, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            return cls(**data)

    def save(self,
             filename: Union[str, Path]):
        """
        Save the experiment as yaml.

        :param filename: Name of the file
        """
        with open(filename, "w") as f:
            yaml.dump(dict(self), f)

    @classmethod
    def create_experiment_directory(cls,
                                    base_dir: Union[str, Path],
                                    label: str) -> Path:
        """Create an experiment directory inside the output directory.

        The created directory is prefixed with a time-stamp indicating when it was created.

        """
        prefix = datetime.datetime.utcnow().strftime(cls.TIMESTAMP_FORMAT)
        experiment_dir = base_dir / f"{prefix}-{label}"
        experiment_dir.mkdir(exist_ok=False, parents=True)

        Experiment.touch_marker(experiment_dir)

        return experiment_dir

    @classmethod
    def validate_experiment_dir(cls, dir: Union[str, Path]) -> Path:
        """
        Validate that a particular directory is an "experiment" directory.

        Experiment directories will have a hidden marker file

        :param dir: Path or name of the directorry
        :return: Directory Path
        """
        experiment_dir = Path(dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"{cls.__name__}.from_directory: {dir} does not exist")

        if not experiment_dir.is_dir():
            raise NotADirectoryError(f"{cls.__name__}.from_directory: {dir}")

        if not (experiment_dir / cls.MARKER_FILE).exists():
            raise NotADirectoryError(f"{cls.__name__}.from_directory: {dir} is not an experiment result directory")

        return experiment_dir

    @classmethod
    def from_directory(cls, dir: Union[str, Path]) -> Dict[str, 'keras.Model']:
        """
        Create an experiment object by loading a directory with experiment artifacts.

        Note that currently keras models will be loaded, when available.

        :param dir: Directory with experiment artifacts
        :return: Dictionary of loaed
        """
        experiment_dir = cls.validate_experiment_dir(dir=dir)

        models = {}

        import keras.models

        from damast.ml.models.base import MODEL_TF_HDF5

        # Load available models
        for dirname in os.listdir(str(experiment_dir)):
            model_hdf5 = experiment_dir / dirname / MODEL_TF_HDF5
            if model_hdf5.exists():
                models[Path(dirname).stem] = keras.models.load_model(model_hdf5)

        return models

    @classmethod
    def touch_marker(cls, dir: Union[str, Path]):
        """
        Create a marker file in a given directory.

        :param dir: Directory where to create the marker file
        """
        path = Path(dir)
        if not path.exists():
            raise FileNotFoundError(f"{cls.__name__}.touch_marker: {path} does not exist")
        elif not path.is_dir():
            raise NotADirectoryError(f"{cls.__name__}.touch_marker: {path} is not a directory")

        with open(path / cls.MARKER_FILE, "a"):
            pass

    @classmethod
    def compute_train_test_validate_groups(cls,
                                           adf: AnnotatedDataFrame,
                                           group: str,
                                           ratios: List[float]) -> List[List[Any]]:
        """
        Given a :class:`damast.AnnotatedDataFrame` and a column name, split the dataframe into groups
        based on the column name. The size of each group is determined by the ratio.

        ..note::
            This function does not modify anything in the incoming dataframe.

        :param adf: The available data in the input data frame
        :param group: In case this deals with timeseries / sequence data, the group will relate to multiple rows
        :param ratios: Ratio between the train, test and validate dataset
        """

        normalized_rates = np.asarray([rate / sum(ratios) for rate in ratios])
        groups = adf[group].unique().select(pl.col(group).shuffle().alias(group)).collect()

        partition_sizes = np.asarray(
            np.round(len(groups) * normalized_rates), dtype=int)
        delta = len(groups) - sum(partition_sizes)
        assert delta <= 1, f"|# of groups {len(groups)} - # of partitions {sum(partition_sizes)}| <= 1"
        if delta == 1:
            rand_idx = random.randint(0, len(partition_sizes) - 1)
            partition_sizes[rand_idx] += 1
        assert len(groups) == sum(partition_sizes)

        from_idx = 0
        partitions = []
        for ps in partition_sizes:
            to_idx = min(from_idx + ps, len(groups))
            partitions.append(groups[from_idx:to_idx])
            from_idx = to_idx
        return partitions

    def compute_features(self,
                         adf: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Compute features for an experiment by running the designated pipeline on input dataframe.

        .. note::
            Stores the transformed data in the experiment directory.

        :param adf: The input dataframe
        """
        adf_with_features = self.learning_task.pipeline.transform(adf)

        self.learning_task.pipeline.save(dir=self.output_directory)
        self.learning_task.pipeline.save_state(
            df=adf_with_features,
            dir=self.output_directory)
        return adf_with_features

    def create_generator(self,
                         adf: AnnotatedDataFrame,
                         group: str,
                         group_ids: List[Any]) -> Sequence:
        """
        Create a generator applicable for :py:mod:`keras` from the input data.

        ..note::
            This function is called whenever you want to train your models, i.e
            :func:`Experiment.train`

        :param adf: The input dataframe
        :param group: Group to extract sequences from
        :param group_ids: Groups used in the generator
        :returns: A triplet of (train, test, validation) generators of the input features
        """

        sta = GroupSequenceAccessor(df=adf,
                                    group_column=group)
        if isinstance(self.learning_task, ForecastTask):
            return sta.to_keras_generator(features=self.learning_task.features,
                                          target=self.learning_task.targets,
                                          sequence_length=self.learning_task.sequence_length,
                                          sequence_forecast=self.learning_task.forecast_length,
                                          groups=group_ids,
                                          batch_size=self._batch_size,
                                          infinite=True)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.create_generator: Sorry, but there is currently no support "
                f"implemented for {self.learning_task.__class__.__name__}")

    def train(self,
              model_instance_description: ModelInstanceDescription,
              train_generator: Sequence,
              validate_generator: Sequence,
              output_dir: Path,
              suffix: str = "",
              epochs: int = 2,
              steps_per_epoch: int = 10,
              validation_steps: int = 12,
              learning_rate: float = 0.1,
              loss_function: str = "mse") -> BaseModel:
        """
        Train a set of machine-learning models on given input data.
        :param suffix: allow to add a training specific suffix to the output folder (otherwise) only the model class name will be used, so only the last training would be kept
        :param model_instance_description: The model instance description for the model that should be trained
        :param train_generator: Generator providing data from training set
        :param validate_generator: Generator providing data from validation set
        :param output_dir: Directory where the model training log, etc. should go
        :param epochs: Number of epochs
        :param steps_per_epoch: Steps per epoch
        :param validation_steps: Number of validation steps
        :param learning_rate: The learning rate for the Stochastic Gradient descent algorithm
        :param loss_function: The loss function

        :return: model instance
        """
        # NOTE: This should probably be an individual step, as we could re-use an existing model to continue training
        if isinstance(self.learning_task, ForecastTask):
            # TODO: the model needs to be applicable to a forecast task - so we might
            model: BaseModel = model_instance_description.model(
                name=f"{model_instance_description.model.__name__}{suffix}",
                output_dir=output_dir,
                features=self.learning_task.features,
                targets=self.learning_task.targets,
                timeline_length=self.learning_task.sequence_length,
                **model_instance_description.parameters
            )
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.create_generator: Sorry, but there is currently no support "
                f"implemented for {self.learning_task.__class__.__name__}")

        model.build(loss_function=loss_function,
                    optimizer=keras.optimizers.SGD(learning_rate=learning_rate))

        # Train model
        model.train(training_data=train_generator,
                    validation_data=validate_generator,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps)
        model.save()
        return model

    def evaluate(self,
                 test_generator: Sequence,
                 steps: int = 1) -> Dict[str, Dict[str, Any]]:

        return Experiment.evaluate_models(models=self._trained_models,
                                          test_generator=test_generator,
                                          steps=steps)

    @classmethod
    def evaluate_models(cls,
                        models: List[BaseModel],
                        test_generator: Sequence,
                        steps: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a list of trained models.

        :param models: List of trained models
        :param test_generator:  The generator providing access to the test data
        :param steps: number of steps that should be used to test the models

        :return: a dictionary mapping the model name to the results, e.g.
                { "Baseline": { "mse": 0.1, "loss": 0.2 } }
        """
        assert models is not None and len(models) > 0

        evaluations_results = {}
        for model in models:
            evaluation_result = model.evaluate(label=model.name,
                                               evaluation_data=test_generator,
                                               steps=steps)
            evaluations_results[model.name] = evaluation_result
        return evaluations_results

    def run(self,
            logging_level: int = INFO,
            report_filename: Optional[Union[str, Path]] = None) -> DataFrame:
        """
        Run the experiment and return evaluation data.

        The data will also be written into a report file into the current experiment folder unless
        specified otherwise.

        :param logging_level: The logging level that should be applied when running this experiment
        :param report_filename: The report will be written into the experiment folder as experiment dictionary including
            and an additional "evaluation" key

        :return: the evaluation data
        """
        _log.setLevel(logging_level)

        # Load data and validate it by loading it into the annotated dataframe
        # We remove values that are outside the Min/Max value defined in the specification
        adf = AnnotatedDataFrame.from_file(self.input_data)

        if not isinstance(self.learning_task, ForecastTask):
            raise NotImplementedError(
                f"{self.__class__.__name__}.run: Sorry, but there is currently only support "
                f"implemented for ForecastTask")

        group_column = self.learning_task.group_column
        if group_column not in adf.column_names:
            raise RuntimeError(f"{self.__class__.__name__}.run: no column '{group_column}' in the given dataset "
                               f"'{self.input_data}' -- found only '{','.join(adf.column_names)}'")

        # If a learning task is used, that requires a sequence length, then filter the data, so that
        # only relevant data (with a minimum length of the given sequence length) will be accounted
        # for this learning task
        if hasattr(self.learning_task, "sequence_length"):
            groups_with_sequence_length = adf.dataframe.group_by(group_column).agg(sequence_length=pl.len())

            filtered_groups = groups_with_sequence_length.filter(pl.col("sequence_length") > self.learning_task.sequence_length)
            permitted_values = filtered_groups.select(group_column).unique().collect()[:,0]
            adf._dataframe = adf.filter(pl.col(group_column).is_in(permitted_values))

        features = self.compute_features(adf)
        train_group, test_group, validate_group = \
            self.compute_train_test_validate_groups(features,
                                                    group=group_column,
                                                    ratios=self._split_data_ratios)

        experiment_dir = self.create_experiment_directory(base_dir=self.output_directory,
                                                          label=self.label)

        self._trained_models = []
        for model_instance_description in self.learning_task.models:
            train_data_gen = self.create_generator(features,
                                                   group=group_column,
                                                   group_ids=train_group)
            validate = self.create_generator(features,
                                             group=group_column,
                                             group_ids=validate_group)

            model = self.train(
                suffix=f"-{self.learning_task.label}",
                model_instance_description=model_instance_description,
                train_generator=train_data_gen,
                validate_generator=validate,
                output_dir=experiment_dir,
                **self.learning_task.training_parameters._asdict()  # noqa
            )
            self._trained_models.append(model)

        # TODO: This should return a 'REPORT / PROTOCOL' of the whole experiment
        test_data_gen = self.create_generator(features,
                                              group=group_column,
                                              group_ids=test_group)

        self._evaluation_report = self.evaluate(test_generator=test_data_gen,
                                                steps=self._evaluation_steps)

        self._timestamp = datetime.datetime.utcnow()

        if report_filename is None:
            filename = experiment_dir / "experiment-report.yaml"
        else:
            filename = report_filename

        self.save(filename=filename)
        return filename

    def __eq__(self, other):
        return dict(self) == dict(other)

    def __iter__(self):
        yield "label", self.label
        yield "input_data", str(self.input_data)
        yield "output_directory", str(self.output_directory)
        yield "learning_task", dict(self.learning_task)
        yield "batch_size", self._batch_size
        yield "evaluation_steps", self._evaluation_steps
        yield "split_data_ratios", self._split_data_ratios
        yield "timestamp", self._timestamp.strftime(self.TIMESTAMP_FORMAT)
        yield "evaluation", self._evaluation_report

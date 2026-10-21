"""
Module containing functionality to setup up an Machine-Learning Experiment
"""
from __future__ import annotations

import datetime
import importlib
import os
import tempfile
from collections.abc import Sequence
from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
import polars as pl
import yaml

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.types import DataFrame
from damast.data_handling.accessors import GroupSequenceAccessor, GroupWindowAccessor, partition_sizes
from damast.ml import keras
from damast.ml.evaluation import classification_metrics, forecast_metrics
from damast.ml.models.base import BaseModel, ModelInstanceDescription
from damast.utils import set_random_seed

basicConfig()
_log: Logger = getLogger(__name__)

__all__ = [
    "ClassificationTask",
    "Experiment",
    "ForecastTask",
    "LearningTask",
    "ModelInstanceDescription",
    "TemporalForecastTask",
    "TrainingParameters"
]


class TrainingParameters(NamedTuple):
    epochs: int = 10
    steps_per_epoch: int = 10
    validation_steps: int = 12
    learning_rate: float = 0.1
    #: Name of a keras loss, None for the learning task's :attr:`LearningTask.default_loss`
    loss_function: str | None = None
    #: Name of a keras optimizer, e.g. ``"sgd"`` or ``"adam"``
    optimizer: str = "sgd"
    #: Names of keras metrics, None for the learning task's :attr:`LearningTask.default_metrics`
    metrics: list[str] | None = None


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
    features: list[str]
    targets: list[str]

    models: list[ModelInstanceDescription]
    training_parameters: dict[str, Any] | TrainingParameters = TrainingParameters(),

    #: Loss and metrics of a training whose parameters name none
    default_loss: ClassVar[str] = "mse"
    default_metrics: ClassVar[list[str]] = ["mse"]

    def __init__(self, *,
                 label: str,
                 pipeline: dict[str, Any] | DataProcessingPipeline,
                 features: list[str],
                 models: list[dict[str, Any] | ModelInstanceDescription],
                 targets: list[str] | None = None,
                 training_parameters: dict[str, Any] | TrainingParameters = TrainingParameters(),
                 ):

        self.label = label

        if isinstance(pipeline, DataProcessingPipeline):
            self.pipeline = pipeline
        elif isinstance(pipeline, dict):
            self.pipeline = DataProcessingPipeline(**pipeline)
        else:
            raise TypeError(f"{self.__class__.__name__}.__init__: could not instantiate DataProcessingPipeline"
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
                raise TypeError(f"{self.__class__.__name__}.__init__: could not instantiate ModelInstanceDescription"
                                 f" from {type(m)}")

        if isinstance(training_parameters, TrainingParameters):
            self.training_parameters = training_parameters
        elif isinstance(training_parameters, dict):
            self.training_parameters = TrainingParameters(**training_parameters)
        else:
            raise TypeError(f"{self.__class__.__name__}.__init__: training_parameters must be either "
                             f"dict or TrainingParameters object")

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
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
        if type(self) is not type(other):
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
                 pipeline: dict[str, Any] | DataProcessingPipeline,
                 group_column: str,
                 features: list[str],
                 sequence_length: int,
                 forecast_length: int,
                 models: list[dict[str, Any] | ModelInstanceDescription],
                 targets: list[str] | None = None,
                 training_parameters: dict[str, Any] | TrainingParameters = TrainingParameters(),
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
        if type(self) is not type(other):
            return False

        for attr in ["pipeline", "features", "targets", "models", "group_column", "sequence_length",
                     "forecast_length", "training_parameters"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __iter__(self):
        yield from super().__iter__()

        yield "group_column", self.group_column
        yield "sequence_length", self.sequence_length
        yield "forecast_length", self.forecast_length


class TemporalForecastTask(ForecastTask):
    """
    Forecast task whose input window and forecast horizon are defined in time rather than by a number of rows.

    Sequences are generated by :class:`damast.data_handling.accessors.GroupWindowAccessor`: each input window
    of duration ``window`` (:math:`H^{-}`) is resampled to ``sequence_length`` (:math:`L^{-}`) equidistant
    points, and the ``forecast_horizon`` (:math:`H^{+}`) following it to ``forecast_length`` (:math:`L^{+}`)
    points. Windows are only drawn inside stretches in which no two consecutive rows lie further apart than
    ``max_gap``, so an interpolation never bridges a larger gap.

    Durations are given as duration string (e.g. ``"30m"``) if ``timestamp_column`` is a datetime, otherwise as a
    number in the unit of ``timestamp_column`` - :class:`datetime.timedelta` is not supported, since it cannot be
    stored in the experiment file.

    :param timestamp_column: Name of the time column
    :param window: Duration of the input window
    :param forecast_horizon: Duration between the end of the input window and the last forecast point
    :param max_gap: Maximum time between two consecutive rows that may be interpolated
    :param step_features: Features that hold the last row's value instead of being interpolated, e.g.
        so that no input looks ahead of its grid point
    :param target_reference: ``"absolute"`` for the targets as they are, ``"last_input"`` for their change
        since the end of the input window - e.g. a displacement from cumulative track coordinates
    """
    timestamp_column: str
    window: str | float
    forecast_horizon: str | float
    max_gap: str | float
    step_features: list[str] | None
    target_reference: str

    def __init__(self, *,
                 timestamp_column: str,
                 window: str | float,
                 forecast_horizon: str | float,
                 max_gap: str | float,
                 step_features: list[str] | None = None,
                 target_reference: str = "absolute",
                 **kwargs):
        super().__init__(**kwargs)
        self.step_features = step_features
        self.target_reference = target_reference

        for name, value in [("window", window), ("forecast_horizon", forecast_horizon), ("max_gap", max_gap)]:
            if isinstance(value, datetime.timedelta):
                raise TypeError(f"{self.__class__.__name__}.__init__: {name} must be a duration string or number,"
                                f" got a timedelta")

        self.timestamp_column = timestamp_column
        self.window = window
        self.forecast_horizon = forecast_horizon
        self.max_gap = max_gap

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ["timestamp_column", "window", "forecast_horizon", "max_gap",
                                "step_features", "target_reference"])

    def __iter__(self):
        yield from super().__iter__()

        yield "timestamp_column", self.timestamp_column
        yield "window", self.window
        yield "forecast_horizon", self.forecast_horizon
        yield "max_gap", self.max_gap
        yield "step_features", self.step_features
        yield "target_reference", self.target_reference


class ClassificationTask(LearningTask):
    """
    Classification of time windows of a group by a label that is constant per group, e.g. the type of a
    vessel from a window of its movement.

    Windows are generated as for :class:`TemporalForecastTask`, by
    :class:`damast.data_handling.accessors.GroupWindowAccessor`: each input window of duration ``window`` is
    resampled to ``sequence_length`` equidistant points, inside stretches without a gap larger than
    ``max_gap``. The target of a window is the one-hot encoded ``label_column`` of its group over
    ``classes``; the model has one output per class, in this order (:attr:`targets`). Groups whose label is
    missing or not one of the classes are left out; a group with more than one label is an error.

    The groups are split into the partitions stratified by their label, so that a rare class is present
    in all of them.

    :param label_column: Column of the label
    :param classes: The labels to learn, in the order of the model's outputs
    :param group_column: Column identifying a group
    :param timestamp_column: Name of the time column
    :param window: Duration of the input window
    :param sequence_length: Number of equidistant points the input window is resampled to
    :param max_gap: Maximum time between two consecutive rows that may be interpolated
    :param step_features: Features that hold the last row's value instead of being interpolated, e.g. a
        one-hot encoded status
    :param balance_classes: If True, training and validation draw the groups so that each class is
        equally likely; the evaluation keeps the data's own distribution
    """
    label_column: str
    classes: list[str]
    group_column: str
    timestamp_column: str
    window: str | float
    sequence_length: int
    max_gap: str | float
    step_features: list[str] | None
    balance_classes: bool

    default_loss: ClassVar[str] = "categorical_crossentropy"
    default_metrics: ClassVar[list[str]] = ["categorical_accuracy"]

    def __init__(self, *,
                 label: str,
                 pipeline: dict[str, Any] | DataProcessingPipeline,
                 features: list[str],
                 models: list[dict[str, Any] | ModelInstanceDescription],
                 label_column: str,
                 classes: list[str],
                 group_column: str,
                 timestamp_column: str,
                 window: str | float,
                 sequence_length: int,
                 max_gap: str | float,
                 step_features: list[str] | None = None,
                 balance_classes: bool = True,
                 targets: list[str] | None = None,
                 training_parameters: dict[str, Any] | TrainingParameters = TrainingParameters()):
        classes = [str(c) for c in classes]
        # 'targets' only comes back from a saved task, as a copy of the classes
        if targets is not None and list(targets) != classes:
            raise ValueError(f"{self.__class__.__name__}.__init__: targets {targets} differ from the"
                             f" classes {classes}")
        super().__init__(label=label, pipeline=pipeline, features=features, models=models,
                         targets=classes, training_parameters=training_parameters)

        self.label_column = label_column
        self.classes = classes
        self.group_column = group_column
        self.timestamp_column = timestamp_column
        self.window = window
        self.sequence_length = sequence_length
        self.max_gap = max_gap
        self.step_features = step_features
        self.balance_classes = balance_classes

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ["label_column", "classes", "group_column", "timestamp_column", "window",
                                "sequence_length", "max_gap", "step_features", "balance_classes",
                                "training_parameters"])

    def __iter__(self):
        yield from super().__iter__()

        yield "label_column", self.label_column
        yield "classes", self.classes
        yield "group_column", self.group_column
        yield "timestamp_column", self.timestamp_column
        yield "window", self.window
        yield "sequence_length", self.sequence_length
        yield "max_gap", self.max_gap
        yield "step_features", self.step_features
        yield "balance_classes", self.balance_classes


#: Learning tasks whose sequences are windows in time rather than a number of rows
_WINDOW_TASKS = (TemporalForecastTask, ClassificationTask)


class Experiment:
    MARKER_FILE: str = ".damast_experiment"
    TIMESTAMP_FORMAT: str = "%Y%m%d-%H%M%S"

    learning_task: LearningTask

    input_data: list[Path]
    output_directory: Path
    label: str

    _batch_size: int
    _split_data_ratios: list[float]

    _timestamp: datetime.datetime
    _evaluation_steps: int
    _evaluation_report: dict[str, dict[str, Any]]

    _trained_models: list[BaseModel]

    def __init__(self,
                 learning_task: dict[str, Any] | LearningTask,
                 input_data: str | Path,
                 output_directory: str | Path = tempfile.gettempdir(),
                 batch_size: int = 2,
                 evaluation_steps=1,
                 split_data_ratios: list[float] = [0.8, 0.1, 0.1],
                 label: str = "damast-ml-experiment",
                 timestamp: str | datetime.datetime = datetime.datetime.now(datetime.timezone.utc),
                 evaluation={},
                 seed: int | None = None,
                 max_evaluation_windows: int | None = None,
                 ):
        """
        :param learning_task:  A description of the machine-learning model setup
        :param input_data:
        :param evaluation_steps: Number of batches the trained models are evaluated on; None for one
            pass over fixed, non-overlapping windows of the test groups (window tasks only, see
            :func:`evaluate_on_windows`)
        :param split_data_ratios: Relative sizes of the (train, test, validate) partitions of the
            input groups; normalized, so that they sum to 1
        :param label:
        :param timestamp:
        :param evaluation:
        :param batch_size: The batch-size
        :param output_directory: Path to direct output to
        :param seed: If given, the seed of all random draws - the split of the groups, the windows and
            the initial weights - see :func:`damast.utils.set_random_seed`
        :param max_evaluation_windows: If given, the evaluation on fixed windows uses at most this many,
            drawn at random
        """
        if isinstance(learning_task, LearningTask):
            self.learning_task = learning_task
        elif isinstance(learning_task, dict):
            self.learning_task = LearningTask.from_dict(data=learning_task)
        else:
            raise TypeError(f"{self.__class__.__name__}.__init__: learning_task must be either "
                             f"dict or LearningTask object, was {type(learning_task)}")

        if type(input_data) is list:
            self.input_data = input_data
        else:
            self.input_data = [ Path(input_data) ]

        self.output_directory = Path(output_directory)
        self.label = label

        self._batch_size = batch_size
        self._evaluation_steps = evaluation_steps
        self._split_data_ratios = split_data_ratios
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.strptime(timestamp, Experiment.TIMESTAMP_FORMAT).replace(
                tzinfo=datetime.timezone.utc)

        self._timestamp = timestamp
        self._evaluation_report = evaluation
        self._trained_models = []
        self.seed = seed
        self.max_evaluation_windows = max_evaluation_windows

    @classmethod
    def from_file(cls,
                  filename: str | Path) -> Experiment:
        if not Path(filename).exists():
            raise FileNotFoundError(f"{cls.__name__}.from_file: the given experiment description file"
                                    f" does not exist: '{filename}'")

        with open(filename, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            return cls(**data)

    def save(self,
             filename: str | Path):
        """
        Save the experiment as yaml.

        :param filename: Name of the file
        """
        with open(filename, "w") as f:
            yaml.dump(dict(self), f)

    @classmethod
    def create_experiment_directory(cls,
                                    base_dir: str | Path,
                                    label: str) -> Path:
        """Create an experiment directory inside the output directory.

        The created directory is prefixed with a time-stamp indicating when it was created.

        """
        prefix = datetime.datetime.now(datetime.timezone.utc).strftime(cls.TIMESTAMP_FORMAT)
        experiment_dir = base_dir / f"{prefix}-{label}"
        experiment_dir.mkdir(exist_ok=False, parents=True)

        Experiment.touch_marker(experiment_dir)

        return experiment_dir

    @classmethod
    def validate_experiment_dir(cls, dir: str | Path) -> Path:
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
    def from_directory(cls, dir: str | Path) -> dict[str, keras.Model]:
        """
        Create an experiment object by loading a directory with experiment artifacts.

        Note that currently keras models will be loaded, when available.

        :param dir: Directory with experiment artifacts
        :return: Dictionary of loaed
        """
        experiment_dir = cls.validate_experiment_dir(dir=dir)

        models = {}

        import keras.models

        from damast.ml.models.base import MODEL_TF_KERAS

        # Load available models
        for dirname in os.listdir(str(experiment_dir)):
            model_keras = experiment_dir / dirname / MODEL_TF_KERAS
            if model_keras.exists():
                models[Path(dirname).stem] = keras.models.load_model(model_keras)

        return models

    @classmethod
    def touch_marker(cls, dir: str | Path):
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
                                           ratios: list[float],
                                           strata: dict[Any, Any] | None = None) -> list[pl.DataFrame]:
        """
        Given a :class:`damast.AnnotatedDataFrame` and a column name, split the dataframe into groups
        based on the column name. The size of each group is determined by the ratio.

        The groups are sorted before they are shuffled, so that the split only depends on the random
        seed (see :func:`damast.utils.set_random_seed`), not on the order the groups were found in.

        ..note::
            This function does not modify anything in the incoming dataframe.

        :param adf: The available data in the input data frame
        :param group: In case this deals with timeseries / sequence data, the group will relate to multiple rows
        :param ratios: Ratio between the train, test and validate dataset
        :param strata: If given, the stratum of each group, e.g. its label: each stratum is split by the
            ratios on its own, so that every partition holds each stratum in proportion. Groups without
            a stratum are left out.
        """
        groups = adf[group].unique().sort(group).collect()
        if strata is None:
            return cls._split(groups.select(pl.col(group).shuffle()), ratios)

        groups = groups.filter(pl.col(group).is_in(list(strata)))
        stratum = pl.Series([strata[g] for g in groups[group].to_list()])
        partitions = [cls._split(groups.filter(stratum == value).select(pl.col(group).shuffle()), ratios)
                      for value in sorted(set(stratum.to_list()), key=str)]
        return [pl.concat([p[i] for p in partitions]) for i in range(len(ratios))]

    @staticmethod
    def _split(groups: pl.DataFrame, ratios: list[float]) -> list[pl.DataFrame]:
        """Cut groups into consecutive partitions of the relative sizes ``ratios``."""
        from_idx = 0
        partitions = []
        for ps in partition_sizes(len(groups), ratios):
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
        # executed now: training reads the frame directly, which would skip the run's checks
        adf_with_features = self.learning_task.pipeline.transform(adf).execute()

        self.learning_task.pipeline.save(dir=self.output_directory)
        self.learning_task.pipeline.save_state(
            df=adf_with_features,
            dir=self.output_directory)
        return adf_with_features

    def create_generator(self,
                         adf: AnnotatedDataFrame,
                         group: str,
                         group_ids: list[Any],
                         infinite: bool = True,
                         windows: pl.DataFrame | None = None) -> Sequence:
        """
        Create a generator applicable for :py:mod:`keras` from the input data.

        ..note::
            This function is called whenever you want to train your models, i.e
            :func:`Experiment.train`

        :param adf: The input dataframe
        :param group: Group to extract sequences from
        :param group_ids: Groups used in the generator
        :param infinite: If True, draw random sequences endlessly, as for training
        :param windows: For a window task, one pass over exactly these windows (see
            :func:`evaluation_windows`)
        :returns: A triplet of (train, test, validation) generators of the input features
        """
        task = self.learning_task
        if isinstance(task, ClassificationTask):
            gwa = GroupWindowAccessor(df=adf, group_column=group, timestamp_column=task.timestamp_column)
            return gwa.to_keras_generator(features=task.features,
                                          label=task.label_column,
                                          classes=task.classes,
                                          groups=group_ids,
                                          window=task.window,
                                          sequence_length=task.sequence_length,
                                          max_gap=task.max_gap,
                                          step_columns=task.step_features,
                                          balance_by_label=task.balance_classes and infinite and windows is None,
                                          batch_size=self._batch_size,
                                          infinite=infinite,
                                          windows=windows)
        elif isinstance(task, TemporalForecastTask):
            gwa = GroupWindowAccessor(df=adf,
                                      group_column=group,
                                      timestamp_column=task.timestamp_column)
            return gwa.to_keras_generator(features=task.features,
                                          target=task.targets,
                                          groups=group_ids,
                                          window=task.window,
                                          sequence_length=task.sequence_length,
                                          forecast_horizon=task.forecast_horizon,
                                          forecast_length=task.forecast_length,
                                          max_gap=task.max_gap,
                                          step_columns=task.step_features,
                                          target_reference=task.target_reference,
                                          batch_size=self._batch_size,
                                          infinite=infinite,
                                          windows=windows)
        elif isinstance(self.learning_task, ForecastTask):
            sta = GroupSequenceAccessor(df=adf,
                                        group_column=group)
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

    def evaluation_windows(self,
                           adf: AnnotatedDataFrame,
                           group: str,
                           group_ids: list[Any] | None,
                           max_windows: int | None = None) -> pl.DataFrame:
        """
        The fixed, non-overlapping windows an evaluation of a window task passes over.

        :param adf: The dataframe with the features
        :param group: Column identifying a group
        :param group_ids: The groups to evaluate on, all if None
        :param max_windows: If given, at most this many windows, drawn at random
        :return: The windows, see :func:`damast.data_handling.accessors.GroupWindowAccessor.tile_windows`
        """
        task = self.learning_task
        if not isinstance(task, _WINDOW_TASKS):
            raise NotImplementedError(f"{self.__class__.__name__}.evaluation_windows: only for a window task,"
                                      f" not for {task.__class__.__name__}")

        gwa = GroupWindowAccessor(df=adf, group_column=group, timestamp_column=task.timestamp_column)
        horizon = None
        if isinstance(task, ClassificationTask):
            # only the groups that have a label among the classes can be evaluated
            labelled = gwa.group_labels(label=task.label_column, classes=task.classes,
                                        groups=None if group_ids is None else gwa._group_values(group_ids))
            group_ids = list(labelled)
        else:
            horizon = task.forecast_horizon
        return gwa.tile_windows(groups=group_ids, window=task.window, forecast_horizon=horizon,
                                max_gap=task.max_gap, max_windows=max_windows)

    def instantiate_model(self,
                          model_instance_description: ModelInstanceDescription,
                          output_dir: Path,
                          suffix: str = "") -> BaseModel:
        """
        Create a model of the learning task, named as :func:`train` names it - so that a model created for
        the output directory of a training finds its checkpoints.

        :param model_instance_description: The model and its parameters
        :param output_dir: The directory of the model's own directory
        :param suffix: Suffix of the model's name
        :return: The model, not yet compiled
        """
        if not isinstance(self.learning_task, ForecastTask | ClassificationTask):
            raise NotImplementedError(
                f"{self.__class__.__name__}.instantiate_model: Sorry, but there is currently no support "
                f"implemented for {self.learning_task.__class__.__name__}")

        return model_instance_description.model(
            name=f"{model_instance_description.model.__name__}{suffix}",
            output_dir=output_dir,
            features=self.learning_task.features,
            targets=self.learning_task.targets,
            timeline_length=self.learning_task.sequence_length,
            **model_instance_description.parameters
        )

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
              loss_function: str | None = None,
              optimizer: str = "sgd",
              metrics: list[str] | None = None) -> BaseModel:
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
        :param learning_rate: The learning rate of the optimizer
        :param loss_function: The loss function, None for the learning task's default
        :param optimizer: Name of a keras optimizer
        :param metrics: Names of keras metrics, None for the learning task's default

        :return: model instance
        """
        # NOTE: This should probably be an individual step, as we could re-use an existing model to continue training
        model = self.instantiate_model(model_instance_description, output_dir=output_dir, suffix=suffix)

        model.build(loss_function=loss_function or self.learning_task.default_loss,
                    metrics=metrics or self.learning_task.default_metrics,
                    optimizer=keras.optimizers.get({"class_name": optimizer,
                                                    "config": {"learning_rate": learning_rate}}))

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
                 steps: int = 1) -> dict[str, dict[str, Any]]:

        return Experiment.evaluate_models(models=self._trained_models,
                                          test_generator=test_generator,
                                          steps=steps)

    @classmethod
    def evaluate_models(cls,
                        models: list[BaseModel],
                        test_generator: Sequence,
                        steps: int = 1) -> dict[str, dict[str, Any]]:
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

    def evaluate_on_windows(self,
                            models: list[BaseModel],
                            adf: AnnotatedDataFrame,
                            group: str,
                            windows: pl.DataFrame,
                            output_dir: Path,
                            dataset_label: str) -> dict[str, dict[str, Any]]:
        """
        Evaluate models in one pass over fixed windows, with the metrics of the learning task.

        Unlike :func:`evaluate_models`, every window counts once and the result is the same in every run.
        The predictions are kept per model, next to the window they were made for, as
        ``<output_dir>/<model>/predictions-<dataset_label>.parquet`` - with the true and predicted values,
        and the last input step of each feature (``last_<feature>``).

        :param models: The trained models
        :param adf: The dataframe with the features
        :param group: Column identifying a group
        :param windows: The windows, see :func:`evaluation_windows`
        :param output_dir: Directory of the predictions
        :param dataset_label: Name of the evaluated data, e.g. ``"test"``
        :return: Per model the number of windows and the metrics, see :mod:`damast.ml.evaluation`
        """
        group_ids = windows[group].unique().to_list()
        results = {}
        for model in models:
            batches = self.create_generator(adf, group=group, group_ids=group_ids, infinite=False, windows=windows)
            y_true, y_pred, last_inputs = model.predict_batches(batches)
            y_pred = y_pred.reshape(y_true.shape)

            directory = Path(output_dir) / model.name
            directory.mkdir(parents=True, exist_ok=True)
            self._predictions_frame(windows, y_true, y_pred, last_inputs).write_parquet(
                directory / f"predictions-{dataset_label}.parquet")
            results[model.name] = {"windows": windows.height, "metrics": self._metrics(y_true, y_pred)}
        return results

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """The metrics of the learning task, see :mod:`damast.ml.evaluation`."""
        if isinstance(self.learning_task, ClassificationTask):
            return classification_metrics(y_true.argmax(axis=1), y_pred.argmax(axis=1),
                                          classes=self.learning_task.classes)
        return forecast_metrics(y_true, y_pred, targets=self.learning_task.targets)

    def _predictions_frame(self, windows: pl.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                           last_inputs: np.ndarray) -> pl.DataFrame:
        """The windows with their true and predicted values, and their last input step."""
        task = self.learning_task
        if isinstance(task, ClassificationTask):
            values = [pl.Series("true_class", [task.classes[i] for i in y_true.argmax(axis=1)]),
                      pl.Series("predicted_class", [task.classes[i] for i in y_pred.argmax(axis=1)]),
                      *[pl.Series(f"probability_{c}", y_pred[:, i]) for i, c in enumerate(task.classes)]]
        else:
            values = [pl.Series(f"{kind}_{t}_{k + 1}", array[:, k, j])
                      for k in range(y_true.shape[1]) for j, t in enumerate(task.targets)
                      for kind, array in (("true", y_true), ("predicted", y_pred))]
        last = [pl.Series(f"last_{f}", last_inputs[:, j]) for j, f in enumerate(task.features)]
        return windows.with_columns(values + last)

    @classmethod
    def evaluate_experiment(cls,
                            run_dir: str | Path,
                            input_data: str | Path | list[str | Path],
                            dataset_label: str,
                            *,
                            groups: list[Any] | None = None,
                            exclude_groups: list[Any] | None = None,
                            max_windows: int | None = None,
                            seed: int | None = None,
                            output_dir: str | Path | None = None) -> Path:
        """
        Evaluate the models of a finished run on other data - e.g. another source - exactly as the run
        evaluated them on its test groups: with its pipeline, its windows and its metrics.

        The pipeline is run on the data with the evaluation directory as base directory, so that the
        statistics it writes do not overwrite those of the training.

        :param run_dir: Directory of the run, holding ``experiment-report.yaml`` and the models
        :param input_data: The data, in the form the run was trained on
        :param dataset_label: Name of the data, e.g. ``"aisdk"``
        :param groups: If given, only these groups
        :param exclude_groups: Groups to leave out, e.g. those the models were trained on
        :param max_windows: If given, at most this many windows, drawn at random
        :param seed: If given, the seed of the random draws, see :func:`damast.utils.set_random_seed`
        :param output_dir: Directory of the results, default ``<run_dir>/evaluations/<dataset_label>``
        :return: Path of ``evaluation.yaml``, holding the number of groups and windows and the metrics
            per model; the predictions are next to it, per model
        """
        run_dir = cls.validate_experiment_dir(run_dir)
        experiment = cls.from_file(Path(run_dir) / "experiment-report.yaml")
        task = experiment.learning_task
        if not isinstance(task, _WINDOW_TASKS):
            raise NotImplementedError(f"{cls.__name__}.evaluate_experiment: only for a window task,"
                                      f" not for {task.__class__.__name__}")
        if seed is not None:
            set_random_seed(seed)
            keras.utils.set_random_seed(seed)

        output_dir = Path(output_dir) if output_dir is not None else Path(run_dir) / "evaluations" / dataset_label
        output_dir.mkdir(parents=True, exist_ok=True)
        task.pipeline.base_dir = output_dir

        inputs = [Path(p) for p in input_data] if isinstance(input_data, list) else [Path(input_data)]
        adf = AnnotatedDataFrame.from_files(inputs)
        features = task.pipeline.transform(adf).execute()

        group = task.group_column
        available = features.lazyframe.select(pl.col(group).unique()).collect()[group].sort()
        selected = available if groups is None else available.filter(available.is_in(list(groups)))
        excluded = 0
        if exclude_groups is not None:
            leave_out = selected.is_in(list(exclude_groups))
            excluded = int(leave_out.sum())
            selected = selected.filter(~leave_out)

        windows = experiment.evaluation_windows(features, group, selected.to_list(), max_windows=max_windows)
        models = [experiment.instantiate_model(mid, output_dir=Path(run_dir), suffix=f"-{task.label}")
                  for mid in task.models]
        results = experiment.evaluate_on_windows(models, features, group, windows,
                                                 output_dir=output_dir, dataset_label=dataset_label)

        report = {
            "run": str(run_dir),
            "dataset": dataset_label,
            "input_data": [str(p) for p in inputs],
            "groups": windows[group].n_unique(),
            "excluded_groups": excluded,
            "windows": windows.height,
            "evaluation": results,
        }
        filename = output_dir / "evaluation.yaml"
        with open(filename, "w") as f:
            yaml.dump(report, f, sort_keys=False)
        return filename

    def run(self,
            logging_level: int = INFO,
            report_filename: str | Path | None = None) -> DataFrame:
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
        if self.seed is not None:
            set_random_seed(self.seed)
            keras.utils.set_random_seed(self.seed)

        # Load data and validate it by loading it into the annotated dataframe
        # We remove values that are outside the Min/Max value defined in the specification
        adf = AnnotatedDataFrame.from_files(self.input_data)

        if not isinstance(self.learning_task, ForecastTask | ClassificationTask):
            raise NotImplementedError(
                f"{self.__class__.__name__}.run: Sorry, but there is currently only support "
                f"implemented for ForecastTask and ClassificationTask")
        if self._evaluation_steps is None and not isinstance(self.learning_task, _WINDOW_TASKS):
            raise ValueError(f"{self.__class__.__name__}.run: evaluation_steps=None (an evaluation on fixed"
                             f" windows) requires a window task, not {self.learning_task.__class__.__name__}")

        group_column = self.learning_task.group_column
        if group_column not in adf.column_names:
            raise RuntimeError(f"{self.__class__.__name__}.run: no column '{group_column}' in the given dataset "
                               f"'{self.input_data}' -- found only '{','.join(adf.column_names)}'")

        # If a learning task is used, that requires a sequence length, then filter the data, so that
        # only relevant data (with a minimum length of the given sequence length) will be accounted
        # for this learning task.
        # A temporal task resamples, so its sequence length is not a row count - its accessor skips
        # groups without a sufficiently long gap-free window itself.
        if not isinstance(self.learning_task, _WINDOW_TASKS):
            groups_with_sequence_length = adf.dataframe.group_by(group_column).agg(sequence_length=pl.len())

            filtered_groups = groups_with_sequence_length.filter(pl.col("sequence_length") > self.learning_task.sequence_length)
            permitted_values = filtered_groups.select(group_column).unique().collect()[:,0]
            adf.lazyframe = adf.filter(pl.col(group_column).is_in(permitted_values.implode()))

        features = self.compute_features(adf)
        strata = None
        if isinstance(self.learning_task, ClassificationTask):
            strata = GroupWindowAccessor(df=features, group_column=group_column,
                                         timestamp_column=self.learning_task.timestamp_column
                                         ).group_labels(label=self.learning_task.label_column,
                                                        classes=self.learning_task.classes)
        train_group, test_group, validate_group = \
            self.compute_train_test_validate_groups(features,
                                                    group=group_column,
                                                    ratios=self._split_data_ratios,
                                                    strata=strata)

        experiment_dir = self.create_experiment_directory(base_dir=self.output_directory,
                                                          label=self.label)
        # the run keeps its own copy of the pipeline and its split, so that it can be evaluated later
        self.learning_task.pipeline.save(dir=experiment_dir)
        pl.concat([partition.with_columns(pl.lit(name).alias("partition"))
                   for name, partition in zip(["train", "test", "validate"],
                                              [train_group, test_group, validate_group])]
                  ).write_parquet(experiment_dir / "groups.parquet")

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
                **self.learning_task.training_parameters._asdict()
            )
            self._trained_models.append(model)

        # TODO: This should return a 'REPORT / PROTOCOL' of the whole experiment
        if self._evaluation_steps is None:
            windows = self.evaluation_windows(features, group=group_column, group_ids=test_group,
                                              max_windows=self.max_evaluation_windows)
            self._evaluation_report = self.evaluate_on_windows(self._trained_models, features,
                                                               group=group_column, windows=windows,
                                                               output_dir=experiment_dir, dataset_label="test")
        else:
            test_data_gen = self.create_generator(features,
                                                  group=group_column,
                                                  group_ids=test_group)

            self._evaluation_report = self.evaluate(test_generator=test_data_gen,
                                                    steps=self._evaluation_steps)

        self._timestamp = datetime.datetime.now(datetime.timezone.utc)

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
        yield "input_data", [str(x) for x in self.input_data]
        yield "output_directory", str(self.output_directory)
        yield "learning_task", dict(self.learning_task)
        yield "batch_size", self._batch_size
        yield "evaluation_steps", self._evaluation_steps
        yield "split_data_ratios", self._split_data_ratios
        yield "timestamp", self._timestamp.strftime(self.TIMESTAMP_FORMAT)
        yield "evaluation", self._evaluation_report
        yield "seed", self.seed
        yield "max_evaluation_windows", self.max_evaluation_windows

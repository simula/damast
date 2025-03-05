from __future__ import annotations

import gc
import glob
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import gettempdir
from typing import ClassVar, NamedTuple, OrderedDict

import keras
import keras.callbacks
import keras.utils
import pandas as pd

from damast.core import DataSpecification
from damast.core.types import DataFrame, XDataFrame

__all__ = [
    "BaseModel",
    "ModelInstanceDescription"
]

HISTORY_FILENAME = 'training-history.csv'
CHECKPOINT_BEST = "checkpoint.best.weights.h5"
MODEL_TF_HDF5 = "model.tf.hdf5"


class BaseModel(ABC):
    """
    BaseModel for Machine Learning Models

    :param name: A name for this model - by default this will be the class name
    :param features: List of the input feature names
    :param targets: List of the target feature names
    :param distribution_strategy: A tensorflow distribution strategy when trying to distribute the training across
                         multiple devices
    :param output_dir: model / experiment specific output directory
    """

    #: Input specification for this model
    input_specs: ClassVar[OrderedDict[str, DataSpecification]]
    features: list[str]

    #: Output specification for this model
    output_specs: ClassVar[OrderedDict[str, DataSpecification]]
    targets: list[str]

    #: Name of this model
    name: str
    #: Underlying keras model
    model: keras.Model

    # Computation
    distribution_strategy: object | None

    #: The directory where all the model output will be stored
    model_dir: Path

    #: History of the training
    history: keras.callbacks.History

    def __init__(self,
                 features: list[str],
                 targets: list[str] | None = None,
                 name: str | None = None,
                 distribution_strategy = None,
                 output_dir: Path = Path(gettempdir())
                 ):
        if not hasattr(self, "input_specs"):
            raise RuntimeError(f"{self.__class__.__name__}.__init__: input_specs are not set")

        if not hasattr(self, "output_specs"):
            raise RuntimeError(f"{self.__class__.__name__}.__init__: output_specs are not set")

        if name is None:
            self.name = self.__class__.__name__

        if distribution_strategy is None:
            if keras.backend.backend() == "tensorflow":
                import tensorflow as tf
                self.distribution_strategy = tf.distribute.get_strategy()
            else:
                self.distribution_strategy = None

        if name is None:
            name = self.__class__.__name__

        if features is not None:
            self.features = features

        if targets is not None:
            self.targets = targets

        self.name = name
        self.model_dir = output_dir / self.name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._init_model()
        assert self.model is not None

    @abstractmethod
    def _init_model(self):
        """
        Initialize the internal keras model, i.e. self.model.
        """
        pass

    def load_weights(self,
                     checkpoint_filepath: str | Path):
        """
        Load the model weight from an existing checkpoint named by the checkpoint_filepath

        :param checkpoint_filepath: Name of the checkpoint filepath (prefix)
        """
        if self.model is None:
            raise RuntimeError(
                f"{self.__class__}.load_weights: you need to initialize the model first via _init_model()")

        self.model.load_weights(checkpoint_filepath)

    def build(self,
              optimizer: str | keras.optimizers.Optimizer = keras.optimizers.Adam(0.001),
              loss_function: str | keras.losses.Loss = "mse",
              metrics: list[str] | list[keras.metrics.Metric] = ["mse"],
              **kwargs):
        """
        Build / Compile the actual model

        :param optimizer: Name of the optimizer or an instance of keras.optimizers.Optimizer
        :param loss_function:  Name of the loss function or an instance of keras.losses.Loss
        :param metrics: List of names of the metrics that shall be used, or list of instance of keras.metrics.Metrics
        :param kwargs: additional arguments that can be forwarded to keras.engine.training.Model.compile
        """
        def _build_impl():
            # Plotting of the model
            self.plot()

            # https://keras.io/api/models/model_training_apis/
            self.model.compile(
                optimizer=optimizer,
                loss=loss_function,
                # see
                metrics=metrics,
                **kwargs
            )
            self.model.summary()

        if self.distribution_strategy and keras.backend.backend() == "tensorflow":
            with self.distribution_strategy.scope():
                _build_impl()
        else:
            _build_impl()

    def plot(self,
             suffix=".png") -> Path:
        """
        Plot the current model instance.

        :param suffix: Suffix that shall be used for the plot file
        :return: Path to the plotted file
        """

        plots_outputdir = self.model_dir / "plots"
        plots_outputdir.mkdir(parents=True, exist_ok=True)

        filename = plots_outputdir / f"{self.name}{suffix}"
        keras.utils.plot_model(model=self.model,
                               to_file=str(filename),
                               show_shapes=True,
                               expand_nested=True)
        return filename

    @property
    def checkpoints_dir(self) -> Path:
        """
        Directory for the checkpoints.

        :return: Path to the checkpoints directory
        """
        return self.model_dir / "checkpoints"

    @property
    def evaluation_file(self) -> Path:
        """
        Path to the evaluation file storing the results

        :return: evaluation file
        """
        return self.model_dir / f'evaluation-{CHECKPOINT_BEST}.csv'

    def train(self,
              training_data: tf.data.Dataset,
              validation_data: tf.data.Dataset,
              monitor: str = "val_loss",
              mode: str = "min",
              epochs: int = 1,
              initial_epoch: int = 0,
              save_history: bool = True,
              **kwargs) -> None:
        try:
            callbacks = []
            # check https://keras.io/api/callbacks/ for available callbacks
            checkpoint_filepath = self.checkpoints_dir / CHECKPOINT_BEST
            if keras.backend.backend() == "tensorflow":
                # https://www.tensorflow.org/tensorboard/get_started
                tensorboard_cb = keras.callbacks.TensorBoard(log_dir=str(self.model_dir / "logs"),
                                                             histogram_freq=1
                                                             )
                callbacks.append(tensorboard_cb)

            model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True)
            callbacks.append(model_checkpoint_cb)

            early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                              restore_best_weights=True)
            callbacks.append(early_stopping_cb)

            # log_hyperparams_cb = hp.KerasCallback(logdir=self.model_dir / "logs",
            #                                       hparams)

            class CleanupCallback(keras.callbacks.Callback):
                def on_train_batch_begin(self, epoch, logs=None):
                    gc.collect()
            cleanup_cb = CleanupCallback()
            callbacks.append(cleanup_cb)

            # Allow to checkpoint per epoch
            self.history = self.model.fit(training_data,
                                          validation_data=validation_data,
                                          epochs=epochs,
                                          initial_epoch=initial_epoch,
                                          callbacks=callbacks,
                                          **kwargs
                                          )
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__}: Training of {self.name} failed") from e

        # Save the history
        if save_history:
            history: DataFrame = XDataFrame.from_dict(self.history.history)
            history.write_csv(f"{self.model_dir / HISTORY_FILENAME}")

    def get_evaluations(self) -> DataFrame:
        """
        Read the evaluation results from the evaluation file.

        :raise FileNotFoundError: If evaluation result file does not exist
        """
        if self.evaluation_file.exists():
            return XDataFrame.read_csv(self.evaluation_file)

        raise FileNotFoundError(f"There is are no evaluation results for {self.name} available: {self.evaluation_file} "
                                "does not exist")

    def save(self) -> Path:
        """
        Save the model in the model directory.

        :return: Path to the saved model.
        """
        filename = self.model_dir / MODEL_TF_HDF5
        self.model.save(filepath=filename)
        return filename

    def evaluate(self,
                 label: str,
                 evaluation_data: tf.data.Dataset,
                 **kwargs) -> dict[str, any]:
        """
        Evaluate this model.

        :param label: Custom label for this evaluation, e.g., can be the name of the dataset being used
        :param evaluation_data: The data that shall be used for evaluation
        :return: DataFrame listing all performed evaluation results for this model
        """
        # Load the best weight
        checkpoint_best = self.checkpoints_dir / CHECKPOINT_BEST
        if glob.glob(f"{checkpoint_best}*"):
            self.load_weights(checkpoint_filepath=checkpoint_best)
        else:
            raise RuntimeError(f"{self.__class__}.evaluate: could not find the checkpoint data"
                               f" with best weights: {checkpoint_best} does not exist")

        # Evaluate the model against the provided data using the weights/checkpoint
        # with the (so far) best performance
        evaluation: dict[str, float] = self.model.evaluate(evaluation_data,
                                                           return_dict=True,
                                                           verbose=0,
                                                           **kwargs)

        evaluation_column_names = ["dataset"]
        evaluation_results: list[any] = [label]
        for name, value in evaluation.items():
            evaluation_column_names.append(name)
            evaluation_results.append(value)

        result_df = pd.DataFrame([evaluation_results], columns=evaluation_column_names)
        if self.evaluation_file.exists():
            result_df.to_csv(self.evaluation_file)

        return evaluation


class ModelInstanceDescription(NamedTuple):
    """
    Provide a description of a model instance to allow serialization.
    """
    model: BaseModel
    parameters: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> ModelInstanceDescription:
        if "module_name" not in data:
            raise ValueError(f"{cls.__name__}.from_dict: missing 'module' specification")
        if "class_name" not in data:
            raise ValueError(f"{cls.__name__}.from_dict: missing 'class' specification")

        module_name = data["module_name"]
        class_name = data["class_name"]

        m_module = importlib.import_module(module_name)
        if not hasattr(m_module, class_name):
            raise ImportError(f"{cls.__name__}.from_dict: could not import '{class_name}' from '{m_module}'")

        model = getattr(m_module, class_name)

        parameters = data.get("parameters", {})
        return ModelInstanceDescription(model=model,
                                        parameters=parameters)

    def __iter__(self):
        yield "module_name", self.model.__module__
        yield "class_name", self.model.__qualname__
        yield "parameters", self.parameters

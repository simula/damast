import os
from pathlib import Path
from typing import Union, Dict


class Experiment:
    MARKER_FILE: str = ".damast_experiment"

    @classmethod
    def validate_experiment_dir(cls, dir: Union[str, Path]):
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
        with open(dir / cls.MARKER_FILE, "a") as f:
            pass

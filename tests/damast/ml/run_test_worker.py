"""
Module to provide a test worker instance, which provides predefined sequence predictions
"""
from damast.ml.experiments import Experiment
from damast.ml.worker import Worker
import numpy as np
from typing import List, Dict

predicted_sequence = np.array([[0, 0]])


class MockMLModel:
    """
    Mock a machine learning model, so that no actual keras model needs to be used
    """
    def predict(self, input_data, steps, verbose):
        return predicted_sequence

    def loss(self, x, y) -> List[float]:
        return [float(i) for i in range(x.shape[0])]


def mock_from_directory(dir) -> Dict[str, object]:
    """
    Mock the directory loading for Experiment to link to the Mocked machine learning model
    :param dir:
    :return:
    """
    return {"custom_model": MockMLModel()}


setattr(Experiment, "from_directory", mock_from_directory)

if __name__ == "__main__":
    worker = Worker()
    worker.listen_and_accept()

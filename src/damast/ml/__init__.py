import importlib
import logging
import os

logger = logging.getLogger(__name__)

# To ensure that autoloading works as expected
# from damast.ml import keras
def backend_available(backend: str) -> bool:
    try:
        importlib.import_module(backend)
        return True
    except ImportError:
        return False

def autodiscover_backend(priority: list[str] = ["torch", "tensorflow", "jax"]):
    for backend in priority:
        if backend_available(backend):
            logger.warning(f"Autoloading backend: {backend} - to explicity select backend set env variable KERAS_BACKEND")
            os.environ['KERAS_BACKEND'] = backend
            break
        else:
            logger.info(f"Ignoring backend: {backend}, library is not available")

if "KERAS_BACKEND" in os.environ:
    backend = os.environ['KERAS_BACKEND']
    if not backend_available(backend):
        raise RuntimeError(
                "Keras backend: select backend '{backend}' is not available. "
                "Please install the required package(s)."
        )
else:
    autodiscover_backend()

import keras  # noqa


def backend_device() -> str:
    """
    Device the keras backend computes on - the backends use a visible GPU by themselves,
    ``CUDA_VISIBLE_DEVICES=`` restricts them to the CPU.

    :return: ``"gpu"`` or ``"cpu"``
    """
    backend = keras.backend.backend()
    if backend == "torch":
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    if backend == "tensorflow":
        import tensorflow as tf
        return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    if backend == "jax":
        import jax
        return "gpu" if jax.default_backend() == "gpu" else "cpu"
    return "cpu"


logger.info(f"Keras backend '{keras.backend.backend()}' computes on: {backend_device()}")

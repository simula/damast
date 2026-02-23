import importlib
import os

import logging

logger = logging.getLogger(__name__)

# To ensure that autoloading works as expected
# from damast.ml import keras

if "KERAS_BACKEND" in os.environ:
    backend = os.environ['KERAS_BACKEND']
    if not backend_available(backend):
        raise RuntimeError("Keras backend: {backend} is not available")

def backend_available(backend: str) -> bool:
    try:
        importlib.import_module(backend)
        return True
    except ImportError as e:
        return False

def autodiscover_backend(priority: list[str] = ["torch", "tensorflow", "jax"]):
    for backend in priority:
        if backend_available(backend):
            logger.warning(f"Autoloading backend: {backend} - to explicity select backend set env variable KERAS_BACKEND")
            os.environ['KERAS_BACKEND'] = backend
            break
        else:
            logger.info(f"Ignoring backend: {backend}, library is not available")

autodiscover_backend()

import keras

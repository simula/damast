"""
Selection of the polars engine that executes damast pipelines - the GPU when available,
otherwise the CPU streaming engine.
"""
from __future__ import annotations

import importlib.util
import logging
import os

import polars

logger = logging.getLogger(__name__)


def gpu_available() -> bool:
    """
    Check whether the polars GPU engine can be used.

    :return: True if CUDA devices are not hidden (``CUDA_VISIBLE_DEVICES`` is not empty or ``-1``)
        and the GPU engine (``cudf_polars``, see the ``cuda`` extra) is installed
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return False
    return importlib.util.find_spec("cudf_polars") is not None


def configure_engine_affinity():
    """
    Set the polars engine affinity: an explicit ``POLARS_ENGINE_AFFINITY`` is kept, otherwise
    ``gpu`` if :func:`gpu_available`, else ``streaming``.
    """
    if "POLARS_ENGINE_AFFINITY" in os.environ:
        logger.info(f"Keeping POLARS_ENGINE_AFFINITY='{os.environ['POLARS_ENGINE_AFFINITY']}'")
        return
    affinity = "gpu" if gpu_available() else "streaming"
    logger.info(f"Using polars engine affinity '{affinity}'"
                " - set CUDA_VISIBLE_DEVICES= or POLARS_ENGINE_AFFINITY to change it")
    polars.Config.set_engine_affinity(affinity)


def default_engine() -> str | polars.GPUEngine:
    """
    Engine to execute a whole pipeline with, following the configured engine affinity.

    With the GPU engine, a query it cannot run falls back to polars' in-memory CPU engine (not
    the streaming one) - for data larger than memory set ``CUDA_VISIBLE_DEVICES=`` or
    ``POLARS_ENGINE_AFFINITY=streaming``.

    :return: A :class:`polars.GPUEngine` for the ``gpu`` affinity, otherwise ``"streaming"``
    """
    if polars.Config.state().get("POLARS_ENGINE_AFFINITY") == "gpu":
        return polars.GPUEngine(raise_on_fail=False)
    return "streaming"

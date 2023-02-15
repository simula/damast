# code=utf-8
"""
This namespace contains the core modules that are required to define metada.
"""

from .dataprocessing import (
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    input,
    output
    )

__all__ = [
    "input",
    "output",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS"
]

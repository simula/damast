# code=utf-8
"""
This namespace contains the core modules that are required to define metada.
"""

from .dataprocessing import (
    DECORATED_DESCRIPTION,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    describe,
    input,
    output
    )

__all__ = [
    "describe",
    "input",
    "output",
    "DECORATED_DESCRIPTION",
    "DECORATED_INPUT_SPECS",
    "DECORATED_OUTPUT_SPECS"
]

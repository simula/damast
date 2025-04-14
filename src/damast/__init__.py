"""Main module"""

from damast import data_handling  # , domains
from damast import cli, core
from damast.core import artifacts, describe, input, output

from .version import __version__, __version_info__

__all__ = [
    "artifacts",
    "cli",
    "core",
    "data_handling",
    "describe",
#    "domains",
    "input",
    "output",
    "__version__",
    "__version_info__"
]

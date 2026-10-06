"""
Shared handling of the labels AIS sources use in place of a standard's codes.

A source may report a coded field as text - 'Tanker' rather than 80, 'Under way using engine'
rather than 0 - and every source spells it its own way. Reducing both the standard's names and
a source's labels to one form lets them be compared without a table per source.
"""

import re

__all__ = ["normalise_label"]

#: A boundary inside a camel case name, e.g. 'UnderWay' or 'AISSart'
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def normalise_label(name: str) -> str:
    """
    Reduce a label or a standard's name to its comparable form.

    Camel case is split first, so that an enum member reduces to the same form as the words a
    source writes: both ``UnderWayUsingEngine`` and ``"Under way using engine"`` become
    ``under_way_using_engine``.

    :param name: The name, as a standard or a source spells it
    :return: Lower case, with each run of non-alphanumeric characters as a single underscore
    """
    split = _CAMEL_BOUNDARY.sub("_", name.strip())
    return re.sub(r"[^0-9a-z]+", "_", split.lower()).strip("_")

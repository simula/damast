# Copyright (C) 2023 Simula Research Laboratory
#
# This file is part of Damast
#
# SPDX-License-Identifier:    BSD-3-Clause
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union

__all__ = [
    "Annotation",
    "Change",
    "History"
]


class Annotation:
    """
    Create an annotation

    :param name: Name / Type of the annotation
    :param value: Value of the annotation
    """

    class Key(str, Enum):
        """
        Enumeration of predefined keys which can be used for the annotation
        """
        Institution = "institution"
        License = "license"
        References = "references"
        Source = "source"
        History = "history"
        Comment = "comment"

    #: Name of the annotation (see also Key)
    name: str

    #: Value of the annotation
    value: Optional[Any] = None

    def __init__(self,
                 name: Union[str, Key],
                 value: Optional[Any] = None
                 ):
        self.name = name
        self.value = value

        # In case we want to enforce some key-related validation, then
        # add a 'validate_<key-name>' function
        if isinstance(self.name, Annotation.Key):
            self.name = self.name.value

        validation_func_name = f"validate_{self.name}"
        if hasattr(self, validation_func_name):
            getattr(self, validation_func_name)()

    def __eq__(self, other) -> bool:
        """
        Override equality operator to consider equality based on properties only.

        :param other: Other object
        :return: True if objects are consider same, otherwise False
        """
        if self.__class__ != other.__class__:
            return False

        if self.value != other.value:
            return False

        if self.name != other.name:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary to represent this object, e.g., to serialise the object.

        :return: dictionary
        """
        return {self.name: self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Annotation:
        """
        Create an instance from a given dictionary.

        :param data: The input dictionary
        :return: Annotation
        :raise RuntimeError: Raises if annotation cannot be created from dictionary
        """
        for k, value in data.items():
            return Annotation(name=k, value=value)

        raise RuntimeError("Annotation: failed to identify item")

    # region Key Validation Functions
    def validate_license(self) -> None:
        """
        Validation function for the license

        :raise ValueError: Raises if a license is given, but it cannot be validated
        """
        if self.value is None or self.value == '':
            raise ValueError("License cannot be empty")

    def validate_comment(self):
        """
        Validation function for the comment

        :raise ValueError: Raises if a comment remains empty.
        """
        if self.value is None or self.value == '':
            raise ValueError("Comment cannot be empty")


# endregion


class Change:
    """
    Representation of a change of data and metadata by transformation pipelines
    """
    TIMESTAMP_FORMAT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"

    title: str
    timestamp: datetime
    description: str

    def __init__(self,
                 title: str,
                 description: str,
                 timestamp: datetime = datetime.utcnow()
                 ):
        # Ensure that we only deal with the required precision
        timestamp_txt = timestamp.strftime(self.TIMESTAMP_FORMAT)
        self.timestamp = datetime.strptime(timestamp_txt, self.TIMESTAMP_FORMAT)

        self.title = title
        self.description = description

    def __repr__(self):
        return f"{self.timestamp.strftime(self.TIMESTAMP_FORMAT)}: {self.title}"

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        if self.title != other.title:
            return False
        if self.timestamp != other.timestamp:
            return False
        if self.description != other.description:
            return False

        return True

    def to_dict(self):
        return {
            "title": self.title,
            "timestamp": self.timestamp.strftime(self.TIMESTAMP_FORMAT),
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        instance = cls(title=data["title"],
                       timestamp=datetime.strptime(data["timestamp"], Change.TIMESTAMP_FORMAT),
                       description=data["description"])

        return instance


class History(Annotation):
    """
    Representation of the history of changes
    """
    changes: List[Change]

    def __init__(self, changes: Optional[List[Change]] = None):
        super().__init__(name=Annotation.Key.History)
        if changes is None:
            self.changes = []
        else:
            self.changes = changes

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        if self.changes != other.changes:
            return False

        return True

    def add_change(self, change: Change):
        self.changes.append(change)

    def to_dict(self) -> Dict[str, Any]:
        changes = []
        for c in self.changes:
            changes.append(c.to_dict())
        return {Annotation.Key.History.value: changes}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        changes = []
        if Annotation.Key.History.value in data:
            for c in data[Annotation.Key.History.value]:
                changes.append(Change.from_dict(data=c))

        return cls(changes=changes)

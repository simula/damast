"""
Module to define Annotation object for data specification
"""
# Copyright (C) 2023-2025 Simula Research Laboratory
#
# This file is part of damast
#
# SPDX-License-Identifier:    BSD-3-Clause
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union

__all__ = ["Annotation", "Change", "History"]


class Annotation:
    """
    Annotation class

    :param name: Name / Type of the annotation (see also :class:`Key`)
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

    def __init__(self, name: Union[str, Key], value: Optional[Any] = None):
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
        Check if two annotations are the same.

        The two objects are considered equal if the other object is a :class:`Annotation`
        is the same for both objects, and the  :py:attr:`name` and :py:attr:`value` of
        the annotation is the same.

        :param other: Other object
        :return: ``True`` if objects are consider same, otherwise ``False``
        """
        if self.__class__ != other.__class__:
            return False

        if self.value != other.value:
            return False

        if self.name != other.name:
            return False

        return True

    def __iter__(self):
        yield self.name, self.value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Annotation:
        """
        Create an instance from a given dictionary.

        .. note::
            The dictionary is assumed to only have one ``(key, value)`` pair,
            where the ``key`` is interpreted as the :attr:`Annotation.name`, the
            ``value`` as the :attr:`Annotation.value`

        :param data: The input dictionary
        :return: Annotation
        :raise RuntimeError: Raises if annotation cannot be created from dictionary
        """
        assert len(data.keys()) == 1, "Input dictionary has more than one object"
        for k, value in data.items():
            return Annotation(name=k, value=value)

    def validate_license(self) -> None:
        """
        Validation function for the license

        :raise ValueError: Raises if a license is given, but it cannot be validated
        """
        if self.value is None or self.value == "":
            raise ValueError("License cannot be empty")

    def validate_comment(self):
        """
        Validation function for the comment

        :raise ValueError: Raises if a comment remains empty.
        """
        if self.value is None or self.value == "":
            raise ValueError("Comment cannot be empty")


class Change:
    """
    Representation of a change of data and metadata by transformation pipelines

    :param title: A concise description of the change
    :param description: A detailed description of the change
    :param timestamp: A timestamp for when the change was mede
    """

    TIMESTAMP_FORMAT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"

    title: str
    timestamp: datetime
    description: str

    def __init__(
        self, title: str, description: str, timestamp: datetime = datetime.utcnow()
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

    def __iter__(self):
        yield "title", self.title
        yield "timestamp", self.timestamp.strftime(self.TIMESTAMP_FORMAT)
        yield "description", self.description

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Initialize a change from a dictionary

        .. note::
            The dictionary is required to have the keys ``"title"``,
            ``"timestamp"`` and ``"description"``.
            All other keys in the input dictionary is ignored

        :param data: Input dictionary
        :return: A change object
        """
        return cls(
            title=data["title"],
            timestamp=datetime.strptime(
                data["timestamp"], Change.TIMESTAMP_FORMAT
            ),
            description=data["description"],
        )


class History(Annotation):
    """
    Representation of the (linear) history of changes

    :param changes: A list of changes (ordered chronologically)
    """

    changes: List[Change]

    def __init__(self, changes: Optional[List[Change]] = None):
        super().__init__(name=Annotation.Key.History)
        if changes is None:
            self.changes = []
        else:
            self.changes = sorted(changes, key=str)

    def __eq__(self, other) -> bool:
        """
        A history is equal to another history of it has the same sequence of changes

        :param other: The other object
        """
        if self.__class__ != other.__class__:
            return False

        return self.changes == other.changes

    def add_change(self, change: Change):
        """
        Extend history with adding a new change

        :param change: The change
        """
        changes = self.changes + [change]
        self.changes = sorted(changes, key=str)

    def __iter__(self):
        yield Annotation.Key.History.value, [dict(c) for c in self.changes]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Initialize a history from a dictionary

        .. note::
            It is assumed that the history is located under
            the key (string representation) of :attr:`Annotation.Key.History`.
            The item under this key should be a list changes (represented as dictionaries).

        .. note::
            If the key  :attr:`Annotation.Key.History` is not present in the dictionary, assume no
            changes present and initialize a clean history

        :param data: The input dictionary
        :return: A history object with the corresponding changes
        """
        changes = []
        if Annotation.Key.History.value in data:
            for c in data[Annotation.Key.History.value]:
                changes.append(Change.from_dict(data=c))

        return cls(changes=changes)

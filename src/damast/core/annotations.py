from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, List


class Annotation:
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

    name: str = None
    value: Any = None

    def __init__(self,
                 name: str,
                 value: Any = None
                 ):
        self.name = name
        self.value = value

        # In case we want to enforce some key-related validation, then
        # add a 'validate_<key-name>' function
        if self.name in Annotation.Key:
            validation_func_name = f"validate_{self.name}"
            if hasattr(self, validation_func_name):
                getattr(self, validation_func_name)()

# region Key Validation Functions
    def validate_license(self):
        """
        Validation function for the license
        """
        if self.value is None or self.value is '':
            raise ValueError("License cannot be empty")

    def validate_comment(self):
        """
        Validation function for the comment
        """
        if self.value is None or self.value is '':
            raise ValueError("Comment cannot be empty")
# endregion


class Change:
    """
    Representation of a change of data and metadata by transformation pipelines
    """
    TIMESTAMP_FORMAT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"

    title: str = None
    timestamp: datetime = None
    description: str = None

    def __init__(self,
                 title: str,
                 description: str,
                 timestamp: datetime = datetime.utcnow()
                 ):
        self.timestamp = timestamp
        self.title = title
        self.description = description

    def __repr__(self):
        return f"{datetime.strptime(self.timestamp, Change.TIMESTAMP_FORMAT)}: {self.title}"


class History(Annotation):
    """
    Representation of the history of changes
    """
    changes: List[Change] = None

    def __init__(self):
        super().__init__(name=Annotation.Key.History)

    def add_change(self, change: Change):
        self.changes.insert(change)

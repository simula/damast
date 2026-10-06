"""
Module which collect all filters that filter the existing data.
"""

import datetime
import numbers
import warnings
# 'operator' is a constructor argument of Filter - import the functions by name
from operator import eq, ge, gt, le, lt, ne
from typing import Any

import numpy as np
import polars as pl

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement
from damast.core.types import XDataFrame

__all__ = [
    "DropMissingOrNan",
    "Filter",
    "FilterWithin",
    "RemoveValueRows"
]

#: Supported comparisons - the first seven are spelled as in 'damast inspect --filter'
OPERATORS = {
    "<": lt, "<=": le, ">": gt, ">=": ge, "==": eq, "!=": ne, "<>": ne,
    "in": lambda expr, value: expr.is_in(value),
    "not in": lambda expr, value: ~expr.is_in(value),
}

#: Operators that compare against a collection of values rather than a single one
MEMBERSHIP_OPERATORS = ["in", "not in"]

#: Operators that are meaningful for a boolean value
EQUALITY_OPERATORS = ["==", "!=", "<>"]


class DropMissingOrNan(PipelineElement):
    """
    Drop rows that do not have a defined value or NaN for a given column.
    """

    @damast.core.describe("Drop rows where this column has a missing (or nan) value")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Drop rows with missing value
        """
        mapped_name = self.get_name("x")
        dataframe = df.lazyframe

        new_dataframe = dataframe.drop_nulls(subset=mapped_name)
        # NaN only exists for float columns - 'drop_nans' raises for e.g. a Datetime column
        if XDataFrame(new_dataframe).dtype(mapped_name).is_float():
            new_dataframe = new_dataframe.drop_nans(subset=mapped_name)

        df.lazyframe = new_dataframe
        return df


class Filter(PipelineElement):
    """
    Filter rows and keep those satisfying a comparison against a fixed value.

    Mirrors a single :func:`polars.LazyFrame.filter` with a binary predicate. The operators are
    the ones :code:`damast inspect --filter` accepts, plus membership - see :data:`OPERATORS`.
    Rows satisfying the comparison are *kept*, so to exclude rows state the complement: keep
    :code:`">=" 999999` in order to drop everything below 999999.

    Example:

        .. highlight:: python
        .. code-block:: python

            pipeline.add("valid_mmsi", Filter(">=", 999999), name_mappings={"x": "mmsi"})
            pipeline.add("cargo", Filter("in", [30, 60, 70, 80]), name_mappings={"x": "ship_type"})

    The value may be a number, a string, a :class:`datetime.date` or a
    :class:`datetime.datetime`, as long as it suits the column's datatype - or, for a membership
    operator, a collection of those.

    :param operator: The comparison to apply, one of :data:`OPERATORS`
    :param value: The value to compare each row against, or the values to test membership in
    :raise ValueError: If the operator is unknown, or the value is None
    """
    _operator: str
    _value: Any

    def __init__(self, operator: str, value: Any):
        super().__init__()

        if operator not in OPERATORS:
            raise ValueError(f"{self.__class__.__name__}.__init__: unknown operator '{operator}' -"
                             f" use one of {sorted(OPERATORS)}")

        if value is None:
            # 'pl.col(x) == None' evaluates to null per row, it is not a null test
            raise ValueError(f"{self.__class__.__name__}.__init__: cannot compare against None -"
                             f" use DropMissingOrNan to remove rows without a value")

        self._operator = operator
        self._value = value

    @property
    def operator(self) -> str:
        return self._operator

    @property
    def value(self) -> Any:
        return self._value

    def _validate_value_type(self, df: AnnotatedDataFrame, column: str):
        """
        Ensure that the value can be compared against the column.

        The frame is lazy, so an unsuitable value would otherwise only surface as a polars error
        on collection - far away from the step that caused it. For a membership operator the
        elements are checked; an empty collection is legal and keeps nothing, so there is
        nothing to check.

        :param df: The dataframe that will be filtered
        :param column: Name of the column to compare
        :raise ValueError: If the value does not suit the column's datatype
        """
        dtype = XDataFrame(df.lazyframe).dtype(column)

        values = list(self._value) if self._operator in MEMBERSHIP_OPERATORS else [self._value]
        for value in values:
            self._validate_one_value(value=value, dtype=dtype, column=column)

    def _validate_one_value(self, value: Any, dtype: Any, column: str):
        """
        Ensure that a single value can be compared against a column of the given datatype.

        :param value: The value to check
        :param dtype: Datatype of the column
        :param column: Name of the column, for the error message
        :raise ValueError: If the value does not suit the datatype
        """
        # bool before the numeric check: a Python bool is a numbers.Real
        if isinstance(value, (bool, np.bool_)):
            expected, matches = "a boolean", dtype == pl.Boolean
            permitted = EQUALITY_OPERATORS + MEMBERSHIP_OPERATORS
            if matches and self._operator not in permitted:
                raise ValueError(f"{self.__class__.__name__}: operator '{self._operator}' is not meaningful"
                                 f" for the boolean value {value!r} - use one of {permitted}")
        # numbers.Real rather than (int, float), so that a numpy scalar - as a caller passing a
        # numpy array of values supplies - is accepted
        elif isinstance(value, numbers.Real):
            expected, matches = "a numeric", dtype.is_numeric()
        elif isinstance(value, str):
            expected, matches = "a string", dtype == pl.String
        elif isinstance(value, (datetime.datetime, datetime.date)):
            expected, matches = "a temporal", dtype.is_temporal()
        else:
            raise ValueError(f"{self.__class__.__name__}: cannot compare against {value!r}"
                             f" of type '{type(value).__name__}'")

        if not matches:
            raise ValueError(f"{self.__class__.__name__}: column '{column}' is {dtype}, but comparing against"
                             f" {value!r} requires {expected} column")

    @damast.core.describe("Filter rows and keep those satisfying a comparison")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Filter rows and keep those satisfying the comparison
        """
        mapped_name = self.get_name("x")
        self._validate_value_type(df=df, column=mapped_name)

        df.lazyframe = df.lazyframe.filter(
            OPERATORS[self._operator](pl.col(mapped_name), self._value)
        )
        return df


class FilterWithin(Filter):
    """
    Filter rows and keep those within given values.

    .. deprecated::
        Superseded by :code:`Filter("in", within_values)`, which this delegates to. Kept so that
        pipelines saved with this step keep loading - it still serialises under its own name and
        with its own parameter.

    :param within_values: list of values to keep
    """
    _within_values: Any

    def __init__(self, within_values: Any):
        warnings.warn('FilterWithin is superseded by Filter("in", within_values)',
                      DeprecationWarning, stacklevel=2)
        super().__init__(operator="in", value=within_values)
        # PipelineElement.prepare_parameters reads one attribute per __init__ keyword, so this
        # is what keeps the saved parameter block identical to that of earlier versions
        self._within_values = within_values

    @property
    def within_values(self):
        return self._within_values


class RemoveValueRows(Filter):
    """
    Remove rows that have the given value for a given column.

    .. deprecated::
        Superseded by :code:`Filter("!=", remove_value)`, which this delegates to. Kept so that
        pipelines saved with this step keep loading - it still serialises under its own name and
        with its own parameter.

    :param remove_value: remove rows with this value.
    """
    _remove_value: Any

    def __init__(self, remove_value: Any):
        warnings.warn('RemoveValueRows is superseded by Filter("!=", remove_value)',
                      DeprecationWarning, stacklevel=2)
        super().__init__(operator="!=", value=remove_value)
        # see FilterWithin.__init__ on why this attribute has to keep its name
        self._remove_value = remove_value

    @property
    def remove_value(self):
        return self._remove_value

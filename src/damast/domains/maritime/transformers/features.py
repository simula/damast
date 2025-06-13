import numpy as np
import polars as pl

import damast.core
from damast.core import AnnotatedDataFrame, DataSpecification
from damast.core.dataprocessing import PipelineElement
from damast.core.types import DataFrame
from damast.domains.maritime.math import great_circle_distance

__all__ = [
    "DeltaDistance",
    "Speed"
]


class DeltaDistance(PipelineElement):
    """
    Given a dataframe with `(latitude, longitude)` data, group messages by given column,
    and sort them by another column. Then compute the distance between two messages, using
    the :func:`damast.domains.maritime.math.great_circle_distance`.

    :param x_shift: True if one should compute the difference in latitude
    :param y_shift: True if one should compute the difference in longitude

    .. note::
        If both :code:`x_shift` and :code:`y_shift` is :code:`True`, one computes the distance between two coordinates.
    """
    _x_shift: bool
    _y_shift: bool

    def __init__(self, x_shift: bool, y_shift: bool):
        self._x_shift = x_shift
        self._y_shift = y_shift

    @property
    def x_shift(self):
        return self._x_shift

    @property
    def y_shift(self):
        return self._y_shift

    @damast.core.describe("Compute the distance between lat/lon given positions")
    @damast.core.input({"group": {"representation_type": int},
                        "sort": {},
                        "x": {"unit": "deg" },
                        "y": {"unit": "deg" }})
    @damast.core.output({"out": {"description": "distance to previous coordinate", "unit": "km"}})
    def transform(self,
                  df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Compute distance between adjacent messages
        """
        dataframe = df._dataframe

        in_x = self.get_name("x")
        in_y = self.get_name("y")
        shift_x = f"{in_x}_shifted"
        shift_y = f"{in_y}_shifted"

        tmp_column = f"{self.__class__.__name__}_tmp"
        assert tmp_column != self.get_name("out")
        if tmp_column in dataframe.compat.column_names:
            raise RuntimeError(f"{self.__class__.__name__}.transform: Dataframe contains {tmp_column}")

        # https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.shift.html
        # shift
        # Group data and sort it. Only shift data within the same group
        groups = dataframe.sort(self.get_name('sort'))\
                    .group_by(self.get_name("group"), maintain_order=True)\
                    .agg(\
                        pl.col("*"),
                        (pl.col(in_x).shift(1)).alias(shift_x),\
                        (pl.col(in_y).shift(1)).alias(shift_y)\
                    )

        expanded = groups.explode(pl.exclude(self.get_name('group')))\
                      .drop_nulls([shift_x, shift_y])\
                      .with_columns(
                        pl.struct(
                            in_x, in_y,
                            shift_x, shift_y).map_elements(
                                lambda x : great_circle_distance(x[in_x], x[in_y], x[shift_x], x[shift_y]),
                                return_dtype=float
                            ).alias(self.get_name('out'))
                      )

        # Drop/Hide unused columns
        df._dataframe = expanded.drop([shift_x, shift_y])
        return df


class Speed(PipelineElement):
    """
    Given a dataframe with delta time and delta distance compute the speed
    """
    @damast.core.describe("Compute the speed")
    @damast.core.input({"delta_distance": {"unit": "km"},
                        "delta_time": {}})
    @damast.core.output({"speed": {"description": "speed of object", "unit": "km / h"}})
    def transform(self,
                  df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Compute distance between adjacent messages
        """
        dataframe = df._dataframe

        delta_distance = self.get_name("delta_distance")
        delta_time = self.get_name("delta_time")

        df._dataframe = dataframe.with_columns(
                (pl.col(delta_distance) / (pl.col(delta_time).dt.total_seconds()/3600)).alias("speed")
                        )

        return df

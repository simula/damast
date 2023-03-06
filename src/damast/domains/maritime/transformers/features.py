from typing import List

import numpy as np
import vaex
from astropy import units

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement
from damast.domains.maritime.data_specification import ColumnName
from damast.domains.maritime.math import great_circle_distance
from damast.core import DataSpecification

__all__ = ["DeltaDistance", ]


class DeltaDistance(PipelineElement):
    """
    Given a dataframe with `(latitude, longitude)` data, group messages by given column, 
    and sort them by another column. Then compute the distance between two messages, using 
    the :func:`damast.domains.maritime.math.great_circle_distance`.

    :param x_shift: True if one should compute the difference in latitude
    :param y_shift: True if one should compute the difference in longitude
    :param inplace: True if the transformer should work on the input dataframe, else return a copy.

    .. note::
        If both `x_shift` and `y_shift` is `True`, one computes the distance between two coordinates.
    """
    _x_shift: bool
    _y_shift: bool
    _inplace: bool

    def __init__(self, x_shift: bool, y_shift: bool, inplace: bool = True):
        self._x_shift = x_shift
        self._y_shift = y_shift
        self._inplace = inplace

    @property
    def x_shift(self):
        return self._x_shift

    @property
    def y_shift(self):
        return self._y_shift

    @damast.core.describe("Compute the ")
    @damast.core.input({"group": {"representation_type": int},
                        "sort": {"representation_type": "datetime64[ns]"},
                        "x": {"unit": units.deg},
                        "y": {"unit": units.deg}})
    @damast.core.output({"out": {"unit": units.km}})
    def transform(self,
                  df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Compute distance between adjacent messages
        """
        if not self._inplace:
            dataframe = df._dataframe.copy()
        else:
            dataframe = df._dataframe

        in_x = self.get_name("x")
        in_y = self.get_name("y")
        shift_x = f"{in_x}_shifted"
        shift_y = f"{in_y}_shifted"

        dataframe["INDEX"] = vaex.vrange(0, len(dataframe), dtype=int)

        shifted_x_array = np.zeros(len(dataframe))
        shifted_y_array = np.zeros(len(dataframe))

        # Group data and sort it. Only shift data within the same group
        groups = dataframe.groupby(by=self.get_name("group"))
        for _, group in groups:
            sorted_group = group.sort(self.get_name("sort"))
            # Add copy of in-column that should be shifted
            sorted_group.add_virtual_column(shift_x, sorted_group[in_x])
            sorted_group.add_virtual_column(shift_y, sorted_group[in_y])
            global_indices = sorted_group["INDEX"].evaluate()

            # Shift columns and assign to global output array
            sorted_group.shift(1, shift_x, inplace=True)
            shifted_x_array[global_indices] = sorted_group[shift_x].evaluate()
            sorted_group.shift(1, f"{self.get_name('y')}_shifted", inplace=True)
            shifted_y_array[global_indices] = sorted_group[shift_y].evaluate()

            del global_indices

        # Add global output array (and mask nans)
        dataframe[shift_x] = np.ma.masked_invalid(shifted_x_array)
        dataframe[shift_y] = np.ma.masked_invalid(shifted_y_array)

        # Add virtual column for greater-circle-distance computation
        dataframe.add_virtual_column(
            self.get_name("out"),
            dataframe.apply(great_circle_distance, [in_x, in_y, shift_x, shift_y], vectorize=True)
        )
        # Drop/Hide unused columns
        dataframe.drop(shift_x, inplace=True)
        dataframe.drop(shift_y, inplace=True)
        dataframe.drop("INDEX", inplace=True)

        # Add unit and data-specification to dataframe
        dataframe.units[self.get_name("out")] = units.km
        new_spec = DataSpecification(self.get_name("out"), unit=units.km)
        if self._inplace:
            df._metadata.columns.append(new_spec)
            return df
        else:
            metadata = df._metadata.columns.copy()
            metadata.append(new_spec)
            return AnnotatedDataFrame(dataframe, metadata=damast.core.MetaData(
                metadata))

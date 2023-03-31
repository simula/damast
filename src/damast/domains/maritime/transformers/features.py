import numpy as np
import vaex
from astropy import units

import damast.core
from damast.core import AnnotatedDataFrame, DataSpecification
from damast.core.dataprocessing import PipelineElement
from damast.domains.maritime.math import great_circle_distance

__all__ = ["DeltaDistance", ]


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

    @damast.core.describe("Compute the ")
    @damast.core.input({"group": {"representation_type": int},
                        "sort": {},
                        "x": {"unit": units.deg},
                        "y": {"unit": units.deg}})
    @damast.core.output({"out": {"unit": units.km}})
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
        tmp_column = f"{self.__class__.__name__}"
        if tmp_column in dataframe.column_names:
            raise RuntimeError(f"{self.__class__.__name__}.transform: Dataframe contains {tmp_column}")

        dataframe[tmp_column] = vaex.vrange(0, len(dataframe), dtype=int)

        shifted_x_array = np.zeros(len(dataframe))
        shifted_y_array = np.zeros(len(dataframe))

        # Group data and sort it. Only shift data within the same group
        groups = dataframe.groupby(by=self.get_name("group"))
        for _, group in groups:
            sorted_group = group.sort(self.get_name("sort"))
            # Add copy of in-column that should be shifted
            sorted_group.add_virtual_column(shift_x, sorted_group[in_x])
            sorted_group.add_virtual_column(shift_y, sorted_group[in_y])
            global_indices = sorted_group[tmp_column].evaluate()

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
        dataframe.drop(tmp_column, inplace=True)

        # Add unit and data-specification to dataframe
        dataframe.units[self.get_name("out")] = units.km
        new_spec = DataSpecification(self.get_name("out"), unit=units.km)
        df._metadata.columns.append(new_spec)
        return df

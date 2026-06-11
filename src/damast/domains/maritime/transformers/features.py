import polars as pl
import polars_h3

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement

__all__ = [
    "DeltaDistance",
    "Heading",
    "Speed",
    "AngularVelocity"
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

        group = self.get_name("group")
        in_x = self.get_name("x")
        in_y = self.get_name("y")
        shift_x = f"{in_x}_shifted"
        shift_y = f"{in_y}_shifted"

        tmp_column = f"{self.__class__.__name__}_tmp"
        assert tmp_column != self.get_name("out")
        if tmp_column in dataframe.compat.column_names:
            raise RuntimeError(f"{self.__class__.__name__}.transform: Dataframe contains {tmp_column}")

        use_columns = []
        target_shift_x = in_x
        target_shift_y = in_y

        if self.x_shift:
            use_columns.append(pl.col(in_x).shift(1).over(group).alias(shift_x))
            target_shift_x = shift_x

        if self.y_shift:
            use_columns.append(pl.col(in_y).shift(1).over(group).alias(shift_y))
            target_shift_y = shift_y

        # https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.shift.html
        # shift
        # Group data and sort it. Only shift data within the same group
        dataframe = dataframe.sort(
                        self.get_name("group"), self.get_name('sort')
                    ).with_columns(
                     *use_columns
                    )

        dataframe = dataframe.drop_nulls(
                          [shift_x, shift_y]
                      ).with_columns(
                        polars_h3.great_circle_distance(in_x, in_y, target_shift_x, target_shift_y, unit="km").alias(self.get_name('out'))
                      )

        # Drop/Hide unused columns
        df._dataframe = dataframe.drop([shift_x, shift_y])
        return df


class Speed(PipelineElement):
    """
    Given a dataframe with delta time and delta distance compute the speed
    """
    @damast.core.describe("Compute the speed")
    @damast.core.input({
        "delta_distance": {"unit": "km"},
        "delta_time": {"unit": "s"}}
    )
    @damast.core.output({"speed": {"description": "speed of object", "unit": "km / h"}})
    def transform(self,
                  df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Compute distance between adjacent messages
        """
        dataframe = df._dataframe

        delta_distance = self.get_name("delta_distance")
        delta_time = self.get_name("delta_time")

        df._dataframe = dataframe.filter(pl.col(delta_time) != 0).with_columns(
                (pl.col(delta_distance) / (pl.col(delta_time)/3600.0)).alias("speed")
        )
        return df


class Heading(PipelineElement):
    """
    Given a dataframe which computes the heading from known lat lon deltas
    """
    @damast.core.describe("Compute the heading between lat/lon given positions")
    @damast.core.input({"group": {"representation_type": int},
                        "sort": {},
                        "lat": {"unit": "deg" },
                        "lon": {"unit": "deg" }})
    @damast.core.output({
        "heading": {"description": "heading between two locations", "representation_type": float, "unit": "rad"},
        "delta_heading": {"description": "delta_heading between two locations", "representation_type": float, "unit": "rad"},
        "angular_velocity": { "description": "angular velocity", "representation_type": float, "unit": "rad / s"}
    })
    def transform(self,
                  df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        """
        Compute distance between adjacent messages
        """
        dataframe = df._dataframe

        group = self.get_name("group")
        sort_column = self.get_name("sort")

        in_lat = self.get_name("lat")
        in_lon = self.get_name("lon")

        # output
        heading = self.get_name("heading")
        delta_heading = self.get_name("delta_heading")
        angular_velocity = self.get_name("angular_velocity")

        delta_lon = "delta_lon"
        shift_lat = "shift_lat" # from this latitude

        dataframe = dataframe.sort(group, sort_column).with_columns(
                    pl.col(in_lon).diff().over(group).alias(delta_lon),
                    pl.col(in_lat).shift(-1).over(group).alias(shift_lat)
                ).filter(
                    pl.col(delta_lon).is_not_null()
                ).filter(
                    pl.col(shift_lat).is_not_null()
                )

        dataframe = dataframe.with_columns(
            pl.arctan2(
                pl.col(delta_lon).radians().sin() * pl.col(in_lat).radians().cos(),
                # Y component
                pl.col(shift_lat).radians().cos() * pl.col(in_lat).radians().sin() -
                pl.col(shift_lat).radians().sin() * pl.col(in_lat).radians().cos() * pl.col(delta_lon).radians().cos()
            ).alias(heading)
        ).drop(delta_lon, shift_lat)

        dataframe = dataframe.sort(group, sort_column).with_columns(
                    pl.col(heading).diff().alias(delta_heading),
                    pl.col(sort_column).diff().dt.total_seconds().alias("_delta_time")
                )

        dataframe = dataframe.with_columns(
                        (pl.col(delta_heading) / pl.col("_delta_time")).alias("angular_velocity")
                    ).drop("_delta_time")

        df._dataframe = dataframe.filter(
                    pl.col(delta_heading).is_not_null()
                )

        return df

class AngularVelocity(PipelineElement):
    """
    Given heading compute the angular velocity
    """
    def __init__(self):
        pass

    @damast.core.describe("Compute the angular velocity in radians/s")
    @damast.core.input({"group": {"representation_type": int},
                        "heading": { "representation_type": float, "unit": "rad" },
                        "time": { } # "representation_type": pl.Datetime }
    })
    @damast.core.output({"angular_velocity": {"description": "angular velocity", "unit": "rad / s"}})
    def transform(self,
                  df: AnnotatedDataFrame) -> AnnotatedDataFrame:

        group = self.get_name("group")
        heading = self.get_name("heading")
        time = self.get_name("time")

        dataframe = df.with_columns(
            pl.col(heading).diff().over(group).alias("_delta_heading"),
            pl.col(time).diff().dt.total_seconds().over(group).alias("_delta_time")
        )

        df._dataframe = dataframe.with_columns(
            (pl.col("_delta_heading") / pl.col("_delta_time")).alias("angular_velocity")
        ).drop("_delta_heading").drop("_delta_time")

        return df


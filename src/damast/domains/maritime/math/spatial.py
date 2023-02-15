# -----------------------------------------------------------
# This file contains multiple mathematical function to normalise data, compute distance
#
# (C) 2020 Pierre Bernabé, Oslo, Norway
# email pierbernabe@simula.no
# -----------------------------------------------------------

import numpy as np
import numpy.typing as npt
from numba import njit

__all__ = ["angle_sat_c",
           "great_circle_distance",
           "bearing",
           "reverse_bearing",
           "decdeg2dms",
           "dms2decdeg",
           "distance_sat_vessel",
           "chord_distance"]

EARTH_RADIUS: int = 6371  # km


@njit
def bearing(lat_1: npt.NDArray[np.float64],
            lon_1: npt.NDArray[np.float64],
            lat_2: npt.NDArray[np.float64],
            lon_2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Computes the bearing/heading between two points.

    The points are defined as
    :math:`(\\phi_1, \\lambda_1)`, :math:`(\\phi_2, \\lambda_2)`
    where :math:`\\phi_i` is the longitude, :math:`\\lambda_i` is the latitude of the points.

    Bearing is defined as the angle between the north-south line of the earth and the line connecting the target
    and the reference point.
    Heading is the angle you are currently navigating in.
    See: https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/

    :param lat_1: A sequence of latitudes (in degrees), :math:`\\phi_1`
    :param lon_1: A sequence of longitudes (in degrees), :math:`\\lambda_1`
    :param lat_2: A sequence of latitudes (in degrees), :math:`\\phi_2`
    :param lon_2: A sequence of longitudes (in degrees), :math:`\\lambda_2`
    :returns: The bearing between two points
    """
    lon_1 = np.radians(lon_1)
    lat_1 = np.radians(lat_1)
    lon_2 = np.radians(lon_2)
    lat_2 = np.radians(lat_2)
    x = np.cos(lat_2) * np.sin(lon_2 - lon_1)
    y = np.cos(lat_1) * np.sin(lat_2) - np.sin(lat_1) * np.cos(lat_2) * np.cos(lon_2 - lon_1)
    return np.degrees(np.arctan2(x, y))


def reverse_bearing(lat_1: npt.NDArray[np.float64],
                    lon_1: npt.NDArray[np.float64],
                    distance: npt.NDArray[np.float64],
                    bearing: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the position of an object, given its initial position, the initial bearing, and the distance
    it will travel.
    See: https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/

    :param lat_1: A sequence of latitudes (in degrees)
    :param lon_1: A sequence of longitudes (in degrees)
    :param distance: The distance traveled for each ship
    :param bearing: The bearing (in degrees) for each ship

    :returns: The new position of the object
"""
    lon_1 = np.radians(lon_1)
    lat_1 = np.radians(lat_1)
    bearing = np.radians(bearing)
    ad = distance / EARTH_RADIUS
    lat_2 = np.arcsin(np.sin(lat_1) * np.cos(ad) + np.cos(lat_1) * np.sin(ad) * np.cos(bearing))
    lon_2 = lon_1 + np.arctan2(np.sin(bearing) * np.sin(ad) * np.cos(lat_1), np.cos(ad) - np.sin(lat_1) * np.sin(lat_2))
    return np.degrees(lat_2), np.degrees(lon_2)


@njit
def decdeg2dms(dd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert decimal degrees (dd) to sexagesimal degrees (degrees, minutes, seconds)

    Uses the formula from https://en.wikipedia.org/wiki/Decimal_degrees#Example:

    .. math::

        D &= \\mathrm{trunc}(dd, 0)\\\\
        M &= \\mathrm{trunc}(60 \\times \\vert dd - D\\vert, 0)\\\\
        S &= 3600 \\times \\vert dd - D \\vert - 60 \\times M\\\\

    :param dd: The decimal degrees
    :returns: The degrees :math:`D`, minutes :math:`M`, seconds of a decimal :math:`S`
    """
    D = np.trunc(dd)
    M = np.trunc(60 * np.abs(dd - D))
    S = 3600 * np.abs(dd - D) - 60 * M
    return D, M, S


def dms2decdeg(degrees: npt.NDArray[np.float64],
               minutes: npt.NDArray[np.float64],
               seconds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert sexagesimal degrees (degrees, minutes, seconds) to decimal degrees (dd).

    Uses the formula from https://en.wikipedia.org/wiki/Decimal_degrees#Example:

    .. math::

        dd = D + \\frac{M}{60} + \\frac{S}{3600}

    :param degrees: The degrees (:math:`D`)
    :param minutes: The minutes (:math:`M`)
    :param seconds: The seconds (:math:`S`)
    :returns: The decimal representation of the latitude/longitude.

    """
    # Get sign of any zero-degree input
    sign = np.ones_like(degrees)
    sign[np.signbit(degrees)] = -1
    return degrees + sign * (minutes / 60 + seconds / 3600)


def great_circle_distance(lat_1: npt.NDArray[np.float64],
                          lon_1: npt.NDArray[np.float64],
                          lat_2: npt.NDArray[np.float64],
                          lon_2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Computes the great circle distance between two points.

    Uses the :math:`(\\phi_1, \\lambda_1)`, :math:`(\\phi_2, \\lambda_2)` using the
    `Haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_.

    .. math::

        d = 2 R \\arcsin \\left(\\sqrt{\\sin^2\\left(\\frac{\\phi_1 - \\phi_2}{2} \\right)
        + \\cos\\phi_1\\cos\\phi_2\\sin^2\\left(\\frac{\\lambda_1 - \\lambda_2}{2} \\right)
        }\\right)

    :param lat_1: A sequence of latitudes (in degrees), :math:`\\phi_1`
    :param lon_1: A sequence of longitudes (in degrees), :math:`\\lambda_1`
    :param lat_2: A sequence of latitudes (in degrees), :math:`\\phi_2`
    :param lon_2: A sequence of longitudes (in degrees), :math:`\\lambda_2`

    :returns: The great circle distance between sets of points """
    lon_1, lat_1, lon_2, lat_2 = map(np.radians, [lon_1, lat_1, lon_2, lat_2])
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.power(np.sin((lat_1 - lat_2) / 2), 2)
                                                + np.cos(lat_1) * np.cos(lat_2) * np.power(np.sin((lon_1 - lon_2) / 2),
                                                                                           2)))


def chord_distance(d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Computes the chord length depending on the greater circle distance.

    This is the shortest distance inside the earth between two points.

    .. math::
        \\mathrm{crd} = 2 R \\sin \\left(\\theta/2 \\right)

    where :math:`\\theta=\\frac{d}{R}`, `d` being the great circle distance.

    :param d: The great circle distance
    :returns: The shortest distance between two points
    """
    return 2 * EARTH_RADIUS * np.sin(d / (2 * EARTH_RADIUS))


def angle_sat_c(c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute angle between sub-satellite location and vessel given the coord length `c`.

    Using the fact that the triangle made by the sub-longitude/latitude of the satellite
    and the straight line through the earth made by extending the line from the satellite through
    the earth's core can be used to determine the angle between the vessel, sub-long/lat of the satellite
    and the satellite.

    .. note::
        This angle is used in :func:`distance_sat_vessel` to determine the distance from the satellite to the vessel.

    :param c: The length of the chord going from the satellite sub-longitude/latitude to the vessel
    :returns: The angle between the vessel, satellite sub-longitude/latitude and the satellite.
    """
    return np.pi - np.arccos(c / (2 * EARTH_RADIUS))


def distance_sat_vessel(v: npt.NDArray[np.float64],
                        s: npt.NDArray[np.float64],
                        alpha: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute distance to satellite from vessel by using the theorem Al-Kashi.

    Given a satellite at Point S and (sub longitude, sub latitude) at Point U and a vessel
    at Point V, the distance from the satellite to the vessel :math:`u=\\vert S-V\\vert`
    is given by

    .. math::

        u^2= s^2 + v^2 - 2 s v \\cos\\alpha

    :param v: The altitude of the satellite over the earth :math:`\\vert U - S\\vert`
    :param s: The `s` is the chord length from the vessel to the satellite
              sub longitude/latitude  :math:`\\vert U - V\\vert`
    :param alpha: The angle :math:`\\alpha=\\angle SUV`
    :returns: The distance :math:`u` from the satellite to the vessel
    """
    return np.sqrt(np.power(s, 2) + np.power(v, 2) - 2 * s * v * np.cos(alpha))

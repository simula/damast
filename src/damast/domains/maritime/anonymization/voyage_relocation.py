"""
Anonymization strategy `VoyageRelocation`: move each voyage to a random place and time.

Each voyage keeps its shape, speeds and timing relative to its own start, so the data stays useful
for learning movement, but it is rotated on the sphere to a random location and heading and shifted
in time. The strategy chains, in this order:

0. `OwnerRisk`, `DropHighOwnerRisk` - drop the vessels likely to identify a private person
   (unless `drop_high_owner_risk=False`)
1. `SegmentVoyages` - drop fixes at rest and split each vessel's track into voyages
2. `SuppressVoyageEndpoints` - drop the fixes close to where a voyage starts or ends
3. `RelocateVoyages` - rotate each voyage to a random place and heading, shift it in time
4. `PseudonymizeVoyages` - replace the voyage by a keyed pseudonym, keep only whitelisted columns

The shape of a long voyage can still be matched against external AIS data, so the strategy
protects less than its scrambled positions suggest; see the docs of `RelocateVoyages`.
"""
import hashlib
import hmac
import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement
from damast.domains.maritime.anonymization.strategy import (
    AISColumns,
    AnonymizationStrategy,
    Step,
    owner_risk_steps,
)
from damast.domains.maritime.anonymization.voyage_k_anonymity import (
    PseudonymizeVoyages,
    SegmentVoyages,
    SuppressVoyageEndpoints,
    load_key,
)

__all__ = [
    "RelocateVoyages",
    "VoyageRelocation",
]

#: Names of the temporary columns holding a voyage's rotation matrix, row by row
_R = [[f"_r{i}{j}" for j in range(3)] for i in range(3)]
_SHIFT = "_shift_ms"


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _to_origin(lat: float, lon: float) -> np.ndarray:
    """Rotation moving (lat, lon), in radians, to (0, 0), with north still pointing north."""
    return _ry(lat) @ _rz(-lon)


def voyage_relocation(key: bytes, voyage: str, start_lat: float, start_lon: float,
                      max_shift_s: float) -> tuple[np.ndarray, float]:
    """
    The random rotation and time shift of a voyage, drawn from a keyed hash of the voyage.

    The rotation moves the voyage's first fix to a point drawn uniformly on the sphere and turns
    the voyage around it by a uniform angle.

    Args:
        key: Secret key of the HMAC
        voyage: Key of the voyage, as `SegmentVoyages` makes it: "mmsi|start time"
        start_lat: Latitude of the voyage's first fix, in degrees
        start_lon: Longitude of the voyage's first fix, in degrees
        max_shift_s: Largest time shift, in seconds, either way

    Returns:
        The 3x3 rotation matrix, acting on unit vectors (x, y, z), and the time shift in seconds
    """
    digest = hmac.new(key, f"relocate|{voyage}".encode(), hashlib.sha256).digest()
    u = [int.from_bytes(digest[8 * i:8 * (i + 1)], "big") / 2 ** 64 for i in range(4)]
    target_lat = math.asin(2 * u[0] - 1)
    target_lon = 2 * math.pi * u[1] - math.pi
    spin = 2 * math.pi * u[2]
    rotation = (_to_origin(target_lat, target_lon).T @ _rx(spin)
                @ _to_origin(math.radians(start_lat), math.radians(start_lon)))
    return rotation, (2 * u[3] - 1) * max_shift_s


def _rotate(rotation: list[list[pl.Expr]], v: tuple[pl.Expr, pl.Expr, pl.Expr]) -> list[pl.Expr]:
    return [rotation[i][0] * v[0] + rotation[i][1] * v[1] + rotation[i][2] * v[2] for i in range(3)]


class RelocateVoyages(PipelineElement):
    """
    Move each voyage to a random place, heading and time.

    Each voyage is rotated on the sphere so that its first fix lands on a random point, and turned
    around that point by a random angle; distances, speeds and turns are kept, the course is turned
    with the voyage. Its times are shifted by a random offset of at most `max_shift_s` either way.
    Relocated voyages may cross land.

    The rotation and shift are drawn from an HMAC of the voyage, keyed as in `PseudonymizeVoyages`.
    Whoever holds the key can undo the relocation, so the output is only pseudonymized; for a public
    release leave `key_env` unset, which draws a random key per run.

    The shape and duration of a voyage are unchanged, so a long voyage can still be matched against
    external AIS data by its shape - keep voyages short and drop their ends with
    `SuppressVoyageEndpoints`.

    Args:
        key_env: Name of the environment variable holding the HMAC key as hex, or None for a
            random key per run
        max_shift_s: Largest time shift, in seconds, either way

    Raises:
        KeyError: On transform, if the environment variable `key_env` is not set
    """

    def __init__(self, key_env: str | None = None, max_shift_s: float = 365 * 86400.0):
        self.key_env = key_env
        self.max_shift_s = max_shift_s

    @describe("Rotate each voyage to a random place and heading, and shift it in time")
    @input({"group": {},
            "timestamp": {},
            "lat": {"unit": "deg"},
            "lon": {"unit": "deg"},
            "course": {"unit": "deg"}})
    @output({"group": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        group = self.get_name("group")
        timestamp = self.get_name("timestamp")
        lat = self.get_name("lat")
        lon = self.get_name("lon")
        course = self.get_name("course")
        key = load_key(self.key_env, self.__class__.__name__)

        def relocation(start: dict) -> dict:
            rotation, shift_s = voyage_relocation(key, start[group], start[lat], start[lon], self.max_shift_s)
            entries = {_R[i][j]: float(rotation[i, j]) for i in range(3) for j in range(3)}
            return {**entries, _SHIFT: round(shift_s * 1000)}

        # each voyage's relocation is drawn once and joined back - there are far fewer voyages than fixes
        dtype = pl.Struct({**{name: pl.Float64 for row in _R for name in row}, _SHIFT: pl.Int64})
        relocations = df.lazyframe.group_by(group).agg(
            pl.col(lat).sort_by(timestamp).first(), pl.col(lon).sort_by(timestamp).first()
        ).select(
            pl.col(group),
            pl.struct(group, lat, lon).map_elements(relocation, return_dtype=dtype).alias("_relocation"),
        ).unnest("_relocation")

        phi, lam, c = (pl.col(name).radians() for name in (lat, lon, course))
        rotation = [[pl.col(name) for name in row] for row in _R]
        x, y, z = _rotate(rotation, (phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin()))
        # direction of the course as a vector: sin(course) * east + cos(course) * north
        east = (-lam.sin(), lam.cos(), pl.lit(0.0))
        north = (-phi.sin() * lam.cos(), -phi.sin() * lam.sin(), phi.cos())
        heading = tuple(c.sin() * e + c.cos() * n for e, n in zip(east, north, strict=True))
        dx, dy, dz = _rotate(rotation, heading)

        lazyframe = df.lazyframe.join(relocations, on=group, how="left", maintain_order="left").with_columns(
            z.clip(-1.0, 1.0).arcsin().alias("_phi"),
            pl.arctan2(y, x).alias("_lam"),
            dx.alias("_dx"), dy.alias("_dy"), dz.alias("_dz"),
        )
        phi, lam = pl.col("_phi"), pl.col("_lam")
        d = (pl.col("_dx"), pl.col("_dy"), pl.col("_dz"))
        to_east = -lam.sin() * d[0] + lam.cos() * d[1]
        to_north = -phi.sin() * lam.cos() * d[0] - phi.sin() * lam.sin() * d[1] + phi.cos() * d[2]
        df.lazyframe = lazyframe.with_columns(
            phi.degrees().alias(lat),
            lam.degrees().alias(lon),
            (pl.arctan2(to_east, to_north).degrees() % 360).alias(course),
            pl.col(timestamp) + pl.duration(milliseconds=pl.col(_SHIFT)),
        ).drop(*(name for row in _R for name in row), _SHIFT, "_phi", "_lam", "_dx", "_dy", "_dz")
        return df


class VoyageRelocation(AnonymizationStrategy):
    """
    Anonymize AIS data by moving each voyage to a random place, heading and time.

    Chains `OwnerRisk` and `DropHighOwnerRisk` (if `drop_high_owner_risk`), `SegmentVoyages`,
    `SuppressVoyageEndpoints`, `RelocateVoyages` and `PseudonymizeVoyages` - the parameters are theirs, see there; `key_env` keys both the relocation and the pseudonyms.
    `columns`, `name` and `base_dir` are those of `AnonymizationStrategy`.

    Example:

    ```python
    released = VoyageRelocation(max_shift_s=30 * 86400).transform(df=adf)
    ```
    """

    def __init__(self, *, key_env: str | None = None, max_gap_s: float = 3600.0, rest_speed: float = 0.5,
                 radius_km: float = 2.0, min_fixes: int = 2, max_shift_s: float = 365 * 86400.0,
                 drop_high_owner_risk: bool = True,
                 columns: AISColumns = AISColumns(), name: str = "anonymize",
                 base_dir: str | Path = tempfile.gettempdir()):
        self.key_env = key_env
        self.drop_high_owner_risk = drop_high_owner_risk
        self.max_gap_s = max_gap_s
        self.rest_speed = rest_speed
        self.radius_km = radius_km
        self.min_fixes = min_fixes
        self.max_shift_s = max_shift_s
        super().__init__(columns=columns, name=name, base_dir=base_dir)

    def steps(self, columns: AISColumns) -> list[Step]:
        c = columns
        return [
            *(owner_risk_steps(c) if self.drop_high_owner_risk else []),
            Step("segment voyages", SegmentVoyages(max_gap_s=self.max_gap_s, rest_speed=self.rest_speed),
                 {"mmsi": c.mmsi, "timestamp": c.timestamp, "speed": c.speed, "status": c.status}),
            Step("suppress voyage endpoints", SuppressVoyageEndpoints(radius_km=self.radius_km,
                                                                      min_fixes=self.min_fixes),
                 {"group": "voyage", "sort": c.timestamp, "lat": c.lat, "lon": c.lon}),
            Step("relocate voyages", RelocateVoyages(key_env=self.key_env, max_shift_s=self.max_shift_s),
                 {"group": "voyage", "timestamp": c.timestamp, "lat": c.lat, "lon": c.lon, "course": c.course}),
            Step("pseudonymize voyages", PseudonymizeVoyages(key_env=self.key_env),
                 {"voyage": "voyage", "timestamp": c.timestamp, "lat": c.lat, "lon": c.lon, "speed": c.speed,
                  "course": c.course, "status": c.status, "ship_type": c.ship_type}),
        ]

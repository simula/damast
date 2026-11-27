"""
Privacy report of AIS data: what in it can identify a vessel or a person.

The report measures the data against the steps of the anonymization strategies, e.g. how many fixes
`KAnonymityFilter` would drop, so it tells before a release how much an anonymization costs, and
after one whether it worked.
"""
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import polars_h3
import yaml

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.domains.maritime.anonymization.owner_risk import OwnerRisk
from damast.domains.maritime.anonymization.strategy import AISColumns, owner_risk_steps
from damast.domains.maritime.anonymization.voyage_k_anonymity import AT_REST_STATUS

__all__ = [
    "IDENTIFIER_NAMES",
    "PrivacyReport",
    "identifier_columns",
]

#: Names of columns, other than MMSI and IMO number, that identify a vessel - compared in lower
#: case without separators, so "callSign" and "Call sign" both match
IDENTIFIER_NAMES: list[str] = ["callsign", "name", "shipname", "vesselname"]
#: Column holding the voyage pseudonym of an anonymized release, see `PseudonymizeVoyages`
VOYAGE_ID: str = "voyage_id"
#: Largest number of decimals the precision of a position is checked for
_MAX_DECIMALS: int = 8


def identifier_columns(names: list[str], columns: AISColumns = AISColumns()) -> list[str]:
    """
    The columns that directly identify a vessel: MMSI, IMO number, and those in `IDENTIFIER_NAMES`.

    Args:
        names: Names of the columns in the data
        columns: Names of the AIS columns in the data

    Returns:
        The identifying columns among `names`, MMSI and IMO number first
    """
    candidates = [columns.mmsi, columns.imo] + [
        n for n in names if re.sub(r"[^a-z]", "", n.lower()) in IDENTIFIER_NAMES]
    return [c for c in dict.fromkeys(candidates) if c in names]


def _percent(part: int, total: int) -> float:
    return round(100.0 * part / total, 2) if total else 0.0


def _shares(vessels: int, all_vessels: int, rows: int, all_rows: int) -> dict[str, Any]:
    """Vessels and rows affected, each with its share - vessels first, since a vessel is what identifies."""
    return {"vessels": vessels, "vessels_percent": _percent(vessels, all_vessels), "vessels_total": all_vessels,
            "rows": rows, "rows_percent": _percent(rows, all_rows)}


def _affected(shares: dict[str, Any], vessel: str | None, what: str) -> str:
    """
    A finding on the vessels with `what`, e.g. '12.5 % of vessels (40 of 320) have fixes at rest (3.1 % of rows)'
    - on rows only without a vessel column, and on voyages in an anonymized release.
    """
    if vessel is None:
        return f"{shares['rows_percent']} % of rows are {what}"
    noun = "voyages" if vessel == VOYAGE_ID else "vessels"
    return (f"{shares['vessels_percent']} % of {noun} ({shares['vessels']} of {shares['vessels_total']}) have {what}"
            f" ({shares['rows_percent']} % of rows)")


@dataclass
class PrivacyReport:
    """
    How identifying AIS data is - built with `PrivacyReport.assess`.

    Each section is a dict of plain values, so the report saves as YAML. A section whose columns
    are missing from the data holds only `skipped`, the reason. The sections:

    - `dataset`: rows, vessels, time span and bounding box
    - `direct_identifiers`: the columns that identify a vessel, with their number of distinct values
    - `owner_risk`: vessels and rows per `OwnerRisk` value
    - `at_rest`: rows at anchor, moored or slower than `rest_speed` - a berth identifies a vessel
    - `precision`: decimals of the positions, smallest time step between two fixes of a vessel
    - `k_anonymity`: rows and vessels in (H3 cell, time bucket) groups with fewer than `k` vessels,
      i.e. what `KAnonymityFilter` drops after `GeneralizeSpaceTime`
    - `minimization`: the columns `PseudonymizeVoyages` would not release

    `findings` sums up the risks found, one sentence each.

    Example:

    ```python
    report = PrivacyReport.assess(adf, k=5, time_bucket="1h")
    print(report)
    report.save_yaml("AIS_2026_07_01.privacy.yaml")
    ```
    """

    settings: dict[str, Any]
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    @classmethod
    def assess(cls, adf: AnnotatedDataFrame, *, columns: AISColumns = AISColumns(), k: int = 5,
               h3_resolution: int = 6, time_bucket: str = "10m", rest_speed: float = 0.5) -> "PrivacyReport":
        """
        Assess the privacy of AIS data.

        Example:

        ```python
        report = PrivacyReport.assess(adf, columns=AISColumns(mmsi="MMSI", lat="Latitude"))
        ```

        Args:
            adf: The data to assess, left unchanged
            columns: Names of the AIS columns in the data
            k: Minimum number of vessels per (cell, time bucket), see `KAnonymityFilter`
            h3_resolution: H3 resolution of the cells, see `GeneralizeSpaceTime`
            time_bucket: Interval time is truncated to, see `GeneralizeSpaceTime`
            rest_speed: Speed, in knots, below which a fix counts as at rest, see `SegmentVoyages`

        Returns:
            The report
        """
        report = cls(settings={"k": k, "h3_resolution": h3_resolution, "time_bucket": time_bucket,
                               "rest_speed": rest_speed})
        present = set(adf.lazyframe.collect_schema().names())
        df = adf.lazyframe.collect()
        if columns.timestamp in present and df[columns.timestamp].dtype == pl.String:
            # e.g. BarentsWatch's ISO 8601 text, with and without fractional seconds
            df = df.with_columns(pl.col(columns.timestamp).str.to_datetime(time_zone="UTC", strict=False))
        # the MMSI counts vessels; in an anonymized release only the voyage pseudonym is left
        vessel = columns.mmsi if columns.mmsi in present else (VOYAGE_ID if VOYAGE_ID in present else None)

        report.sections["dataset"] = report._dataset(df, columns, vessel)
        report.sections["direct_identifiers"] = report._direct_identifiers(df, columns)
        report.sections["owner_risk"] = report._owner_risk(adf, columns, present)
        report.sections["at_rest"] = report._at_rest(df, columns, present, vessel, rest_speed)
        report.sections["precision"] = report._precision(df, columns, present, vessel)
        report.sections["k_anonymity"] = report._k_anonymity(df, columns, present, vessel, k,
                                                             h3_resolution, time_bucket)
        report.sections["minimization"] = report._minimization(df, columns)
        return report

    def _dataset(self, df: pl.DataFrame, columns: AISColumns, vessel: str | None) -> dict[str, Any]:
        section: dict[str, Any] = {"rows": df.height, "columns": df.width,
                                   "vessels": df[vessel].n_unique() if vessel else None}
        if columns.timestamp in df.columns:
            section["start"] = str(df[columns.timestamp].min())
            section["end"] = str(df[columns.timestamp].max())
        if columns.lat in df.columns and columns.lon in df.columns:
            section["bbox"] = {"lat": [df[columns.lat].min(), df[columns.lat].max()],
                               "lon": [df[columns.lon].min(), df[columns.lon].max()]}
        return section

    def _direct_identifiers(self, df: pl.DataFrame, columns: AISColumns) -> dict[str, Any]:
        found = {c: df[c].n_unique() for c in identifier_columns(df.columns, columns)}
        if found:
            self.findings.append(f"Direct identifiers present: {', '.join(found)}")
        return {"columns": found}

    def _owner_risk(self, adf: AnnotatedDataFrame, columns: AISColumns, present: set[str]) -> dict[str, Any]:
        missing = [c for c in (columns.mmsi, columns.ship_type, columns.imo) if c not in present]
        if missing:
            return {"skipped": f"missing columns: {', '.join(missing)}"}
        columns = columns.available(list(present))

        with tempfile.TemporaryDirectory() as base_dir:
            pipeline = DataProcessingPipeline(name="owner risk", base_dir=base_dir)
            for step in owner_risk_steps(columns)[:2]:  # without the step that drops
                pipeline.add(step.name, step.transformer, name_mappings=step.name_mappings)
            risk = pipeline.transform(df=adf.copy()).lazyframe.select(columns.mmsi, "owner_risk").collect()

        counts = risk.group_by("owner_risk").agg(pl.len().alias("rows"),
                                                 pl.col(columns.mmsi).n_unique().alias("vessels"))
        vessels = risk[columns.mmsi].n_unique()
        section: dict[str, Any] = {"report_class": columns.report_class, "length": columns.length}
        for level in (OwnerRisk.HIGH, OwnerRisk.MEDIUM, OwnerRisk.LOW):
            row = counts.filter(pl.col("owner_risk") == level)
            section[level] = _shares(int(row["vessels"].sum()), vessels, int(row["rows"].sum()), risk.height)

        high = section[OwnerRisk.HIGH]
        if high["vessels"]:
            self.findings.append(f"{high['vessels_percent']} % of vessels ({high['vessels']} of {vessels}) likely"
                                 f" identify a private person (owner risk high) - {high['rows_percent']} % of rows")
        return section

    def _at_rest(self, df: pl.DataFrame, columns: AISColumns, present: set[str], vessel: str | None,
                 rest_speed: float) -> dict[str, Any]:
        conditions = []
        if columns.status in present:
            conditions.append(pl.col(columns.status).is_in(AT_REST_STATUS))
        if columns.speed in present:
            conditions.append(pl.col(columns.speed) < rest_speed)
        if not conditions:
            return {"skipped": f"missing columns: {columns.status}, {columns.speed}"}

        resting = df.filter(pl.any_horizontal(conditions).fill_null(False))
        if vessel is None:
            section = {"rows": resting.height, "rows_percent": _percent(resting.height, df.height)}
        else:
            section = _shares(resting[vessel].n_unique(), df[vessel].n_unique(), resting.height, df.height)

        if resting.height:
            self.findings.append(_affected(section, vessel, "fixes at rest")
                                 + " - berths and anchorages identify a vessel")
        return section

    def _precision(self, df: pl.DataFrame, columns: AISColumns, present: set[str],
                   vessel: str | None) -> dict[str, Any]:
        section: dict[str, Any] = {}
        for name in (columns.lat, columns.lon):
            if name not in present or not df[name].dtype.is_float():
                continue
            values = df[name].drop_nulls()
            section[f"{name}_decimals"] = next(
                (d for d in range(_MAX_DECIMALS + 1) if ((values - values.round(d)).abs() < 1e-9).all()),
                _MAX_DECIMALS)
        if vessel and columns.timestamp in present and df[columns.timestamp].dtype.is_temporal():
            step = df.sort(vessel, columns.timestamp).select(
                pl.col(columns.timestamp).diff().over(vessel).dt.total_seconds(fractional=True).alias("step")
            ).filter(pl.col("step") > 0)["step"].min()
            section["min_time_step_s"] = step

        decimals = max((v for k, v in section.items() if k.endswith("_decimals")), default=0)
        if decimals == _MAX_DECIMALS:
            self.findings.append(f"Positions keep their full precision, {_MAX_DECIMALS} decimals or more")
        elif decimals >= 4:
            self.findings.append(f"Positions are given to {decimals} decimals, about {111_000 / 10 ** decimals:g} m")
        return section

    def _k_anonymity(self, df: pl.DataFrame, columns: AISColumns, present: set[str], vessel: str | None,
                     k: int, h3_resolution: int, time_bucket: str) -> dict[str, Any]:
        missing = [c for c in (columns.timestamp, columns.lat, columns.lon) if c not in present]
        if vessel is None:
            missing.append(columns.mmsi)
        if missing:
            return {"skipped": f"missing columns: {', '.join(missing)}"}

        # the cells and buckets of GeneralizeSpaceTime, the group sizes of KAnonymityFilter
        groups = ["_cell", "_bucket"]
        sizes = df.select(
            vessel,
            polars_h3.latlng_to_cell(columns.lat, columns.lon, h3_resolution).alias("_cell"),
            pl.col(columns.timestamp).dt.truncate(time_bucket).alias("_bucket"),
        ).with_columns(pl.col(vessel).n_unique().over(groups).alias("_vessels"))
        below = sizes.filter(pl.col("_vessels") < k)

        section = {"vessel_column": vessel,
                   "groups": sizes.select(groups).n_unique(),
                   "groups_below_k": below.select(groups).n_unique() if below.height else 0,
                   "below_k": _shares(below[vessel].n_unique(), df[vessel].n_unique(), below.height, df.height)}

        if below.height:
            self.findings.append(_affected(section["below_k"], vessel, f"fixes that are not {k}-anonymous in"
                                           f" (H3 resolution {h3_resolution}, {time_bucket}) cells"))
        return section

    def _minimization(self, df: pl.DataFrame, columns: AISColumns) -> dict[str, Any]:
        released = {VOYAGE_ID, columns.timestamp, columns.lat, columns.lon, columns.speed, columns.course,
                    columns.status, columns.ship_type}
        not_released = [c for c in df.columns if c not in released]
        if not_released:
            self.findings.append(f"Columns not needed for a release: {', '.join(not_released)}")
        return {"not_released": not_released}

    def to_dict(self) -> dict[str, Any]:
        """
        The report as plain values.

        Returns:
            Dict of `settings`, each section and `findings`
        """
        return {"settings": self.settings, **self.sections, "findings": self.findings}

    def save_yaml(self, filename: str | Path) -> Path:
        """
        Save the report as YAML.

        Args:
            filename: Path of the file, its parent directories are created

        Returns:
            The path of the file
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(self))
        return path

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

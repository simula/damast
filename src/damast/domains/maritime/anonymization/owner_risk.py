"""
Whether the movements of a vessel can identify a private person, assessed from AIS data alone.

Ship and owner registers are not used: the open ones (Fiskeridirektoratet's vessel register,
Brønnøysund's Enhetsregisteret) return the names of private owners and sole proprietors with every
answer, so they cannot be queried without retrieving personal data. Without them a vessel is
assumed to be linked to a person unless its AIS data shows that it is a commercial ship.
"""
import polars as pl

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement
from damast.domains.maritime.ais.labels import normalise_label
from damast.domains.maritime.ais.vessel_types import ShipType

__all__ = [
    "OwnerRisk",
    "valid_imo",
]

#: MMSI prefixes of devices carried by a person: 970 AIS-SART, 972 man overboard, 974 EPIRB-AIS
PERSONAL_DEVICE_PREFIXES: list[str] = ["970", "972", "974"]
#: MMSI prefix of aids to navigation - not a vessel
AID_TO_NAVIGATION_PREFIX: str = "99"
#: Ship types of sailing and pleasure craft
PLEASURE_CRAFT_TYPES: list[int] = [36, 37]
#: Ship type of fishing vessels - often owned by a sole proprietor, whether or not it has an IMO number
FISHING_TYPE: int = 30


def valid_imo(imo: pl.Expr) -> pl.Expr:
    """
    Whether an IMO number is valid: seven digits, the last being the check digit.

    The check digit is the last digit of the sum of the first six digits weighted 7, 6, ..., 2.

    Args:
        imo: Expression of the IMO number, as integer

    Returns:
        Boolean expression, False for null
    """
    digit = [(imo // 10 ** (6 - i)) % 10 for i in range(7)]
    checksum = sum(d * (7 - i) for i, d in enumerate(digit[:6]))
    return (imo.is_between(1_000_000, 9_999_999) & (checksum % 10 == digit[6])).fill_null(False)


def _imo_number(imo: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """The IMO number as integer - from text it is read as digits after an optional 'IMO' prefix."""
    if dtype != pl.String:
        return imo.cast(pl.Int64, strict=False)
    return imo.str.extract(r"^\s*(?:IMO)?\s*(\d+)\s*$").cast(pl.Int64, strict=False)


def _ship_type_codes(labels: pl.Series) -> pl.Series:
    """Codes of ship types written as labels, e.g. 'Pleasure' - null for an unknown label."""
    mapping = ShipType.get_mapping()
    codes = {label: mapping.get(normalise_label(label)) for label in labels.unique().drop_nulls().to_list()}
    return labels.replace_strict(codes, default=None, return_dtype=pl.Int64)


class OwnerRisk(PipelineElement):
    """
    Add `owner_risk`: whether a vessel's movements can identify a private person.

    Assessed per MMSI over all its rows, from AIS data only - see the module for why registers are
    not used. The values:

    - "high": a device carried by a person (MMSI 970, 972 or 974...), or a sailing or pleasure craft
      (ship type 36 or 37); with `report_class`, also a class B transponder without a valid IMO
      number - typically a leisure boat or a small one-person fishing boat
    - "low": an aid to navigation (MMSI 99...), or a ship with a valid IMO number that is neither a
      fishing vessel nor a pleasure craft - in practice run by a company
    - "medium": everything else, which may well be owned or crewed by a single person, e.g. small
      boats without an IMO number and fishing vessels

    "low" is not "none": the crew of a commercial ship can still be identifiable.

    Sources encode the inputs differently, and each is accepted as it comes:

    | Source         | ship_type               | imo                         | report_class               |
    |----------------|-------------------------|-----------------------------|----------------------------|
    | BarentsWatch   | `shipType`, code        | `imoNumber`, integer        | `reportClass`, "A"/"B"     |
    | AISDK          | `Ship type`, label      | `IMO`, text or "Unknown"    | `Type of mobile`, "Class B"|
    | MarineCadastre | `VesselType`, code      | `IMO`, text, "IMO9074729"   | `TransceiverClass`, "A"/"B"|

    A ship type label, as AISDK writes it, is translated with `ShipType`; an unknown label counts as
    no ship type. MarineCadastre data before 2018 may use other vessel type codes - check them first.

    BarentsWatch's open data leaves out fishing vessels under 15 m and leisure craft under 45 m, so
    few of its vessels are "high"; AISDK and MarineCadastre include them.

    Example:

    ```python
    pipeline.add("owner risk", OwnerRisk(report_class="Type of mobile"),
                 name_mappings={"mmsi": "MMSI", "ship_type": "Ship type", "imo": "IMO"})
    ```

    Args:
        report_class: Name of the column holding the AIS transponder class ("A"/"B", "Class A"/
            "Class B"), or None if the data has none
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __init__(self, report_class: str | None = None):
        self.report_class = report_class

    @describe("Assess per MMSI whether the vessel's movements can identify a private person")
    @input({"mmsi": {"representation_type": int},
            "ship_type": {},
            "imo": {}})
    @output({"owner_risk": {"representation_type": str,
                            "description": "whether the vessel can identify a private person: high, medium or low"}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        mmsi = self.get_name("mmsi")
        ship_type_column = self.get_name("ship_type")
        imo_column = self.get_name("imo")
        schema = df.lazyframe.collect_schema()

        ship_type = pl.col(ship_type_column)
        if schema[ship_type_column] == pl.String:
            ship_type = ship_type.map_batches(_ship_type_codes, return_dtype=pl.Int64)
        lazyframe = df.lazyframe.with_columns(
            ship_type.alias("_ship_type"),
            _imo_number(pl.col(imo_column), schema[imo_column]).alias("_imo"),
        )

        def any_row(condition: pl.Expr) -> pl.Expr:
            return condition.fill_null(False).any().over(mmsi)

        mmsi_text = pl.col(mmsi).cast(pl.String)
        has_imo = any_row(valid_imo(pl.col("_imo")))
        personal = any_row(mmsi_text.str.slice(0, 3).is_in(PERSONAL_DEVICE_PREFIXES)
                           | pl.col("_ship_type").is_in(PLEASURE_CRAFT_TYPES))
        if self.report_class is not None:
            transponder = (pl.col(self.report_class).cast(pl.String).str.to_uppercase()
                           .str.replace(r"^\s*CLASS\s*", "").str.strip_chars())
            personal = personal | (any_row(transponder == "B") & ~has_imo)
        commercial = has_imo & ~any_row(pl.col("_ship_type").is_in([FISHING_TYPE, *PLEASURE_CRAFT_TYPES]))
        no_vessel = mmsi_text.str.starts_with(AID_TO_NAVIGATION_PREFIX)

        df.lazyframe = lazyframe.with_columns(
            pl.when(personal).then(pl.lit(self.HIGH))
            .when(commercial | no_vessel).then(pl.lit(self.LOW))
            .otherwise(pl.lit(self.MEDIUM))
            .alias(self.get_name("owner_risk"))
        ).drop("_ship_type", "_imo")
        return df

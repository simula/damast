import pytest

from damast.domains.maritime.ais.vessel_types import (
    DriftingLonglines,
    Fishing,
    PotsAndTraps,
    ShipType,
    VesselType,
)


def test_vessel_types():
    assert Fishing.typename() == "fishing"
    assert PotsAndTraps.typename() == "pots_and_traps"

    vessel_types = VesselType.get_types()

    assert len(vessel_types) > 0

    vessel_type_names = [x.typename() for x in vessel_types]
    assert "fishing" in vessel_type_names

    VesselType._initialize_types()
    assert VesselType._all_types is not None
    print(VesselType._all_types)

    id = VesselType.to_id(klass=Fishing)
    assert VesselType.by_id(identifier=id) == Fishing

    assert VesselType.by_name(name="drifting_longlines") == DriftingLonglines
    assert VesselType.to_id(klass=DriftingLonglines) == DriftingLonglines.to_id()
    assert VesselType["drifting_longlines"] == DriftingLonglines.to_id(klass=DriftingLonglines)

    with pytest.raises(KeyError):
        VesselType.by_name(name="foo")


# --- ShipType: the ITU-R M.1371 code mapping ------------------------------------------------

#: Every spelling the Danish AIS (aisdk) source uses, with the code the standard assigns it
AISDK_SHIP_TYPES = [
    ("Cargo", 70), ("Fishing", 30), ("Passenger", 60), ("Undefined", 0), ("Tanker", 80),
    ("Sailing", 36), ("Pleasure", 37), ("Other", 90), ("Tug", 52), ("Pilot", 50),
    ("HSC", 40), ("SAR", 51), ("Dredging", 33), ("Military", 35), ("Law enforcement", 55),
    ("Towing", 31), ("Port tender", 53), ("Diving", 34), ("Not party to conflict", 59),
    ("Towing long/wide", 32), ("Medical", 58), ("Anti-pollution", 54), ("WIG", 20),
    ("Spare 2", 57), ("Spare 1", 56),
]


@pytest.mark.parametrize("label,code", AISDK_SHIP_TYPES)
def test_ship_type_resolves_a_sources_spelling(label: str, code: int):
    """
    A source that writes the type as a label rather than a code has to land on the standard,
    or its rows are lost. These are the spellings aisdk actually uses.
    """
    assert ShipType.to_code(label) == code


def test_ship_type_reserved_has_no_single_code():
    """The standard reserves a range, so 'Reserved' names no one code - the caller decides."""
    assert ShipType.by_code(5) == ShipType.RESERVED

    with pytest.raises(KeyError, match="not a known ship type"):
        ShipType.to_code("Reserved")


@pytest.mark.parametrize("code,expected", [
    (0, "not_available"),
    (5, "reserved"),
    (20, "wing_in_ground"),
    (37, "pleasure_craft"),
    (38, "reserved"),
    (80, "tanker"),
    (81, "tanker_hazardous_a"),
    (84, "tanker_hazardous_d"),
    (86, "tanker_reserved"),
    (89, "tanker"),
    (99, "other"),
])
def test_ship_type_by_code(code: int, expected: str):
    assert ShipType.by_code(code) == expected


def test_by_code_covers_every_legal_value():
    """A code is two digits, so nothing in range may be unnamed."""
    assert all(isinstance(ShipType.by_code(code), str) for code in range(100))


@pytest.mark.parametrize("code", [-1, 100, 1.5, True, "80"])
def test_by_code_rejects_what_is_not_a_code(code):
    with pytest.raises(ValueError, match="is not a code between 0 and 99"):
        ShipType.by_code(code)


@pytest.mark.parametrize("code,expected", [
    (80, 80), (84, 80), (89, 80),   # the hazard subdivisions collapse onto the category
    (37, 37),                       # a distinct type is its own category
    (5, None), (38, None),          # reserved belongs to no category
])
def test_ship_type_category(code: int, expected: int | None):
    assert ShipType.category(code) == expected


def test_codes_spans_the_decade_of_a_subdivided_category():
    assert ShipType.codes("tanker") == list(range(80, 90))
    assert ShipType.codes("Cargo") == list(range(70, 80))
    # a distinct type is a single code, not a decade
    assert ShipType.codes("tug") == [52]


def test_to_code_ignores_case_and_punctuation():
    for spelling in ["search_and_rescue", "Search And Rescue", "SEARCH-AND-RESCUE", " sar "]:
        assert ShipType.to_code(spelling) == 51


def test_to_code_resolves_a_hazard_subdivision():
    assert ShipType.to_code("tanker_hazardous_a") == 81
    assert ShipType.by_code(81) == "tanker_hazardous_a"


def test_to_code_rejects_an_unknown_type():
    with pytest.raises(KeyError, match="not a known ship type"):
        ShipType.to_code("submarine")


def test_get_mapping_covers_names_and_aliases():
    mapping = ShipType.get_mapping()

    assert mapping["tanker"] == 80
    assert mapping["tanker_hazardous_a"] == 81
    assert mapping["sar"] == 51
    assert mapping["hsc"] == 40
    # reserved names a range, so it is not in a name-to-code mapping
    assert ShipType.RESERVED not in mapping
    # every entry round-trips back to a name of its own category
    for name, code in mapping.items():
        assert ShipType.category(code) is not None, name

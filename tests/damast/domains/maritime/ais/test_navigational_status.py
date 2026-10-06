import pytest

from damast.domains.maritime.ais.navigational_status import AISNavigationalStatus

#: Every spelling the Danish AIS (aisdk) source uses, with the code the standard assigns it
AISDK_NAV_STATUS = [
    ("Under way using engine", 0), ("At anchor", 1), ("Not under command", 2),
    ("Restricted maneuverability", 3), ("Constrained by her draught", 4), ("Moored", 5),
    ("Aground", 6), ("Engaged in fishing", 7), ("Under way sailing", 8),
    ("Reserved for future amendment [HSC]", 9), ("Reserved for future amendment [WIG]", 10),
    ("Power-driven vessel towing astern", 11),
    ("Power-driven vessel pushing ahead or towing alongside", 12),
    ("Reserved for future use", 13), ("AIS-SART (active)", 14), ("Unknown value", 15),
]


@pytest.mark.parametrize("label,code", AISDK_NAV_STATUS)
def test_to_code_resolves_a_sources_spelling(label: str, code: int):
    """A source reporting the status as text has to land on the standard, or its rows are lost."""
    assert AISNavigationalStatus.to_code(label) == code


def test_to_code_ignores_case_and_punctuation():
    for spelling in ["AtAnchor", "at_anchor", "At Anchor", " AT-ANCHOR "]:
        assert AISNavigationalStatus.to_code(spelling) == 1


def test_to_code_rejects_an_unknown_status():
    with pytest.raises(KeyError, match="not a known navigational status"):
        AISNavigationalStatus.to_code("under way using sails")


def test_get_mapping_covers_every_member():
    mapping = AISNavigationalStatus.get_mapping()

    assert set(mapping.values()) == set(AISNavigationalStatus.get_values())
    # the aliases are extra spellings, not extra codes
    assert len(mapping) > len(list(AISNavigationalStatus))

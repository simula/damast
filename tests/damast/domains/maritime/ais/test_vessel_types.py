import pytest

from damast.domains.maritime.ais.vessel_types import (
    DriftingLonglines,
    Fishing,
    PotsAndTraps,
    VesselType
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

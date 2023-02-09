from damast.domains.maritime.data_specification import MMSI


def test_mmsi():
    assert MMSI.min_value < MMSI.max_value

    mmsi = MMSI(mmsi=200000000)
    assert mmsi.country_iso_code == 200
    assert mmsi.national_id == 0

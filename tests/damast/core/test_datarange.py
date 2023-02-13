import pytest

from damast.core.datarange import MinMax, CyclicMinMax, DataRange, ListOfValues


@pytest.mark.parametrize(["min", "max", "value", "is_in_range"],
                         [
                             [0, 1, 0, True],
                             [0, 1, 1, True],
                             [0, 1, -1, False],
                             [-1.2, 2.7, -1.2, True],
                             [-1.2, 2.7, 2.7, True],
                             [-1.2, 2.7, -2.7, False],
                             [-1.2, 2.7, 2.71, False],
                             [-1, 2, 1.5, True],
                             [-1, 2, -1.0, True],
                             [-1, 2, 2.0, True],
                             [-1, 2, 2.5, False]
                         ])
def test_min_max(min, max, value, is_in_range):
    mm = MinMax(min=min, max=max)
    assert mm.is_in_range(value) == is_in_range
    if is_in_range:
        assert value in mm
    else:
        assert value not in mm

    cm = CyclicMinMax(min=min, max=max)
    assert cm.is_in_range(value) == is_in_range
    if is_in_range:
        assert value in cm
    else:
        assert value not in cm

@pytest.mark.parametrize(["data", "expected_instance"],
                         [
                             [{"ListOfValues": [0,1,2]}, ListOfValues([0,1,2])],
                             [{"MinMax": {"min": 0, "max": 1}}, MinMax(0,1)],
                             [{"CyclicMinMax": {"min": 0, "max": 1}}, CyclicMinMax(0, 1)]
                         ])
def test_data_range_from_dict(data, expected_instance):

    instance = DataRange.from_dict(data=data)
    assert instance == expected_instance
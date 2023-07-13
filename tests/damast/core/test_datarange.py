import numpy as np
import pytest

from damast.core.datarange import CyclicMinMax, DataRange, ListOfValues, MinMax


@pytest.mark.parametrize(["min", "max", "value", "is_in_range", "allow_missing"],
                         [
                             [10, 10, 10, True, True],
                             [0, 1, 0, True, True],
                             [0, 1, 1, True, True],
                             [0, 1, -1, False, True],
                             [-1.2, 2.7, -1.2, True, True],
                             [-1.2, 2.7, 2.7, True, True],
                             [-1.2, 2.7, -2.7, False, True],
                             [-1.2, 2.7, 2.71, False, True],
                             [-1, 2, 1.5, True, True],
                             [-1, 2, -1.0, True, True],
                             [-1, 2, 2.0, True, True],
                             [-1, 2, 2.5, False, True],
                             [0, 1, None, True, True],
                             [0, 1, None, False, False],
                             [np.timedelta64(0), np.timedelta64(10), np.timedelta64("NaT"), True, True],
                             [np.timedelta64(0), np.timedelta64(10), np.timedelta64("NaT"), False, False]
])
def test_min_max(min, max, value, is_in_range, allow_missing):
    mm = MinMax(min=min, max=max, allow_missing=allow_missing)
    assert mm.is_in_range(value=value) == is_in_range
    if is_in_range:
        assert value in mm
    else:
        assert value not in mm

    cm = CyclicMinMax(min=min, max=max, allow_missing=allow_missing)
    assert cm.is_in_range(value=value) == is_in_range
    if is_in_range:
        assert value in cm
    else:
        assert value not in cm


@pytest.mark.parametrize(["data", "expected_instance"],
                         [
                             [{"ListOfValues": [0, 1, 2]}, ListOfValues([0, 1, 2])],
                             [{"MinMax": {"min": 0, "max": 1}}, MinMax(0, 1)],
                             [{"CyclicMinMax": {"min": 0, "max": 1}}, CyclicMinMax(0, 1)]
])
def test_data_range_from_dict(data, expected_instance):

    instance = DataRange.from_dict(data=data)
    assert instance == expected_instance

    data_dict = instance.to_dict()
    instance = DataRange.from_dict(data=data_dict)
    assert expected_instance == instance

import numpy as np

from damast.domains.maritime.math.spatial import (
    great_circle_distance,
    haversine_distance,
)


def test_haversine_distance_is_great_circle_distance():
    """haversine_distance is an alias for great_circle_distance (same formula, same function)."""
    assert haversine_distance is great_circle_distance


def test_haversine_distance_known_value():
    # Oslo (59.9139 N, 10.7522 E) to Bergen (60.3913 N, 5.3221 E), ~300 km apart
    distance = haversine_distance(59.9139, 10.7522, 60.3913, 5.3221)
    assert np.isclose(distance, 306, atol=5)


def test_haversine_distance_zero_for_identical_points():
    assert haversine_distance(59.9139, 10.7522, 59.9139, 10.7522) == 0.0

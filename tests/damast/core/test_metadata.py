import datetime

import astropy.units as units
import pytest

from damast.core.annotations import Annotation, Change, History
from damast.core.metadata import (
    MinMax,
    DataSpecification,
    MetaData,
    DataCategory
)


@pytest.mark.parametrize(["min", "max", "value", "is_in_range"],
                         [
                             [0, 1, 0, True],
                             [0, 1, 1, True],
                             [0, 1, -1, False],
                             [-1.2, 2.7, -1.2, True],
                             [-1.2, 2.7, 2.7, True],
                             [-1.2, 2.7, -2.7, False],
                             [-1.2, 2.7, 2.71, False],
                         ])
def test_min_max(min, max, value, is_in_range):
    mm = MinMax(min=min, max=max)
    assert mm.is_in_range(value) == is_in_range


@pytest.mark.parametrize(["name", "category", "is_optional",
                          "abbreviation", "representation_type",
                          "missing_value", "unit", "precision",
                          "value_range", "value_meanings",
                          "raises"],
                         [
                             ["test-data-spec", 0, False,
                              "tds", int, -1, units.m, 0.01,
                              MinMax(0, 100), {0: 'min value', 100: 'max value'},
                              False],
                             ["test-data-spec", 0, False,
                              "tds", int, -1, units.m, 0.01,
                              MinMax(0, 100), {-10: 'min value', 100: 'max value'},
                              True]
                         ])
def test_data_specification(name, category, is_optional,
                            abbreviation, representation_type,
                            missing_value, unit, precision,
                            value_range, value_meanings,
                            raises):
    exception = None
    try:
        ds = DataSpecification(name=name,
                               category=category,
                               is_optional=is_optional,
                               abbreviation=abbreviation,
                               representation_type=representation_type,
                               missing_value=missing_value,
                               unit=unit,
                               precision=precision,
                               value_range=value_range,
                               value_meanings=value_meanings
                               )

        assert ds.name == name
        assert ds.category == category
        assert ds.abbreviation == abbreviation
        assert ds.representation_type == representation_type
        assert ds.missing_value == missing_value
        assert ds.unit == unit
        assert ds.precision == precision
        assert ds.value_range == value_range
    except ValueError as ve:
        exception = ve

    if raises:
        assert exception is not None
    else:
        assert exception is None


def test_annotation():
    with pytest.raises(ValueError) as e:
        Annotation(name=Annotation.Key.Comment)

    with pytest.raises(ValueError) as e:
        Annotation(name=Annotation.Key.License)

    a = Annotation(name=Annotation.Key.Comment,
                   value="test-comment")

    assert a.name == Annotation.Key.Comment
    assert a.value == "test-comment"


def test_change():
    title = "new-change"
    description = "elaborate description of change"

    c = Change(title, description)
    assert c.title == title
    assert c.description == description
    assert type(c.timestamp) is datetime.datetime
    assert c.timestamp < datetime.datetime.utcnow()


def test_history():
    history = History()
    assert history.name == Annotation.Key.History


def test_metadata():
    column_a = DataSpecification(name="a", category=DataCategory.STATIC)
    column_b = DataSpecification(name="b", category=DataCategory.DYNAMIC)

    columns = [column_a, column_b]

    annotation_history = History()
    annotation_comment = Annotation(name=Annotation.Key.Comment,
                                    value="test-comment")

    annotations = [
        annotation_history,
        annotation_comment
    ]

    md = MetaData(columns=columns,
                  annotations=annotations)

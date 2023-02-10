import datetime

import astropy.units as units
import pytest

from damast.core.annotations import Annotation, Change, History
from damast.core.metadata import (
    DataSpecification,
    MetaData,
    DataCategory
)

from damast.core.datarange import MinMax, CyclicMinMax



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
                             ["test-data-spec-out-of-range", 0, False,
                              "tds", int, -1, units.m, 0.01,
                              MinMax(0, 100), {-10: 'min value', 100: 'max value'},
                              True],
                             ["latitude-in-range", 1, False,
                             "lat", float, None, units.deg, 0.01,
                             CyclicMinMax(-90, 90), {-90.0: 'min value', 90.0: 'max value'},
                             False],
                             ["latitude-out-of-range", 1, False,
                              "lat", float, None, units.deg, 0.01,
                              CyclicMinMax(-90, 90), {-91.0: 'min value', 90.0: 'max value'},
                              True],
                             ["longitude-in-range", 1, False,
                             "lon", float, None, units.deg, 0.01,
                             CyclicMinMax(-180.0, 180.0), {-180.0: 'min value', 180.0: 'max value'},
                             False]
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

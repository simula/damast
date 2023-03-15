import datetime

import astropy.units as units
import pytest
import yaml

from damast.core.annotations import Annotation, Change, History
from damast.core.datarange import CyclicMinMax, MinMax
from damast.core.metadata import (
    DataCategory,
    DataSpecification,
    MetaData,
    Status
    )


@pytest.mark.parametrize(["name", "category", "is_optional",
                          "abbreviation", "representation_type",
                          "missing_value", "unit", "precision",
                          "value_range", "value_meanings",
                          "raises"],
                         [
                             ["test-data-spec", DataCategory.STATIC, False,
                              "tds", int, -1, units.m, 0.01,
                              MinMax(0, 100), {0: 'min value', 100: 'max value'},
                              False],
                             ["test-data-spec-out-of-range", DataCategory.STATIC, False,
                              "tds", int, -1, units.m, 0.01,
                              MinMax(0, 100), {-10: 'min value', 100: 'max value'},
                              True],
                             ["latitude-in-range", DataCategory.DYNAMIC, False,
                              "lat", float, None, units.deg, 0.01,
                              CyclicMinMax(-90, 90), {-90.0: 'min value', 90.0: 'max value'},
                              False],
                             ["latitude-out-of-range", DataCategory.DYNAMIC, False,
                              "lat", float, None, units.deg, 0.01,
                              CyclicMinMax(-90, 90), {-91.0: 'min value', 90.0: 'max value'},
                              True],
                             ["longitude-in-range", DataCategory.DYNAMIC, False,
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


@pytest.mark.parametrize(["name", "category", "is_optional",
                          "abbreviation", "representation_type",
                          "missing_value", "unit", "precision",
                          "value_range", "value_meanings"
                          ],
                         [
                             ["test-data-spec", DataCategory.DYNAMIC, False,
                              "tds", float, -1, units.m, 0.01,
                              MinMax(0.0, 100.0), {0.0: "minimum", 100.0: "maximum"}]
])
def test_data_specification_read_write(name, category, is_optional,
                                       abbreviation, representation_type,
                                       missing_value, unit, precision,
                                       value_range, value_meanings, tmp_path):
    ds = DataSpecification(name=name,
                           category=category,
                           is_optional=is_optional,
                           abbreviation=abbreviation,
                           representation_type=representation_type,
                           missing_value=missing_value,
                           unit=unit,
                           precision=precision,
                           value_range=value_range,
                           value_meanings=value_meanings)

    ds_dict = ds.to_dict()
    ds_yaml = tmp_path / "test_data_specification_read_write-ds.yaml"
    ds_yaml = "/tmp/test_data_specification_read_write-ds.yaml"

    assert ds_dict["name"] == name

    with open(ds_yaml, "w") as f:
        yaml.dump(ds_dict, f)

    ds_loaded_dict = None
    with open(ds_yaml, "r") as f:
        ds_loaded_dict = yaml.load(f, Loader=yaml.SafeLoader)

    assert ds_loaded_dict["name"] == name

    ds_loaded = DataSpecification.from_dict(data=ds_loaded_dict)

    assert ds_loaded == ds


@pytest.mark.parametrize(["dataspec", "other_dataspec", "error_msg"],
                         [
                             [DataSpecification(name="a",
                                                unit=units.m),
                              DataSpecification(name="a", category=DataCategory.DYNAMIC),
                              None],
                             [DataSpecification(name="a",
                                                representation_type=float),
                              DataSpecification(name="a",
                                                category=DataCategory.DYNAMIC,
                                                representation_type=int),
                              "'representation_type' differs"]
])
def test_data_specification_merge(dataspec, other_dataspec, error_msg):
    if error_msg is None:
        dataspec.merge(other=other_dataspec)
        assert dataspec.name == dataspec.name
    else:
        with pytest.raises(ValueError, match=error_msg):
            dataspec.merge(other=other_dataspec)


@pytest.mark.parametrize(["dataspec", "other_dataspec", "error_type"],
                         [
                             [DataSpecification(name="a",
                                                unit=units.m),
                              DataSpecification(name="a",
                                                unit=units.m,
                                                category=DataCategory.DYNAMIC),
                              None],
                             [DataSpecification(name="a",
                                                unit=units.m),
                              DataSpecification(name="a",
                                                category=DataCategory.DYNAMIC,
                                                unit=units.deg),
                              'unit'],
                             [DataSpecification(name="a",
                                                representation_type=float),
                              DataSpecification(name="a",
                                                category=DataCategory.DYNAMIC,
                                                representation_type=int),
                              "representation_type"]
])
def test_data_specification_fulfillment(dataspec, other_dataspec, error_type):
    f = dataspec.get_fulfillment(other_dataspec)
    if error_type is None:
        assert f.is_met()
    else:
        assert not f.is_met()
        assert f.status[error_type]["status"] == Status.FAIL


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


def test_metadata_read_write(tmp_path):
    column_a = DataSpecification(name="a", category=DataCategory.STATIC)
    column_b = DataSpecification(name="b", category=DataCategory.DYNAMIC)

    columns = [column_a, column_b]

    annotation_history = History()
    annotation_comment = Annotation(name=Annotation.Key.Comment,
                                    value="test-comment")

    annotation_change = Change(title="minor-change",
                               description="spelling of word changed")

    annotation_history.add_change(annotation_change)

    annotations = {
        Annotation.Key.History.value: annotation_history,
        Annotation.Key.Comment.value: annotation_comment
    }

    md = MetaData(columns=columns,
                  annotations=annotations)

    metadata_yaml = tmp_path / "test_metadata-md.yaml"
    md.save_yaml(filename=metadata_yaml)
    assert metadata_yaml.exists()

    loaded_md = MetaData.load_yaml(filename=metadata_yaml)
    assert md == loaded_md

import datetime
import decimal
import importlib.util
import sys

import polars
import pydantic
import pytest

from damast.core.data_description import CyclicMinMax, ListOfValues, MinMax
from damast.core.metadata import DataSpecification, MetaData
from damast.core.pydantic_export import PydanticExporter


def _load_module_from_path(name, path):
    """
    Load a module from a file the same way PipelineElement's PluginManager does - a module
    must be registered in sys.modules *before* exec_module for pydantic to be able to resolve
    the postponed (`from __future__ import annotations`) forward references in its fields.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exporter() -> PydanticExporter:
    return PydanticExporter()


@pytest.mark.parametrize(["representation_type", "expected"], [
    [polars.Int8, int],
    [polars.Int64, int],
    [polars.UInt64, int],
    [polars.Float32, float],
    [polars.Float64, float],
    [polars.String, str],
    [polars.Utf8, str],
    [polars.Categorical, str],
    [polars.Enum, str],
    [polars.Boolean, bool],
    [polars.Binary, bytes],
    [polars.Decimal, decimal.Decimal],
    [polars.Date, datetime.date],
    [polars.Datetime, datetime.datetime],
    [polars.Time, datetime.time],
    [polars.Duration, datetime.timedelta],
    [int, int],
    [str, str],
    [float, float],
    # DataType instances (as returned by LazyFrame.collect_schema().dtypes()), not classes
    [polars.Int64(), int],
    [polars.Float64(), float],
    [polars.Datetime("us"), datetime.datetime],
])
def test_resolve_python_type(exporter, representation_type, expected):
    assert exporter.resolve_python_type(representation_type) is expected


def test_resolve_python_type_none_raises(exporter):
    with pytest.raises(ValueError):
        exporter.resolve_python_type(None)


@pytest.mark.parametrize("unmapped", [polars.List, polars.Struct, polars.Array, polars.Object])
def test_resolve_python_type_unmapped_polars_type_raises(exporter, unmapped):
    # bare (uninstantiated) dtype classes carry no field info, so even Struct cannot be resolved
    with pytest.raises(TypeError):
        exporter.resolve_python_type(unmapped)


def test_resolve_python_type_struct_builds_nested_model(exporter):
    dtype = polars.Struct({"a": polars.Int64, "b": polars.String})

    model = exporter.resolve_python_type(dtype, name="Position")

    assert issubclass(model, pydantic.BaseModel)
    assert model.__name__ == "Position"
    assert model.model_fields["a"].annotation is int
    assert model.model_fields["b"].annotation is str

    record = model(a=1, b="x")
    assert record.a == 1
    with pytest.raises(pydantic.ValidationError):
        model(a="not-an-int", b="x")


def test_resolve_python_type_struct_default_name_is_unique(exporter):
    dtype = polars.Struct({"a": polars.Int64})

    first = exporter.resolve_python_type(dtype)
    second = exporter.resolve_python_type(dtype)

    assert first.__name__ != second.__name__


def test_resolve_python_type_nested_struct(exporter):
    dtype = polars.Struct({"a": polars.Int64, "inner": polars.Struct({"b": polars.Float64})})

    model = exporter.resolve_python_type(dtype, name="Outer")

    inner_model = model.model_fields["inner"].annotation
    assert issubclass(inner_model, pydantic.BaseModel)
    assert inner_model.model_fields["b"].annotation is float

    record = model(a=1, inner={"b": 2.5})
    assert record.inner.b == 2.5


def test_resolve_python_type_struct_invalid_field_name_raises(exporter):
    dtype = polars.Struct({"not a valid name": polars.Int64})
    with pytest.raises(ValueError):
        exporter.resolve_python_type(dtype)


@pytest.fixture
def ais_like_metadata() -> MetaData:
    return MetaData(columns=[
        DataSpecification(name="mmsi", representation_type=int, is_optional=False,
                          description="Maritime Mobile Service Identity",
                          value_range=MinMax(100000000, 999999999)),
        DataSpecification(name="lon", representation_type=polars.Float64, is_optional=False,
                          value_range=CyclicMinMax(-180.0, 180.0)),
        DataSpecification(name="nav_status", representation_type=str, is_optional=True,
                          missing_value="unknown",
                          value_range=ListOfValues(["moored", "underway", "anchored"])),
        DataSpecification(name="date_time_utc", representation_type=polars.Datetime, is_optional=True),
    ])


@pytest.fixture
def struct_metadata() -> MetaData:
    return MetaData(columns=[
        DataSpecification(name="mmsi", representation_type=int, is_optional=False),
        DataSpecification(
            name="position",
            representation_type=polars.Struct({"lat": polars.Float64, "lon": polars.Float64}),
            is_optional=False,
        ),
    ])


def test_to_pydantic_model_valid_record(exporter, ais_like_metadata):
    Record = exporter.to_pydantic_model(ais_like_metadata, name="AISMessage")

    record = Record(mmsi=123456789, lon=10.5, nav_status="moored", date_time_utc=None)
    assert record.mmsi == 123456789
    assert record.nav_status == "moored"


def test_to_pydantic_model_optional_field_uses_default(exporter, ais_like_metadata):
    Record = exporter.to_pydantic_model(ais_like_metadata, name="AISMessage")

    record = Record(mmsi=123456789, lon=10.5)
    assert record.nav_status == "unknown"


def test_to_pydantic_model_rejects_out_of_range(exporter, ais_like_metadata):
    Record = exporter.to_pydantic_model(ais_like_metadata, name="AISMessage")

    with pytest.raises(pydantic.ValidationError):
        Record(mmsi=1, lon=10.5)


def test_to_pydantic_model_rejects_invalid_literal(exporter, ais_like_metadata):
    Record = exporter.to_pydantic_model(ais_like_metadata, name="AISMessage")

    with pytest.raises(pydantic.ValidationError):
        Record(mmsi=123456789, lon=10.5, nav_status="not-a-real-status")


def test_to_pydantic_model_rejects_missing_required_field(exporter, ais_like_metadata):
    Record = exporter.to_pydantic_model(ais_like_metadata, name="AISMessage")

    with pytest.raises(pydantic.ValidationError):
        Record(lon=10.5)


def test_to_pydantic_model_missing_representation_type_raises(exporter):
    metadata = MetaData(columns=[DataSpecification(name="x")])
    with pytest.raises(ValueError):
        exporter.to_pydantic_model(metadata)


def test_to_pydantic_model_struct_column(exporter, struct_metadata):
    Record = exporter.to_pydantic_model(struct_metadata, name="Vessel")

    record = Record(mmsi=123456789, position={"lat": 59.9, "lon": 10.5})
    assert record.position.lat == 59.9

    with pytest.raises(pydantic.ValidationError):
        Record(mmsi=123456789, position={"lat": "not-a-float", "lon": 10.5})


def test_generate_pydantic_source_is_valid_python(exporter, ais_like_metadata):
    source = exporter.generate_pydantic_source(ais_like_metadata, class_name="AISMessage")
    assert "class AISMessage(pydantic.BaseModel):" in source
    assert "mmsi: int = pydantic.Field(..., description=" in source
    assert "import datetime" in source

    compile(source, "<generated>", "exec")


def test_generate_pydantic_source_rejects_non_identifier_column(exporter):
    metadata = MetaData(columns=[DataSpecification(name="not a valid name", representation_type=int)])
    with pytest.raises(ValueError):
        exporter.generate_pydantic_source(metadata, class_name="X")


def test_generate_pydantic_source_struct_column_defines_nested_class(exporter, struct_metadata):
    source = exporter.generate_pydantic_source(struct_metadata, class_name="Vessel")

    assert "class Position(pydantic.BaseModel):" in source
    assert "    lat: float" in source
    assert "    lon: float" in source
    assert "class Vessel(pydantic.BaseModel):" in source
    assert "    position: Position = pydantic.Field(...)" in source

    # nested class must be defined before it is referenced
    assert source.index("class Position(pydantic.BaseModel):") < source.index("class Vessel(pydantic.BaseModel):")

    compile(source, "<generated>", "exec")


def test_export_pydantic_module_writes_importable_file(exporter, tmp_path, ais_like_metadata):
    path = exporter.export_pydantic_module(ais_like_metadata, class_name="AISMessage",
                                           path=tmp_path / "generated" / "ais_message.py")

    assert path.exists()

    module = _load_module_from_path("ais_message_export_test", path)

    record = module.AISMessage(mmsi=123456789, lon=10.5, nav_status="moored")
    assert record.mmsi == 123456789

    with pytest.raises(pydantic.ValidationError):
        module.AISMessage(mmsi=1, lon=10.5)


def test_export_pydantic_module_struct_column_roundtrips(exporter, tmp_path, struct_metadata):
    path = exporter.export_pydantic_module(struct_metadata, class_name="Vessel",
                                           path=tmp_path / "vessel.py")

    module = _load_module_from_path("vessel_struct_export_test", path)

    record = module.Vessel(mmsi=123456789, position={"lat": 59.9, "lon": 10.5})
    assert record.position.lat == 59.9

    with pytest.raises(pydantic.ValidationError):
        module.Vessel(mmsi=123456789, position={"lat": "not-a-float", "lon": 10.5})


def test_export_pydantic_module_does_not_write_file_on_error(exporter, tmp_path):
    metadata = MetaData(columns=[DataSpecification(name="x")])
    path = tmp_path / "should_not_exist.py"

    with pytest.raises(ValueError):
        exporter.export_pydantic_module(metadata, class_name="X", path=path)

    assert not path.exists()


def test_dynamic_and_static_models_agree_on_valid_and_invalid_records(exporter, tmp_path, ais_like_metadata):
    """
    The dynamic (to_pydantic_model) and static (generate_pydantic_source/export_pydantic_module)
    paths are built from the same per-column mapping - this checks they cannot silently drift
    apart on real validation decisions, not just on the mapping in isolation.
    """
    DynamicRecord = exporter.to_pydantic_model(ais_like_metadata, name="AISMessage")
    path = exporter.export_pydantic_module(ais_like_metadata, class_name="AISMessage", path=tmp_path / "ais_message.py")

    module = _load_module_from_path("ais_message_agree_test", path)
    StaticRecord = module.AISMessage

    cases = [
        dict(mmsi=123456789, lon=10.5, nav_status="moored"),
        dict(mmsi=1, lon=10.5, nav_status="moored"),  # out of range
        dict(mmsi=123456789, lon=10.5, nav_status="bogus"),  # bad literal
        dict(mmsi=123456789, lon=10.5),  # optional default
        dict(mmsi=123456789, lon=180.0),  # boundary: inclusive max
        dict(mmsi=123456789, lon=180.0000001),  # just past boundary
    ]

    for case in cases:
        dyn_ok = True
        try:
            DynamicRecord(**case)
        except pydantic.ValidationError:
            dyn_ok = False

        static_ok = True
        try:
            StaticRecord(**case)
        except pydantic.ValidationError:
            static_ok = False

        assert dyn_ok == static_ok, f"dynamic/static disagreement for {case}"

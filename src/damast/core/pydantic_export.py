"""
Generate pydantic models - dynamically at runtime, or as exported Python source - from a
`MetaData` specification.

Where `MetaData` describes a dataframe's columnar schema (vectorized value ranges, aggregate
statistics), the models produced here describe a single *row*/record of that schema. This is
useful for validating an individual incoming record (e.g. a single message before it is batched
into a dataframe), or for interop with tools that expect a pydantic model (FastAPI, JSON schema
export, IDE type-checking).

All entry points are methods of `PydanticExporter`. Both `PydanticExporter.to_pydantic_model`
(dynamic) and `PydanticExporter.generate_pydantic_source` (static) are built from the same
per-column mapping, so the two cannot drift apart from each other.

.. note::
    Only column-level constraints that make sense for a single record are carried over:
    ``representation_type`` (as the field type), ``is_optional``/``missing_value`` (as
    optionality/default) and ``value_range`` (as ``ge``/``le`` for `MinMax`/`CyclicMinMax`, or a
    ``Literal`` for `ListOfValues`). ``value_stats`` describes the whole column (e.g. mean,
    stddev) and has no per-record meaning, so it is not translated into a validation constraint.

    A ``polars.Struct`` column is supported *type-only*: its named sub-fields are translated into
    a nested `pydantic.BaseModel` recursively, but - since polars' ``Struct`` field only carries
    a name and a dtype - none of ``value_range``/``missing_value``/``description`` can be applied
    to sub-fields, only to the top-level column.
"""
from __future__ import annotations

import ast
import datetime
import decimal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import polars
import polars.datatypes
import pydantic

from .data_description import ListOfValues, MinMax
from .metadata import DataSpecification, MetaData

__all__ = ["PydanticExporter"]

#: Polars types that map to a Python type by exact match (aliases, e.g. Utf8 == String,
#: collapse naturally since they are literally the same class object)
_EXACT_POLARS_TYPE_MAP: dict[type, type] = {
    polars.Boolean: bool,
    polars.String: str,
    polars.Categorical: str,
    polars.Enum: str,
    polars.Binary: bytes,
    polars.Decimal: decimal.Decimal,
    polars.Date: datetime.date,
    polars.Datetime: datetime.datetime,
    polars.Time: datetime.time,
    polars.Duration: datetime.timedelta,
}

#: Python source (module to import, or None; annotation text) for a subset of types that are
#: not builtins and therefore need an explicit import in generated source
_PYTHON_TYPE_SOURCE: dict[type, tuple[Optional[str], str]] = {
    decimal.Decimal: ("decimal", "decimal.Decimal"),
    datetime.date: ("datetime", "datetime.date"),
    datetime.datetime: ("datetime", "datetime.datetime"),
    datetime.time: ("datetime", "datetime.time"),
    datetime.timedelta: ("datetime", "datetime.timedelta"),
}


def _pascal_case(name: str) -> str:
    """Turn a ``snake_case`` (or arbitrary) column/field name into a ``PascalCase`` class name
    fragment, used to name nested struct models."""
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def _is_struct_model(py_type: Any) -> bool:
    """Whether ``py_type`` is a nested pydantic model generated for a `polars.Struct` field."""
    return isinstance(py_type, type) and issubclass(py_type, pydantic.BaseModel)


def _type_source(py_type: type) -> tuple[Optional[str], str]:
    """
    Resolve the (module to import, annotation text) needed to reference ``py_type`` in
    generated source.
    """
    if py_type in _PYTHON_TYPE_SOURCE:
        return _PYTHON_TYPE_SOURCE[py_type]

    if py_type.__module__ == "builtins":
        return None, py_type.__name__

    raise TypeError(
        f"PydanticExporter: no known Python source representation for type '{py_type}'"
    )


class PydanticExporter:
    """
    Generates pydantic models - dynamically at runtime, or as exported Python source - from a
    `MetaData` specification.

    Example:

    ```python
    exporter = PydanticExporter()
    Record = exporter.to_pydantic_model(adf.metadata, name="AISMessage")
    ```

    A single instance may be reused across multiple exports; it only carries a counter used to
    name anonymous (unnamed) nested struct models, so reuse just keeps those names unique.
    """

    @dataclass
    class ResolvedField:
        """
        The single per-column mapping shared by `PydanticExporter.to_pydantic_model` and
        `PydanticExporter.generate_pydantic_source`, so the dynamic model and the generated
        source describe identical fields.
        """
        name: str
        py_type: type
        optional: bool
        default: Any
        literal_values: Optional[list] = None
        field_kwargs: dict = field(default_factory=dict)

    def __init__(self) -> None:
        self._anonymous_struct_count = 0

    def resolve_python_type(self, representation_type: Any, *, name: Optional[str] = None) -> type:
        """
        Resolve a `DataSpecification.representation_type` to a plain Python type suitable
        for a pydantic field annotation.

        Example:

        ```python
        exporter = PydanticExporter()
        exporter.resolve_python_type(polars.Int64)  # -> int
        exporter.resolve_python_type(str)           # -> str
        ```

        Args:
            representation_type: A plain Python type (as resolved by
                `DataSpecification.resolve_representation_type` for builtin names), a polars
                `polars.datatypes.DataType` subclass (as resolved for polars type names), a
                `polars.datatypes.DataType` *instance* (as returned e.g. by
                `LazyFrame.collect_schema().dtypes()` and therefore commonly found in
                metadata inferred from real data via `AnnotatedDataFrame.infer_annotation`), or a
                `polars.Struct` instance with named fields
            name: Class name to use if ``representation_type`` is a `polars.Struct` and a nested
                `pydantic.BaseModel` needs to be generated for it. Defaults to an auto-generated,
                unique ``AnonymousStructN`` name.

        Returns:
            The corresponding Python type - a nested `pydantic.BaseModel` subclass for a
            `polars.Struct`

        Raises:
            ValueError: If `representation_type` is `None`, or (for a `polars.Struct`) a field
                name is not a valid Python identifier
            TypeError: If `representation_type` is not a type (or DataType instance), or is a
                polars type with no known Python type mapping
        """
        if representation_type is None:
            raise ValueError("PydanticExporter.resolve_python_type: 'representation_type' is not set")

        if isinstance(representation_type, polars.Struct):
            return self._resolve_struct_type(representation_type, name=name)

        if isinstance(representation_type, polars.DataType):
            # a DataType instance rather than the class itself, e.g. as returned by
            # LazyFrame.collect_schema().dtypes() - possibly parameterized (Datetime(time_unit=...))
            representation_type = type(representation_type)

        if not isinstance(representation_type, type):
            raise TypeError(
                f"PydanticExporter.resolve_python_type: expected a type, got {representation_type!r}"
            )

        if not issubclass(representation_type, polars.DataType):
            # already a plain Python type, e.g. int, float, str, bool - as resolved for builtin names
            return representation_type

        if representation_type in _EXACT_POLARS_TYPE_MAP:
            return _EXACT_POLARS_TYPE_MAP[representation_type]

        if issubclass(representation_type, polars.datatypes.IntegerType):
            return int

        if issubclass(representation_type, polars.datatypes.FloatType):
            return float

        raise TypeError(
            "PydanticExporter.resolve_python_type: no known Python type mapping for polars type"
            f" '{representation_type.__name__}'"
        )

    def _resolve_struct_type(
        self, dtype: polars.Struct, *, name: Optional[str]
    ) -> type[pydantic.BaseModel]:
        """Recursively build a type-only nested `pydantic.BaseModel` for a `polars.Struct`
        instance - one field per named struct member, without constraints (polars' ``Field``
        carries only a name and a dtype, nothing else to translate)."""
        if name is None:
            name = f"AnonymousStruct{self._anonymous_struct_count}"
            self._anonymous_struct_count += 1

        fields: dict[str, tuple[type, Any]] = {}
        for struct_field in dtype.fields:
            if not struct_field.name.isidentifier():
                raise ValueError(
                    "PydanticExporter.resolve_python_type: struct field"
                    f" '{struct_field.name}' is not a valid Python identifier and cannot be"
                    " used as a nested field name"
                )
            nested_name = f"{name}{_pascal_case(struct_field.name)}"
            py_type = self.resolve_python_type(struct_field.dtype, name=nested_name)
            fields[struct_field.name] = (py_type, ...)

        return pydantic.create_model(name, **fields)

    def _build_field_plan(self, spec: DataSpecification) -> PydanticExporter.ResolvedField:
        try:
            py_type = self.resolve_python_type(
                spec.representation_type, name=_pascal_case(spec.name)
            )
        except (TypeError, ValueError) as e:
            raise type(e)(f"PydanticExporter: column '{spec.name}': {e}") from e

        literal_values = None
        field_kwargs: dict[str, Any] = {}

        if spec.description:
            field_kwargs["description"] = spec.description

        if isinstance(spec.value_range, ListOfValues):
            literal_values = list(spec.value_range.values)
        elif isinstance(spec.value_range, MinMax):
            # CyclicMinMax is a plain MinMax subclass with identical range semantics elsewhere in
            # damast (see data_description.py) - min/max bounds apply the same way here
            field_kwargs["ge"] = spec.value_range.min
            field_kwargs["le"] = spec.value_range.max

        optional = bool(spec.is_optional)
        default = spec.missing_value if spec.missing_value is not None else (None if optional else ...)

        return PydanticExporter.ResolvedField(
            name=spec.name,
            py_type=py_type,
            optional=optional,
            default=default,
            literal_values=literal_values,
            field_kwargs=field_kwargs,
        )

    def to_pydantic_model(self, metadata: MetaData, name: str = "Record") -> type[pydantic.BaseModel]:
        """
        Dynamically create a pydantic model describing a single record/row that conforms to
        ``metadata``, using `pydantic.create_model`.

        Example:

        ```python
        exporter = PydanticExporter()
        Record = exporter.to_pydantic_model(adf.metadata, name="AISMessage")
        Record(mmsi=123456789, lon=10.5, lat=59.9)
        ```

        Args:
            metadata: The dataframe metadata to derive fields from
            name: Class name for the generated model

        Returns:
            A new `pydantic.BaseModel` subclass

        Raises:
            TypeError: If a column's representation_type has no known Python type mapping
            ValueError: If a column's representation_type is not set
        """
        field_definitions: dict[str, tuple[Any, Any]] = {}
        for spec in metadata.columns:
            plan = self._build_field_plan(spec)

            annotation = Literal[tuple(plan.literal_values)] if plan.literal_values else plan.py_type
            if plan.optional:
                annotation = Optional[annotation]

            field_definitions[plan.name] = (
                annotation,
                pydantic.Field(default=plan.default, **plan.field_kwargs),
            )

        return pydantic.create_model(name, **field_definitions)

    @staticmethod
    def _collect_struct_class_sources(
        model: type[pydantic.BaseModel], class_sources: dict[str, str], extra_imports: set[str]
    ) -> None:
        """Populate ``class_sources`` (class name -> source text) with ``model`` and any nested
        struct models it contains, innermost first, so each class is defined before it is
        referenced. Also collects any imports its (possibly nested) field types need."""
        if model.__name__ in class_sources:
            return

        lines = [f"class {model.__name__}(pydantic.BaseModel):"]
        for field_name, info in model.model_fields.items():
            field_type = info.annotation
            if _is_struct_model(field_type):
                PydanticExporter._collect_struct_class_sources(field_type, class_sources, extra_imports)
                type_text = field_type.__name__
            else:
                module_name, type_text = _type_source(field_type)
                if module_name:
                    extra_imports.add(module_name)
            lines.append(f"    {field_name}: {type_text}")

        class_sources[model.__name__] = "\n".join(lines)

    def generate_pydantic_source(self, metadata: MetaData, class_name: str) -> str:
        """
        Generate the Python source code of a pydantic model describing a single record/row that
        conforms to `metadata` - the static counterpart to `to_pydantic_model`, built from the
        same per-column mapping so the two cannot drift apart.

        The returned string is not written to disk - see `export_pydantic_module` for that.

        Example:

        ```python
        exporter = PydanticExporter()
        source = exporter.generate_pydantic_source(adf.metadata, class_name="AISMessage")
        print(source)
        ```

        Args:
            metadata: The dataframe metadata to derive fields from
            class_name: Class name for the generated model

        Returns:
            Python source code defining `class <class_name>(pydantic.BaseModel): ...`, plus any
            nested struct models it references

        Raises:
            TypeError: If a column's representation_type has no known Python type mapping
            ValueError: If a column's representation_type is not set, or a column name is not a
                valid Python identifier
        """
        plans = []
        extra_imports: set[str] = set()
        struct_class_sources: dict[str, str] = {}
        for spec in metadata.columns:
            if not spec.name.isidentifier():
                raise ValueError(
                    f"PydanticExporter.generate_pydantic_source: column '{spec.name}' is not a valid"
                    " Python identifier and cannot be used as a field name in generated source"
                )

            plan = self._build_field_plan(spec)
            plans.append(plan)

            if plan.literal_values:
                continue

            if _is_struct_model(plan.py_type):
                self._collect_struct_class_sources(plan.py_type, struct_class_sources, extra_imports)
            else:
                module_name, _ = _type_source(plan.py_type)
                if module_name:
                    extra_imports.add(module_name)

        lines = [
            "# Auto-generated by damast.core.pydantic_export from a MetaData specification.",
            "# Do not edit by hand - regenerate instead.",
            "from __future__ import annotations",
            "",
        ]
        for module_name in sorted(extra_imports):
            lines.append(f"import {module_name}")

        lines += [
            "",
            "from typing import Literal, Optional",
            "",
            "import pydantic",
            "",
        ]

        for struct_source in struct_class_sources.values():
            lines += ["", struct_source, ""]

        lines += ["", f"class {class_name}(pydantic.BaseModel):"]

        for plan in plans:
            if plan.literal_values:
                type_text = f"Literal[{', '.join(repr(v) for v in plan.literal_values)}]"
            elif _is_struct_model(plan.py_type):
                type_text = plan.py_type.__name__
            else:
                _, type_text = _type_source(plan.py_type)

            if plan.optional:
                type_text = f"Optional[{type_text}]"

            field_args = ["..."] if plan.default is ... else [f"default={plan.default!r}"]
            for key, value in plan.field_kwargs.items():
                field_args.append(f"{key}={value!r}")

            lines.append(f"    {plan.name}: {type_text} = pydantic.Field({', '.join(field_args)})")

        source = "\n".join(lines) + "\n"

        try:
            ast.parse(source)
        except SyntaxError as e:
            raise RuntimeError(
                f"PydanticExporter.generate_pydantic_source: generated invalid Python source -- {e}"
            ) from e

        return source

    def export_pydantic_module(self, metadata: MetaData, class_name: str, path: str | Path) -> Path:
        """
        Generate a pydantic model from ``metadata`` and write it to a Python source file.

        Example:

        ```python
        exporter = PydanticExporter()
        exporter.export_pydantic_module(adf.metadata, class_name="AISMessage", path="ais_message.py")
        ```

        Args:
            metadata: The dataframe metadata to derive fields from
            class_name: Class name for the generated model
            path: Destination `.py` file - parent directories are created as needed

        Returns:
            The path that was written
        """
        source = self.generate_pydantic_source(metadata, class_name=class_name)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return path

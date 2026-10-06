import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import damast.core
import damast.data_handling.transformers.filters
from damast.data_handling.transformers.filters import (
    MEMBERSHIP_OPERATORS,
    OPERATORS,
    Filter,
)
from damast.domains.maritime.ais.data_generator import AISTestData, AISTestDataSpec
from damast.domains.maritime.data_specification import ColumnName


@pytest.fixture()
def adf() -> damast.core.AnnotatedDataFrame:
    test_data = AISTestData(number_of_trajectories=10, min_length=25, max_length=200)
    return damast.core.AnnotatedDataFrame(test_data.dataframe, damast.core.MetaData.from_dict(AISTestDataSpec))


@pytest.mark.parametrize("inplace", [True, False])
def test_remove_values(tmpdir, adf: damast.core.AnnotatedDataFrame, inplace: bool):
    """
    Test that removal of sources work on test data
    """
    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )

    pipeline.add("Remove rows with ground as source", damast.data_handling.transformers.filters.RemoveValueRows("g"),
                 name_mappings={"x": ColumnName.SOURCE})

    original_sources = adf[ColumnName.SOURCE].collect()
    num_sources = len(original_sources)
    num_invalid_sources = len(original_sources.filter(pl.col(ColumnName.SOURCE) == "g"))
    num_valid_sources = num_sources - num_invalid_sources

    new_adf = pipeline.transform(adf)

    filtered_sources = new_adf[ColumnName.SOURCE].collect()
    assert len(filtered_sources) == num_valid_sources
    assert len(filtered_sources.filter(pl.col(ColumnName.SOURCE) != "s")) == 0

    original_df_length = len(adf.filter(pl.col(ColumnName.SOURCE) == "g").collect())
    if inplace:
        # frame has been updated, so no invalid entries should be left
        assert original_df_length == 0, "Inplace update should remove all values"
    else:
        # frame has been not been updated inplace, so invalid entries should be still left
        assert original_df_length == num_invalid_sources, "Inplace update should keep invalid sources"


@pytest.mark.parametrize("inplace", [True, False])
def test_drop_missing(tmpdir,  adf: damast.core.AnnotatedDataFrame, inplace: bool):

    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )
    pipeline.add("Remove rows with ground as source",
                 damast.data_handling.transformers.filters.DropMissingOrNan(),
                 name_mappings={"x": ColumnName.DATE_TIME_UTC})


    num_missing = adf.dataframe.select(ColumnName.DATE_TIME_UTC).null_count().collect()[0,0]

    assert num_missing > 0
    new_adf = pipeline.transform(adf)

    new_num_missing = new_adf.dataframe.select(ColumnName.DATE_TIME_UTC).null_count().collect()[0,0]

    assert new_num_missing == 0
    num_missing_post = adf.dataframe.select(ColumnName.DATE_TIME_UTC).null_count().collect()[0,0]

    if inplace:
        assert num_missing_post == 0
    else:
        assert num_missing == num_missing_post


def test_drop_missing_datetime(tmpdir):
    """
    NaN is a float-only concept - dropping it must be skipped for a (non-string) Datetime
    column, which polars' 'drop_nans' rejects with an InvalidOperationError.
    """
    df = pl.LazyFrame({"timestamp": [datetime(2020, 1, 1), None, datetime(2020, 1, 2)]})
    adf = damast.core.AnnotatedDataFrame(
        df,
        metadata=damast.core.MetaData([damast.core.DataSpecification(name="timestamp")]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )

    pipeline = damast.core.DataProcessingPipeline(name="drop missing timestamps", base_dir=Path(tmpdir))
    pipeline.add("drop_missing_timestamp",
                 damast.data_handling.transformers.filters.DropMissingOrNan(),
                 name_mappings={"x": "timestamp"})

    new_adf = pipeline.transform(adf)

    assert len(new_adf.dataframe.collect()) == 2
    assert new_adf.dataframe.select("timestamp").null_count().collect()[0, 0] == 0


@pytest.mark.parametrize("inplace", [True, False])
def test_filter_within(tmpdir,  adf: damast.core.AnnotatedDataFrame, inplace: bool):

    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )
    unique_values = adf.select("message_nr").unique().collect().to_numpy().flatten()
    assert len(unique_values) > 1

    pipeline.add("Filter rows within message types",
                 damast.data_handling.transformers.filters.FilterWithin(unique_values[:1]),
                 name_mappings={"x": "message_nr"})

    num_all = len(adf.dataframe)
    num_eq = len(adf.filter(pl.col("message_nr") == unique_values[[0]]).collect())
    new_adf = pipeline.transform(adf)

    assert len(new_adf.dataframe) == num_eq
    assert len(new_adf.filter(pl.col("message_nr") == unique_values[[0]]).collect()) == num_eq
    if inplace:
        assert len(adf.dataframe.collect()) == len(new_adf.dataframe.collect())
    else:
        assert len(adf.dataframe.collect()) == num_all


# The expected semantics, written out independently of filters.OPERATORS
COMPARISONS = [
    ("<", lambda value, threshold: value < threshold),
    ("<=", lambda value, threshold: value <= threshold),
    (">", lambda value, threshold: value > threshold),
    (">=", lambda value, threshold: value >= threshold),
    ("==", lambda value, threshold: value == threshold),
    ("!=", lambda value, threshold: value != threshold),
    ("<>", lambda value, threshold: value != threshold),
]


@pytest.mark.parametrize("operator,keeps", COMPARISONS)
def test_filter_operators(tmpdir, adf: damast.core.AnnotatedDataFrame, operator: str, keeps):
    """Every operator keeps exactly the rows that satisfy it."""
    mmsis = adf.dataframe.select("mmsi").collect().to_series().to_list()
    threshold = sorted(mmsis)[len(mmsis) // 2]
    expected = sum(1 for value in mmsis if keeps(value, threshold))
    assert 0 < expected < len(mmsis), "the threshold must actually split the data"

    pipeline = damast.core.DataProcessingPipeline(name="compare", base_dir=Path(tmpdir))
    pipeline.add("filter_mmsi", Filter(operator=operator, value=threshold),
                 name_mappings={"x": "mmsi"})

    new_adf = pipeline.transform(adf)
    assert len(new_adf.dataframe.collect()) == expected


@pytest.mark.parametrize("inplace", [True, False])
def test_filter_drops_values_below_a_bound(tmpdir, inplace: bool):
    """
    The motivating case: exclude 'mmsi < 999999' by keeping its complement.

    The generated test data holds only realistic 9-digit identifiers, so the implausible ones
    this is meant to remove have to be put there on purpose.
    """
    adf = damast.core.AnnotatedDataFrame(
        pl.LazyFrame({"mmsi": [0, 12345, 999998, 999999, 1000000, 227991212]}),
        metadata=damast.core.MetaData([damast.core.DataSpecification(name="mmsi")]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )

    pipeline = damast.core.DataProcessingPipeline(name="valid mmsi", base_dir=Path(tmpdir),
                                                  inplace_transformation=inplace)
    pipeline.add("valid_mmsi", Filter(operator=">=", value=999999),
                 name_mappings={"x": "mmsi"})

    new_adf = pipeline.transform(adf)

    assert new_adf.dataframe.collect().to_series().to_list() == [999999, 1000000, 227991212]
    if not inplace:
        assert len(adf.dataframe.collect()) == 6


def test_filter_on_a_string_column(tmpdir, adf: damast.core.AnnotatedDataFrame):
    """'<>' is an alias of '!=', and a string value compares against a string column."""
    num_not_ground = len(adf.filter(pl.col(ColumnName.SOURCE) != "g").collect())

    pipeline = damast.core.DataProcessingPipeline(name="source", base_dir=Path(tmpdir))
    pipeline.add("not_ground", Filter(operator="<>", value="g"),
                 name_mappings={"x": ColumnName.SOURCE})

    new_adf = pipeline.transform(adf)
    assert len(new_adf.dataframe.collect()) == num_not_ground


def test_filter_on_a_datetime_column(tmpdir):
    """A temporal cutoff - the non-numeric case the design turns on."""
    df = pl.LazyFrame({"timestamp": [datetime(2026, 6, 1), datetime(2026, 6, 2), datetime(2026, 6, 3)]})
    adf = damast.core.AnnotatedDataFrame(
        df,
        metadata=damast.core.MetaData([damast.core.DataSpecification(name="timestamp")]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )

    pipeline = damast.core.DataProcessingPipeline(name="cutoff", base_dir=Path(tmpdir))
    pipeline.add("after_cutoff", Filter(operator=">=", value=datetime(2026, 6, 2)),
                 name_mappings={"x": "timestamp"})

    new_adf = pipeline.transform(adf)
    assert new_adf.dataframe.collect().to_series().to_list() == [datetime(2026, 6, 2), datetime(2026, 6, 3)]


def test_filter_rejects_an_unusable_configuration():
    """A bad filter fails when the pipeline is built, not part-way through a run."""
    with pytest.raises(ValueError, match="unknown operator '=<'"):
        Filter(operator="=<", value=1)

    with pytest.raises(ValueError, match="use DropMissingOrNan"):
        Filter(operator="==", value=None)


def test_filter_rejects_a_value_the_column_cannot_hold(tmpdir, adf: damast.core.AnnotatedDataFrame):
    """A lazy frame would otherwise only fail on collection, far from the offending step."""
    pipeline = damast.core.DataProcessingPipeline(name="mismatch", base_dir=Path(tmpdir))
    pipeline.add("bad_compare", Filter(operator=">", value="not-a-number"),
                 name_mappings={"x": "mmsi"})

    with pytest.raises(RuntimeError, match="column 'mmsi' is Int64.*requires a string column"):
        pipeline.transform(adf)


@pytest.mark.parametrize("value", [999999, datetime(2026, 6, 2)])
def test_filter_survives_a_pipeline_round_trip(tmpdir, value):
    """The operator and the value must come back out of the saved *.damast.ppl unchanged."""
    column = "mmsi" if isinstance(value, int) else "timestamp"
    pipeline = damast.core.DataProcessingPipeline(name="roundtrip", base_dir=Path(tmpdir))
    pipeline.add("compare", Filter(operator=">=", value=value), name_mappings={"x": column})

    filename = pipeline.save(Path(tmpdir))
    loaded = damast.core.DataProcessingPipeline.load(filename)

    transformer = list(loaded.processing_graph.nodes())[-1].transformer
    assert transformer.operator == ">="
    assert transformer.value == value
    assert type(transformer.value) is type(value)


# --- Filter subsumes the deprecated FilterWithin / RemoveValueRows ---------------------------

def _int_adf(values: list[int]) -> damast.core.AnnotatedDataFrame:
    return damast.core.AnnotatedDataFrame(
        pl.LazyFrame({"t": values}),
        metadata=damast.core.MetaData([damast.core.DataSpecification(name="t")]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )


def _rows(tmpdir, name: str, element) -> list:
    pipeline = damast.core.DataProcessingPipeline(name=name, base_dir=Path(tmpdir))
    pipeline.add(name, element, name_mappings={"x": "t"})
    return pipeline.transform(df=_int_adf([10, 20, 30, 40])).dataframe.collect().to_series().to_list()


@pytest.mark.parametrize("operator,expected", [
    ("in", [20, 30]),
    ("not in", [10, 40]),
])
def test_filter_membership(tmpdir, operator: str, expected: list):
    assert _rows(tmpdir, "member", Filter(operator, [20, 30])) == expected


def test_filter_membership_on_a_string_column(tmpdir, adf: damast.core.AnnotatedDataFrame):
    num_ground = len(adf.filter(pl.col(ColumnName.SOURCE) == "g").collect())

    pipeline = damast.core.DataProcessingPipeline(name="sources", base_dir=Path(tmpdir))
    pipeline.add("only_ground", Filter("in", ["g"]), name_mappings={"x": ColumnName.SOURCE})

    assert len(pipeline.transform(adf).dataframe.collect()) == num_ground


def test_filter_membership_accepts_an_empty_collection(tmpdir):
    """'is_in([])' keeps nothing - legal, and there is no element whose type could be checked."""
    assert _rows(tmpdir, "none", Filter("in", [])) == []


def test_filter_membership_validates_the_elements(tmpdir):
    """A list would otherwise fall into the 'cannot compare against' branch unchecked."""
    pipeline = damast.core.DataProcessingPipeline(name="bad", base_dir=Path(tmpdir))
    pipeline.add("bad", Filter("in", ["not-a-number"]), name_mappings={"x": "t"})

    with pytest.raises(RuntimeError, match="column 't' is Int64.*requires a string column"):
        pipeline.transform(df=_int_adf([10, 20]))


def test_deprecated_wrappers_match_the_new_spelling(tmpdir):
    with pytest.warns(DeprecationWarning, match='Filter\\("in", within_values\\)'):
        within = damast.data_handling.transformers.filters.FilterWithin([20, 30])
    with pytest.warns(DeprecationWarning, match='Filter\\("!=", remove_value\\)'):
        remove = damast.data_handling.transformers.filters.RemoveValueRows(20)

    assert _rows(tmpdir, "a", within) == _rows(tmpdir, "b", Filter("in", [20, 30]))
    assert _rows(tmpdir, "c", remove) == _rows(tmpdir, "d", Filter("!=", 20))


def test_deprecated_wrappers_keep_their_serialised_form(tmpdir):
    """
    Pipelines already on disk carry 'class_name: FilterWithin' and a 'within_values' parameter -
    e.g. the replication package's examples/prepare.damast.ppl - so both have to stay as they are.
    """
    import yaml

    pipeline = damast.core.DataProcessingPipeline(name="legacy", base_dir=Path(tmpdir))
    with pytest.warns(DeprecationWarning):
        pipeline.add("within", damast.data_handling.transformers.filters.FilterWithin([20, 30]),
                     name_mappings={"x": "t"})
    filename = pipeline.save(Path(tmpdir))

    saved = yaml.safe_load(filename.read_text())
    step = [n for n in saved["processing_graph"]["nodes"] if n["name"] == "within"][0]["transformer"]
    assert step["class_name"] == "FilterWithin"
    assert step["parameters"] == {"within_values": [20, 30]}

    reloaded = damast.core.DataProcessingPipeline.load(filename)
    assert reloaded.transform(df=_int_adf([10, 20, 30, 40])
                              ).dataframe.collect().to_series().to_list() == [20, 30]


# --- Filter: cases the operator/round-trip tests above do not reach -------------------------

def _typed_adf(column: str, values: list) -> damast.core.AnnotatedDataFrame:
    return damast.core.AnnotatedDataFrame(
        pl.LazyFrame({column: values}),
        metadata=damast.core.MetaData([damast.core.DataSpecification(name=column)]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )


def _filtered(tmpdir, element, column: str, values: list) -> list:
    pipeline = damast.core.DataProcessingPipeline(name="f", base_dir=Path(tmpdir))
    pipeline.add("f", element, name_mappings={"x": column})
    return pipeline.transform(
        df=_typed_adf(column, values)).dataframe.collect().to_series().to_list()


@pytest.mark.parametrize("operator,value,expected", [
    (">=", 20, [20, 30]),
    ("!=", 20, [10, 30]),
    ("in", [20], [20]),
    ("not in", [20], [10, 30]),
])
def test_filter_drops_rows_whose_value_is_null(tmpdir, operator: str, value, expected: list):
    """
    A null never satisfies a comparison, not even a negated one - 'pl.col(x) != 20' and
    '~is_in([20])' are both null for a null x, so the row goes. Use DropMissingOrNan if the
    intent is to be explicit about it.
    """
    assert _filtered(tmpdir, Filter(operator, value), "t", [10, None, 20, 30]) == expected


@pytest.mark.parametrize("value", [np.int64(20), np.float64(20.0)])
def test_filter_accepts_a_numpy_scalar(tmpdir, value):
    """
    The value type is validated against the column, and a numpy scalar is not a Python int or
    float - so the check is on numbers.Real. Callers do pass numpy, e.g. from an array of ids.
    """
    assert _filtered(tmpdir, Filter(">=", value), "t", [10, 20, 30]) == [20, 30]


def test_filter_accepts_a_numpy_array_of_values(tmpdir):
    assert _filtered(tmpdir, Filter("in", np.array([20, 30])), "t", [10, 20, 30]) == [20, 30]


def test_filter_on_a_boolean_column(tmpdir):
    assert _filtered(tmpdir, Filter("==", True), "b", [True, False, True]) == [True, True]
    assert _filtered(tmpdir, Filter("in", [False]), "b", [True, False]) == [False]


def test_filter_rejects_an_ordering_comparison_on_a_boolean(tmpdir):
    """True < False is not a question worth asking - only equality and membership are."""
    pipeline = damast.core.DataProcessingPipeline(name="bad_bool", base_dir=Path(tmpdir))
    pipeline.add("bad_bool", Filter("<", True), name_mappings={"x": "b"})

    with pytest.raises(RuntimeError, match=re.escape("operator '<' is not meaningful")):
        pipeline.transform(df=_typed_adf("b", [True, False]))


def test_filter_is_exported_from_the_transformers_package():
    from damast.data_handling.transformers import Filter as ReExported

    assert ReExported is Filter


def test_membership_operators_are_all_known_operators():
    """MEMBERSHIP_OPERATORS selects from OPERATORS - a typo in either would silently widen it."""
    assert set(MEMBERSHIP_OPERATORS) <= set(OPERATORS)

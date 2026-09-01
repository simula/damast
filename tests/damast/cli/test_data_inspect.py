from argparse import ArgumentParser

import astropy.units as units
import pandas as pd
import polars
import pytest

from damast.cli.data_inspect import DataInspectParser
from damast.core.annotations import Annotation
from damast.core.data_description import MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataCategory, DataSpecification, MetaData


@pytest.fixture()
def metadata():
    # "height" already has a value_range but no value_stats; "letter" has neither.
    column_spec_height = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m,
                                    value_range=MinMax(min=0, max=40))
    column_spec_letter = DataSpecification(name="letter",
                                    category=DataCategory.STATIC)

    comment = Annotation(name=Annotation.Key.Comment, value="test dataframe")
    return MetaData(columns=[column_spec_height, column_spec_letter], annotations=[comment])


@pytest.fixture()
def polars_dataframe():
    data = [[0, "a"], [1, "b"], [2, "c"]]
    pandas_df = pd.DataFrame(data, columns=["height", "letter"])
    return polars.from_pandas(pandas_df)


def test_fill_missing_value_stats_computes_only_whats_missing(metadata, polars_dataframe):
    adf = AnnotatedDataFrame(dataframe=polars_dataframe, metadata=metadata)

    height_spec = adf.metadata["height"]
    original_range = height_spec.value_range
    assert height_spec.value_stats is None

    parser = DataInspectParser(parser=ArgumentParser())
    generated_fields = parser.fill_missing_value_stats(adf)

    # value_range was already set - filled in only what was missing, left it untouched
    assert height_spec.value_range == original_range
    assert height_spec.value_stats is not None
    assert generated_fields["height"] == {"value_stats"}

    # "letter" had neither - value_range is computable (categories), value_stats is not
    # (non-numeric), so only value_range is filled in and marked as generated
    letter_spec = adf.metadata["letter"]
    assert letter_spec.value_range is not None
    assert letter_spec.value_stats is None
    assert generated_fields["letter"] == {"value_range"}


def test_fill_missing_value_stats_skips_fully_populated_columns(polars_dataframe):
    from damast.core.data_description import NumericValueStats

    height_spec = DataSpecification(
        name="height",
        value_range=MinMax(min=0, max=40),
        value_stats=NumericValueStats(
            mean=1.0, stddev=0.5, median=1.0, lower_quantile=0.5, upper_quantile=1.5, total_count=3
        ),
    )
    letter_spec = DataSpecification(name="letter", value_range=MinMax(min="a", max="c"))
    adf = AnnotatedDataFrame(
        dataframe=polars_dataframe,
        metadata=MetaData(columns=[height_spec, letter_spec]),
    )

    parser = DataInspectParser(parser=ArgumentParser())
    generated_fields = parser.fill_missing_value_stats(adf)

    assert generated_fields == {}


def test_fill_missing_value_stats_marks_all_fields_when_metadata_is_inferred(metadata, polars_dataframe):
    # e.g. a plain CSV with no spec file: AnnotatedDataFrame.from_files infers this same
    # shape of metadata and sets metadata_inferred=True - none of it came from a spec file,
    # so every populated field should be marked as generated, not just the ones actually
    # missing beforehand.
    adf = AnnotatedDataFrame(dataframe=polars_dataframe, metadata=metadata, metadata_inferred=True)

    parser = DataInspectParser(parser=ArgumentParser())
    generated_fields = parser.fill_missing_value_stats(adf)

    # "height" already had a value_range (as if inferred earlier) - still marked, since it
    # is not backed by an actual spec file either.
    assert generated_fields["height"] == {"value_range", "value_stats"}
    assert generated_fields["letter"] == {"value_range"}


def test_metadata_inferred_true_for_a_csv_without_a_spec_file(tmp_path):
    csv_path = tmp_path / "no_spec.csv"
    csv_path.write_text("height;letter\n0;a\n1;b\n2;c\n")

    adf = AnnotatedDataFrame.from_files(files=[str(csv_path)], metadata_required=False)

    assert adf.metadata_inferred is True

    parser = DataInspectParser(parser=ArgumentParser())
    generated_fields = parser.fill_missing_value_stats(adf)

    # infer_annotation already computed these - fill_missing_value_stats must still mark them
    assert adf.metadata["height"].value_stats is not None
    assert generated_fields["height"] == {"value_range", "value_stats"}

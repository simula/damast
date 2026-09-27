
import polars
import pytest
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.metadata import DataSpecification, MetaData

from damast.domains.maritime.transformers import MMSIPatternClassifier
from damast.domains.maritime.transformers.mmsi_pattern_classifier import load_pattern_rules

# Declaration order matters (first match wins) - see MMSIPatternClassifier. "009990000" must
# precede "00", which must precede "0", or the more specific rules are never reached.
MAPPING = {
    "009990000": "all-coast-stations",
    "00": "coast-station",
    "0": "group-ship-call",
    "111": "SAR",
    "98": "craft-associated-with-parent-ship",
    "99": "aid-to-navigation",
}


def write_mapping_file(path, mapping):
    # single-quoted, so a pattern containing a regex backslash escape (e.g. '\d') is not
    # mistaken by YAML for one of its own (unsupported) escape sequences
    path.write_text("\n".join(f"'{pattern}': {category}" for pattern, category in mapping.items()))
    return path


@pytest.fixture
def mapping_file(tmp_path):
    return write_mapping_file(tmp_path / "mmsi_categories.yaml", MAPPING)


@pytest.fixture
def adf() -> AnnotatedDataFrame:
    # 111234567 -> SAR, 982345678 -> craft-associated-..., 993456789 -> aid-to-navigation,
    # 2345678   -> coast station 002345678 (2 leading 0s stripped by int storage) - must match
    #              "00" (coast-station), not "0" (group-ship-call), despite both being
    #              patterns of it,
    # 9990000   -> the reserved all-coast-stations identity 009990000 (2 leading 0s
    #              stripped) - must match the full 9-digit pattern, not the shorter "00",
    # 23523456  -> group ship call 023523456 (1 leading 0 stripped) - must not be swallowed by
    #              the "0" -> "00" disambiguation above,
    # 235678901 -> no rule matches -> default_category
    df = polars.DataFrame({
        "mmsi": [111234567, 982345678, 993456789, 2345678, 9990000, 23523456, 235678901],
    })
    metadata = MetaData(columns=[DataSpecification("mmsi", representation_type=int)])
    return AnnotatedDataFrame(df, metadata)


def test_load_pattern_rules(mapping_file):
    assert load_pattern_rules(mapping_file) == MAPPING


def test_classifies_by_first_matching_pattern_in_file_order(adf, mapping_file, tmp_path):
    pipeline = DataProcessingPipeline(name="example", base_dir=str(tmp_path))
    pipeline.add("classify mmsi",
                 MMSIPatternClassifier(mapping_file=mapping_file),
                 name_mappings={"x": "mmsi"})

    result = pipeline.transform(df=adf)
    assert result.lazyframe.collect()["mmsi_category"].to_list() == [
        "SAR", "craft-associated-with-parent-ship", "aid-to-navigation", "coast-station",
        "all-coast-stations", "group-ship-call", "unknown",
    ]


def test_custom_default_category(adf, mapping_file, tmp_path):
    pipeline = DataProcessingPipeline(name="example", base_dir=str(tmp_path))
    pipeline.add("classify mmsi",
                 MMSIPatternClassifier(mapping_file=mapping_file, default_category="ship"),
                 name_mappings={"x": "mmsi"})

    result = pipeline.transform(df=adf)
    assert result.lazyframe.collect()["mmsi_category"].to_list()[-1] == "ship"


def test_named_group_lets_a_more_specific_rule_take_precedence(tmp_path):
    # a coast station whose optional 6th digit is "1" (Annex 1 S2 §3a) should be reported as
    # "coast-station-proper", but any other 6th digit falls through to the generic
    # "coast-station" rule - only possible because the specific rule is listed first
    mapping_file = write_mapping_file(tmp_path / "mmsi_categories.yaml", {
        r"00(?P<mid>\d{3})1": "coast-station-proper",
        "00": "coast-station",
    })
    df = polars.DataFrame({"mmsi": [2351234, 2359234]})  # 002351234, 002359234
    metadata = MetaData(columns=[DataSpecification("mmsi", representation_type=int)])
    adf = AnnotatedDataFrame(df, metadata)

    pipeline = DataProcessingPipeline(name="example", base_dir=str(tmp_path))
    pipeline.add("classify mmsi",
                 MMSIPatternClassifier(mapping_file=mapping_file),
                 name_mappings={"x": "mmsi"})

    result = pipeline.transform(df=adf)
    assert result.lazyframe.collect()["mmsi_category"].to_list() == [
        "coast-station-proper", "coast-station",
    ]


def test_a_pattern_listed_after_a_more_general_one_is_shadowed(tmp_path):
    # same two rules as above, but in the wrong order - the specific rule can now never be
    # reached, since the general "00" rule always matches first; this is the documented
    # trade-off of first-match-wins over automatic longest-match
    mapping_file = write_mapping_file(tmp_path / "mmsi_categories.yaml", {
        "00": "coast-station",
        r"00(?P<mid>\d{3})1": "coast-station-proper",
    })
    df = polars.DataFrame({"mmsi": [2351234]})  # 002351234 - would be "coast-station-proper"
    metadata = MetaData(columns=[DataSpecification("mmsi", representation_type=int)])
    adf = AnnotatedDataFrame(df, metadata)

    pipeline = DataProcessingPipeline(name="example", base_dir=str(tmp_path))
    pipeline.add("classify mmsi",
                 MMSIPatternClassifier(mapping_file=mapping_file),
                 name_mappings={"x": "mmsi"})

    result = pipeline.transform(df=adf)
    assert result.lazyframe.collect()["mmsi_category"].to_list() == ["coast-station"]


def test_invalid_pattern_raises_value_error(tmp_path):
    mapping_file = write_mapping_file(tmp_path / "mmsi_categories.yaml", {"(": "broken"})

    with pytest.raises(ValueError, match=r"invalid regex pattern"):
        MMSIPatternClassifier(mapping_file=mapping_file)

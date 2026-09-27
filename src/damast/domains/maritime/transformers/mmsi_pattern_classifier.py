"""
A MMSI classifier based on ITU-R M.585-10 rules / recommendations
"""
import re
from pathlib import Path

import polars
import yaml

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import describe, input, output
from damast.core.transformations import PipelineElement


def load_pattern_rules(mapping_file: str | Path) -> dict[str, str]:
    """
    Load MMSI-pattern -> category rules from a YAML mapping file.

    Example:
        ```yaml
        # mmsi_categories.yaml - see examples/mmsi_categories.yaml for the full set of
        # patterns defined in Recommendation ITU-R M.585-10. More specific patterns are
        # listed before the more general ones they would otherwise shadow.
        '00(?P<mid>\\d{3})1': coast-station      # Annex 1, S2 §3a: 00-MID-1XXX
        '00': coast-station                      # Annex 1, S2 §1:  00-MID-XXXX (generic)
        '111': sar-aircraft                      # Annex 1, S3 §1:  111-MID-XXX
        '009990000': all-coast-stations          # Annex 1, S2 §8:  fixed identity
        ```

    Args:
        mapping_file: Path to a YAML file mapping MMSI patterns to category labels. Each
            pattern is a regular expression (standard `re` syntax, including named groups
            like `(?P<mid>...)`) matched against the start of the (9-digit) MMSI - it does not
            need to match the whole string, so `"111"` matches any MMSI starting with those
            three digits. Patterns are tried in the order they appear in the file and the
            *first* match wins - unlike a plain prefix, a longer/more specific regex does not
            automatically take precedence, so a pattern that is a special case of an earlier,
            more general one (e.g. `'00(?P<mid>\\d{3})1'` vs `'00'`) must be listed *before*
            it, or it will never be reached.

    Returns:
        Mapping of pattern (as string) to category label, in file order

    Raises:
        FileNotFoundError: If `mapping_file` does not exist
    """
    with open(mapping_file) as f:
        rules = yaml.safe_load(f) or {}

    return {str(pattern): str(category) for pattern, category in rules.items()}


class MMSIPatternClassifier(PipelineElement):
    """
    Classify a MMSI column by matching it against patterns from a mapping file.

    MMSI numbers are not only assigned to ships - fixed digit patterns are reserved for other
    kinds of AIS transmitters, e.g. ``111MIDxxx`` for SAR aircraft or ``99MIDxxxx`` for aids to
    navigation, per Recommendation ITU-R M.585-10 (04/2026) - see
    examples/mmsi_categories.yaml for the full set of patterns it defines. This transformer
    looks up each MMSI's category from a user-supplied pattern -> label mapping (see
    `load_pattern_rules`) and adds it as a new column; it never drops rows - a MMSI matching no
    rule gets `default_category`.

    Example:
        ```python
        from damast.core.dataprocessing import DataProcessingPipeline
        from damast.domains.maritime.transformers.mmsi_pattern_classifier import MMSIPatternClassifier

        pipeline = DataProcessingPipeline(name="example", base_dir=".")
        pipeline.add("classify mmsi",
                     MMSIPatternClassifier(mapping_file="mmsi_categories.yaml"),
                     name_mappings={"x": "mmsi"})
        ```

    Args:
        mapping_file: YAML file mapping MMSI patterns (regexes) to category labels - see
            `load_pattern_rules`. Re-read every time the transformer is constructed, including
            when a saved pipeline using it is loaded elsewhere, so keep it alongside the
            pipeline (or use an absolute path)
        default_category: Category assigned to a MMSI that matches no rule in `mapping_file`

    Raises:
        ValueError: If `mapping_file` contains a pattern that is not a valid regular
            expression
    """

    def __init__(self, mapping_file: str | Path | None = None, default_category: str = "unknown"):
        if mapping_file is None:
            self.mapping_file = Path(__file__).parent.parent / "ais" / "mmsi_categories.yaml"
        else:
            self.mapping_file = str(mapping_file)

        self.default_category = default_category
        self._rules = load_pattern_rules(self.mapping_file)

        # tried in file order - first match wins, see load_pattern_rules
        self._compiled_patterns: list[tuple[re.Pattern, str]] = []
        for pattern, category in self._rules.items():
            try:
                # wrapped in a non-capturing group so e.g. a top-level '|' in the pattern
                # doesn't escape the '^' anchor
                compiled = re.compile(r"^(?:" + pattern + r")")
            except re.error as e:
                raise ValueError(
                    f"{self.__class__.__name__}: invalid regex pattern {pattern!r} in "
                    f"'{self.mapping_file}': {e}"
                ) from e
            self._compiled_patterns.append((compiled, category))

    def _categorize(self, mmsi: int | None) -> str:
        if mmsi is None:
            return self.default_category

        # A MMSI is always 9 digits, but two categories (group ship call, coast station)
        # start with a "0" that a plain integer column silently drops - zfill it back before
        # matching patterns, or those two categories (and the all-coast-stations identity)
        # could never match.
        mmsi_str = str(mmsi).zfill(9)
        for pattern, category in self._compiled_patterns:
            if pattern.match(mmsi_str):
                return category

        return self.default_category

    @describe("Classify a MMSI column by matching it against patterns from a mapping file")
    @input({"x": {}})
    @output({"{{x}}_category": {"representation_type": str, "description": "MMSI-derived vessel category - from ITU-R M.585-10." }})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        feature = self.get_name("x")
        result = self.get_name("{{x}}_category")
        df.lazyframe = df.lazyframe.with_columns(
            polars.col(feature)
            .map_elements(self._categorize, return_dtype=polars.String)
            .alias(result)
        )
        return df

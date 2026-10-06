import datetime

import pytest
import yaml

from damast.viz.stats_export import StatsExporter


def _stats(**overrides) -> dict:
    """Two steps' worth of statistics, as the default trackers record them."""
    start = datetime.datetime(2026, 6, 1, 12, 0, tzinfo=datetime.timezone.utc)
    stats = {
        "df": {
            "start_time": start,
            "end_time": start + datetime.timedelta(seconds=2),
            "processing_time_in_s": 2.0,
            "input_dataframe_length": {"df": 100},
            "output_dataframe_length": 100,
            "rows_removed": 0,
        },
        "filter": {
            "start_time": start + datetime.timedelta(seconds=2),
            "end_time": start + datetime.timedelta(seconds=5),
            "processing_time_in_s": 3.0,
            "input_dataframe_length": {"df": 100},
            "output_dataframe_length": 40,
            "rows_removed": 60,
        },
    }
    stats.update(overrides)
    return stats


def _with_keys(stats: dict) -> dict:
    stats["df"]["keys"] = {"mmsi": {"unique_in": 9, "unique_out": 9, "unique_removed": 0}}
    stats["filter"]["keys"] = {"mmsi": {"unique_in": 9, "unique_out": 3, "unique_removed": 6}}
    return stats


def test_panels_cover_the_recorded_measures():
    panels = StatsExporter(_with_keys(_stats())).panels()

    assert [panel.title for panel in panels] == [
        "Rows", "Distinct 'mmsi'", "Duration per step", "Cumulative duration"]


def test_key_panel_is_absent_without_a_key_tracker():
    """The default tracker set records no keys, so there is nothing to draw for them."""
    titles = [panel.title for panel in StatsExporter(_stats()).panels()]

    assert "Distinct 'mmsi'" not in titles
    assert "Rows" in titles


def test_cumulative_duration_accumulates_in_execution_order():
    panels = {panel.title: panel for panel in StatsExporter(_stats()).panels()}

    assert panels["Duration per step"].series["duration"][0] == [2.0, 3.0]
    assert panels["Cumulative duration"].series["cumulative"][0] == [2.0, 5.0]


def test_steps_follow_execution_order_not_the_reports_key_order():
    """
    A report written before save_stats stopped alphabetising its keys has lost the order the
    statistics only make sense in - start_time recovers it.
    """
    alphabetised = dict(sorted(_stats().items()))
    assert list(alphabetised) == ["df", "filter"]

    renamed = {"zzz_first": _stats()["df"], "aaa_second": _stats()["filter"]}
    exporter = StatsExporter(dict(sorted(renamed.items())))

    assert list(exporter.processing_stats) == ["aaa_second", "zzz_first"]
    assert exporter.steps == ["zzz_first", "aaa_second"]
    # the series have to follow the steps, or they would be plotted against the wrong labels
    panels = {panel.title: panel for panel in exporter.panels()}
    assert panels["Duration per step"].series["duration"][0] == [2.0, 3.0]


def test_steps_keep_insertion_order_without_timestamps():
    stats = _stats()
    for step in stats.values():
        del step["start_time"]

    assert StatsExporter(stats).steps == ["df", "filter"]


def test_a_join_totals_the_rows_of_its_inputs():
    stats = _stats()
    stats["filter"]["input_dataframe_length"] = {"df": 100, "other": 25}

    panels = {panel.title: panel for panel in StatsExporter(stats).panels()}
    assert panels["Rows"].series["entering"][0] == [100, 125]


def test_export_svg_writes_a_figure(tmp_path):
    written = StatsExporter(_with_keys(_stats()), name="prepare").export_svg(tmp_path / "run.svg")

    assert written.exists()
    content = written.read_text()
    assert content.lstrip().startswith("<?xml")
    assert "prepare - per-step statistics" in content


def test_export_svg_creates_the_parent_directory(tmp_path):
    written = StatsExporter(_stats()).export_svg(tmp_path / "nested" / "dir" / "run.svg")

    assert written.exists()


def test_export_rejects_statistics_with_nothing_to_draw(tmp_path):
    exporter = StatsExporter({"a": {}, "b": {}})

    with pytest.raises(ValueError, match="no statistics to draw"):
        exporter.export_svg(tmp_path / "run.svg")


def test_from_report_reads_what_save_stats_wrote(tmp_path):
    report = tmp_path / "prepare.stats.yaml"
    report.write_text(yaml.dump({"name": "prepare", "steps": _stats()}, sort_keys=False))

    exporter = StatsExporter.from_report(report)

    assert exporter.name == "prepare"
    assert exporter.steps == ["df", "filter"]


def test_from_report_rejects_a_file_that_is_not_a_report(tmp_path):
    other = tmp_path / "other.yaml"
    other.write_text(yaml.dump({"name": "prepare"}))

    with pytest.raises(ValueError, match="not a pipeline statistics report"):
        StatsExporter.from_report(other)

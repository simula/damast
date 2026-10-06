"""
Render what a `DataProcessingPipeline` run did to the data, step by step.

Takes the statistics the pipeline's trackers collected - see
`damast.core.dataprocessing.DataProcessingPipeline.processing_stats` and the report
`save_stats` writes - and draws them as panels sharing one x axis of pipeline steps, so the
point at which rows or entities disappear is visible against where the time went.

Each measure gets its own panel rather than a second y axis: rows, distinct keys and seconds
do not share a scale, and neither do a per-step duration and its running total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["StatsExporter"]

#: Chart surface and ink, one shade off each other so grid and axes stay recessive
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_MUTED = "#898781"
_AXIS = "#c3c2b7"

#: Categorical slots 1 and 2. Every panel uses them with one fixed meaning - slot 1 is what
#: entered a step, slot 2 what left it - so the two are never a different pair panel to panel.
_SERIES_IN = "#2a78d6"
_SERIES_OUT = "#eb6834"

#: Width of a bar within its step slot, leaving a surface gap between neighbours
_BAR_WIDTH = 0.38


@dataclass
class _Panel:
    """One measure over the steps: its title, y label and the series to draw."""
    title: str
    y_label: str
    #: label -> (values, colour); a value may be None where a step did not report it
    series: dict[str, tuple[list[float | None], str]] = field(default_factory=dict)
    #: draw as bars per step, else as a line
    bars: bool = True


class StatsExporter:
    """
    Draw a pipeline run's per-step statistics as an SVG.

    Example:

        .. highlight:: python
        .. code-block:: python

            pipeline.transform(df=adf)
            StatsExporter(pipeline.processing_stats).export_svg("run.svg")

            # or from the report a run left behind
            StatsExporter.from_report("results/prepare.stats.yaml").export_svg("run.svg")

    :param processing_stats: Per-step statistics, keyed by step name in execution order
    :param name: Name of the pipeline, used in the figure title
    """

    def __init__(self, processing_stats: dict[str, dict[str, Any]], name: str | None = None):
        self.processing_stats = processing_stats
        self.name = name

    @classmethod
    def from_report(cls, path: str | Path) -> StatsExporter:
        """
        Load the report written by
        :func:`damast.core.dataprocessing.DataProcessingPipeline.save_stats`.

        :param path: Path of the ``*.stats.yaml`` report
        :return: An exporter for the statistics it holds
        :raise ValueError: If the file holds no per-step statistics
        """
        report = yaml.safe_load(Path(path).read_text())
        if not isinstance(report, dict) or "steps" not in report:
            raise ValueError(f"{cls.__name__}.from_report: '{path}' is not a pipeline statistics"
                            f" report - expected a mapping with a 'steps' entry")

        return cls(processing_stats=report["steps"] or {}, name=report.get("name"))

    @property
    def steps(self) -> list[str]:
        """
        The step names, in the order the pipeline ran them.

        Insertion order already is execution order, but a report written before ``save_stats``
        stopped alphabetising its keys has lost that - so where every step recorded a
        ``start_time``, that is what the order is taken from.
        """
        starts = {name: stats.get("start_time") for name, stats in self.processing_stats.items()}
        if starts and all(start is not None for start in starts.values()):
            return sorted(starts, key=lambda name: starts[name])

        return list(self.processing_stats)

    def _values(self, *path: str) -> list[float | None]:
        """
        One value per step, reaching into the nested statistics by key path.

        :param path: Keys to follow per step, e.g. ``("keys", "mmsi", "unique_in")``
        :return: The value per step, None where that step did not report it
        """
        values: list[float | None] = []
        for step in self.steps:
            value: Any = self.processing_stats[step]
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value if isinstance(value, (int, float)) and not isinstance(value, bool)
                          else None)
        return values

    def _tracked_keys(self) -> list[str]:
        """The key columns any step reported on, in first-seen order."""
        columns: list[str] = []
        for step in self.steps:
            for column in self.processing_stats[step].get("keys", {}):
                if column not in columns:
                    columns.append(column)
        return columns

    def panels(self) -> list[_Panel]:
        """
        The panels to draw, leaving out every measure this run did not record - a pipeline
        without a `damast.core.tracking.KeyTracker` simply has no key panel.

        :return: The panels, in drawing order
        """
        panels = []

        # 'input_dataframe_length' is per datasource; a join has several, so total them
        rows_in = [sum(self.processing_stats[step].get("input_dataframe_length", {}).values()) or None
                   for step in self.steps]
        rows_out = self._values("output_dataframe_length")
        if any(v is not None for v in rows_in + rows_out):
            panels.append(_Panel(
                title="Rows", y_label="rows",
                series={"entering": (rows_in, _SERIES_IN), "remaining": (rows_out, _SERIES_OUT)}))

        for column in self._tracked_keys():
            unique_in = self._values("keys", column, "unique_in")
            unique_out = self._values("keys", column, "unique_out")
            panels.append(_Panel(
                title=f"Distinct '{column}'", y_label=f"distinct {column}",
                series={"entering": (unique_in, _SERIES_IN),
                        "remaining": (unique_out, _SERIES_OUT)}))

        durations = self._values("processing_time_in_s")
        if any(v is not None for v in durations):
            panels.append(_Panel(
                title="Duration per step", y_label="seconds",
                series={"duration": (durations, _SERIES_IN)}))

            running = 0.0
            cumulative: list[float | None] = []
            for value in durations:
                running += value or 0.0
                cumulative.append(running)
            panels.append(_Panel(
                title="Cumulative duration", y_label="seconds",
                series={"cumulative": (cumulative, _SERIES_OUT)}, bars=False))

        return panels

    def export_svg(self, path: str | Path) -> Path:
        """
        Draw the panels and write them as an SVG.

        :param path: Where to write the figure
        :return: The path written
        :raise ValueError: If the statistics hold nothing to draw
        """
        import matplotlib.pyplot as plt

        figure = self.build_figure()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format=path.suffix.lstrip(".") or "svg", facecolor=_SURFACE)
        plt.close(figure)
        return path

    def build_figure(self):
        """
        Build the figure, so that a caller can embed or rasterise it instead of writing an SVG.

        :return: The :class:`matplotlib.figure.Figure` holding one panel per recorded measure
        :raise ValueError: If the statistics hold nothing to draw
        """
        # imported here, so that importing this module does not pull in a backend
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        panels = self.panels()
        if not panels:
            raise ValueError(f"{self.__class__.__name__}.build_figure: no statistics to draw -"
                             f" the run recorded nothing for its {len(self.steps)} step(s)")

        steps = self.steps
        positions = range(len(steps))
        # the x band has to fit the rotated step names, so grow the figure with the step count
        figure, axes = plt.subplots(
            nrows=len(panels), ncols=1, sharex=True, squeeze=False,
            figsize=(max(7.0, 0.62 * len(steps)), 1.85 * len(panels) + 1.6),
        )
        figure.patch.set_facecolor(_SURFACE)

        for panel, axis in zip(panels, (a for row in axes for a in row)):
            axis.set_facecolor(_SURFACE)
            offsets = self._bar_offsets(len(panel.series)) if panel.bars else None

            for index, (label, (values, colour)) in enumerate(panel.series.items()):
                if panel.bars:
                    axis.bar([p + offsets[index] for p in positions],
                             [v if v is not None else 0 for v in values],
                             width=_BAR_WIDTH, label=label, color=colour, linewidth=0)
                else:
                    axis.plot(list(positions), values, label=label, color=colour,
                              linewidth=2.0, marker="o", markersize=4,
                              markeredgecolor=_SURFACE, markeredgewidth=1.5)

            axis.set_title(panel.title, loc="left", fontsize=10, color=_TEXT_PRIMARY, pad=6)
            axis.set_ylabel(panel.y_label, fontsize=8, color=_TEXT_SECONDARY)
            axis.yaxis.set_major_formatter(FuncFormatter(self._format_tick))
            self._style_axis(axis)

            # a legend names the series only where there is more than one to tell apart
            if len(panel.series) > 1:
                legend = axis.legend(loc="upper right", fontsize=8, frameon=False,
                                     labelcolor=_TEXT_SECONDARY, handlelength=1.2)
                legend.set_zorder(5)

        bottom = axes[-1][0]
        bottom.set_xticks(list(positions))
        bottom.set_xticklabels(steps, rotation=40, ha="right", fontsize=8, color=_TEXT_SECONDARY)
        bottom.set_xlabel("pipeline step", fontsize=9, color=_TEXT_SECONDARY, labelpad=6)

        title = f"{self.name} - per-step statistics" if self.name else "Per-step statistics"
        figure.suptitle(title, x=0.01, ha="left", fontsize=12, color=_TEXT_PRIMARY)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    @staticmethod
    def _bar_offsets(count: int) -> list[float]:
        """
        Where each series' bar sits within a step's slot, with a gap between neighbours.

        :param count: Number of series in the panel
        :return: The offset per series
        """
        span = _BAR_WIDTH * count
        return [-span / 2 + _BAR_WIDTH * (index + 0.5) for index in range(count)]

    @staticmethod
    def _format_tick(value: float, _position: int) -> str:
        """Abbreviate a tick, so that a row count of 30 million does not crowd the axis."""
        for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if abs(value) >= threshold:
                return f"{value / threshold:.3g}{suffix}"
        return f"{value:.3g}"

    @staticmethod
    def _style_axis(axis) -> None:
        """Hairline, solid, recessive chrome - a dashed grid reads as a threshold."""
        axis.grid(axis="y", color=_AXIS, linewidth=0.5, linestyle="-", alpha=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(_AXIS)
            axis.spines[side].set_linewidth(0.6)
        axis.tick_params(colors=_MUTED, labelsize=8, length=3, width=0.6)

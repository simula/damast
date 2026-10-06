"""
Visualize a pipeline - either its structure, or what a run of it did to the data.

Which of the two is drawn follows from the input file: a saved pipeline (``*.damast.ppl``)
has a structure to render, a run's statistics report (``*.stats.yaml``, written by
`damast.core.dataprocessing.DataProcessingPipeline.save_stats`) has per-step numbers to plot.
"""

from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.dataprocessing import (
    DAMAST_PIPELINE_SUFFIX,
    DAMAST_STATS_SUFFIX,
    DataProcessingPipeline,
)
from damast.viz.mermaid_export import MermaidExporter
from damast.viz.stats_export import StatsExporter
from damast.viz.svg_export import SvgExporter

#: Default file extension per output format
_EXTENSIONS = {"svg": ".svg", "html": ".html", "mermaid": ".mmd"}

#: Formats each kind of input can be drawn as, the first being its default
_FORMATS = {
    "stats": ["svg"],
    "pipeline": ["svg", "html", "mermaid"],
}


def resolve_kind(filename: Path) -> str:
    """
    Decide what the given file holds, from its name.

    :param filename: The file to visualize
    :return: ``"stats"`` or ``"pipeline"``
    :raise ValueError: If the suffix matches neither
    """
    name = filename.name
    if name.endswith(DAMAST_STATS_SUFFIX):
        return "stats"
    if name.endswith(DAMAST_PIPELINE_SUFFIX):
        return "pipeline"

    raise ValueError(f"Cannot visualize '{filename}': expected a pipeline ('{DAMAST_PIPELINE_SUFFIX}')"
                     f" or a run's statistics ('{DAMAST_STATS_SUFFIX}')")


class DataVisualizeParser(BaseParser):
    """Argparser for the 'viz' subcommand."""

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast viz - visualize a pipeline or a run of it"

        parser.add_argument("-f", "--filename", type=str, required=True,
                            help=f"A saved pipeline ('{DAMAST_PIPELINE_SUFFIX}') to draw the structure"
                                 f" of, or a run's statistics ('{DAMAST_STATS_SUFFIX}') to plot")
        parser.add_argument("-o", "--output-file", type=str, default=None,
                            help="Where to write the result, by default next to the input file")
        parser.add_argument("--format", type=str, default=None,
                            choices=sorted({f for formats in _FORMATS.values() for f in formats}),
                            help="Output format - 'svg' for statistics; 'svg', 'html' or 'mermaid'"
                                 " for a pipeline (default: svg)")

    def execute(self, args):
        super().execute(args)

        filename = Path(args.filename)
        if not filename.exists():
            raise FileNotFoundError(f"File '{filename}' does not exist")

        kind = resolve_kind(filename)
        permitted = _FORMATS[kind]
        output_format = args.format or permitted[0]
        if output_format not in permitted:
            raise ValueError(f"Cannot write a {kind} visualization as '{output_format}' -"
                             f" available: {', '.join(permitted)}")

        output_file = Path(args.output_file) if args.output_file else \
            filename.parent / f"{filename.name.split('.')[0]}{_EXTENSIONS[output_format]}"

        if kind == "stats":
            written = StatsExporter.from_report(filename).export_svg(output_file)
        else:
            pipeline = DataProcessingPipeline.load(filename)
            if output_format == "svg":
                written = SvgExporter(pipeline).export_svg(output_file)
            elif output_format == "html":
                written = MermaidExporter(pipeline).export_html(output_file)
            else:
                written = MermaidExporter(pipeline).export_mermaid(output_file)

        print(f"Written: {written}")

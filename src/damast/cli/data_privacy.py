import dataclasses
from argparse import ArgumentParser
from pathlib import Path

from damast.core.dataframe import AnnotatedDataFrame
from damast.domains.maritime.anonymization import AISColumns, PrivacyReport
from damast.utils.io import Archive

from .base import BaseParser


def ais_columns(mappings: list[str] | None) -> AISColumns:
    """
    `AISColumns` from 'field=column' arguments, e.g. 'mmsi=MMSI'.

    Args:
        mappings: The arguments, or None for the default columns

    Returns:
        The column names

    Raises:
        ValueError: If an argument is not 'field=column' or names no field of `AISColumns`
    """
    fields = {f.name for f in dataclasses.fields(AISColumns)}
    kwargs = {}
    for mapping in mappings or []:
        key, sep, value = mapping.partition("=")
        if not sep or key not in fields:
            raise ValueError(f"--column '{mapping}' must be 'field=column' with field one of: {', '.join(sorted(fields))}")
        kwargs[key] = value
    return AISColumns(**kwargs)


class DataPrivacyParser(BaseParser):
    """
    Argparser for the privacy report of AIS data, see `PrivacyReport`

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast privacy - report what in AIS data can identify a vessel or person"
        parser.add_argument("-f", "--files",
                            help="Files or patterns of the (annotated) data file to assess",
                            nargs="+",
                            type=str,
                            required=True)
        parser.add_argument("-o", "--output-file",
                            help="Save the report as YAML to this file",
                            default=None)
        parser.add_argument("--column",
                            help="Name of an AIS column in the data as 'field=column', e.g. 'mmsi=MMSI'"
                                 " - fields as damast.domains.maritime.anonymization.AISColumns",
                            nargs="+",
                            default=None)
        parser.add_argument("-k", type=int, default=5,
                            help="Minimum number of vessels per (cell, time bucket) (default: %(default)s)")
        parser.add_argument("--h3-resolution", type=int, default=6,
                            help="H3 resolution of the cells (default: %(default)s)")
        parser.add_argument("--time-bucket", default="10m",
                            help="Interval time is truncated to, as polars duration (default: %(default)s)")
        parser.add_argument("--rest-speed", type=float, default=0.5,
                            help="Speed in knots below which a fix is at rest (default: %(default)s)")

    def execute(self, args):
        super().execute(args)
        columns = ais_columns(args.column)

        with Archive(filenames=args.files) as input_files:
            files = [x for x in input_files if AnnotatedDataFrame.get_supported_format(Path(x).suffix)]
            if not files:
                raise RuntimeError(f"Privacy report is not supported for input files: {input_files=}")

            adf = AnnotatedDataFrame.from_files(files=files, metadata_required=False)
            report = PrivacyReport.assess(adf, columns=columns, k=args.k, h3_resolution=args.h3_resolution,
                                          time_bucket=args.time_bucket, rest_speed=args.rest_speed)

        print(report)
        if args.output_file:
            print(f"Written: {report.save_yaml(args.output_file)}")

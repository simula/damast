"""
'damast-maritime convert': 'damast convert' with a privacy mode for AIS data.
"""
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import yaml

from damast.cli.data_converter import DataConvertParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DAMAST_PIPELINE_SUFFIX
from damast.core.metadata import MetaData
from damast.core.partitioning import SaveAs
from damast.domains.maritime.anonymization import (
    AISColumns,
    AnonymizationStrategy,
    DataMinimization,
    VoyageKAnonymity,
    VoyageRelocation,
    identifier_columns,
)

from .privacy import add_assessment_arguments, ais_columns, assess

#: Values of --privacy: no privacy handling, only a report, or a strategy applied to the data
PRIVACY_MODES: list[str] = ["off", "report", "minimize", "k-anonymity", "relocation"]
#: Suffix of the privacy report written next to the output
PRIVACY_REPORT_SUFFIX: str = ".privacy.yaml"


def reannotated(df: pl.DataFrame | pl.LazyFrame, reference: MetaData) -> AnnotatedDataFrame:
    """
    The data with its metadata inferred again, e.g. after a column changed type - the units and
    descriptions of `reference` are kept.
    """
    return AnnotatedDataFrame(df, AnnotatedDataFrame.infer_annotation(df, reference=reference))


def privacy_strategy(args, columns: AISColumns) -> AnonymizationStrategy | None:
    """
    The strategy of a --privacy mode, configured from the arguments.

    Args:
        args: The parsed arguments
        columns: Names of the AIS columns in the data

    Returns:
        The strategy, or None for the modes "off" and "report", which leave the data as it is
    """
    if args.privacy == "minimize":
        return DataMinimization(rest_speed=args.rest_speed, decimals=args.decimals,
                                time_resolution=args.time_resolution, columns=columns)
    if args.privacy == "k-anonymity":
        return VoyageKAnonymity(key_env=args.privacy_key_env, rest_speed=args.rest_speed,
                                h3_resolution=args.h3_resolution, time_bucket=args.time_bucket, k=args.k,
                                columns=columns)
    if args.privacy == "relocation":
        return VoyageRelocation(key_env=args.privacy_key_env, rest_speed=args.rest_speed, columns=columns)
    return None


class MaritimeConvertParser(DataConvertParser):
    """
    Argparser of 'damast convert' with the --privacy options, see `prepare_export`

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast-maritime convert - data conversion with a privacy mode for AIS data"
        privacy = parser.add_argument_group(
            "privacy", "Assess, minimize or anonymize AIS data while converting it - see 'damast-maritime privacy'."
                       " Every mode but 'off' writes a privacy report of the input and the output.")
        privacy.add_argument("--privacy",
                             default="off",
                             choices=PRIVACY_MODES,
                             help="'report': only write the report; 'minimize': drop high owner risk vessels,"
                                  " fixes at rest and identifiers but the MMSI, coarsen position and time - not"
                                  " anonymous; 'k-anonymity', 'relocation': anonymize with VoyageKAnonymity or"
                                  " VoyageRelocation (default: %(default)s)")
        privacy.add_argument("--privacy-report",
                             default=None,
                             help="Path of the privacy report (default: next to the output file, with suffix"
                                  f" {PRIVACY_REPORT_SUFFIX}) - the pipeline that was applied is saved next to it")
        privacy.add_argument("--privacy-key-env",
                             default=None,
                             help="Environment variable holding the hex key of the pseudonyms and relocations"
                                  " - without it, a random key is drawn and the output is not reproducible")
        privacy.add_argument("--decimals", type=int, default=4,
                             help="'minimize': decimals positions are rounded to (default: %(default)s)")
        privacy.add_argument("--time-resolution", default="1m",
                             help="'minimize': interval time is truncated to (default: %(default)s)")
        add_assessment_arguments(privacy)

    def privacy_report_path(self, args, save_as: SaveAs | None) -> Path:
        """
        Where the privacy report goes: --privacy-report, or next to a single output file.

        Raises:
            ValueError: If the output is not one combined file, or partitioned without --privacy-report
        """
        if save_as is None:
            raise ValueError("--privacy needs all input combined into one output: use --save-as or --output-file"
                             " - a vessel's track can span input files")
        if args.privacy_report:
            return Path(args.privacy_report)
        if save_as.strategy is not None:
            raise ValueError("--privacy with a partitioned --save-as needs --privacy-report")
        return save_as.path.with_suffix(PRIVACY_REPORT_SUFFIX)

    def apply_privacy(self, adf: AnnotatedDataFrame, args, report_path: Path) -> AnnotatedDataFrame:
        """
        Apply the --privacy mode, and write the privacy report of input and output.

        With a strategy, its pipeline is saved next to the report, so the conversion can be
        reproduced with 'damast process' - the key itself is never saved, only --privacy-key-env.

        Args:
            adf: The data to convert
            args: The parsed arguments
            report_path: Path of the report

        Returns:
            The data to export
        """
        names = adf.column_names
        columns = ais_columns(args.column).available(names)

        report = {"privacy": args.privacy, "input": assess(adf, columns, args).to_dict()}
        strategy = privacy_strategy(args, columns)
        if strategy is not None:
            if adf.lazyframe.collect_schema().get(columns.timestamp) == pl.String:
                adf = reannotated(adf.lazyframe.with_columns(
                    pl.col(columns.timestamp).str.to_datetime(time_zone="UTC", strict=False)), adf.metadata)
            data = strategy.transform(df=adf)
            # collected once: a partitioned export must not run the strategy - and draw a random key - per partition
            adf = reannotated(data.lazyframe.collect(), data.metadata)
            if args.privacy == "minimize":
                # identifiers but the MMSI - a pipeline step cannot drop them, see DataMinimization
                report["dropped_columns"] = [c for c in identifier_columns(names, columns) if c != columns.mmsi]
                adf = adf.drop(report["dropped_columns"])
            report["output"] = assess(adf, strategy.columns, args).to_dict()
            stem = report_path.name.removesuffix(PRIVACY_REPORT_SUFFIX).removesuffix(report_path.suffix)
            pipeline_path = strategy.save(report_path.parent).rename(
                report_path.with_name(stem + DAMAST_PIPELINE_SUFFIX))
            report["pipeline"] = str(pipeline_path)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
        print(f"Privacy report: {report_path}")
        return adf

    def prepare_export(self, adf: AnnotatedDataFrame, args, save_as: SaveAs) -> AnnotatedDataFrame:
        if args.privacy == "off":
            return adf
        return self.apply_privacy(adf, args, self.privacy_report_path(args, save_as))

    def execute(self, args):
        if args.privacy != "off":
            # fail before loading: the per-file conversion has no combined output to protect
            save_as = SaveAs.parse(args.save_as) if args.save_as else (
                SaveAs(Path(args.output_file)) if args.output_file else None)
            self.privacy_report_path(args, save_as)
        super().execute(args)

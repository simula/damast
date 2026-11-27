import dataclasses
import logging
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import yaml
from tqdm import tqdm

from damast.core.dataframe import AnnotatedDataFrame, COMPRESSION_CODECS
from damast.core.dataprocessing import DAMAST_PIPELINE_SUFFIX
from damast.core.metadata import MetaData, ValidationMode
from damast.core.partitioning import SaveAs
from damast.domains.maritime.anonymization import (
    AISColumns,
    AnonymizationStrategy,
    DataMinimization,
    VoyageKAnonymity,
    VoyageRelocation,
    identifier_columns,
)
from damast.utils.io import Archive

from .base import BaseParser
from .data_privacy import add_assessment_arguments, ais_columns, assess

logger = logging.getLogger(__name__)


def stem_output_file(file: str | Path, *, directory: Path | None, suffix: str) -> Path:
    """
    Output filename schema for one-file-in/one-file-out conversion: the input file's own
    stem, under `directory` (or the current directory if `None`), with `suffix`.

    :param file: The input file whose stem to reuse
    :param directory: Output directory, or `None` for the current directory
    :param suffix: Output file extension, e.g. ``.parquet``
    """
    directory = directory if directory is not None else Path(".")
    return directory / f"{Path(file).stem}{suffix}"


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


class DataConvertParser(BaseParser):
    """
    Argparser for converting CSV files to AnnotatedDataframes (.parquet and .yml files)

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast convert - data conversion subcommand called"
        parser.add_argument("-f", "--files",
                            help="Files or patterns of the (annotated) data file that should be converted",
                            nargs="+",
                            type=str,
                            required=True
                            )
        parser.add_argument("-m", "--metadata-input",
                            help="The metadata input file",
                            default=None,
                            required=False
                            )
        parser.add_argument("-o", "--output-file",
                            help="The output file either: .parquet, .hdf5 DEPRECATED: use --save-as instead",
                            required=False
                            )
        parser.add_argument("--output-dir",
                            help="The output directory. DEPRECATED: use --save-as instead",
                            required=False,
                            )
        parser.add_argument("--output-type",
                            help="The output file type: .parquet or .hdf5 (default: %(default)s)",
                            default=".parquet",
                            required=False,
                            )
        parser.add_argument("-s","--save-as",
                            type=str,
                            default=None,
                            required=False,
                            help="Combine all input files and save the result according to this"
                                 " spec, instead of --output-file/--output-dir. A plain path saves"
                                 " one file; use 'time:<column>+<interval>:<template>',"
                                 " 'column:<column>:<template>', or"
                                 " 'time+column:<column>+<interval>+<column>:<template>' to instead"
                                 " save one file per partition - see"
                                 " damast.core.partitioning.SaveAs.parse"
                                 " Example: "
                                 " --save-as 'time+column:msgtime+daily+MMSI:data/sourcename/by-mmsi/{MMSI}/{MMSI}-%%Y-%%m-%%d'"
        )
        parser.add_argument("-c", "--compression-type",
                            default="zstd",
                            choices=[x.lower() for x in COMPRESSION_CODECS],
                            )
        parser.add_argument("-l", "--compression-level",
                            default=None,
                            type=int
        )
        parser.add_argument("--validation-mode",
                            default="update_data",
                            choices=[x.value.lower() for x in ValidationMode],
                            help="Define the validation mode (default: %(default)s)")

        privacy = parser.add_argument_group(
            "privacy", "Assess, minimize or anonymize AIS data while converting it - see 'damast privacy'."
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

    def validate(self, adf: AnnotatedDataFrame, args):
        """
        Validate the AnnotatedDataFrame if request through arguments
        """
        if not args.metadata_input:
            return

        if not Path(args.metadata_input).exists():
            raise FileNotFoundError(f"metadata-input: '{args.metadata_input}' does not exist")

        metadata = MetaData.load_yaml(filename=args.metadata_input)

        try:
            validation_mode = ValidationMode[args.validation_mode.upper()]
        except KeyError:
            raise ValueError(f"--validation-mode has invalid argument."
                             f" Select from: {[x.value.lower() for x in ValidationMode]}")

        adf._metadata = metadata
        adf.validate_metadata(validation_mode)

        if validation_mode == ValidationMode.UPDATE_DATA:
            # ensure that metadata like stats are regenerated
            adf.validate_metadata(ValidationMode.UPDATE_METADATA)

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
        columns = ais_columns(args.column)
        names = adf.column_names
        if columns.report_class not in names:
            columns = dataclasses.replace(columns, report_class=None)

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
            pipeline_path = strategy.save(report_path.parent).rename(report_path.with_name(stem + DAMAST_PIPELINE_SUFFIX))
            report["pipeline"] = str(pipeline_path)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
        print(f"Privacy report: {report_path}")
        return adf

    def execute(self, args):
        super().execute(args)

        files_stats = self.get_files_stats(args.files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        if sum(bool(x) for x in (args.output_dir, args.output_file, args.save_as)) > 1:
            raise ValueError("--output-dir, --output-file and --save-as cannot be used together")

        with Archive(filenames=args.files) as input_files:
            files = [x for x in input_files if AnnotatedDataFrame.get_supported_format(Path(x).suffix)]
            if not files:
                raise RuntimeError(f"Conversion is not supported for input files: {input_files=}")

            created_files = []
            # --output-file is a plain path, never the -save-as partitioning DSL -
            # construct SaveAs directly rather than through SaveAs.parse().
            save_as = SaveAs.parse(args.save_as) if args.save_as else (
                SaveAs(Path(args.output_file)) if args.output_file else None)
            report_path = self.privacy_report_path(args, save_as) if args.privacy != "off" else None

            if save_as is not None:
                # this conversion reads all of the data to write it out, so materialising it
                # up front keeps the metadata and the validation from reading it again - except
                # for a partitioned export, which streams: that takes memory of its own
                adf = AnnotatedDataFrame.from_files(
                        files=files,
                        metadata_required=False,
                        materialize=save_as.strategy is None,
                    )

                self.validate(adf, args)
                if report_path is not None:
                    adf = self.apply_privacy(adf, args, report_path)

                written = save_as.export(adf,
                                         compression=args.compression_type,
                                         compression_level=args.compression_level
                )
                created_files = written if isinstance(written, list) else [written]

                print(adf.head(10).collect())
                if len(created_files) <= 20:
                    print(f"Written: {created_files}")
                else:
                    print(f"Written: {len(created_files)} files: {created_files[0]}...{created_files[-1]}")
            else:
                output_dir = Path(args.output_dir) if args.output_dir else None
                if output_dir:
                    # Create multiple output files
                    output_dir.mkdir(parents=True, exist_ok=True)

                for file in tqdm(files, desc="Files"):
                    adf = AnnotatedDataFrame.from_file(
                            filename=file,
                            metadata_required=False,
                            materialize=True,
                        )

                    # One output file per input file, named by that file's own stem - not a
                    # PartitionStrategy: there is only ever one row-group here (the whole file
                    # loaded on its own), so there is nothing to split by row value.
                    output_file = stem_output_file(file, directory=output_dir, suffix=args.output_type)

                    self.validate(adf, args)

                    adf.export(filename=output_file,
                               compression=args.compression_type,
                               compression_level=args.compression_level)
                    created_files.append(output_file)

                    print(f"Filename: {output_file.resolve()}")
                    print(adf.head(10).collect())

                if len(created_files) <= 20:
                    print(f"Written: {created_files}")
                else:
                    print(f"Written: {len(created_files)} files: {created_files[0]}...{created_files[-1]}")


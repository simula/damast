import logging
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from damast.core.dataframe import AnnotatedDataFrame, COMPRESSION_CODECS
from damast.core.metadata import MetaData, ValidationMode
from damast.core.partitioning import SaveAs
from damast.utils.io import Archive

from .base import BaseParser

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
            if args.save_as or args.output_file:
                # --output-file is a plain path, never the -save-as partitioning DSL -
                # construct SaveAs directly rather than through SaveAs.parse().
                save_as = SaveAs.parse(args.save_as) if args.save_as else SaveAs(Path(args.output_file))

                adf = AnnotatedDataFrame.from_files(
                        files=files,
                        metadata_required=False,
                    )

                self.validate(adf, args)

                written = save_as.export(adf,
                                         compression=args.compression_type,
                                         compression_level=args.compression_level
                )
                created_files = written if isinstance(written, list) else [written]

                print(adf.head(10).collect())
                print(f"Written: {created_files}")
            else:
                output_dir = Path(args.output_dir) if args.output_dir else None
                if output_dir:
                    # Create multiple output files
                    output_dir.mkdir(parents=True, exist_ok=True)

                for file in tqdm(files, desc="Files"):
                    adf = AnnotatedDataFrame.from_file(
                            filename=file,
                            metadata_required=False,
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

                print(f"Written: {created_files}")


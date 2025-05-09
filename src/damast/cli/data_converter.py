import glob
from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import MetaData, ValidationMode


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
        parser.add_argument("-o", "--output",
                            help="The output file either: .parquet, .hdf5",
                            required=True
                            )
        parser.add_argument("--validation-mode",
                            default="readonly",
                            choices=[x.value.lower() for x in ValidationMode],
                            help="Define the validation mode")

    def execute(self, args):
        super().execute(args)

        files_stats = self.get_files_stats(args.files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        metadata = None

        adf = AnnotatedDataFrame.from_files(
                files=args.files,
                metadata_required=False
            )

        if args.metadata_input:
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

        print(adf.head(10).collect())
        adf.save(filename=args.output)

        if not Path(args.output).exists():
            raise FileNotFoundError(f"Failed to write {args.output}")

        print(f"Written to: {args.output}")





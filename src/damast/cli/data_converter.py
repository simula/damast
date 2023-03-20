from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame


class DataConvertParser(BaseParser):
    """
    Argparser for converting CSV files to AnnotatedDataframes (.h5 and .yml files)

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast convert - data conversion subcommand called"
        parser.add_argument("-c", "--csv-input", action="append", help="The csv input file(s)", required=True)
        parser.add_argument("-m", "--metadata-input", help="The metadata input file", required=True)
        parser.add_argument("-o", "--output", help="The output (*.hdf5) file", required=True)

    def execute(self, args):
        super().execute(args)

        for path in args.csv_input:
            if not Path(path).exists():
                raise FileNotFoundError(f"csv-input: '{path}' does not exist")

        if not Path(args.metadata_input).exists():
            raise FileNotFoundError(f"metadata-input: '{args.metadata_input}' does not exist")

        AnnotatedDataFrame.convert_csv_to_adf(csv_filenames=args.csv_input,
                                              metadata_filename=args.metadata_input,
                                              output_filename=args.output)

import glob
from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import ValidationMode


class DataConvertParser(BaseParser):
    """
    Argparser for converting CSV files to AnnotatedDataframes (.h5 and .yml files)

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast convert - data conversion subcommand called"
        parser.add_argument("-c", "--csv-input",
                            action="append",
                            help="The csv input file(s); enclosed in quotes accepts * as wildcard "
                                 "for directories of filenames",
                            required=True)
        parser.add_argument("-m", "--metadata-input",
                            help="The metadata input file",
                            required=True
                            )
        parser.add_argument("-o", "--output",
                            help="The output (*.hdf5) file",
                            required=True
                            )
        parser.add_argument("--validation-mode",
                            default="readonly",
                            choices=[x.value.lower() for x in ValidationMode],
                            help="Define the validation mode")

    def execute(self, args):
        super().execute(args)

        csv_filenames = []
        for path in args.csv_input:
            files = [f for f in glob.iglob(path)]
            if len(files) == 0:
                raise FileNotFoundError(f"csv-input: '{path}' does not match an existing filename")
            csv_filenames.extend(files)

        if not Path(args.metadata_input).exists():
            raise FileNotFoundError(f"metadata-input: '{args.metadata_input}' does not exist")

        try:
            validation_mode = ValidationMode[args.validation_mode.upper()]
        except KeyError:
            raise ValueError(f"--validation-mode has invalid argument."
                             f" Select from: {[x.value.lower() for x in ValidationMode]}")

        AnnotatedDataFrame.convert_csv_to_adf(csv_filenames=csv_filenames,
                                              metadata_filename=args.metadata_input,
                                              output_filename=args.output,
                                              validation_mode=validation_mode,
                                              progress=True)

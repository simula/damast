import os
from argparse import ArgumentParser

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame


class DataInspectParser(BaseParser):
    """
    Argparser for inspecting AnnotatedDataFrame encoded as .h5

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast inspect - data inspection subcommand called"
        parser.add_argument("-f", "--filename",
                            help="Filename of the *.hdf5 annotated data file that should be inspected",
                            required=True
                            )

    def execute(self, args):
        super().execute(args)
        stat_result = os.stat(args.filename)
        print(f"Loading dataframe of size: {stat_result.st_size / (1024**2):.2f} MB")
        adf = AnnotatedDataFrame.from_file(filename=args.filename)
        print(adf.metadata.to_str())
        print("\n\nFirst 10 rows:")
        print(adf.head(10))

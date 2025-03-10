import os
from argparse import ArgumentParser
from pathlib import Path

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

        base = Path(args.filename).parent
        stem = Path(args.filename).stem
        files = [x for x in base.glob(stem)]

        sum_st_size = 0
        for idx, file in enumerate(files):
            stat_result = os.stat(str(file))
            sum_st_size += stat_result.st_size

        print(f"Loading dataframe ({len(files)} files) of total size: {sum_st_size / (1024**2):.2f} MB")

        adf = AnnotatedDataFrame.from_file(filename=args.filename)
        print(adf.metadata.to_str())
        print("\n\nFirst 10 rows:")
        print(adf.head(10).collect())

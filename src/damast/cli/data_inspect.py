import re
from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame


class DataInspectParser(BaseParser):
    """
    Argparser for inspecting AnnotatedDataFrame

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast inspect - data inspection subcommand called"
        parser.add_argument("-f", "--filename",
                            help="Filename or pattern of the (annotated) data file that should be inspected",
                            required=True
                            )

    def execute(self, args):
        super().execute(args)

        base = Path(args.filename).parent
        name = Path(args.filename).name
        files = [x for x in base.glob(name)]

        sum_st_size = 0
        for idx, file in enumerate(files, start=1):
            sum_st_size += file.stat().st_size

        print(f"Loading dataframe ({len(files)} files) of total size: {sum_st_size / (1024**2):.2f} MB")

        try:
            adf = AnnotatedDataFrame.from_file(filename=args.filename)

            print(adf.metadata.to_str())
            print("\n\nFirst 10 rows:")
            print(adf.head(10).collect())
        except RuntimeError as e:
            if re.search(r"metadata is missing", str(e)) is not None:
                print(e)
            else:
                raise

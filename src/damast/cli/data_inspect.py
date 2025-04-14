import datetime as dt
import re
from argparse import ArgumentParser
from pathlib import Path

import polars as pl

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

        parser.add_argument("--filter", type=str, help="Filter based on column data, e.g., mmsi==120123")
        parser.add_argument("--head", type=int, default=10, help="First this number of rows, default is 10")

    def expand_filter_arg(self, adf: AnnotatedDataFrame, arg: str):
        if arg in adf.column_names:
            return f"pl.col('{arg}')"

        m = re.match(r"datetime\((.*)\)", arg)
        if m:
            return f"dt.datetime.fromisoformat({m.group(1)})"

        return arg

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
            adf = AnnotatedDataFrame.from_file(filename=args.filename, metadata_required=False)

            if args.filter:
                m = re.match(r"([^=<>]+)([=><]+)([^=<>]*)", args.filter)
                if m:
                    lhs = m.group(1).strip()
                    op = m.group(2).strip()
                    rhs = m.group(3).strip()

                    lhs = self.expand_filter_arg(adf, lhs)
                    rhs = self.expand_filter_arg(adf, rhs)
                    print(f"adf._dataframe.filter({lhs} {op} {rhs})")
                    adf._dataframe = eval(f"adf._dataframe.filter({lhs} {op} {rhs})")

            print(adf.metadata.to_str())
            print(f"\n\nFirst {args.head} rows:")
            print(adf.head(n=args.head).collect())
        except RuntimeError as e:
            if re.search(r"metadata is missing", str(e)) is not None:
                print(e)
            else:
                raise

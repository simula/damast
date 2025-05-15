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
        parser.add_argument("-f", "--files",
                            help="Files or patterns of the (annotated) data file that should be inspected (space separated)",
                            nargs="+",
                            type=str,
                            required=True
                            )

        parser.add_argument("--filter",
                help="Filter based on column data, e.g., mmsi==120123",
                action="append"
        )
        parser.add_argument("--head", type=int, default=10, help="First this number of rows, default is 10")
        parser.add_argument("--tail", type=int, default=10, help="Print number of rows from the end, default is 10")

        parser.add_argument("--columns",
                help="Show/Select these columns",
                nargs="+",
                type=str,
                required=False
        )

    def expand_filter_arg(self, adf: AnnotatedDataFrame, arg: str):
        if arg in adf.column_names:
            return f"pl.col('{arg}')"

        m = re.match(r"datetime\((.*)\)", arg)
        if m:
            return f"dt.datetime.fromisoformat({m.group(1)})"

        return arg

    def execute(self, args):
        super().execute(args)

        files = sorted(args.files)
        files_stats = self.get_files_stats(args.files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        try:
            adf = AnnotatedDataFrame.from_files(files=files, metadata_required=False)

            if args.filter:
                filter_values = ""
                for filter_expression in args.filter:
                    m = re.match(r"([^=<>]+)([=><]+)([^=<>]*)", filter_expression)
                    if m:
                        lhs = m.group(1).strip()
                        op = m.group(2).strip()
                        rhs = m.group(3).strip()

                        lhs = self.expand_filter_arg(adf, lhs)
                        rhs = self.expand_filter_arg(adf, rhs)

                        print(f"   .filter({lhs} {op} {rhs})")
                        filter_values += f".filter({lhs} {op} {rhs})"
                    else:
                        logger.warning(f"Filter expression invalid: {filter_expression}")
                adf._dataframe = eval(f"adf._dataframe{filter_values}")

            print(adf.metadata.to_str(columns=args.columns))
            print(f"\n\nFirst {args.head} and last {args.tail} rows:")
            df = adf._dataframe
            if args.columns:
                df = df.select(args.columns)

            with pl.Config(tbl_rows=args.head):
                print(df.head(n=args.head).collect())
            with pl.Config(tbl_rows=args.tail):
                print(df.tail(n=args.tail).collect())

        except RuntimeError as e:
            if re.search(r"metadata is missing", str(e)) is not None:
                print(e)
            else:
                raise

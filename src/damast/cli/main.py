"""
Main argument parser and CLI entry point.
"""
import traceback as tb
import warnings
from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.cli.data_annotate import DataAnnotateParser
from damast.cli.data_converter import DataConvertParser
from damast.cli.data_inspect import DataInspectParser
from damast.cli.data_processing import DataProcessingParser
from damast.cli.experiment import ExperimentParser


class MainParser(ArgumentParser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "damast - data processing with annotation"

        self.add_argument("-w", "--workdir", default=str(Path(".").resolve()))
        self.add_argument("-v", "--verbose", action="store_true")
        self.add_argument("--loglevel", dest="loglevel", type=int, default=10, help="Set loglevel to display")
        self.add_argument("--logfile", dest="logfile", type=str, default=None,
                          help="Set file for saving log (default prints to terminal)")

        self.subparsers = self.add_subparsers(help='sub-command help')

    def attach_subcommand_parser(self,
                                 subcommand: str,
                                 help: str,
                                 parser_klass: BaseParser
                                 ):
        parser = self.subparsers.add_parser(subcommand, help=help)
        parser_klass(parser=parser)


def run():
    """
    Run the main command line interface
    """
    main_parser = MainParser()
    main_parser.attach_subcommand_parser(subcommand="annotate",
                                         help="Annotate a dataframe",
                                         parser_klass=DataAnnotateParser)

    convert_help_desc = "Convert a dataset (set of .csv-files) to a .h5-file (containing the data)" +\
        " and .yml-file (containing data specification)"
    main_parser.attach_subcommand_parser(subcommand="convert",
                                         help=convert_help_desc,
                                         parser_klass=DataConvertParser)

    experiment_help_desc = "Run a predefined experiment"
    main_parser.attach_subcommand_parser(subcommand="experiment",
                                         help=experiment_help_desc,
                                         parser_klass=ExperimentParser)

    inspect_help_desc = "Inspect a dataset given by filename or pattern and associated metadata"
    main_parser.attach_subcommand_parser(subcommand="inspect",
                                         help=inspect_help_desc,
                                         parser_klass=DataInspectParser)

    main_parser.attach_subcommand_parser(subcommand="process",
                                         help="Process data by running a predefined pipeline",
                                         parser_klass=DataProcessingParser)


    args = main_parser.parse_args()
    if hasattr(args, "active_subparser"):
        try:
            args.active_subparser.execute(args)
        except Exception as e:
            if args.verbose:
                tb.print_exception(e)
            else:
                print(f"\033[91mError: {e}\033[00m")
    else:
        main_parser.print_help()


if __name__ == "__main__":
    run()

from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.cli.data_processing import DataProcessingParser


class MainParser(ArgumentParser):
    subparser: ArgumentParser = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "damast - data processing with annotation"

        self.add_argument("-w", "--workdir", default=str(Path(".").resolve()))
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
    main_parser = MainParser()
    main_parser.attach_subcommand_parser(subcommand="process",
                                         help="Process data",
                                         parser_klass=DataProcessingParser)
    args = main_parser.parse_args()
    args.active_subparser.execute(args)


if __name__ == "__main__":
    run()

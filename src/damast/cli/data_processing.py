from argparse import ArgumentParser

from damast.cli.base import BaseParser


class DataProcessingParser(BaseParser):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast process - processing data subcommand called"
        # parser.set_defaults(which='process')
        parser.add_argument("-f", "--filter", dest="filter", action="store_true", default=False,
                            help="Filter data")

    def execute(self, args):
        super().execute(args)

        print("This functionality has not yet been implemented")

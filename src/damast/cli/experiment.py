from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser


class ExperimentParser(BaseParser):
    """
    Argparser for handling experiments

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast experiment - allow to execute experiments"
        parser.add_argument("-f", "--filename",
                            help="Filename of the experiment that should be executed",
                            required=True
                            )
        parser.add_argument("-o", "--output-dir", dest="output_dir",
                            default=str((Path() / "output").resolve()),
                            help="Absolute path to output folder",
                            required=True)

    def execute(self, args):
        super().execute(args)

        from damast.ml.experiments import Experiment

        experiment = Experiment.from_file(args.filename)
        experiment.output_directory = args.output_dir

        experiment.run(logging_level=args.loglevel)

import argparse
from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseParser(ABC):
    @abstractmethod
    def __init__(self, parser: ArgumentParser):
        parser.add_argument('--active_subparser',
                            default=self,
                            action='store',
                            help=argparse.SUPPRESS)

    def execute(self, args):
        print("Subparser: {args.active_subparser}")
        pass

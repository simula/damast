"""
Module containing the BaseParser functionality, in order to simplify the usage of subparsers.
"""
import argparse
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilesStats:
    # Sum of file sizes in MB
    total_size: int
    files: list[str]

    @property
    def number_of_files(self):
        return len(self.files)


class BaseParser(ABC):

    @abstractmethod
    def __init__(self, parser: ArgumentParser):
        parser.add_argument('--active_subparser',
                            default=self,
                            action='store',
                            help=argparse.SUPPRESS)

    def execute(self, args):
        print(f"Subparser: {args.active_subparser.__class__.__name__}")
        pass

    def get_files_stats(self, files: list[str | Path]) -> FilesStats:
        """
        Get basic stats for the list of files given
        """
        expanded_pattern = []
        for file in files:
            base = Path(file).parent
            name = Path(file).name
            expanded_pattern += [x for x in base.glob(name)]

        sum_st_size = 0
        for idx, file in enumerate(expanded_pattern, start=1):
            sum_st_size += file.stat().st_size

        return FilesStats(total_size=round(sum_st_size / (1024**2), ndigits=0), files=expanded_pattern)

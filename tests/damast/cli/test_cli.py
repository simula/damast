from argparse import ArgumentParser
from pytest_console_scripts import ScriptRunner
import pytest
import re
import sys

from pathlib import Path

import damast.cli.main as cli_main

from damast.cli.data_converter import DataConvertParser
from damast.cli.data_inspect import DataInspectParser
from damast.cli.data_processing import DataProcessingParser
from damast.cli.experiment import ExperimentParser

@pytest.fixture
def subparsers():
    return [
        "inspect",
        "process",
        "convert",
        "experiment"
    ]

def test_help(subparsers, capsys, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['damast'])
    cli_main.run()
    captured = capsys.readouterr()

    for subparser in subparsers:
        assert re.search(subparser, captured.out)


@pytest.mark.parametrize("name, klass", [
    [ "inspect", DataInspectParser ],
    [ "process", DataProcessingParser ],
    [ "convert", DataConvertParser ],
    [ "experiment", ExperimentParser ],
])
def test_subparser(name, klass, script_runner):
    result = script_runner.run(['damast', name, "--help"])
    assert result.returncode == 0

    test_parser = ArgumentParser()
    subparser = klass(parser=test_parser)

    for a in test_parser._actions:
        if a.help == "==SUPPRESS==":
            continue

        for option in a.option_strings:
            assert re.search(option, result.stdout) is not None, f"Should have {option=}"


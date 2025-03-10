import sys

import damast.cli.main as cli_main


def test_help(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['damast'])
    cli_main.run()


import damast.cli.main as cli_main
import sys

def test_help(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['damast'])
    cli_main.run()


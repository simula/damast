"""
Entry point of 'damast-maritime'.
"""
from damast.cli.main import MainParser, execute
from damast.domains.maritime.cli.convert import MaritimeConvertParser
from damast.domains.maritime.cli.privacy import PrivacyParser


def run():
    """
    Run the command line interface of the maritime domain
    """
    main_parser = MainParser(description="damast-maritime - data processing of AIS data")
    main_parser.attach_subcommand_parser(subcommand="convert",
                                         help="Convert a dataset, with a privacy mode for AIS data",
                                         parser_klass=MaritimeConvertParser)
    main_parser.attach_subcommand_parser(subcommand="privacy",
                                         help="Report what in an AIS dataset can identify a vessel or person",
                                         parser_klass=PrivacyParser)
    execute(main_parser)


if __name__ == "__main__":
    run()

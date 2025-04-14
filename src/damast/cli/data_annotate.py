import re
from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.annotations import Annotation
from damast.core.dataframe import DAMAST_SPEC_SUFFIX, AnnotatedDataFrame
from damast.core.metadata import DataSpecification, MetaData


class DataAnnotateParser(BaseParser):
    """
    Argparser for creation of the specification / annotating an AnnotatedDataFrame

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast annotate - data annotation subcommand called"
        parser.add_argument("-f", "--filename",
                            help="Filename or pattern of the (annotated) data file that should be annotated",
                            required=True
                            )
        parser.add_argument("-i", "--interactive",
                            action="store_true",
                            help="Perform the annotation interactively",
                            default=False,
                            required=False)

        parser.add_argument("-o", "--output-dir",
                            help="Output directory",
                            default=None,
                            required=False)

        parser.add_argument("--output-spec-file",
                            help="The spec file name - if provided with path, it will override output-dir",
                            default=None,
                            required=False)


    def execute(self, args):
        super().execute(args)

        base = Path(args.filename).parent
        name = Path(args.filename).name
        files = [x for x in base.glob(name)]

        sum_st_size = 0
        for idx, file in enumerate(files, start=1):
            sum_st_size += file.stat().st_size

        print(f"Loading dataframe ({len(files)} files) of total size: {sum_st_size / (1024**2):.2f} MB")

        adf = AnnotatedDataFrame.from_file(filename=args.filename, metadata_required=False)
        print(adf.head(10).collect())

        metadata_filename = Path(args.filename).with_suffix(DAMAST_SPEC_SUFFIX)
        output_dir = Path(args.filename).parent
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_spec_file:
            spec_file = Path(args.output_spec_file)
            if spec_file.is_absolute():
                spec_file.parent.mkdir(parents=True, exist_ok=True)
                metadata_filename = spec_file.with_suffix(DAMAST_SPEC_SUFFIX)
            else:
                metadata_filename = output_dir / spec_file.with_suffix(DAMAST_SPEC_SUFFIX)
        else:
            metadata_filename = output_dir / metadata_filename.name

        metadata = AnnotatedDataFrame.infer_annotation(df=adf.dataframe)
        metadata.add_annotation(
                Annotation(
                    name=Annotation.Key.Source,
                    value=args.filename
                )
        )
        metadata.save_yaml(metadata_filename)

        print(f"Created {metadata_filename}")

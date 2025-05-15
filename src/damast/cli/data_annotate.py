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

        parser.description = "damast annotate - extract (default) or apply annotation to dataset"
        parser.add_argument("-f", "--files",
                            help="Files or patterns of the (annotated) data file that should be annotated",
                            nargs="+",
                            type=str,
                            required=True
                            )
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

        files_stats = self.get_files_stats(args.files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        adf = AnnotatedDataFrame.from_files(files=args.files, metadata_required=False)
        print(adf.head(10).collect())

        metadata_filename = MetaData.specfile(args.files)
        output_dir = Path(metadata_filename).parent
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

        metadata = AnnotatedDataFrame.infer_annotation(df=adf)
        metadata.add_annotation(
                Annotation(
                    name=Annotation.Key.Source,
                    value=args.files
                )
        )
        metadata.save_yaml(metadata_filename)

        print(f"Created {metadata_filename}")

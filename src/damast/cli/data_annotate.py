import re
from argparse import Action, ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.annotations import Annotation
from damast.core.dataframe import DAMAST_SPEC_SUFFIX, AnnotatedDataFrame
from damast.core.metadata import DataSpecification, MetaData


class SetTxtFieldAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        name = option_string.lstrip('--set-')
        if not hasattr(namespace, "update_metadata"):
            setattr(namespace, "update_metadata", MetaData(columns=[]))

        for value in values:
            column, column_value = value.split(":", 1)
            if column not in namespace.update_metadata:
                ds = DataSpecification(name=column)
                namespace.update_metadata.columns.append(ds)

            setattr(namespace.update_metadata[column], name, column_value)

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

        str_options = ["description", "abbreviation", "unit", "representation_type"]
        for field_name in str_options:
            parser.add_argument(f"--set-{field_name}",
                                nargs="+",
                                action=SetTxtFieldAction,
                                metavar="COLUMN:VALUE",
                                help=f"Set {field_name} in spec for a column and value",
                                required=False)

        parser.add_argument("--inplace",
                            help="Update the dataset inplace (only possible for a single file)",
                            action="store_true",
                            required=False)
        parser.add_argument("--apply",
                help="Update the annotation inference and rewrite the metadata to the dataset",
                action="store_true",
                required=False,
        )



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
            spec_file = Path(args.output_spec_file.replace(DAMAST_SPEC_SUFFIX, ''))
            if spec_file.is_absolute():
                spec_file.parent.mkdir(parents=True, exist_ok=True)
                metadata_filename = spec_file.with_suffix(DAMAST_SPEC_SUFFIX)
            else:
                metadata_filename = output_dir / spec_file.with_suffix(DAMAST_SPEC_SUFFIX)
        else:
            metadata_filename = output_dir / metadata_filename.name

        metadata = AnnotatedDataFrame.infer_annotation(df=adf)

        if hasattr(args, "update_metadata"):
            metadata = metadata.merge(args.update_metadata, strategy=DataSpecification.MergeStrategy.OTHER)

        metadata.add_annotation(
                Annotation(
                    name=Annotation.Key.Source,
                    value=args.files
                )
        )

        if hasattr(args, "update_metadata") or args.apply:
            if len(args.files) == 1:
                output_file = args.files[0]
                for column_spec in args.update_metadata.columns:
                    if column_spec.representation_type:
                        representation_type = adf.set_dtype(column_spec.name, column_spec.representation_type)
                        metadata[column_spec.name].representation_type = representation_type

                adf._metadata = metadata
                if not args.inplace:
                    output_file = output_dir / (Path(output_file).stem + "-annotated" + Path(output_file).suffix)
                    print(f"Creating {output_file}")
                else:
                    print(f"Updating {output_file}")
                adf.export(output_file)
            else:
                raise ValueError("Cannot update metadata for multiple files")

            metadata_filename = Path(output_file).parent / (Path(output_file).stem + DAMAST_SPEC_SUFFIX)
            metadata.save_yaml(metadata_filename)
        else:
            metadata.save_yaml(metadata_filename)
        print(f"Created {metadata_filename}")

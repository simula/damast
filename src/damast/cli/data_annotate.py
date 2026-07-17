from argparse import Action, ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.annotations import Annotation
from damast.core.dataframe import DAMAST_SPEC_SUFFIX, AnnotatedDataFrame
from damast.core.metadata import (
    DataCategory,
    DataSpecification,
    MetaData,
    ValidationMode,
)


class SetTxtFieldAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        name = option_string.lstrip('--set-')
        if not hasattr(namespace, "update_metadata"):
            setattr(namespace, "update_metadata", MetaData(columns=[]))

        for value in values:
            column, column_value = value.split(":", 1)
            if name == "category":
                if column_value not in DataCategory:
                    raise ValueError(f"Allowed values for category: {','.join([x.value for x in DataCategory])}")
                column_value = DataCategory[column_value.upper()]
            elif name == "precision":
                column_value = float(column_value)
            elif name == "is_optional":
                if column_value.lower() in ['true', '1', 'y', 'yes']:
                    column_value = True
                elif column_value.lower() in ['false', '0', 'n', 'no']:
                    column_value = False
                else:
                    raise ValueError("Allowed valued for is_optional: 'true' or 'false'")

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

        parser.add_argument("-m", "--metadata-input",
                            help="The metadata that is applied to the data",
                            default=None,
                            required=False
                            )

        parser.add_argument("-o", "--output-dir",
                            help="Output directory",
                            default=None,
                            required=False)

        parser.add_argument("--output-spec-file",
                            help="The spec file name - if provided with path, it will override output-dir",
                            default=None,
                            required=False)

        str_options = [
                    "abbreviation",
                    "category",
                    "description",
                    "is_optional",
                    "precision",
                    "representation_type",
                    "unit",
        ]
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
                help="Update the annotation, infer and rewrite the metadata to the dataset (implicitly used when using --set-X option)",
                action="store_true",
                required=False,
        )


    def execute(self, args):
        super().execute(args)

        files_stats = self.get_files_stats(args.files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        update_metadata = args.apply or hasattr(args, 'update_metadata')
        if update_metadata:
            if args.inplace:
                for file in args.files:
                    self.update(args, [file])
            else:
                self.update(args, args.files)
        else:
            self.extract(args, args.files)


    def extract(self, args, files):
        if not files:
            raise ValueError("Missing files for updating metadata")

        adf = AnnotatedDataFrame.from_files(files=files, metadata_required=False, validation_mode=ValidationMode.IGNORE)
        print(adf.head(10).collect())

        metadata_filename = MetaData.specfile(files)
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
        metadata.add_annotation(
                Annotation(
                    name=Annotation.Key.Source,
                    value=files
                )
        )

        metadata.save_yaml(metadata_filename)
        print(f"Created {metadata_filename}")


    def update(self, args, files):
        if not files:
            raise ValueError("Missing files for updating metadata")

        if len(files) > 1:
            raise ValueError("Cannot update metadata for multiple files")

        output_file = files[0]
        adf = AnnotatedDataFrame.from_files(files=files, metadata_required=False, validation_mode=ValidationMode.IGNORE)
        if args.metadata_input:
            metadata = MetaData.load_yaml(args.metadata_input)
        else:
            metadata = AnnotatedDataFrame.infer_annotation(df=adf)

        metadata.add_annotation(
                Annotation(
                    name=Annotation.Key.Source,
                    value=files
                )
        )

        print("Loaded dataframe")
        print(adf.head(10).collect())

        if hasattr(args, "update_metadata"):
            metadata = metadata.merge(args.update_metadata,
                                      strategy=DataSpecification.MergeStrategy.OTHER)
            for x in metadata.columns:
                if x.name not in adf.column_names:
                    raise ValueError(f"Column '{x.name}' does not exist")

            for column_spec in args.update_metadata.columns:
                if column_spec.representation_type:
                    representation_type = adf.set_dtype(column_spec.name, column_spec.representation_type)
                    metadata[column_spec.name].representation_type = representation_type
                    metadata[column_spec.name].update_datarange_and_stats(adf.lazyframe, column_spec.name)

        adf._metadata = metadata
        adf.validate_metadata(ValidationMode.UPDATE_DATA)
        adf.validate_metadata(ValidationMode.UPDATE_METADATA)

        if not args.inplace:
            metadata_filename = MetaData.specfile(files)

            output_dir = Path(metadata_filename).parent
            if args.output_dir is not None:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / (Path(output_file).stem + "-annotated" + Path(output_file).suffix)
            print(f"Creating new {output_file}")
        else:
            print(f"Updating (inplace) {output_file}")

        adf.export(output_file)

        metadata_filename = Path(output_file).parent / (Path(output_file).stem + DAMAST_SPEC_SUFFIX)
        metadata.save_yaml(metadata_filename)
        print(f"Created {metadata_filename}")

        print("Updated dataframe")
        adf  = AnnotatedDataFrame.from_files(files=[output_file])
        print(adf.head(10).collect())


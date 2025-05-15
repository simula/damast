import glob
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import MetaData, ValidationMode


class DataConvertParser(BaseParser):
    """
    Argparser for converting CSV files to AnnotatedDataframes (.parquet and .yml files)

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast convert - data conversion subcommand called"
        parser.add_argument("-f", "--files",
                            help="Files or patterns of the (annotated) data file that should be converted",
                            nargs="+",
                            type=str,
                            required=True
                            )
        parser.add_argument("-m", "--metadata-input",
                            help="The metadata input file",
                            default=None,
                            required=False
                            )
        parser.add_argument("-o", "--output-file",
                            help="The output file either: .parquet, .hdf5",
                            required=False
                            )
        parser.add_argument("--output-dir",
                            help="The output directory",
                            required=False,
                            )
        parser.add_argument("--output-type",
                            help="The output file type: .parquet or .hdf5",
                            required=False,
                            )
        parser.add_argument("--validation-mode",
                            default="readonly",
                            choices=[x.value.lower() for x in ValidationMode],
                            help="Define the validation mode")
    def execute(self, args):
        super().execute(args)

        files = args.files
        files_stats = self.get_files_stats(files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        if args.output_dir and args.output_file:
            raise ValueError("--output-dir and --output-file cannot be used together")

        metadata = None

        expanded_files = []
        local_mount = Path(".local-mount")
        local_mount.mkdir(parents=True, exist_ok=True)
        mounted_dirs = []
        for file in files:
            if Path(file).suffix in [".tar", ".gz", ".zip"]:
                target_mount = local_mount / Path(file).name
                target_mount.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ratarmount", file, target_mount])
                mounted_dirs.append(target_mount)

                decompressed_files = [x for x in Path(target_mount).glob("*")]
                expanded_files += decompressed_files

        if expanded_files:
            files = expanded_files

        if args.metadata_input:
            if not Path(args.metadata_input).exists():
                raise FileNotFoundError(f"metadata-input: '{args.metadata_input}' does not exist")
            metadata = MetaData.load_yaml(filename=args.metadata_input)

            try:
                validation_mode = ValidationMode[args.validation_mode.upper()]
            except KeyError:
                raise ValueError(f"--validation-mode has invalid argument."
                                 f" Select from: {[x.value.lower() for x in ValidationMode]}")
            adf._metadata = metadata
            adf.validate_metadata(validation_mode)

        created_files = []
        try:
            if args.output_dir:
                # Create multiple output files
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                for file in files:
                    adf = AnnotatedDataFrame.from_files(
                            files=file,
                            metadata_required=False,
                        )
                    output_file = output_dir / f"{Path(file).stem}{args.output_type}"
                    adf.save(filename=output_file)
                    created_files.append(output_file)
            else:
                # Create single output file
                adf = AnnotatedDataFrame.from_files(
                        files=files,
                        metadata_required=False,
                    )
                adf.save(filename=args.output_file)
                created_files.append(args.output_file)

            print(adf.head(10).collect())
            print(f"Written: {created_files}")
        except Exception as e:
            raise
        finally:
            for mounted_dir in mounted_dirs:
                subprocess.run(["umount", mounted_dir])






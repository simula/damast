import logging
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import MetaData, ValidationMode
from damast.utils.io import Archive

logger = logging.getLogger(__name__)

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
                            help="The output file type: .parquet (default) or .hdf5 ",
                            default=".parquet",
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
        archive = None
        try:
            archive = Archive(filenames=files)
            extracted_files = archive.mount()
            if extracted_files:
                files = [x for x in extracted_files if AnnotatedDataFrame.get_supported_format(Path(x).suffix)]

            created_files = []
            if args.output_dir:
                # Create multiple output files
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            for file in tqdm(files, desc="Files"):
                adf = AnnotatedDataFrame.from_file(
                        filename=file,
                        metadata_required=False,
                    )

                if args.output_dir:
                    output_file = output_dir / f"{Path(file).stem}{args.output_type}"
                else:
                    output_file = Path(args.output_file)

                adf.save(filename=output_file)
                created_files.append(output_file)

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

                print(f"Filename: {output_file.resolve()}")
                print(adf.head(10).collect())

            print(f"Written: {created_files}")
        except Exception as e:
            raise
        finally:
            try:
                archive.umount()
            except Exception as e:
                logger.warning(f"Unmount failed {e}")


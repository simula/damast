from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DAMAST_PIPELINE_SUFFIX, DataProcessingPipeline
from damast.utils.io import Archive


class DataProcessingParser(BaseParser):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast process - apply an existing pipeline"

        parser.add_argument("--input-data",
                            help="Input file(s) to process",
                            nargs="+",
                            type=str,
                            required=True
        )
        parser.add_argument("--pipeline", help="Pipeline (*.damast.ppl) file to apply to the data", required=True)

        parser.add_argument("--output-file",
                        help="Save the result in the given (*.parquet) file",
                        required=False)

    def execute(self, args):
        super().execute(args)

        with Archive(filenames=args.input_data) as input_files:
            files = [x for x in input_files if AnnotatedDataFrame.get_supported_format(Path(x).suffix)]
            if not files:
                raise RuntimeError(f"Processing is not possible for input files: {input_files=}")

            adf = AnnotatedDataFrame.from_files(
                    files,
                    metadata_required=False
            )

            pipeline_path = Path(args.pipeline)
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline {pipeline_path} does not exist")

            if not str(pipeline_path).endswith(DAMAST_PIPELINE_SUFFIX):
                raise ValueError(f"File suffix of pipeline file is not matching {DAMAST_PIPELINE_SUFFIX}")


            pipeline = DataProcessingPipeline.load(pipeline_path)
            new_adf = pipeline.transform(adf)

            print(new_adf.head().collect())
            print(new_adf.tail().collect())

            if args.output_file:
                path = Path(args.output_file)
                path.parent.resolve().mkdir(parents=True, exist_ok=True)

                new_adf.save(filename=path)
                print(f"Saved {path.resolve()}")

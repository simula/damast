from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.constants import DAMAST_DEFAULT_DATASOURCE
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DAMAST_PIPELINE_SUFFIX, DataProcessingPipeline
from damast.utils.io import Archive


def resolve_input_data(raw_groups: list[list[str]] | None, datasource_names: list[str]) -> dict[str, list[str]]:
    """
    Resolve repeated '--input-data' occurrences into a mapping of datasource name to its
    file(s).

    :param raw_groups: One list of tokens per '--input-data' occurrence, e.g.
        [["1.zip"], ["osint_events=2.parquet", "3.parquet"]]
    :param datasource_names: The pipeline's actual datasource names, in execution order - the
        first is always the default datasource ('df')
    :raise RuntimeError: If no input was given, a required 'name=' prefix is missing, a name
        is unknown or repeated, or a declared datasource is left unsupplied
    """
    if not raw_groups:
        raise RuntimeError(
            "--input-data is required (unless --describe is given) - this pipeline requires"
            f" input for datasource(s): {', '.join(datasource_names)}"
        )

    # backward-compatible bare form: a single-datasource pipeline, no occurrence uses 'name=' -
    # every file across every occurrence goes to that one datasource
    if len(datasource_names) == 1 and all("=" not in group[0] for group in raw_groups):
        return {datasource_names[0]: [f for group in raw_groups for f in group]}

    resolved: dict[str, list[str]] = {}
    for group in raw_groups:
        first, *rest = group
        if "=" not in first:
            raise RuntimeError(
                "--input-data: this pipeline requires more than one named datasource"
                f" ({', '.join(datasource_names)}) - prefix each --input-data occurrence with"
                f" 'name=', e.g. '{datasource_names[0]}={first}'"
            )

        name, _, first_file = first.partition("=")
        if name not in datasource_names:
            raise RuntimeError(
                f"--input-data: unknown datasource '{name}' - this pipeline declares:"
                f" {', '.join(datasource_names)}"
            )
        if name in resolved:
            raise RuntimeError(f"--input-data: datasource '{name}' was given more than once")

        files = ([first_file] if first_file else []) + rest
        if not files:
            raise RuntimeError(f"--input-data: no files given for datasource '{name}'")
        resolved[name] = files

    missing = [name for name in datasource_names if name not in resolved]
    if missing:
        raise RuntimeError(
            f"--input-data: missing input for datasource(s) {', '.join(missing)} - this"
            f" pipeline requires: {', '.join(datasource_names)}"
        )
    return resolved


def load_dataframes(input_data: dict[str, list[str]]) -> dict[str, AnnotatedDataFrame]:
    """
    Load one :class:`AnnotatedDataFrame` per datasource, from its resolved file(s).
    """
    dataframes = {}
    for name, raw_files in input_data.items():
        with Archive(filenames=raw_files) as input_files:
            files = [x for x in input_files if AnnotatedDataFrame.get_supported_format(Path(x).suffix)]
            if not files:
                raise RuntimeError(f"Processing is not possible for input files: {raw_files=}")

            dataframes[name] = AnnotatedDataFrame.from_files(files, metadata_required=False)
    return dataframes



class DataProcessingParser(BaseParser):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast process - apply an existing pipeline"

        parser.add_argument("--input-data",
                            help="Input file(s) for a datasource - 'FILE...' for a"
                                 " single-datasource pipeline, or repeated"
                                 " '--input-data NAME=FILE...' (one per datasource, e.g."
                                 " 'osint_events=1.parquet 2.parquet') for a pipeline that"
                                 " requires more than one. Not required if --describe is given.",
                            nargs="+",
                            action="append",
                            type=str,
                            required=False
        )
        parser.add_argument("--pipeline", help="Pipeline (*.damast.ppl) file to apply to the data", required=True)

        parser.add_argument("--output-file",
                        help="Save the result of a pipeline in the given (*.parquet) file",
                        default=None,
                        required=False)

        parser.add_argument("--base-dir",
                        help="Save pipeline artifacts relative to the given base directory (default: %(default)s)",
                        default=".")

        parser.add_argument("--describe",
                        help="Print the pipeline's interface (required datasources and their"
                             " columns) and steps, then exit - no --input-data needed",
                        action="store_true",
                        default=False)

    def execute(self, args):
        super().execute(args)

        pipeline_path = Path(args.pipeline)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline {pipeline_path} does not exist")

        if not str(pipeline_path).endswith(DAMAST_PIPELINE_SUFFIX):
            raise ValueError(f"File suffix of pipeline file is not matching {DAMAST_PIPELINE_SUFFIX}")

        pipeline = DataProcessingPipeline.load(pipeline_path)
        if args.base_dir:
            pipeline.base_dir = args.base_dir

        if args.describe:
            print(pipeline.describe())
            return

        datasource_names = [n.name for n in pipeline.processing_graph.datasource_nodes()]
        input_data = resolve_input_data(args.input_data, datasource_names)
        dataframes = load_dataframes(input_data)

        new_adf = pipeline.transform(df=dataframes.pop(DAMAST_DEFAULT_DATASOURCE), **dataframes)

        print(new_adf.head().collect())
        print(new_adf.tail().collect())

        path = Path(pipeline.base_dir) / f"{pipeline.name}.parquet"
        if args.output_file:
            path = Path(args.output_file)

        path.parent.resolve().mkdir(parents=True, exist_ok=True)

        new_adf.save(filename=path)
        print(f"Saved {path.resolve()}")

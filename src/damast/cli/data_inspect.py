import datetime
import logging
import re
from argparse import ArgumentParser
from pathlib import Path

import polars as pl

import damast  # noqa
from damast.cli.base import BaseParser
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import ValidationMode
from damast.utils.io import Archive

logger = logging.getLogger(__name__)

class DataInspectParser(BaseParser):
    """
    Argparser for inspecting AnnotatedDataFrame

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast inspect - data inspection subcommand called"
        parser.add_argument("-f", "--files",
                            help="Files or patterns of the (annotated) data file that should be inspected (space separated)",
                            nargs="+",
                            type=str,
                            required=True
                            )

        parser.add_argument("--filter",
                help="Filter based on column data, e.g., mmsi==120123, allowed operators !=,<,>,==,<=,>=,<> or =~ for a regex, e.g.,'name =~ '^SAR*'",
                nargs="+",
                type=str,
                required=False
        )
        parser.add_argument("--head", type=int, default=10, help="First this number of rows, default is 10")
        parser.add_argument("--tail", type=int, default=10, help="Print number of rows from the end, default is 10")
        parser.add_argument("--column-count", type=int, default=10, help="Number of columns to show")
        parser.add_argument("--column-width", type=int, default=None, help="Column width to show")

        parser.add_argument("--columns",
                help="Show/Select these columns",
                nargs="+",
                type=str,
                required=False
        )

        parser.add_argument("--validation-mode",
                            default="readonly",
                            choices=[x.value.lower() for x in ValidationMode],
                            help="Define the validation mode")

    def fill_missing_value_stats(self, adf: AnnotatedDataFrame) -> dict[str, set[str]]:
        """
        Compute a column's 'value_range'/'value_stats' on the fly wherever the loaded
        metadata left them unset, without touching either field where it is already set.

        If `adf.metadata_inferred` is True (no metadata spec file was found for the input,
        so its metadata was inferred from the data itself - see
        `damast.core.dataframe.AnnotatedDataFrame.metadata_inferred`), every populated
        'value_range'/'value_stats' is reported as generated too, not just the ones this
        call fills in, since none of it came from an actual spec file either way.

        Args:
            adf: The dataframe whose metadata columns to fill in, updated in place

        Returns:
            Column name to the set of field names ('value_range' and/or 'value_stats') that
            are not backed by an actual metadata spec - for
            `damast.core.metadata.MetaData.to_str`'s ``generated_fields``
        """
        generated_fields: dict[str, set[str]] = {}
        for column_spec in adf.metadata.columns:
            original_range = column_spec.value_range
            original_stats = column_spec.value_stats

            if original_range is None or original_stats is None:
                column_spec.update_datarange_and_stats(adf.lazyframe, column_spec.name)

                if not (original_range is None and column_spec.value_range is not None):
                    column_spec.value_range = original_range
                if not (original_stats is None and column_spec.value_stats is not None):
                    column_spec.value_stats = original_stats

            computed = set()
            for field_name, original_value in (("value_range", original_range), ("value_stats", original_stats)):
                if getattr(column_spec, field_name) is None:
                    continue
                if original_value is None or adf.metadata_inferred:
                    computed.add(field_name)

            if computed:
                generated_fields[column_spec.name] = computed

        return generated_fields

    def expand_filter_arg(self, adf: AnnotatedDataFrame, arg: str):
        if arg in adf.column_names:
            return f"pl.col('{arg}')"

        m = re.match(r"datetime\(([^,]*)(,\s*(.*))?\)", arg)
        if m:
            time_zone = m.group(3)
            return f"pl.lit({m.group(1)}).str.to_datetime(time_zone={time_zone})"

        return arg

    def execute(self, args):
        super().execute(args)

        files_stats = self.get_files_stats(args.files)
        print(f"Loading dataframe ({files_stats.number_of_files} files) of total size: {files_stats.total_size} MB")

        try:
            with Archive(filenames=args.files) as input_files:
                files = [x for x in input_files if AnnotatedDataFrame.get_supported_format(Path(x).suffix)]

                if not files:
                    raise RuntimeError(f"Inspection is not supported for input files: {input_files=}")

                try:
                    validation_mode = ValidationMode[args.validation_mode.upper()]
                except KeyError:
                    raise ValueError(f"--validation-mode has invalid argument."
                                     f" Select from: {[x.value.lower() for x in ValidationMode]}")

                adf = AnnotatedDataFrame.from_files(files=files, metadata_required=False, validation_mode=validation_mode)

                if args.filter:
                    filter_values = ""
                    for filter_expression in args.filter:
                        m = re.match(r"([^!=<>~]+)([!=><][=~]?)([^!=<>~]*)", filter_expression)
                        if m:
                            lhs = m.group(1).strip()
                            op = m.group(2).strip()
                            rhs = m.group(3).strip()

                            lhs = self.expand_filter_arg(adf, lhs)

                            new_filter = ""
                            if rhs in ["null", "None"]:
                                if op == "==":
                                    new_filter = f"{lhs}.is_null()"
                                elif op == "!=":
                                    new_filter = f"{lhs}.is_not_null()"
                                else:
                                    logger.warning("Filter expression invalid: operator must be either '==' or '!='")
                                    continue
                            elif op == "=~":
                                new_filter = f"{lhs}.str.contains(r'{rhs}')"
                            else:
                                rhs = self.expand_filter_arg(adf, rhs)
                                new_filter = f"{lhs} {op} {rhs}"

                            print(f"   .filter({new_filter})")
                            filter_values += f".filter({new_filter})"
                        else:
                            logger.warning(f"Filter expression invalid: {filter_expression}")

                    safe_globals = {
                        '__builtins__': None,
                        'adf': adf,
                        'dt': datetime,
                        'pl': pl
                    }
                    adf.lazyframe = eval(f"adf.lazyframe{filter_values}", safe_globals)
                    # Refresh value_range/value_stats in place to match the filtered rows,
                    adf.validate_metadata(validation_mode=ValidationMode.UPDATE_METADATA)

                generated_fields = self.fill_missing_value_stats(adf)
                print(adf.metadata.to_str(columns=args.columns, generated_fields=generated_fields))
                print(f"\n\nFirst {args.head} and last {args.tail} rows:")
                df = adf.lazyframe

                df = df.select(pl.all().name.map(lambda name: name + f" ({unit})" if name in adf.metadata and (unit := getattr(adf.metadata[name], "unit", None)) else name))

                if args.columns:
                    df = df.select(args.columns)

                with pl.Config(tbl_rows=args.head, tbl_cols=args.column_count, fmt_str_lengths=args.column_width):
                    print(df.head(n=args.head).collect())
                with pl.Config(tbl_rows=args.tail, tbl_cols=args.column_count, fmt_str_lengths=args.column_width):
                    print(df.tail(n=args.tail).collect())

        except RuntimeError as e:
            if re.search(r"metadata is missing", str(e)) is not None:
                print(e)
            else:
                raise

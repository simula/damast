import logging
from argparse import ArgumentParser
from pathlib import Path

from damast.cli.base import BaseParser
from damast.core.watch import WatchConfig, WatchJob, WatchResult

logger = logging.getLogger(__name__)


class DataWatchParser(BaseParser):
    """
    Argparser for watching directories and running a configured command on completed files

    :param parser: The base parser
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser)

        parser.description = "damast watch - watch directories for completed files and act on them"
        parser.add_argument("-c", "--config",
                            help="YAML watch config file, defining one or more jobs",
                            required=False,
                            )
        parser.add_argument("--job",
                            help="Restrict to job(s) with this name (repeatable);"
                                 " default: run every job in the config",
                            action="append",
                            default=None,
                            )
        parser.add_argument("--dry-run",
                            help="List files that are ready to be processed, without running anything",
                            action="store_true",
                            default=False,
                            )
        parser.add_argument("--create-config",
                            help="Interactively create the watch config file given via --config,"
                                 " instead of running a scan",
                            action="store_true",
                            default=False,
                            )

    def print_result(self, name: str, result: WatchResult, dry_run: bool):
        if result.config_error:
            print(f"[{name}] config error: {result.config_error}")
            return

        if dry_run:
            print(f"[{name}] would process {len(result.would_process)} file(s), "
                  f"{len(result.not_ready)} not yet ready")
            for f in result.would_process:
                print(f"  {f}")
            return

        print(f"[{name}] processed={len(result.processed)} failed={len(result.failed)} "
              f"not_ready={len(result.not_ready)}")
        for f in result.processed:
            print(f"  ok: {f}")
        for f in result.failed:
            print(f"  FAILED: {f}")

    def execute(self, args):
        super().execute(args)

        if args.create_config:
            config_path = Path(args.config) if args.config else Path("watch-config.yaml")
            WatchConfig.create_interactively(config_path)
            return

        jobs = WatchConfig.load(Path(args.config)).jobs

        if args.job:
            job_names = {job.name for job in jobs}
            unknown = [name for name in args.job if name not in job_names]
            if unknown:
                raise ValueError(f"--job: unknown job(s) {unknown} - config defines: {sorted(job_names)}")

            jobs = [job for job in jobs if job.name in args.job]

        results = WatchJob.run_jobs(jobs, dry_run=args.dry_run)

        for name, result in results.items():
            self.print_result(name, result, args.dry_run)

        failed_jobs = {name: result for name, result in results.items() if result.has_failures}
        if failed_jobs:
            raise RuntimeError(f"damast watch: {len(failed_jobs)} job(s) had failures: {sorted(failed_jobs)}")

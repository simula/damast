"""
Module to watch directories and trigger file processors

A WatchConfig defines one or more jobs, where a job does the following:
1. scans a source directory for files matching a pattern,
2. considers a file "ready" once it has not been modified for a
configurable quiet period, and
3. then runs a job-specific command on it.

The command is a plain argv list (no shell) with placeholders for the ready file and the job's output
directory. On success the source file is moved into the job's ``processed_dir``, on
failure into its ``failed_dir`` alongside an error log.

This module intentionally has no notion of specific file types, conversions, or pipelines - the configured
command owns that, e.g., typically invoking ``damast convert``/``damast process``.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
import traceback as tc
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tqdm import tqdm
from typing_extensions import Self

__all__ = [
    "WatchJob",
    "WatchResult",
    "WatchConfig",
]

logger = logging.getLogger(__name__)

DAMAST_WATCH_DEFAULT_PATTERN: str = "*.csv"
DAMAST_WATCH_DEFAULT_QUIET_PERIOD_IN_S: float = 1800


class WatchJob(BaseModel):
    """
    A single watch job: a source directory to scan and the command to run on ready files.

    `name`, `target_dir`, `processed_dir` and `failed_dir` default from `source_dir` when
    omitted.

    :param name: Unique identifier for this job, used in logs and CLI ``--job`` filtering
    :param source_dir: Directory to scan for candidate files
    :param command: Argv list to run for each ready file - see `render_command`
        for the available placeholders
    :param target_dir: Directory made available to the command as ``{output_dir}``
    :param pattern: Glob pattern (relative to `source_dir`) for candidate files
    :param quiet_period_seconds: Seconds since the last modification before a file is
        considered complete (config key: ``quiet_period``)
    :param processed_dir: Where successfully handled source files are moved
    :param failed_dir: Where source files that raised an error are moved, alongside a
        ``<file>.error.log``
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str
    source_dir: Path
    command: list[str]
    target_dir: Path
    processed_dir: Path
    failed_dir: Path
    pattern: str = DAMAST_WATCH_DEFAULT_PATTERN
    quiet_period_seconds: float = Field(DAMAST_WATCH_DEFAULT_QUIET_PERIOD_IN_S, alias="quiet_period")

    @classmethod
    def expand_envvars(cls, txt: str) -> str:
            resolved_txt = txt
            m = re.match(r".*\${(?P<envvar>[a-zA-Z_]+)}.*", resolved_txt)
            if m:
                envvar = m.group("envvar")
                if envvar not in os.environ:
                    raise RuntimeError(f"WatchJob: variable ${envvar} is not available in environment")
                resolved_txt = re.sub(r"\${" + envvar + "}", os.environ[envvar], resolved_txt)

            for home_vars in [r"~", r"{home}"]:
                resolved_txt = re.sub(home_vars, str(Path.home()), resolved_txt)

            return resolved_txt

    @classmethod
    def expand_path(cls, path: Path | str) -> Path:
            resolved_name = str(Path(path))
            return Path(cls.expand_envvars(txt=resolved_name))

    @model_validator(mode="before")
    @classmethod
    def _default_dirs_from_source_dir(cls, data: object) -> object:
        """Fill in 'name'/'target_dir'/'processed_dir'/'failed_dir' from 'source_dir' if absent."""
        if not isinstance(data, dict) or "source_dir" not in data:
            return data

        data = dict(data)
        source_dir = cls.expand_path(Path(data["source_dir"]))

        data.setdefault("name", source_dir.name)
        data.setdefault("target_dir", source_dir)
        data.setdefault("processed_dir", source_dir / "processed")
        data.setdefault("failed_dir", source_dir / "failed")
        return data

    @model_validator(mode="after")
    def _auto_expand_var(self) -> Self:
        """Ensure that home directory and variable are being resolved"""
        for dirname in ["source_dir", "target_dir", "processed_dir", "failed_dir"]:
            path = getattr(self, dirname)
            expanded_path = self.expand_path(path)

            if expanded_path != path:
                setattr(self, dirname, Path(expanded_path))
        return self

    def unique_path(self, dirname: Path, name: str) -> Path:
        """Return `target_dir / name`, disambiguated with a timestamp suffix if it already exists."""
        target = dirname / name
        if not target.exists():
            return target

        stem, suffix = Path(name).stem, Path(name).suffix
        return dirname / f"{stem}.{int(time.time())}{suffix}"

    def find_ready_files(self, now: float | None = None) -> tuple[list[Path], list[Path]]:
        """
        Split this job's candidate files into ready and not-yet-ready ones.

        Args:
            now: Reference time (epoch seconds); defaults to `time.time()`

        Returns:
            `(ready, not_ready)`, `ready` sorted oldest-modified first

        Raises:
            FileNotFoundError: If `source_dir` does not exist
        """
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"watch: source_dir '{self.source_dir}' does not exist")

        if now is None:
            now = time.time()

        excluded_dirs = {self.processed_dir.resolve(), self.failed_dir.resolve()}

        ready: list[tuple[float, Path]] = []
        not_ready: list[Path] = []
        for candidate in self.source_dir.glob(self.pattern):
            if not candidate.is_file() or candidate.resolve().parent != self.source_dir.resolve():
                continue
            if candidate.resolve().parent in excluded_dirs:
                continue

            mtime = candidate.stat().st_mtime
            if now - mtime >= self.quiet_period_seconds:
                ready.append((mtime, candidate))
            else:
                not_ready.append(candidate)

        ready.sort(key=lambda item: item[0])
        return [candidate for _, candidate in ready], not_ready

    def render_command(self, input_path: Path) -> list[str]:
        """
        Substitute placeholders into this job's command for a single ready file.

        Args:
            input_path: The ready file, available as ``{input}``, ``{stem}`` and ``{name}``;
                `target_dir` is available as ``{output_dir}``

        Returns:
            The argv list with every token's placeholders substituted
        """
        variables = {
            "input": str(input_path),
            "output_dir": str(self.target_dir),
            "stem": input_path.stem,
            "name": input_path.name,
        }

        tokens = [str(self.expand_envvars(token)) for token in self.command]
        return [token.format(**variables) for token in tokens]

    def run_command(self, csv_path: Path) -> subprocess.CompletedProcess:
        """
        Run this job's command on a single ready file.

        Output is logged line by line at DEBUG level as it runs - available for checking
        (e.g. via ``--log-level DEBUG`` or ``--log-file``) without spamming the console during
        a normal run; `WatchJob.run` shows a `tqdm` progress bar instead.

        Args:
            csv_path: The ready file to run it on

        Returns:
            The completed subprocess, with `stdout` holding its combined stdout/stderr; `stderr`
            is always `None` since both are interleaved into `stdout` for line-by-line logging

        Raises:
            RuntimeError: If the command exits with a non-zero status - the message includes
                the command, exit code, and captured output
        """
        argv = self.render_command(csv_path)
        # Force UTF-8 for the child's stdio: a piped stdout otherwise falls back to the
        # platform's legacy encoding (e.g. Windows' ANSI code page), which raises a
        # UnicodeEncodeError on non-ASCII output such as polars' box-drawing table borders.
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )

        lines: list[str] = []
        assert process.stdout is not None  # guaranteed by stdout=PIPE above
        for line in process.stdout:
            lines.append(line)
            logger.debug(f"[{self.name}] {line.rstrip()}")
        process.wait()

        output = "".join(lines)
        if process.returncode != 0:
            raise RuntimeError(
                f"command failed (exit {process.returncode}): {' '.join(argv)}\n{output}"
            )

        return subprocess.CompletedProcess(argv, process.returncode, output, None)

    def move_to_processed(self, csv_path: Path) -> Path:
        """
        Move a successfully handled file into `processed_dir`.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        target = self.unique_path(self.processed_dir, csv_path.name)
        return csv_path.rename(target)

    def move_to_failed(self, csv_path: Path, error: BaseException) -> Path:
        """
        Move a file that failed processing into `failed_dir`, alongside a `<file>.error.log`.
        """
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        target = self.unique_path(self.failed_dir, csv_path.name)
        moved = csv_path.rename(target)

        error_log = moved.with_name(moved.name + ".error.log")
        error_log.write_text(f"{type(error).__name__}: {error}\n\n{''.join(tc.format_exception(error))}")

        return moved

    def run(self, dry_run: bool = False, now: float | None = None) -> "WatchResult":
        """
        Run a single watch cycle for this job.

        Shows a `tqdm` progress bar naming this job and the file currently being processed;
        each command's own output is only logged at DEBUG level (see `run_command`), so it
        doesn't interleave with or spam the progress bar.

        Args:
            dry_run: If True, only report what would be processed - no command is run and
                no file is moved
            now: Reference time passed through to `find_ready_files`

        Returns:
            The outcome of this cycle. A missing `source_dir` is reported as
            `WatchResult.config_error` rather than raised, so a multi-job scan can continue.
        """
        try:
            ready, not_ready = self.find_ready_files(now=now)
        except FileNotFoundError as e:
            return WatchResult(config_error=str(e))

        if dry_run:
            return WatchResult(not_ready=not_ready, would_process=ready)

        result = WatchResult(not_ready=not_ready)
        with tqdm(ready, desc=self.name, unit="file") as progress:
            for csv_path in progress:
                progress.set_description_str(f"[{self.name}] {csv_path.name}")
                try:
                    self.run_command(csv_path)
                    result.processed.append(self.move_to_processed(csv_path))
                except Exception as e:
                    logger.exception(f"watch job '{self.name}': failed to process '{csv_path}'")
                    result.failed.append(self.move_to_failed(csv_path, e))

        return result

    @classmethod
    def run_jobs(cls, jobs: list[WatchJob], dry_run: bool = False) -> dict[str, WatchResult]:
        """
        Run a single watch cycle across all jobs.

        Args:
            jobs: The jobs to scan, e.g. from `WatchConfig.load`
            dry_run: Forwarded to `WatchJob.scan_once` for every job

        Returns:
            One `WatchResult` per job, keyed by job name. A job with a configuration problem
            (e.g. a missing source directory) does not prevent the other jobs from running.
        """
        return {job.name: job.run(dry_run=dry_run) for job in jobs}

    @classmethod
    def prompt(
        cls,
        input_fn: Callable[[str], str] | None = None,
        print_fn: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Interactively build one job's raw config dict - the shape `WatchConfig.load` reads
        from a config's 'jobs' list.

        Example:

        ```python
        job = WatchJob.prompt()
        ```

        Args:
            input_fn: Called with a prompt string, returns the raw answer; defaults to the builtin
                `input`, resolved at call time so it can be monkeypatched - overridable for testing
            print_fn: Used for validation messages; defaults to `print` - overridable for testing

        Returns:
            A job dict with 'name', 'source_dir', 'target_dir', 'pattern', 'quiet_period' and
            'command' (parsed with `shlex.split` from the single line the user typed)
        """
        input_fn = input_fn or input
        print_fn = print_fn or print

        def ask(prompt: str, default: str | None = None) -> str:
            suffix = f" [{default}]" if default else ""
            return input_fn(f"{prompt}{suffix}: ").strip() or (default or "")

        def ask_required(prompt: str) -> str:
            answer = ask(prompt)
            while not answer:
                print_fn(f"{prompt} is required.")
                answer = ask(prompt)
            return answer

        def ask_number(prompt: str, default: float) -> float:
            while True:
                raw = ask(prompt, str(default))
                try:
                    value = float(raw)
                    return int(value) if value.is_integer() else value
                except ValueError:
                    print_fn(f"'{raw}' is not a number.")

        source_dir = ask_required("Source directory to watch")
        name = ask("Job name", Path(source_dir).name)
        target_dir = ask("Target directory (available to the command as {output_dir})", source_dir)
        pattern = ask("File pattern", DAMAST_WATCH_DEFAULT_PATTERN)
        quiet_period = ask_number(
            "Quiet period in seconds before a file is considered complete", DAMAST_WATCH_DEFAULT_QUIET_PERIOD_IN_S
        )
        command_line = ask_required("Command to run on each ready file (use {input}/{output_dir}/{stem}/{name})")

        return {
            "name": name,
            "source_dir": source_dir,
            "target_dir": target_dir,
            "pattern": pattern,
            "quiet_period": quiet_period,
            "command": shlex.split(command_line),
        }


class WatchResult(BaseModel):
    """
    Outcome of a single watch cycle for one job.
    """
    processed: list[Path] = Field(default_factory=list)
    failed: list[Path] = Field(default_factory=list)
    not_ready: list[Path] = Field(default_factory=list)
    would_process: list[Path] = Field(default_factory=list)
    config_error: str | None = None

    @property
    def has_failures(self) -> bool:
        return bool(self.failed) or self.config_error is not None


class WatchConfig(BaseModel):
    """
    A watch config: the `WatchJob` s declared in a watch config YAML file.
    """
    jobs: list[WatchJob]

    @model_validator(mode="after")
    def _check_jobs(self) -> "WatchConfig":
        if not self.jobs:
            raise ValueError("requires a non-empty 'jobs' list")

        seen_names: set[str] = set()
        for job in self.jobs:
            if job.name in seen_names:
                raise ValueError(f"duplicate job name '{job.name}'")
            seen_names.add(job.name)

        return self

    @classmethod
    def _read_yaml_jobs(cls, path: Path) -> list[dict]:
        """Read the raw 'jobs' list from a watch config file, or [] if it doesn't exist yet."""
        path = Path(path)
        if not path.exists():
            return []

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return list(data.get("jobs") or [])

    @classmethod
    def load(cls, path: Path) -> "WatchConfig":
        """
        Load a watch config file, with defaults applied to each job.

        Args:
            path: The YAML watch config file

        Returns:
            The config, with its jobs in the order they are declared

        Raises:
            FileNotFoundError: If `path` does not exist
            ValueError: If the config has no (or an empty) 'jobs' list, a job is missing
                'source_dir' or 'command', or two jobs share the same name
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"watch config '{path}' does not exist")

        try:
            return cls(jobs=cls._read_yaml_jobs(path))
        except ValidationError as e:
            raise ValueError(f"watch config '{path}': {e}") from e

    @classmethod
    def create_interactively(
        cls,
        path: Path,
        input_fn: Callable[[str], str] | None = None,
        print_fn: Callable[[str], None] | None = None,
    ) -> Path | None:
        """
        Interactively build one or more jobs and write them to a watch config file.

        If `path` already exists, the caller is asked whether to append the new job(s) to its
        existing ones or overwrite it.

        Args:
            path: The watch config file to write
            input_fn: Called with a prompt string, returns the raw answer; defaults to the builtin
                `input`, resolved at call time so it can be monkeypatched - overridable for testing
            print_fn: Used for status messages; defaults to `print` - overridable for testing

        Returns:
            `path` once written, or `None` if the caller cancelled

        Raises:
            ValueError: If the resulting jobs would be invalid, e.g. a name collides with an
                existing job when appending
        """
        input_fn = input_fn or input
        print_fn = print_fn or print

        path = Path(path)
        jobs: list[dict] = []

        if path.exists():
            prompt = f"'{path}' already exists. [a]ppend, [o]verwrite, [c]ancel? [a]: "
            choice = input_fn(prompt).strip().lower() or "a"
            if choice.startswith("c"):
                print_fn("Cancelled.")
                return None
            if choice.startswith("a"):
                jobs = cls._read_yaml_jobs(path)

        while True:
            print_fn(f"--- job {len(jobs) + 1} ---")
            jobs.append(WatchJob.prompt(input_fn, print_fn))

            again = input_fn("Add another job? [y/N]: ").strip().lower()
            if not again.startswith("y"):
                break

        cls(jobs=jobs)  # validate before writing, e.g. catches a duplicate job name

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump({"jobs": jobs}, f, sort_keys=False)

        print_fn(f"Wrote {len(jobs)} job(s) to '{path}'")
        return path

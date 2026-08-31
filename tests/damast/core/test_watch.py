import os
import sys
import time
from pathlib import Path

import pytest
import yaml

from damast.core.watch import (
    WatchConfig,
    WatchJob,
)


def _write_file(path, mtime_offset_seconds: float = 0, content: str = "data"):
    """Write a file and set its mtime to now + mtime_offset_seconds (negative = older)."""
    path.write_text(content)
    now = time.time()
    os.utime(path, (now + mtime_offset_seconds, now + mtime_offset_seconds))
    return path


COPY_COMMAND = [sys.executable, "-c",
                "import shutil, sys; shutil.copy(sys.argv[1], sys.argv[2])",
                "{input}", "{output_dir}/{stem}.copy"]

FAIL_COMMAND = [sys.executable, "-c", "import sys; sys.exit(1)"]


# --- WatchConfig.load -------------------------------------------------------------------

def test_load_watch_config_applies_defaults(tmp_path):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{"source_dir": str(source_dir), "command": ["true"]}],
    }))

    jobs = WatchConfig.load(config_path).jobs

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "incoming"
    assert job.target_dir == source_dir
    assert job.processed_dir == source_dir / "processed"
    assert job.failed_dir == source_dir / "failed"
    assert job.pattern == "*.csv"
    assert job.quiet_period_seconds == 1800


def test_load_watch_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        WatchConfig.load(tmp_path / "no-such-config.yaml")


def test_load_watch_config_requires_jobs(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({"jobs": []}))

    with pytest.raises(ValueError, match="non-empty 'jobs' list"):
        WatchConfig.load(config_path)


def test_load_watch_config_requires_source_dir(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({"jobs": [{"command": ["true"]}]}))

    with pytest.raises(ValueError, match="source_dir"):
        WatchConfig.load(config_path)


def test_load_watch_config_requires_command(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({"jobs": [{"source_dir": str(tmp_path)}]}))

    with pytest.raises(ValueError, match="command"):
        WatchConfig.load(config_path)


def test_load_watch_config_rejects_duplicate_names(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [
            {"name": "a", "source_dir": str(tmp_path), "command": ["true"]},
            {"name": "a", "source_dir": str(tmp_path), "command": ["true"]},
        ],
    }))

    with pytest.raises(ValueError, match="duplicate job name"):
        WatchConfig.load(config_path)


# --- WatchJob.find_ready_files ------------------------------------------------------------

def test_find_ready_files_respects_quiet_period(tmp_path):
    fresh = _write_file(tmp_path / "fresh.csv", mtime_offset_seconds=0)
    old = _write_file(tmp_path / "old.csv", mtime_offset_seconds=-3601)
    job = WatchJob(source_dir=tmp_path, command=["true"], quiet_period_seconds=3600)

    ready, not_ready = job.find_ready_files()

    assert ready == [old]
    assert not_ready == [fresh]


def test_find_ready_files_boundary_is_ready(tmp_path):
    now = time.time()
    boundary = tmp_path / "boundary.csv"
    boundary.write_text("data")
    os.utime(boundary, (now - 3600, now - 3600))
    job = WatchJob(source_dir=tmp_path, command=["true"], quiet_period_seconds=3600)

    ready, not_ready = job.find_ready_files(now=now)

    assert ready == [boundary]
    assert not_ready == []


def test_find_ready_files_excludes_processed_and_failed_subdirs(tmp_path):
    processed_dir = tmp_path / "processed"
    failed_dir = tmp_path / "failed"
    processed_dir.mkdir()
    failed_dir.mkdir()

    _write_file(processed_dir / "done.csv", mtime_offset_seconds=-3601)
    _write_file(failed_dir / "broken.csv", mtime_offset_seconds=-3601)
    old = _write_file(tmp_path / "old.csv", mtime_offset_seconds=-3601)

    job = WatchJob(source_dir=tmp_path, command=["true"], quiet_period_seconds=3600,
                   processed_dir=processed_dir, failed_dir=failed_dir)

    ready, not_ready = job.find_ready_files()

    assert ready == [old]
    assert not_ready == []


def test_find_ready_files_missing_directory_raises(tmp_path):
    job = WatchJob(source_dir=tmp_path / "does-not-exist", command=["true"])

    with pytest.raises(FileNotFoundError):
        job.find_ready_files()


def test_find_ready_files_orders_oldest_first(tmp_path):
    newer = _write_file(tmp_path / "newer.csv", mtime_offset_seconds=-3601)
    older = _write_file(tmp_path / "older.csv", mtime_offset_seconds=-7200)
    job = WatchJob(source_dir=tmp_path, command=["true"], quiet_period_seconds=3600)

    ready, _ = job.find_ready_files()

    assert ready == [older, newer]


# --- WatchJob.render_command --------------------------------------------------------------

def test_render_command_substitutes_placeholders(tmp_path):
    input_path = tmp_path / "2026-08-30.csv"
    output_dir = tmp_path / "out"
    job = WatchJob(
        source_dir=tmp_path,
        command=["damast", "convert", "-f", "{input}", "-o", "{output_dir}/{stem}.parquet", "--name", "{name}"],
        target_dir=output_dir,
    )

    argv = job.render_command(input_path)

    assert argv == [
        "damast", "convert", "-f", str(input_path),
        "-o", f"{output_dir}/2026-08-30.parquet",
        "--name", "2026-08-30.csv",
    ]


def test_render_command_passes_through_tokens_without_placeholders(tmp_path):
    job = WatchJob(source_dir=tmp_path, command=["damast", "--describe"])

    argv = job.render_command(Path("x.csv"))

    assert argv == ["damast", "--describe"]


# --- WatchJob.run_jobs ------------------------------------------------------

@pytest.fixture
def job_dirs(tmp_path):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    return source_dir, target_dir


def test_run_job_success_moves_to_processed(job_dirs):
    source_dir, target_dir = job_dirs
    ready_file = _write_file(source_dir / "2026-08-30.csv", mtime_offset_seconds=-3601)

    job = WatchJob(name="j", source_dir=source_dir, command=list(COPY_COMMAND), target_dir=target_dir,
                   quiet_period_seconds=3600, processed_dir=source_dir / "processed",
                   failed_dir=source_dir / "failed")

    result = job.run()

    assert result.config_error is None
    assert (target_dir / "2026-08-30.copy").exists()
    assert result.processed == [source_dir / "processed" / "2026-08-30.csv"]
    assert (source_dir / "processed" / "2026-08-30.csv").exists()
    assert not ready_file.exists()


def test_run_job_failure_moves_to_failed_with_error_log(job_dirs):
    source_dir, target_dir = job_dirs
    _write_file(source_dir / "broken.csv", mtime_offset_seconds=-3601)

    job = WatchJob(name="j", source_dir=source_dir, command=list(FAIL_COMMAND), target_dir=target_dir,
                   quiet_period_seconds=3600, processed_dir=source_dir / "processed",
                   failed_dir=source_dir / "failed")

    result = job.run()

    assert result.processed == []
    assert result.failed == [source_dir / "failed" / "broken.csv"]
    error_log = source_dir / "failed" / "broken.csv.error.log"
    assert error_log.exists()
    assert "exit 1" in error_log.read_text()


def test_run_job_continues_after_one_failure(job_dirs):
    source_dir, target_dir = job_dirs
    _write_file(source_dir / "a-broken.csv", mtime_offset_seconds=-3601)
    _write_file(source_dir / "b-good.csv", mtime_offset_seconds=-3601)

    job = WatchJob(name="j", source_dir=source_dir,
                   command=[sys.executable, "-c",
                            "import sys; sys.exit(1 if 'a-broken' in sys.argv[1] else 0)", "{input}"],
                   target_dir=target_dir, quiet_period_seconds=3600,
                   processed_dir=source_dir / "processed", failed_dir=source_dir / "failed")

    result = job.run()

    assert len(result.failed) == 1
    assert len(result.processed) == 1
    assert "a-broken" in result.failed[0].name
    assert "b-good" in result.processed[0].name


def test_run_job_dry_run_touches_nothing(job_dirs):
    source_dir, target_dir = job_dirs
    ready_file = _write_file(source_dir / "2026-08-30.csv", mtime_offset_seconds=-3601)

    job = WatchJob(name="j", source_dir=source_dir, command=list(COPY_COMMAND), target_dir=target_dir,
                   quiet_period_seconds=3600, processed_dir=source_dir / "processed",
                   failed_dir=source_dir / "failed")

    result = job.run(dry_run=True)

    assert result.would_process == [ready_file]
    assert result.processed == []
    assert ready_file.exists()
    assert not (target_dir / "2026-08-30.copy").exists()


def test_run_job_missing_source_dir_sets_config_error(tmp_path):
    job = WatchJob(name="j", source_dir=tmp_path / "missing", command=["true"], target_dir=tmp_path,
                   processed_dir=tmp_path / "processed", failed_dir=tmp_path / "failed")

    result = job.run()

    assert result.config_error is not None
    assert result.has_failures


def test_run_all_jobs_even_if_one_has_a_config_error(job_dirs, tmp_path):
    source_dir, target_dir = job_dirs
    _write_file(source_dir / "2026-08-30.csv", mtime_offset_seconds=-3601)

    good_job = WatchJob(name="good", source_dir=source_dir, command=list(COPY_COMMAND), target_dir=target_dir,
                        quiet_period_seconds=3600, processed_dir=source_dir / "processed",
                        failed_dir=source_dir / "failed")
    broken_job = WatchJob(name="broken", source_dir=tmp_path / "missing", command=["true"], target_dir=tmp_path,
                          processed_dir=tmp_path / "processed", failed_dir=tmp_path / "failed")

    results = WatchJob.run_jobs([broken_job, good_job])

    assert results["broken"].config_error is not None
    assert results["good"].processed


# --- WatchJob.move_to_processed -------------------------------------------------------------

def test_move_to_processed_avoids_name_collision(tmp_path):
    processed_dir = tmp_path / "processed"
    job = WatchJob(source_dir=tmp_path, command=["true"], processed_dir=processed_dir)

    first = _write_file(tmp_path / "same-name.csv")
    moved_first = job.move_to_processed(first)

    second = _write_file(tmp_path / "same-name.csv")
    moved_second = job.move_to_processed(second)

    assert moved_first.exists()
    assert moved_second.exists()
    assert moved_first != moved_second


# --- WatchJob.prompt / WatchConfig.create_interactively -----------------------------------

def _fake_input(answers):
    """Return an input_fn that yields each of `answers` in turn, ignoring the prompt text."""
    remaining = list(answers)

    def input_fn(prompt):
        return remaining.pop(0)

    return input_fn


def test_prompt_job_config_uses_defaults_on_blank_answers(tmp_path):
    source_dir = str(tmp_path / "incoming")
    answers = _fake_input([
        source_dir,   # source dir (required)
        "",           # name -> default (basename)
        "",           # target dir -> default (source dir)
        "",           # pattern -> default
        "",           # quiet period -> default
        "damast convert -f {input} -o {output_dir}",  # command
    ])

    job = WatchJob.prompt(input_fn=answers, print_fn=lambda _: None)

    assert job == {
        "name": Path(source_dir).name,
        "source_dir": source_dir,
        "target_dir": source_dir,
        "pattern": "*.csv",
        "quiet_period": 1800,
        "command": ["damast", "convert", "-f", "{input}", "-o", "{output_dir}"],
    }


def test_prompt_job_config_reprompts_on_missing_required_answer():
    answers = _fake_input([
        "",                    # source dir - blank, must reprompt
        "/data/incoming",      # source dir - now given
        "ais",                 # name
        "/data/out",           # target dir
        "*.csv",               # pattern
        "3600",                # quiet period
        "",                    # command - blank, must reprompt
        "damast process --pipeline p.damast.ppl --input-data {input}",
    ])
    messages = []

    job = WatchJob.prompt(input_fn=answers, print_fn=messages.append)

    assert job["source_dir"] == "/data/incoming"
    assert job["quiet_period"] == 3600
    assert job["command"][0] == "damast"
    assert any("required" in m for m in messages)


def test_prompt_job_config_reprompts_on_invalid_number():
    answers = _fake_input([
        "/data/incoming",
        "ais",
        "/data/out",
        "*.csv",
        "not-a-number",
        "1800",
        "damast convert -f {input}",
    ])
    messages = []

    job = WatchJob.prompt(input_fn=answers, print_fn=messages.append)

    assert job["quiet_period"] == 1800
    assert any("not a number" in m for m in messages)


def test_create_watch_config_interactively_writes_one_job(tmp_path):
    config_path = tmp_path / "watch.yaml"
    source_dir = str(tmp_path / "incoming")
    answers = _fake_input([
        source_dir, "", "", "", "",
        "damast convert -f {input} -o {output_dir}",
        "n",  # no more jobs
    ])

    written = WatchConfig.create_interactively(config_path, input_fn=answers, print_fn=lambda _: None)

    assert written == config_path
    jobs = WatchConfig.load(config_path).jobs
    assert len(jobs) == 1
    assert jobs[0].source_dir == Path(source_dir)


def test_create_watch_config_interactively_writes_multiple_jobs(tmp_path):
    config_path = tmp_path / "watch.yaml"
    answers = _fake_input([
        str(tmp_path / "a"), "", "", "", "", "damast convert -f {input}",
        "y",
        str(tmp_path / "b"), "", "", "", "", "damast convert -f {input}",
        "n",
    ])

    WatchConfig.create_interactively(config_path, input_fn=answers, print_fn=lambda _: None)

    jobs = WatchConfig.load(config_path).jobs
    assert {j.name for j in jobs} == {"a", "b"}


def test_create_watch_config_interactively_cancel_leaves_existing_config_untouched(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text("jobs:\n  - name: existing\n    source_dir: /x\n    command: [true]\n")
    original = config_path.read_text()

    result = WatchConfig.create_interactively(config_path, input_fn=_fake_input(["c"]), print_fn=lambda _: None)

    assert result is None
    assert config_path.read_text() == original


def test_create_watch_config_interactively_appends_to_existing_config(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{"name": "existing", "source_dir": str(tmp_path / "x"), "command": ["true"]}],
    }))

    answers = _fake_input([
        "a",  # append
        str(tmp_path / "new"), "", "", "", "",
        "damast convert -f {input}",
        "n",
    ])

    WatchConfig.create_interactively(config_path, input_fn=answers, print_fn=lambda _: None)

    jobs = WatchConfig.load(config_path).jobs
    assert {j.name for j in jobs} == {"existing", "new"}


def test_create_watch_config_interactively_overwrites_existing_config(tmp_path):
    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{"name": "existing", "source_dir": str(tmp_path / "x"), "command": ["true"]}],
    }))

    answers = _fake_input([
        "o",  # overwrite
        str(tmp_path / "new"), "", "", "", "",
        "damast convert -f {input}",
        "n",
    ])

    WatchConfig.create_interactively(config_path, input_fn=answers, print_fn=lambda _: None)

    jobs = WatchConfig.load(config_path).jobs
    assert {j.name for j in jobs} == {"new"}

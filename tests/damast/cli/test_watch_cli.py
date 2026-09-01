import builtins
import os
import sys
import time
from pathlib import Path

import yaml

from damast.core.watch import WatchConfig


def _write_old_file(path, content: str = "data", age_seconds: float = 10):
    path.write_text(content)
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


def test_watch_convert_end_to_end(data_path, tmp_path, script_runner):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    target_dir = tmp_path / "out"
    target_dir.mkdir()

    csv_file = source_dir / "test_ais.csv"
    csv_file.write_bytes((data_path / "test_ais.csv").read_bytes())
    old = time.time() - 10
    os.utime(csv_file, (old, old))

    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{
            "name": "ais",
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "quiet_period": 1,
            "command": ["damast", "convert", "-f", "{input}", "--output-dir", "{output_dir}"],
        }],
    }))

    result = script_runner.run(["damast", "watch", "--config", str(config_path)])

    assert result.returncode == 0, result.stdout
    assert (target_dir / "test_ais.parquet").exists()
    assert (source_dir / "processed" / "test_ais.csv").exists()
    assert not csv_file.exists()


def test_watch_dry_run_leaves_files_in_place(data_path, tmp_path, script_runner):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    csv_file = _write_old_file(source_dir / "test_ais.csv", (data_path / "test_ais.csv").read_text())

    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{
            "name": "ais",
            "source_dir": str(source_dir),
            "quiet_period": 1,
            "command": ["damast", "convert", "-f", "{input}", "--output-dir", "{output_dir}"],
        }],
    }))

    result = script_runner.run(["damast", "watch", "--config", str(config_path), "--dry-run"])

    assert result.returncode == 0, result.stdout
    assert csv_file.exists()
    assert "test_ais.csv" in result.stdout
    assert not (source_dir / "test_ais.parquet").exists()


def test_watch_job_filter_rejects_unknown_name(tmp_path, script_runner):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()

    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{"name": "ais", "source_dir": str(source_dir), "command": [sys.executable, "-c", "pass"]}],
    }))

    result = script_runner.run(["damast", "watch", "--config", str(config_path), "--job", "no-such-job"])

    assert result.returncode != 0
    assert "unknown job" in result.stdout


def test_watch_failing_command_yields_nonzero_exit_code(tmp_path, script_runner):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    _write_old_file(source_dir / "broken.csv")

    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{
            "name": "broken",
            "source_dir": str(source_dir),
            "quiet_period": 1,
            "command": [sys.executable, "-c", "import sys; sys.exit(1)"],
        }],
    }))

    result = script_runner.run(["damast", "watch", "--config", str(config_path)])

    assert result.returncode != 0
    assert (source_dir / "failed" / "broken.csv").exists()


def test_watch_create_config_end_to_end(tmp_path, script_runner, monkeypatch):
    config_path = tmp_path / "watch.yaml"
    source_dir = tmp_path / "incoming"

    answers = iter([
        str(source_dir),  # source dir
        "",                # name -> default
        "",                # target dir -> default
        "",                # pattern -> default
        "",                # quiet period -> default
        "damast convert -f {input} -o {output_dir}",  # command
        "n",               # no more jobs
    ])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    result = script_runner.run(["damast", "watch", "--config", str(config_path), "--create-config"])

    assert result.returncode == 0, result.stdout
    assert config_path.exists()

    jobs = WatchConfig.load(config_path).jobs
    assert len(jobs) == 1
    assert jobs[0].source_dir == Path(source_dir)
    assert jobs[0].command == ["damast", "convert", "-f", "{input}", "-o", "{output_dir}"]


def test_watch_no_ready_files_returns_zero(tmp_path, script_runner):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    # fresh file, well within the quiet period
    (source_dir / "still-writing.csv").write_text("data")

    config_path = tmp_path / "watch.yaml"
    config_path.write_text(yaml.dump({
        "jobs": [{
            "name": "ais",
            "source_dir": str(source_dir),
            "quiet_period": 3600,
            "command": [sys.executable, "-c", "pass"],
        }],
    }))

    result = script_runner.run(["damast", "watch", "--config", str(config_path)])

    assert result.returncode == 0, result.stdout
    assert (source_dir / "still-writing.csv").exists()

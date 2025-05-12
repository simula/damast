import subprocess
from functools import partial
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import polars
import psutil
import pytest
import sys

from damast.ml.scheduler import Job, JobScheduler


@pytest.fixture()
def job(tmp_path):
    # Generate a simple sequence dataset, that can be loaded also by the triggered worker
    column_names = ["a", "b", "c", "d"]
    data = []
    for i in range(100):
        row = [i for _ in column_names]
        data.append(row)

    pandas_df = pd.DataFrame(data, columns=column_names)
    df = polars.from_pandas(pandas_df)
    data_filename = tmp_path / "test_sequence.parquet"
    df.write_parquet(data_filename)

    job_dict = {
        "id": -1,
        "experiment_dir": "experiment_dir",
        "model_name": "custom_model",
        "features": ["a", "b"],
        "target": ["c", "d"],
        "sequence_length": 5,
        "data_filename": str(data_filename)
    }
    return Job(**job_dict)


@pytest.fixture()
def worker_process():
    # Ensure that no previous worker job is running
    for proc in psutil.process_iter():
        if "run_test_worker" in proc.name():
            proc.kill()

    run_test_worker = Path(__file__).parent / "run_test_worker.py"
    assert run_test_worker.exists()

    worker_process = subprocess.Popen(["python", str(run_test_worker)])
    yield worker_process
    worker_process.terminate()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Scheduler requires a UNIX system")
def test_scheduler_worker_fail(job):
    # Ensure that no previous worker job is running
    for proc in psutil.process_iter():
        if "run_test_worker" in proc.name():
            proc.kill()

    job_scheduler = JobScheduler()
    with pytest.raises(RuntimeError):
        job_scheduler.start(job)

@pytest.mark.skipif(sys.platform.startswith("win"), reason="Scheduler requires a UNIX system")
def test_scheduler_worker(monkeypatch, job, worker_process):
    job_scheduler = JobScheduler()
    for _ in range(60):
        try:
            job_scheduler.start(job)
            break
        except Exception as e:
            pass
        finally:
            sleep(1)
    assert job.id != -1, "Job has been successfully started"

    responses = Job.wait_for_responses(status_collector=partial(job_scheduler.get_status, job.id),
                                       predicate_responses=lambda x: len(x) > 1)
    response = responses[0]
    assert response.id == job.id
    assert response.predicted_sequence == [0, 0]

    job_scheduler.stop(job.id)

    Job.wait_for_status(status_collector=partial(job_scheduler.get_status, job.id),
                        match_status=Job.Status.STOPPED)

from pathlib import Path
from time import sleep
from functools import partial

import numpy as np
import pandas as pd
import pytest
import vaex
import subprocess

from damast.ml.scheduler import JobScheduler, Job


def test_scheduler_worker(monkeypatch, tmp_path):
    # Generate a simple sequence dataset, that can be loaded also by the triggered worker
    column_names = ["a", "b", "c", "d"]
    data = []
    for i in range(0, 100):
        row = []
        for col in column_names:
            row.append(i)
        data.append(row)

    pandas_df = pd.DataFrame(data, columns=column_names)
    df = vaex.from_pandas(pandas_df)
    data_filename = tmp_path / "test_sequence.hdf5"
    df.export(data_filename)

    run_test_worker = Path(__file__).parent / "run_test_worker.py"
    assert run_test_worker.exists()

    job_dict = {
        "id": -1,
        "experiment_dir": "experiment_dir",
        "model_name": "custom_model",
        "features": ["a", "b"],
        "target": ["c", "d"],
        "sequence_length": 5,
        "data_filename": str(data_filename)
    }
    job = Job(**job_dict)
    job_scheduler = JobScheduler()
    with pytest.raises(RuntimeError):
        job_scheduler.start(job)

    worker_process = subprocess.Popen(["python", str(run_test_worker)])
    for i in range(0, 10):
        try:
            job_scheduler.start(job)
            break
        except Exception as e:
            assert True, f"Waiting for worker: {e}"
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

    worker_process.terminate()

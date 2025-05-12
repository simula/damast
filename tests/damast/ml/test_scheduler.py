import select
import socket
from time import sleep
from typing import Iterable
import pytest
import sys

from damast.ml.scheduler import Job, ResponseCollector


def test_job():
    job_dict = {
        "id": 1,
        "experiment_dir": "experiment_dir",
        "model_name": "model_name",
        "features": ["a", "b"],
        "target": ["c", "d"],
        "sequence_length": 50,
        "data_filename": "data_filename"
    }

    job = Job(**job_dict)

    for key, value in job_dict.items():
        assert getattr(job, key) == value

    encoded_job = job.encode()

    decoded_job = Job.decode(encoded_job)
    assert job == decoded_job


def test_job_response():
    job_response_dict = {
        "id": 1,
        "timepoint": 2,
        "loss": 1.0,
        "actual_sequence": [0, 1, 2],
        "predicted_sequence": [3]
    }
    job_response = Job.Response(**job_response_dict)

    for key, value in job_response_dict.items():
        assert getattr(job_response, key) == value

    encoded_job_response = job_response.encode()
    decoded_job_response = Job.Response.decode(encoded_job_response)

    assert job_response == decoded_job_response

@pytest.mark.skipif(sys.platform.startswith("win"), reason="Scheduler requires a UNIX system")
def test_response_collector(tmp_path, monkeypatch):
    class MockSelect:
        call_count = 0

        def reset_count(self):
            self.call_count = 0

        def select_ready(self, a: Iterable, b: Iterable, c: Iterable, timeout: float):
            if self.call_count == 0:
                self.call_count += 1
                return [True]
            else:
                return [False]

    class MockSock:
        do_exit = False

        def exit(self):
            self.do_exit = True

        def recvmsg(self, length: int):
            if self.do_exit:
                return ["BYE".encode()]
            if length == 4:
                return [msg_size.to_bytes(4, byteorder="little")]
            else:
                return [msg]

    mock_select = MockSelect()
    monkeypatch.setattr(select, "select", mock_select.select_ready)
    soc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    rc = ResponseCollector(job_id=1, sock=soc)
    collected_responses, status = rc.get_status()

    assert status == Job.Status.NOT_STARTED
    assert collected_responses == []

    job_response_dict = {
        "id": 1,
        "timepoint": 2,
        "loss": 1.0,
        "actual_sequence": [0, 1, 2],
        "predicted_sequence": [3]
    }
    job_response = Job.Response(**job_response_dict)
    msg = job_response.encode()
    msg_size = len(msg)

    mock_sock = MockSock()
    monkeypatch.setattr(socket.socket, "recvmsg", mock_sock.recvmsg)

    rc.start()
    collected_responses = Job.wait_for_status(status_collector=rc.get_status,
                                              match_status=Job.Status.RUNNING,
                                              timeout_in_s=10)

    assert len(collected_responses) == 1
    assert collected_responses[0] == job_response

    mock_sock.exit()
    mock_select.reset_count()

    collected_responses = Job.wait_for_status(status_collector=rc.get_status,
                                              match_status=Job.Status.STOPPED,
                                              timeout_in_s=10)
    assert len(collected_responses) == 1

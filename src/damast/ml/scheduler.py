from __future__ import annotations

import json
import select
import socket
from enum import Enum
from pathlib import Path
from threading import Thread, Lock
from typing import List, Any, Dict, Callable, Tuple

PREDICT_FILE_SOCKET = "/tmp/.damast-ais-showcase"


class Job:
    """
    Representation for a Prediction/Classification job
    """
    id: int

    #: The experiment directory which will be used to look for available models
    experiment_dir: Path
    #: The model that shall be loaded
    model_name: str
    #: The (input) features
    features: List[str]
    #: The target features that will be predicted
    target: List[str]
    #: The length of the sequence
    sequence_length: int
    #: The path to the vaex-readable file containing the sequences to run the prediction on
    data_filename: Path

    class Response:
        """
        Response to a job, when it has finished
        :param id: Identifier for the job that this response belong to
        :param timepoint: Related timepoint or index value for this sequence
        :param loss: The computed loss when comparing prediction with actual input
        :param actual_sequence: the actual sequence that shall be used for prediction
        :param predicted_sequence: the predicted features / sequence
        """
        #: id of the job that triggered this response
        id: int
        #: Timepoint / Index in the sequence
        timepoint: int
        #: The computed loss for this prediction
        loss: float
        #: The actual input sequence
        actual_sequence: List[Any]
        #: The predicted sequence or features
        predicted_sequence: List[Any]

        def __init__(self,
                     id: int,
                     timepoint: int,
                     loss: float,
                     actual_sequence: List[Any],
                     predicted_sequence: List[Any]):
            self.id = id
            self.timepoint = timepoint
            self.loss = loss
            self.actual_sequence = actual_sequence
            self.predicted_sequence = predicted_sequence

        def encode(self) -> str:
            """
            Encode this object into a transferable string
            :return: encoded string
            """
            return json.dumps(self.__dict__).encode()

        @classmethod
        def decode(cls, bytes: str) -> Job.Response:
            """
            Decode an encoded string / bytes into a :class:`Job.Response`
            :param bytes:
            :return: Instance of :class:`Job.Response`
            """
            values = json.loads(bytes.decode())
            return cls(**values)

    def __init__(self,
                 id: int,
                 experiment_dir: str,
                 model_name: str,
                 features: List[str],
                 target: List[str],
                 sequence_length: int,
                 data_filename: str):
        self.id = id
        self.experiment_dir = experiment_dir
        self.model_name = model_name
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.data_filename = data_filename

    def encode(self):
        return json.dumps(self.__dict__).encode()

    @classmethod
    def decode(cls, bytes):
        values = json.loads(bytes.decode())
        return cls(**values)


class ControlCommand(str, Enum):
    BYE = "BYE"
    STOP = "STOP"


class ResponseCollector:
    job_id: int

    sock: socket
    thread: Thread

    responses: List[Job.Response]
    lock: Lock
    status: Status

    class Status(str, Enum):
        NOT_STARTED = "NOT STARTED"
        RUNNING = "RUNNING"
        FINISHED = "FINISHED"
        ABORTED = "ABORTED"

    def __init__(self,
                 job_id: int,
                 sock: socket):
        self.job_id = job_id
        self.sock = sock
        self.thread = None

        self.status = self.Status.NOT_STARTED
        self.responses = []

    def get_status(self) -> Tuple[List[Job.Response], Status]:
        return self.responses, self.status

    def start(self):
        """
        Start the collection of results.
        :return:
        """
        self.thread = Thread(target=self.read_responses)
        self.thread.start()

    def read_responses(self):
        """
        Read all the responses from a given connection until the control command BYE is being received.
        """
        if self.sock is None:
            raise RuntimeError(f"{self.__class__.__name__}.read_responses: (job: {self.job_id}): no socket available")

        stop = False
        while not stop:
            self.status = self.Status.RUNNING
            try:
                ready = select.select([self.sock], [], [], 0.01)
                if ready[0]:
                    # Get the length field first,
                    # then extract the data
                    # exit when encountering 'BYE'
                    msg = self.sock.recvmsg(4)
                    msg_size = int.from_bytes(bytes=msg[0], byteorder="little")
                    received_bytes = self.sock.recvmsg(msg_size)[0]
                    if len(received_bytes) == 0:
                        raise ConnectionAbortedError(f"{self.__class__.__name__}.read_responses:"
                                                         f" connection was reset")

                    self.responses.append(Job.Response.decode(bytes=received_bytes))
            except Exception:
                self.status = self.Status.ABORTED
                raise

        self.status = self.Status.FINISHED


class JobScheduler:
    #: Main count of the current job id
    _job_id: int

    # region Job Mapping
    #: Mapping of job id to socket
    _sockets: Dict[int, socket]
    #: Mapping of job id to job
    _jobs: Dict[int, Job]
    #: Mapping of job id to the collecting threads
    _collectors: Dict[int, ResponseCollector]

    # endregion

    def __init__(self):

        self._job_id = 0
        self._sockets: Dict[int, socket] = {}

        self._jobs = {}
        self._collectors = {}

    def stop(self, job_id: int):
        """
        Send the STOP request to the Worker to stop the job related thread

        :param job_id: Job which shall be stopped
        """
        if job_id in self._sockets:
            sock = self._sockets[job_id]
            try:
                sock.sendmsg([ControlCommand.STOP.value.encode()])
            except BrokenPipeError:
                pass

    def start(self, job: Job) -> Job:
        """
        Trigger the execution of a job.

        This will send the job description to a worker which will start a job
        related thread.

        :param job: Job description.
        """
        soc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            soc.connect(PREDICT_FILE_SOCKET)
        except ConnectionRefusedError:
            raise RuntimeError(f"{self.__class__.__name__}.start: no worker available")
        soc.setblocking(0)

        job.id = self._job_id
        self._job_id += 1

        self._sockets[job.id] = soc
        self._jobs[job.id] = job

        soc.sendmsg([job.encode()])

        # start collection thread at the same time
        self._collect(job.id)

    def get_status(self, job_id) -> Tuple[List[Job.Response], ResponseCollector.Status]:
        """
        Get the current status of the job.

        :param job_id: id of the job for which the status shall be retrieved
        :return: List of response and status description of the collector
        """
        if job_id not in self._collectors:
            return [], ResponseCollector.Status.NOT_STARTED

        collector = self._collectors[job_id]
        return collector.get_status()

    def _collect(self, job_id: int):
        """
        Start the collection of the responses for a job

        :param job_id: Job id
        """
        collector = ResponseCollector(job_id, self._sockets[job_id])
        self._collectors[job_id] = collector
        collector.start()

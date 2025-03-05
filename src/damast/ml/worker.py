"""
Module containing the worker functionality to perform a learning task
"""
import datetime as dt
import select
import socket
from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path
from threading import Event, Thread
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from damast.data_handling.accessors import SequenceIterator
from damast.ml.experiments import Experiment
from damast.ml.scheduler import PREDICT_FILE_SOCKET, ControlCommand, Job

_log: Logger = getLogger(__name__)


class Worker:
    threads: Dict[int, Thread]
    stop_event: Event

    def __init__(self):
        self.threads = {}
        self.stop_event = Event()
        self.stop_event.clear()

    def predict(self,
                job_id: int,
                model_name: str,
                experiment_dir: Path,
                df: pd.DataFrame,
                features: List[str],
                target: List[str],
                sequence_length: int,
                connection: socket.socket,
                stop_event: Event,
                update_callback: Optional[Callable] = None
                ):
        models = Experiment.from_directory(experiment_dir)
        current_model = models[model_name]

        sta = SequenceIterator(df=df)

        # Generate a sequence of length + the intended forecasted values
        gen_predict = sta.to_keras_generator(features=features,
                                             target=target,
                                             sequence_length=sequence_length)

        loss = []
        timepoints = []
        timepoint = 0

        result = []

        try:
            _log.info(f"{self.__class__.__name__}({job_id}): run prediction")
            while True:
                if stop_event.is_set():
                    _log.info(f"{self.__class__.__name__}({job_id}): external stop request received")
                    break

                # Allow to stop the worker from outside
                ready = select.select([connection], [], [], 0.01)
                if ready[0]:
                    msg = connection.recvmsg(4)
                    # Get the length field
                    data = msg[0].decode()
                    if type(data) == str and data == ControlCommand.STOP.value:
                        break

                data = next(gen_predict)
                if data is None:
                    break

                X, y = data
                # Since the model requires 'batches' to be fed, provide one batch, which one step
                input_data = X[np.newaxis, :, :]
                predicted_sequence = current_model.predict(input_data,
                                                           steps=1,
                                                           verbose=0)

                actual_sequence = y
                computed_loss = float(current_model.loss(actual_sequence, predicted_sequence[0])[0])

                if update_callback is not None:
                    response = Job.Response(
                        id=job_id,
                        timepoint=timepoint,
                        loss=computed_loss,
                        predicted_sequence=predicted_sequence[0].tolist(),
                        actual_sequence=actual_sequence.tolist()
                    )
                    update_callback(response)

                loss.append(computed_loss)
                timepoints.append(timepoint)
                timepoint += 1
                result.append([timepoint, computed_loss, predicted_sequence[0], actual_sequence])
        except StopIteration:
            pass

        return result

    def listen_and_accept(self):
        das_socket_path = Path(PREDICT_FILE_SOCKET)
        das_socket_path.unlink(missing_ok=True)

        das_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        das_socket.bind(PREDICT_FILE_SOCKET)

        while True:
            das_socket.listen(1)
            _log.info(f"{self.__class__.__name__}: waiting for connection on {PREDICT_FILE_SOCKET}...")
            try:
                connection, client_address = das_socket.accept()

                _log.info(f"{self.__class__.__name__}: receiving job information ...")
                msg = connection.recvmsg(2064)
                job = Job.decode(msg[0])

                _log.info(f"{self.__class__.__name__}: creating thread for {job}")
                t = Thread(target=self.execute, args=(connection, client_address, job, self.stop_event))
                t.name = f"{self.__class__.__name__}({job.id}"
                self.threads[job.id] = t
                t.start()
            except Exception:
                _log.info(f"{self.__class__.__name__}: exception encountered - requesting all threads to stop")
                self.stop_event.set()
                raise

    def execute(self, connection, client_address, job, stop_event: Event):
        _log.info(f"{self.__class__.__name__}.execute: starting job {job.id}")

        def update(response: Job.Response):
            encoded_response = response.encode()
            msg_size = len(encoded_response)

            _log.debug(f"{self.__class__.__name__}({job.id}): sendmsg: size: {msg_size}, response: {response}")
            connection.sendmsg([msg_size.to_bytes(4, 'little'), encoded_response])

        try:
            _log.info(f"{self.__class__.__name__}({job.id}): loading data")
            start = dt.datetime.utcnow()
            df = pl.scan_parquet(job.data_filename)
            _log.info(
                f"{self.__class__.__name__}({job.id}): loading data "
                f"[done after {(dt.datetime.utcnow() - start).total_seconds()} seconds]")

            self.predict(job_id=job.id,
                         model_name=job.model_name,
                         experiment_dir=job.experiment_dir,
                         features=job.features,
                         target=job.target,
                         sequence_length=job.sequence_length,
                         df=df,
                         connection=connection,
                         stop_event=stop_event,
                         update_callback=update)

            connection.sendmsg([ControlCommand.BYE.value.encode()])
        except BrokenPipeError as e:
            raise ConnectionError(f"{self.__class__.__name__}.run_prediction:"
                                  f" failed to communicate the results due to a connection error - {e}") from e
        finally:
            connection.close()


if __name__ == "__main__":
    basicConfig()
    _log.setLevel(INFO)

    w = Worker()
    w.listen_and_accept()

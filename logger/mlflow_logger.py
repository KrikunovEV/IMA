from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE
import multiprocessing as mp

from logger import commands


def _send(queue: mp.Queue, command, instance, *args, **kwargs):
    queue.put((command, instance, args, kwargs))


def load_model(path: str):
    import torch
    return torch.load(path)


def load_model_from_artifacts(exp_id: str, run_id: str, state_dict_path: str):
    return load_model(f'mlruns/{exp_id}/{run_id}/artifacts/{state_dict_path}')


class Worker:
    def __init__(self, logger, experiment_name):
        self.logger = logger
        self.instances = {}
        self.exp_id = self._find_experiment_id(experiment_name)
        self.global_run_id = self.create_run('global')
        self.set_tag(self.global_run_id)
        self.terminated = RunStatus.to_string(RunStatus.FINISHED)

    def _find_experiment_id(self, experiment_name):
        exp = self.logger.get_experiment_by_name(experiment_name)
        if exp is not None:
            return exp.experiment_id
        else:
            exp_count = 0
            while True:
                exp_name = experiment_name + str(exp_count)
                exp = self.logger.get_experiment_by_name(exp_name)
                exp_count += 1
                if exp is None:
                    return self.logger.create_experiment(exp_name)

    def create_run(self, name: str):
        return self.logger.create_run(self.exp_id, tags={MLFLOW_RUN_NAME: name}).info.run_id

    def terminate_run(self, run_id):
        self.logger.set_terminated(run_id, self.terminated)

    def set_tag(self, run_id):
        self.logger.set_tag(run_id, MLFLOW_RUN_NOTE, f'run_id: {run_id}')


def _mlflow_worker(message_queue: mp.Queue, experiment_name: str):
    worker = Worker(logger=MlflowClient(), experiment_name=experiment_name)

    while True:
        cmd_func, instance, args, kwargs = message_queue.get()

        if instance is None:
            raise Exception(f'Logger: init() was not called yet.')

        if cmd_func(worker, instance, *args, **kwargs):
            break


class RunLogger:
    INSTANCE_COUNT = 0

    def __init__(self, queue: mp.Queue, run_name: str, metrics: tuple, train: bool = True):
        self._queue = queue
        self._instance = f'{mp.current_process().name}_{RunLogger.INSTANCE_COUNT}'
        _send(self._queue, commands.init, self._instance, run_name, metrics)
        self.set_mode(train)
        RunLogger.INSTANCE_COUNT += 1

    def set_mode(self, train: bool):
        _send(self._queue, commands.set_mode, self._instance, train)

    def log(self, data: dict):
        _send(self._queue, commands.log, self._instance, data)

    def all(self):
        _send(self._queue, commands.on_all, self._instance)

    def call(self, func: str, data):
        _send(self._queue, commands.call, self._instance, func, data)

    def param(self, params: dict):
        _send(self._queue, commands.param, self._instance, params)

    def artifact(self, src_path: str, dest_dir: str = None):
        _send(self._queue, commands.artifact, self._instance, src_path, dest_dir)

    def artifacts(self, src_path: str, dest_dir: str = None):
        _send(self._queue, commands.artifacts, self._instance, src_path, dest_dir)

    def deinit(self):
        _send(self._queue, commands.deinit, self._instance)


class LoggerServer:
    DEFAULT_EXPERIMENT_NAME = '_default'

    def __init__(self, experiment_name: str = DEFAULT_EXPERIMENT_NAME):
        mp.set_start_method('spawn')  # to work on Windows and Ubuntu
        self.manager = mp.Manager()
        self._queue = self.manager.Queue()
        self._process = mp.Process(target=_mlflow_worker, args=(self._queue, experiment_name))
        self._instance = 'server'

    def start(self):
        self._process.start()

    def stop(self):
        _send(self._queue, commands.stop, self._instance)
        self._process.join()

    @property
    def queue(self):
        return self._queue

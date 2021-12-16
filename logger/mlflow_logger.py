from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE
import multiprocessing as mp

from logger import commands
from logger.metric import IMetric


def _send(queue: mp.Queue, command, instance, *args, **kwargs):
    queue.put((command, instance, args, kwargs))


def load_model_from_artifacts(exp_id: str, run_id: str, state_dict_path: str):
    import torch
    return torch.load(f'mlruns/{exp_id}/{run_id}/artifacts/{state_dict_path}')


def load_model(path: str):
    import torch
    return torch.load(path)


class Worker:
    def __init__(self, logger, experiment_name):
        self.logger = logger
        self.instances = {}
        self.exp_id = self._find_experiment_id(experiment_name)
        self.global_run_id = logger.create_run(self.exp_id, tags={MLFLOW_RUN_NAME: 'global'}).info.run_id
        self.terminated = RunStatus.to_string(RunStatus.FINISHED)

    def _find_experiment_id(self, experiment_name):
        _exp = self.logger.get_experiment_by_name(experiment_name)
        if _exp is not None:
            return _exp.experiment_id
        else:
            _exp_count = 0
            while True:
                _exp_name = experiment_name + str(_exp_count)
                _exp = self.logger.get_experiment_by_name(_exp_name)
                _exp_count += 1
                if _exp is None:
                    return self.logger.create_experiment(_exp_name)

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

    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._instance = None
        self._id = RunLogger.INSTANCE_COUNT
        RunLogger.INSTANCE_COUNT += 1

    def init(self, run_name: str, train: bool, *args: IMetric):
        self._instance = f'{mp.current_process().name}_{self._id}'
        _send(self._queue, commands.init, self._instance, run_name, args)
        self.set_mode(train)

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

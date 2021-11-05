from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE

import multiprocessing as mp
from enum import Enum, unique
from typing import Any

from logger.metric import IMetric


@unique
class CMD(Enum):
    INIT = 'init'
    LOG = 'log'
    STOP = 'stop'
    MODE = 'set_mode'
    ALL = 'all'
    CALL = 'call'
    PARAM = 'param'
    ART = 'artifact'
    ARTS = 'artifacts'
    DEINIT = 'deinit'


def _send(queue: mp.Queue, instance: str, cmd: CMD, data: Any):
    queue.put((instance, cmd, data))


def load_model_from_artifacts(exp_id: str, run_id: str, state_dict_path: str):
    import torch
    return torch.load(f'mlruns/{exp_id}/{run_id}/artifacts/{state_dict_path}')


def load_model(path: str):
    import torch
    return torch.load(path)


def _mlflow_worker(message_queue: mp.Queue, experiment_name: str):
    # initialize mlflow client
    logger = MlflowClient()

    # try to create unique experiment if experiment_name does not exist otherwise continue
    _exp = logger.get_experiment_by_name(experiment_name)
    if _exp is not None:
        exp_id = _exp.experiment_id
    else:
        _exp_count = 0
        while True:
            _exp_name = experiment_name + str(_exp_count)
            _exp = logger.get_experiment_by_name(_exp_name)
            _exp_count += 1
            if _exp is None:
                exp_id = logger.create_experiment(_exp_name)
                break

    # other meta information
    instances = dict()
    terminated = RunStatus.to_string(RunStatus.FINISHED)

    while True:
        instance, cmd, data = message_queue.get()

        if instance not in instances and cmd != CMD.INIT and cmd != CMD.STOP:
            raise Exception(f'Logger: trying to .{cmd.value}(), but .{CMD.INIT.value}() was not called yet.')

        if cmd == CMD.INIT:
            # data: tuple (str run_name, IMetric args)
            if instance in instances:
                raise Exception(f'Logger warning: two .{CMD.INIT.value}() calls are occurred.')
            else:
                metrics = list(data[1])
                if len(metrics) > 0:
                    run_id = logger.create_run(exp_id, tags={MLFLOW_RUN_NAME: data[0]}).info.run_id
                    logger.set_tag(run_id, MLFLOW_RUN_NOTE, f'run_id: {run_id}')
                    for metric in metrics:
                        metric.set_mlflow(logger, run_id)
                    instances[instance] = dict(run_id=run_id, metrics=metrics)
                else:
                    raise Exception(f'Logger: .{CMD.INIT}() has to input at least one \'metric\'.')
        elif cmd == CMD.LOG:
            # data: dict data
            for key, value in data.items():
                for metric in instances[instance]['metrics']:
                    if metric.is_able_to_log() and metric.key == key:
                        metric.on_log(value)
        elif cmd == CMD.MODE:
            # data: bool train
            for metric in instances[instance]['metrics']:
                metric.set_mode(data)
        elif cmd == CMD.ALL:
            # data: None
            for metric in instances[instance]['metrics']:
                if hasattr(metric, 'on_all'):
                    metric.on_all()
        elif cmd == CMD.CALL:
            # data: tuple (str func, data)
            for metric in instances[instance]['metrics']:
                if hasattr(metric, f'{data[0]}'):
                    getattr(metric, f'{data[0]}')(data[1])
        elif cmd == CMD.PARAM:
            # data: dict params
            for key, value in data.items():
                logger.log_param(instances[instance]['run_id'], key, value)
        elif cmd == CMD.ART:
            # data: tuple (str src_path, str dest_path)
            logger.log_artifact(run_id=instances[instance]['run_id'], local_path=data[0], artifact_path=data[1])
        elif cmd == CMD.ARTS:
            # data: tuple (str src_path, str dest_path)
            logger.log_artifacts(run_id=instances[instance]['run_id'], local_dir=data[0], artifact_path=data[1])
        elif cmd == CMD.STOP:
            # data: None
            break
        elif cmd == CMD.DEINIT:
            # data: None
            if instance in instances:
                logger.set_terminated(instances[instance]['run_id'], terminated)
                instances.pop(instance)


class RunLogger:
    INSTANCE_COUNT = 0

    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._instance = f'{mp.current_process().name}_{RunLogger.INSTANCE_COUNT}'
        RunLogger.INSTANCE_COUNT += 1

    def init(self, run_name: str, train: bool, *args: IMetric):
        _send(self._queue, self._instance, CMD.INIT, (run_name, args))
        self.set_mode(train)

    def set_mode(self, train: bool):
        _send(self._queue, self._instance, CMD.MODE, train)

    def log(self, data: dict):
        _send(self._queue, self._instance, CMD.LOG, data)

    def all(self):
        _send(self._queue, self._instance, CMD.ALL, None)

    def call(self, func: str, data):
        _send(self._queue, self._instance, CMD.CALL, (func, data))

    def param(self, params: dict):
        _send(self._queue, self._instance, CMD.PARAM, params)

    def artifact(self, src_path: str, dest_dir: str = None):
        _send(self._queue, self._instance, CMD.ART, (src_path, dest_dir))

    def artifacts(self, src_path: str, dest_dir: str = None):
        _send(self._queue, self._instance, CMD.ARTS, (src_path, dest_dir))

    def deinit(self):
        _send(self._queue, self._instance, CMD.DEINIT, None)


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
        _send(self._queue, self._instance, CMD.STOP, None)
        self._process.join()

    @property
    def queue(self):
        return self._queue

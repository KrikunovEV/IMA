import time
from mlflow.entities import Metric as MLFlowMetric
from mlflow.utils.validation import MAX_ENTITIES_PER_BATCH


def get_time_ms():
    return int(time.time() * 1000.)


class IMetric:
    POSTFIX_EVAL = '_eval'

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = False, is_global: bool = False):
        if not log_on_eval and not log_on_train:
            raise Exception(f'Both log_on_train and log_on_eval can not be False')

        self._train_step = 0
        self._eval_step = 0
        self._train = None
        self._logger = None
        self._run_id = None

        self._key = key
        self._suffix_key = key if suffix == '' else f'{suffix}_{key}'
        self._fullname = self._suffix_key
        self._log_on_train = log_on_train
        self._log_on_eval = log_on_eval
        self._epoch_counter = epoch_counter
        self._is_global = is_global
        self._train_counter = 0
        self._eval_counter = 0

    def set_mlflow(self, client, run_id):
        self._logger = client
        self._run_id = run_id

    def set_mode(self, train: bool):
        if self._train == train:
            return
        self._train = train
        self._fullname = self._suffix_key

        if self._epoch_counter:
            if train:
                self._train_counter += 1
                self._fullname += f'_epoch{self._train_counter}'
            else:
                self._eval_counter += 1
                self._fullname += f'_epoch{self._eval_counter}'

        if self._log_on_eval and not train:
            self._fullname += IMetric.POSTFIX_EVAL

    def is_able_to_log(self):
        return (self._train and self._log_on_train) or (not self._train and self._log_on_eval)

    def _get_step(self):
        if self._train:
            step = self._train_step
            self._train_step += 1
        else:
            step = self._eval_step
            self._eval_step += 1
        return step

    @property
    def key(self):
        return self._key

    @property
    def is_global(self):
        return self._is_global

    @property
    def logger(self):
        return self._logger

    def on_log(self, value):
        raise NotImplementedError

    def on_all(self):
        raise NotImplementedError


class Metric(IMetric):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super().__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)

    def on_log(self, value):
        self.logger.log_metric(run_id=self._run_id,
                               key=self._fullname,
                               value=value,
                               timestamp=get_time_ms(),
                               step=self._get_step())

    def on_all(self):
        pass


class BatchMetric(IMetric):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super().__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self._metrics = []

    def on_log(self, value):
        self._metrics.append(MLFlowMetric(
            key=self._fullname,
            value=value,
            timestamp=get_time_ms(),
            step=self._get_step()
        ))

        # mlflow constraint
        if len(self._metrics) == MAX_ENTITIES_PER_BATCH:
            self.on_all()

    def on_all(self):
        if len(self._metrics) > 0:
            self.logger.log_batch(run_id=self._run_id, metrics=self._metrics)
            self._metrics = []

import pickle
import os
import torch
import numpy as np

from logger.metric import IMetric


class IArtifact(IMetric):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 is_global: bool = False):
        super().__init__(key, suffix, log_on_train, log_on_eval, True, is_global)
        self._dest_dir = ''
        self._tmp_dir = '_temp'
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._salt = np.random.randint(10000000)

    def set_mode(self, train: bool):
        if self._train == train:
            return
        self._train = train
        self._train_step = 0
        self._eval_step = 0

        if train:
            self._train_counter += 1
            self._dest_dir = f'epoch{self._train_counter}'
        else:
            self._eval_counter += 1
            self._dest_dir = f'epoch{self._eval_counter}'

        if self._log_on_eval and not train:
            self._dest_dir += self.POSTFIX_EVAL

    def on_log(self, value):
        raise NotImplementedError

    def on_all(self):
        pass


class PickleArt(IArtifact):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 is_global: bool = False):
        super().__init__(key, suffix, log_on_train, log_on_eval, is_global)

    def on_log(self, data):
        fullname = os.path.join(self._tmp_dir, f'{self._salt}_{self._fullname}_step{self._get_step()}.pickle')
        with open(fullname, 'wb') as f:
            pickle.dump(data, f)
        self._logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self._dest_dir)
        os.remove(fullname)


class ModelArt(IArtifact):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 is_global: bool = True):
        super().__init__(key, suffix, log_on_train, log_on_eval, is_global)

    def on_log(self, state):
        fullname = os.path.join(self._tmp_dir, f'{self._salt}_{self._fullname}_step{self._get_step()}.pt')
        torch.save(state, fullname)
        self._logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self._dest_dir)
        os.remove(fullname)


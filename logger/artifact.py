import pickle
import os
import torch

from logger.metric import IMetric


class IArtifact(IMetric):
    TEMP_DIR = '_temp'

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 log_in_dir: bool = False, is_global: bool = False):
        # epoch counter always True!
        super().__init__(key, suffix, log_on_train, log_on_eval, True, is_global)
        self.dest_dir = ''
        self._log_in_dir = log_in_dir

        while True:
            self._tmp_dir = os.path.join(IArtifact.TEMP_DIR, f'{torch.randint(1000000, size=(1,)).item()}')
            if not os.path.exists(self._tmp_dir):
                os.makedirs(self._tmp_dir, exist_ok=True)
                break

    def set_mode(self, train: bool):
        if self._train == train:
            return
        self._train = train
        self._train_step = 0
        self._eval_step = 0
        self._fullname = self._suffix_key

        if train:
            self._train_counter += 1
            postfix = f'epoch{self._train_counter}'
            if self._log_in_dir:
                self.dest_dir = postfix
            else:
                self._fullname += f'_{postfix}'
        else:
            self._eval_counter += 1
            postfix = f'epoch{self._eval_counter}'
            if self._log_in_dir:
                self.dest_dir = postfix + (self.POSTFIX_EVAL if self._log_on_eval else '')
            else:
                self._fullname += f'_{postfix}' + (self.POSTFIX_EVAL if self._log_on_eval else '')

    def prepare_name(self):
        return os.path.join(self._tmp_dir, f'{self._fullname}_step{self._get_step()}')

    def on_log(self, value):
        raise NotImplementedError

    def on_all(self):
        pass


class PickleArt(IArtifact):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 log_in_dir: bool = False, is_global: bool = False):
        super().__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)

    def on_log(self, data):
        fullname = f'{self.prepare_name()}.pickle'
        with open(fullname, 'wb') as f:
            pickle.dump(data, f)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)


class ModelArt(IArtifact):

    def __init__(self, key: str, suffix: str = '', log_on_train: bool = True, log_on_eval: bool = True,
                 log_in_dir: bool = False, is_global: bool = True):
        super().__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)

    def on_log(self, state):
        fullname = f'{self.prepare_name()}.pt'
        torch.save(state, fullname)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)

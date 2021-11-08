import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import imageio

from logger import Metric, IArtifact


class SumMetric(Metric):
    def __init__(self, key: str, log_on_train: bool = True, log_on_eval: bool = True, epoch_counter: bool = True):
        super(SumMetric, self).__init__(key, log_on_train, log_on_eval, epoch_counter)
        self.sum = 0

    # def set_mode(self, train: bool):
    #     self.sum = 0
    #     super(SumMetric, self).set_mode(train)

    def on_log(self, value):
        self.sum += value
        super(SumMetric, self).on_log(self.sum)


class ActionMap(IArtifact):

    OFFEND: int = 0
    DEFEND: int = 1

    def __init__(self, key: str, players: int, labels: list):
        super(ActionMap, self).__init__(key)
        self.players = players
        self.labels = labels
        self.OM = np.zeros((players, players), dtype=np.int)
        self.DM = np.zeros((players, players), dtype=np.int)

    def on_log(self, actions):
        for row, action in enumerate(actions):
            self.OM[row, action[self.OFFEND]] += 1
            self.DM[row, action[self.DEFEND]] += 1

    def on_all(self):
        pass

    def action_map(self, data):
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title('offends heatmap')
        ax[1].set_title('defends heatmap')
        sn.heatmap(self.OM, cmap='Reds', xticklabels=self.labels, yticklabels=self.labels,
                   ax=ax[0], square=True, cbar=False, annot=True, fmt='d')
        sn.heatmap(self.DM, cmap='Blues', xticklabels=self.labels, yticklabels=self.labels,
                   ax=ax[1], square=True, cbar=False, annot=True, fmt='d')
        fig.tight_layout()

        fullname = f'{self._fullname}_step{self._get_step()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self._logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self._dest_dir)
        os.remove(fullname)

        self.OM = np.zeros((self.players, self.players), dtype=np.int)
        self.DM = np.zeros((self.players, self.players), dtype=np.int)


class PolicyViaTime(IArtifact):

    def __init__(self, key: str, players: int, labels: list, log_on_eval: bool = False):
        super(PolicyViaTime, self).__init__(key, log_on_eval=log_on_eval)
        self.players = players
        self.labels = labels
        self.temp_dir = 'gif'
        self.gif_name = os.path.join(self.temp_dir, f'{self._fullname}.gif')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.filenames = []
        self.frame = 0

    def on_log(self, policies):
        self.frame += 1
        if self.frame % 8 == 0:
            a_policies, d_policies = policies
            n = len(self.labels)
            fig, ax = plt.subplots(2, n, figsize=(16, 9), sharex=True, sharey=True)
            for i in range(n):
                ax[0][i].set_title(f'{self.labels[i]} agent\'s offend policy')
                ax[1][i].set_title(f'{self.labels[i]} agent\'s defend policy')
                ax[0][i].set_ylim(-0.01, 1.01)
                ax[1][i].set_ylim(-0.01, 1.01)
                ax[0][i].set_xticks([int(label) for label in self.labels])
                ax[1][i].set_xticks([int(label) for label in self.labels])
                colors = ['b' for _ in range(n)]
                colors[np.argmax(a_policies[i])] = 'r'
                ax[0][i].bar(np.arange(n) + 1, a_policies[i], color=colors)
                colors = ['b' for _ in range(n)]
                colors[np.argmax(d_policies[i])] = 'r'
                ax[1][i].bar(np.arange(n) + 1, d_policies[i], color=colors)
            fig.tight_layout()
            self.filenames.append(os.path.join(self.temp_dir, f'{len(self.filenames)}.png'))
            plt.savefig(self.filenames[-1])
            plt.close(fig)

    def on_all(self):
        pass

    def policy_via_time(self, data):
        with imageio.get_writer(self.gif_name, mode='I') as writer:
            for filename in self.filenames:
                writer.append_data(imageio.imread(filename))
                os.remove(filename)

        self._logger.log_artifact(run_id=self._run_id, local_path=self.gif_name, artifact_path=self._dest_dir)
        os.remove(self.gif_name)
        os.rmdir(self.temp_dir)

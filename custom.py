import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import cv2 as cv

from logger import Metric, IArtifact


class SumMetric(Metric):
    def __init__(self, key: str, suffix: str = 'sum', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super(SumMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self.sum_train = 0
        self.sum_eval = 0

    def on_log(self, value):
        if self._train:
            self.sum_train += value
            _sum = self.sum_train
        else:
            self.sum_eval += value
            _sum = self.sum_eval
        super(SumMetric, self).on_log(_sum)


class CoopsMetric(IArtifact):
    def __init__(self, key: str, suffix: str):
        super(CoopsMetric, self).__init__(key, suffix, False, True, True)
        self.coops = []

    def set_mode(self, train: bool):
        super(CoopsMetric, self).set_mode(train)
        if not train:
            self.coops.append([0, 0, 0])

    def on_log(self, actions):
        if actions[0][0] == actions[1][0] and actions[0][1] == actions[1][1]:
            self.coops[-1][0] += 1
        elif actions[1][0] == actions[2][0] and actions[1][1] == actions[2][1]:
            self.coops[-1][1] += 1
        elif actions[0][0] == actions[2][0] and actions[0][1] == actions[2][1]:
            self.coops[-1][2] += 1

    def coop_bars(self, data):
        epochs = np.arange(len(self.coops)) + 1
        np_coops = np.array(self.coops)
        fig, ax = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
        ax[0].set_title('1 2 vs 3')
        ax[1].set_title('2 3 vs 1')
        ax[2].set_title('1 3 vs 2')
        for i in range(3):
            ax[i].set_xlabel('epoch')
            ax[i].set_ylabel('# of coops')
            ax[i].set_ylim(-0.01, 100.01)
            ax[i].bar(epochs, np_coops[:, i])
            ax[i].set_xticks(epochs)
            for e in epochs:
                ax[i].text(e, np_coops[e - 1, i], f'{np_coops[e - 1, i]}', fontsize=6, ha='center')
        fig.tight_layout()
        fullname = os.path.join(self._tmp_dir, f'{self._salt}_{self._fullname}_coops.png')
        plt.savefig(fullname)
        plt.close(fig)
        self._logger.log_artifact(run_id=self._run_id, local_path=fullname)
        os.remove(fullname)


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

    def action_map(self, data):
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title('offends heatmap')
        ax[1].set_title('defends heatmap')
        sn.heatmap(self.OM, cmap='Reds', xticklabels=self.labels, yticklabels=self.labels,
                   ax=ax[0], square=True, cbar=False, annot=True, fmt='d')
        sn.heatmap(self.DM, cmap='Blues', xticklabels=self.labels, yticklabels=self.labels,
                   ax=ax[1], square=True, cbar=False, annot=True, fmt='d')
        fig.tight_layout()

        fullname = os.path.join(self._tmp_dir, f'{self._salt}_{self._fullname}_step{self._get_step()}.png')
        plt.savefig(fullname)
        plt.close(fig)
        self._logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self._dest_dir)
        os.remove(fullname)

        self.OM = np.zeros((self.players, self.players), dtype=np.int)
        self.DM = np.zeros((self.players, self.players), dtype=np.int)


class PolicyViaTime(IArtifact):

    def __init__(self, key: str, players: int, labels: list):
        super(PolicyViaTime, self).__init__(key, log_on_eval=False)
        self.players = players
        self.labels = labels
        self.filenames = []
        self.frame = 0

    def on_log(self, data):
        self.frame += 1
        if self.frame % 50 == 0:
            a_policies, d_policies = data
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
            fullname = os.path.join(self._tmp_dir, f'{self._salt}_{len(self.filenames)}.png')
            self.filenames.append(fullname)
            plt.savefig(fullname)
            plt.close(fig)

    def policy_via_time(self, data):
        fullname = os.path.join(self._tmp_dir, f'{self._salt}_{self._fullname}.avi')
        writer = cv.VideoWriter(fullname, cv.VideoWriter_fourcc(*'DIVX'), 2, (1600, 900))

        for i, filename in enumerate(self.filenames):
            writer.write(cv.imread(filename))
            os.remove(filename)
        writer.release()

        self._logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self._dest_dir)
        os.remove(fullname)

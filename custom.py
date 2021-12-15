import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import cv2 as cv
from utils import greater_divisor

from logger import Metric, BatchMetric, IArtifact, CooperationTask


class BatchSumMetric(BatchMetric):
    def __init__(self, key: str, suffix: str = 'sum', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super(BatchSumMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self.sum_train = 0
        self.sum_eval = 0

    def on_log(self, value):
        if self._train:
            self.sum_train += value
            _sum = self.sum_train
        else:
            self.sum_eval += value
            _sum = self.sum_eval
        super(BatchSumMetric, self).on_log(_sum)


class BatchAvgMetric(BatchMetric):
    def __init__(self, key: str, avg: int, suffix: str = 'avg', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super(BatchAvgMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self.values = []
        self.avg = avg

    def on_log(self, value):
        self.values.append(value)
        if len(self.values) == self.avg:
            self.on_all()

    def on_all(self):
        if len(self.values) > 0:
            super(BatchAvgMetric, self).on_log(np.mean(self.values))
            super(BatchAvgMetric, self).on_all()
            self.values = []


class BatchSumAvgMetric(BatchMetric):
    def __init__(self, key: str, avg: int, suffix: str = 'sum_avg', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super(BatchSumAvgMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self.values = []
        self.sum_train = 0
        self.sum_eval = 0
        self.avg = avg

    def on_log(self, value):
        self.values.append(value)
        if len(self.values) == self.avg:
            mean_value = np.mean(self.values)
            if self._train:
                self.sum_train += mean_value
                _sum = self.sum_train
            else:
                self.sum_eval += mean_value
                _sum = self.sum_eval
            super(BatchSumAvgMetric, self).on_log(_sum)
            self.values = []

    def on_all(self):
        if len(self.values) > 0:
            mean_value = np.mean(self.values)
            if self._train:
                self.sum_train += mean_value
                _sum = self.sum_train
            else:
                self.sum_eval += mean_value
                _sum = self.sum_eval
            super(BatchSumAvgMetric, self).on_log(_sum)
            self.values = []
        super(BatchSumAvgMetric, self).on_all()


class CoopsMetric(IArtifact, CooperationTask):
    def __init__(self, key: str, suffix: str):
        IArtifact.__init__(self, key, suffix, log_on_train=False, log_on_eval=True, is_global=True)
        CooperationTask.__init__(self)
        self.coops = []

    def set_mode(self, train: bool):
        super(CoopsMetric, self).set_mode(train)
        if not train:
            self.coops.append(0)

    def on_log(self, actions):
        self.coops[-1] = self.coops[-1] + \
                         np.array([1 if np.array_equal(actions[:, current], actions[:, neighbor])
                                   else 0
                                   for current, neighbor in self.get_cooperation_relation(actions.shape[-1])])

    def coop_bars(self, data):
        epochs = np.arange(1, len(self.coops) + 1)
        np_coops = np.array(self.coops)
        gd = greater_divisor(np_coops.shape[-1])
        fig_size = (gd, int(np_coops.shape[-1]/gd))
        fig, ax = plt.subplots(fig_size[0], fig_size[1], figsize=(16, 9), sharey=True)
        ax = ax.reshape(-1)
        for i, (cur, nbr) in enumerate(self.get_cooperation_relation(np_coops.shape[-1]) + 1):
            self._draw_bars(ax[i], f'{cur}&{nbr}', np_coops[:, i], xticks_step=5)
        ax = ax.reshape(fig_size)

        fig.tight_layout()
        fullname = os.path.join(self._tmp_dir, f'{self._salt}_{self._fullname}_coops.png')
        plt.savefig(fullname)
        plt.close(fig)
        self._logger.log_artifact(run_id=self._run_id, local_path=fullname)
        os.remove(fullname)


class AvgCoopsMetric(IArtifact, CooperationTask):
    def __init__(self, key: str, repeats, epochs):
        IArtifact.__init__(self, key, suffix='', log_on_train=False, log_on_eval=True, is_global=False)
        CooperationTask.__init__(self)

        self.coops = []
        self.repeats = repeats
        self.epochs = epochs

    def on_log(self, game_actions):
        for epoch_actions in game_actions:
            self.coops.append(0)
            for actions in epoch_actions:
                self.coops[-1] = self.coops[-1] + \
                                 np.array([1 if np.array_equal(actions[:, current], actions[:, neighbor])
                                           else 0
                                           for current, neighbor in self.get_cooperation_relation(actions.shape[-1])])

    def avg_coop_bars(self, data):

        fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
        self._draw_bars(ax[0], 'AVG_COOPS by type', np.mean(self.coops, axis=0),
                        xticks=self.get_cooperation_relation(self.coops[0].shape[-1]) + 1,
                        xlabel='cooperation type')

        self._draw_bars(ax[1], 'AVG_COOPS by epoch', np.sum(self.coops, axis=1) / self.repeats, xticks_step=5)

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
        self.OM += actions[self.OFFEND]
        self.DM += actions[self.DEFEND]

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
            # fig, ax = plt.subplots(2, n, figsize=(16, 9), sharex=True, sharey=True)
            fig, ax = plt.subplots(2, n, figsize=(16, 9))
            for i in range(n):
                ax[0][i].set_title(f'{self.labels[i]} agent\'s offend policy')
                ax[1][i].set_title(f'{self.labels[i]} agent\'s defend policy')
                ax[0][i].set_ylim(-0.01, 1.01)
                ax[1][i].set_ylim(-0.01, 1.01)
                ax[0][i].set_xticks([int(label[0]) for label in self.labels], self.labels)
                ax[1][i].set_xticks([int(label[0]) for label in self.labels], self.labels)
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


if __name__ == '__main__':

    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    fig, axs = plt.subplots(5)
    axs[0].plot(x, y)
    axs[0].set_title('Axis [0, 0]')
    axs[1].plot(x, y, 'tab:orange')
    axs[1].set_title('Axis [0, 1]')
    axs[2].plot(x, -y, 'tab:green')
    axs[2].set_title('Axis [1, 0]')
    #axs[1, 1].plot(x, -y, 'tab:red')
    #axs[1, 1].set_title('Axis [1, 1]')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    axs = axs.reshape(-1,2)

    plt.show()

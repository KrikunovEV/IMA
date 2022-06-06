import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import cv2 as cv
from utils import greater_divisor
# from screeninfo import get_monitors

from logger import BatchMetric, IArtifact, Metric
from config import Config


class EMAMetric(Metric):
    def __init__(self, key: str, ema: float, suffix: str = 'ema', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super(EMAMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        if ema < 0 or ema >= 1.:
            raise Exception('ema value must be in range [0, 1)')
        self.ema = ema
        self.value = None

    def on_log(self, value):
        self.value = value if self.value is None else self.ema * value + (1 - self.ema) * self.value
        super(EMAMetric, self).on_log(self.value)


class BatchSumMetric(BatchMetric):
    def __init__(self, key: str, suffix: str = 'sum', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = False, is_global: bool = False):
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


class AvgMetric(Metric):
    def __init__(self, key: str, avg: int, suffix: str = 'avg', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = True, is_global: bool = False):
        super(AvgMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self.values = []
        self.avg = avg

    def on_log(self, value):
        self.values.append(value)
        if len(self.values) == self.avg:
            super(AvgMetric, self).on_log(np.mean(self.values))
            self.values = []


class BatchAvgMetric(BatchMetric):
    def __init__(self, key: str, avg: int, suffix: str = 'avg', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = False, is_global: bool = False):
        super(BatchAvgMetric, self).__init__(key, suffix, log_on_train, log_on_eval, epoch_counter, is_global)
        self.values = []
        self.avg = avg

    def on_log(self, value):
        self.values.append(value)
        if len(self.values) == self.avg:
            super(BatchAvgMetric, self).on_log(np.mean(self.values))
            self.values = []

    def on_all(self):
        if len(self.values) > 0:
            super(BatchAvgMetric, self).on_log(np.mean(self.values))
            self.values = []
        super(BatchAvgMetric, self).on_all()


class BatchSumAvgMetric(BatchMetric):
    def __init__(self, key: str, avg: int, suffix: str = 'sum_avg', log_on_train: bool = True, log_on_eval: bool = True,
                 epoch_counter: bool = False, is_global: bool = False):
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


class AvgCoopsArtifact(IArtifact):
    def __init__(self, key: str, cfg: Config, suffix: str = '', log_on_train: bool = True,
                 log_on_eval: bool = True, log_in_dir: bool = False, is_global: bool = False):
        IArtifact.__init__(self, key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        self.coops = np.zeros(3)
        self.labels = ['агенты 1 и 2', 'агенты 1 и 3', 'агенты 2 и 3']
        self.cfg = cfg
        self.fontsize = 20

    def on_log(self, game_actions):
        for actions in game_actions:
            if actions[0][0] == actions[1][0] and actions[0][1] == actions[1][1]:
                self.coops[0] += 1
            if actions[0][0] == actions[2][0] and actions[0][1] == actions[2][1]:
                self.coops[1] += 1
            if actions[1][0] == actions[2][0] and actions[1][1] == actions[2][1]:
                self.coops[2] += 1

    def avg_coop_bars(self):
        self.coops = self.coops / (self.cfg.repeats * self.cfg.test_episodes)

        fig, ax = plt.subplots(figsize=(16, 9))
        bar = ax.bar([0, 1, 2], self.coops, tick_label=self.labels)
        ax.bar_label(bar, fmt='%.2f', size=self.fontsize)
        ax.set_ylabel('количество коопераций', size=self.fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=self.fontsize)
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()

        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)


class SumArtifact(IArtifact):
    def __init__(self, key: str, players_labels: list, suffix: str = '', log_on_train: bool = True,
                 log_on_eval: bool = True, log_in_dir: bool = False, is_global: bool = False):
        super(SumArtifact, self).__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        self.players = len(players_labels)
        self.players_labels = players_labels
        self.fontsize = 24
        self.fontsize2 = 20
        self.colors = ['r', 'g', 'b']

        self.rewards = [[] for _ in range(self.players)]

    def on_log(self, rewards):
        for metric, r in zip(self.rewards, rewards):
            metric.append(r)

    def all(self):
        if not self.is_able_to_log():
            return

        fig, ax = plt.subplots(figsize=(16, 9))
        for rewards, label, color in zip(self.rewards, self.players_labels, self.colors):
            ax.plot(np.cumsum(rewards), c=color, label=label, linewidth=3.)
        ax.set_xlabel('эпизод', size=self.fontsize)
        ax.set_ylabel('кумулятивная награда', size=self.fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=self.fontsize2)
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize2)
        ax.legend(fontsize=self.fontsize)
        fig.tight_layout()

        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)


class ActionMap(IArtifact):
    OFFEND: int = 0
    DEFEND: int = 1

    def __init__(self, key: str, players_labels: list, acts_labels: list, is_ond: bool, suffix: str = '',
                 log_on_train: bool = True, log_on_eval: bool = True, log_in_dir: bool = False,
                 is_global: bool = False):
        super(ActionMap, self).__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        self.acts = len(acts_labels)
        self.acts_labels = acts_labels
        self.players = len(players_labels)
        self.players_labels = players_labels
        self.actions = np.zeros((self.players, self.acts), dtype=int)
        self.actions2 = np.zeros((self.players, self.acts), dtype=int)
        self.is_ond = is_ond
        self.fontsize = 20
        self.sn_fontscale = 2.75

    def on_log(self, actions):
        for i in range(len(actions)):
            if self.is_ond:
                self.actions[i, actions[i][0]] += 1
                self.actions2[i, actions[i][1]] += 1
            else:
                self.actions[i, actions[i]] += 1

    def action_map(self):
        if not self.is_able_to_log():
            return

        if self.is_ond:
            fig, ax = plt.subplots(1, 2, figsize=(16, 9))
            ax[0].set_title('Нападение', fontsize=self.fontsize)
            ax[1].set_title('Защита', fontsize=self.fontsize)
            sn.heatmap(self.actions, cmap='Reds', xticklabels=self.acts_labels, yticklabels=self.players_labels,
                       ax=ax[0], square=True, cbar=False, annot=True, fmt='d', annot_kws={"size": self.fontsize})
            sn.heatmap(self.actions2, cmap='Blues', xticklabels=self.acts_labels, yticklabels=self.players_labels,
                       ax=ax[1], square=True, cbar=False, annot=True, fmt='d', annot_kws={"size": self.fontsize})
            ax[0].tick_params(axis='both', which='minor', labelsize=self.fontsize)
            ax[0].tick_params(axis='both', which='major', labelsize=self.fontsize)
            ax[1].tick_params(axis='both', which='minor', labelsize=self.fontsize)
            ax[1].tick_params(axis='both', which='major', labelsize=self.fontsize)
        else:
            fig, ax = plt.subplots(figsize=(16, 9))
            sn.heatmap(self.actions, cmap='Reds', xticklabels=self.acts_labels, yticklabels=self.players_labels,
                       ax=ax, square=True, cbar=False, annot=True, fmt='d', annot_kws={"size": self.fontsize})
            ax.tick_params(axis='both', which='minor', labelsize=self.fontsize)
            ax.tick_params(axis='both', which='major', labelsize=self.fontsize)

        fig.tight_layout()

        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)

        self.actions = np.zeros((self.players, self.acts), dtype=int)
        self.actions2 = np.zeros((self.players, self.acts), dtype=int)


class EMAArtifact(IArtifact):
    def __init__(self, key: str, players_labels: list, suffix: str = '', log_on_train: bool = True,
                 log_on_eval: bool = True, log_in_dir: bool = False, is_global: bool = False):
        super(EMAArtifact, self).__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        self.players = len(players_labels)
        self.players_labels = players_labels
        self.fontsize = 24
        self.fontsize2 = 20
        self.colors = ['r', 'g', 'b']
        self.alpha = 0.01

        self.rewards = [[] for _ in range(self.players)]

    def on_log(self, rewards):
        for metric, r in zip(self.rewards, rewards):
            if len(metric) == 0:
                metric.append(r)
            else:
                metric.append(self.alpha * r + (1 - self.alpha) * metric[-1])

    def ema_plots(self, changes=None):
        if not self.is_able_to_log():
            return

        fig, ax = plt.subplots(figsize=(16, 9))
        for rewards, label, color in zip(self.rewards, self.players_labels, self.colors):
            ax.plot(rewards, c=color, label=label, linewidth=3., zorder=1)
        if changes is not None:
            changes = np.array(changes)
            rs = np.array(self.rewards[0])[changes]
            ax.scatter(changes, rs, s=100, c='black', marker='o', label='changes', zorder=2)
            for i, (x, y) in enumerate(zip(changes, rs)):
                ax.text(x, y + 0.1, f'{i + 1}\n({int(x)})', c='black', size=16, zorder=3,
                        ha='center', va='center', ma='center')
        ax.set_ylim(-1., 1.)
        ax.set_xlabel('эпизод', size=self.fontsize)
        ax.set_ylabel('награда', size=self.fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=self.fontsize2)
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize2)
        ax.legend(fontsize=self.fontsize)
        fig.tight_layout()

        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)


if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(16, 9))
    data = [np.arange(100) * np.random.rand() for _ in range(3)]
    colors = ['r', 'g', 'b']
    for rewards, label, color in zip(data, colors, colors):
        ax.plot(rewards, c=color, label=label, linewidth=3., zorder=1)
    ax.set_xlabel('эпизод', size=20)
    ax.set_ylabel('награда', size=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=20)
    fig.tight_layout()
    plt.savefig('base_ema_reward_epoch1_step0.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 9))
    data = [np.arange(100) * np.random.rand() for _ in range(3)]
    colors = ['r', 'g', 'b']
    for rewards, label, color in zip(data, colors, colors):
        ax.plot(rewards, c=color, label=label, linewidth=3., zorder=1)
    ax.set_xlabel('эпизод', size=20)
    ax.set_ylabel('награда', size=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=20)
    fig.tight_layout()
    plt.savefig('base2_ema_reward_epoch1_eval_step0.png')
    plt.close(fig)

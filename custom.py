import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import cv2 as cv
from utils import greater_divisor
from screeninfo import get_monitors

from logger import BatchMetric, IArtifact
from config import Config


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


class CooperationTask:
    def __init__(self):
        self._cooperation_relation = None

    def get_cooperation_relation(self, shape):
        if self._cooperation_relation is None:
            indices = np.arange(shape)
            self._cooperation_relation = np.array([(current, neighbor)
                                                   for current in indices for neighbor in indices[current + 1:]])
        return self._cooperation_relation

    @staticmethod
    def _draw_bars(ax, title, x, xticks=None, xticks_step=1, ylim=(-0.01, 100.01), xlabel='epoch', ylabel='# of coops'):
        num_x = np.arange(1, x.size + 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.bar(num_x, x)
        ax.set_xticks(num_x[::xticks_step])
        if xticks is not None:
            ax.set_xticklabels(xticks)

        # for j in range(len(num_x[::10])):
        #     text = (num_x[::10])[j]
        #     mean_value = np.mean(np_coops[j * 20: (j + 1) * 20, i])
        #     ax[i].text(text, mean_value, f'{mean_value}', fontsize=6, ha='center')

        # for j in range(len(epochs[::10])):
        #     text = (epochs[::10] + 1)[j]
        #     mean_value = np.mean(np_coops[j * 20: (j + 1) * 20, i])
        #     ax[i].text(text, mean_value, f'{mean_value}', fontsize=6, ha='center')


class CoopsMetric(IArtifact, CooperationTask):
    def __init__(self, key: str, suffix: str = 'coops', log_on_train: bool = True, log_on_eval: bool = True,
                 log_in_dir: bool = False, is_global: bool = False):
        IArtifact.__init__(self, key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
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

    def coop_bars(self):
        np_coops = np.array(self.coops)
        gd = greater_divisor(np_coops.shape[-1])
        fig_size = (gd, int(np_coops.shape[-1] / gd))
        fig, ax = plt.subplots(fig_size[0], fig_size[1], figsize=(16, 9), sharey=True)
        ax = ax.reshape(-1)
        for i, (cur, nbr) in enumerate(self.get_cooperation_relation(np_coops.shape[-1]) + 1):
            self._draw_bars(ax[i], f'{cur}&{nbr}', np_coops[:, i], xticks_step=1)
        ax = ax.reshape(fig_size)

        fig.tight_layout()
        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)


class AvgCoopsMetric(IArtifact, CooperationTask):
    def __init__(self, key: str, cfg: Config, suffix: str = 'avg_coops', log_on_train: bool = True,
                 log_on_eval: bool = True, log_in_dir: bool = False, is_global: bool = False):
        IArtifact.__init__(self, key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        CooperationTask.__init__(self)
        self.coops = []
        self.cfg = cfg

    def on_log(self, game_actions):
        for epoch_actions in game_actions:
            self.coops.append(0)
            for actions in epoch_actions:
                self.coops[-1] = self.coops[-1] + \
                                 np.array([1 if np.array_equal(actions[:, current], actions[:, neighbor])
                                           else 0
                                           for current, neighbor in self.get_cooperation_relation(actions.shape[-1])])

    def avg_coop_bars(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
        self._draw_bars(ax[0], 'AVG_COOPS by type', np.mean(self.coops, axis=0),
                        xticks=self.get_cooperation_relation(self.coops[0].shape[-1]) + 1,
                        xlabel='cooperation type')

        by_epoch = np.sum(self.coops, axis=1).reshape((-1, self.cfg.epochs))
        by_epoch = np.mean(by_epoch, axis=0)
        self._draw_bars(ax[1], 'AVG_COOPS by epoch', by_epoch, xticks_step=1)

        fig.tight_layout()
        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)


class ActionMap(IArtifact):
    OFFEND: int = 0
    DEFEND: int = 1

    def __init__(self, key: str, players: int, labels: list, suffix: str = '', log_on_train: bool = True,
                 log_on_eval: bool = True, log_in_dir: bool = False, is_global: bool = False):
        super(ActionMap, self).__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        self.players = players
        self.labels = labels
        self.OM = np.zeros((players, players), dtype=np.int)
        self.DM = np.zeros((players, players), dtype=np.int)

    def on_log(self, actions):
        self.OM += actions[self.OFFEND]
        self.DM += actions[self.DEFEND]

    def action_map(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title('offends heatmap')
        ax[1].set_title('defends heatmap')
        sn.heatmap(self.OM, cmap='Reds', xticklabels=self.labels, yticklabels=self.labels,
                   ax=ax[0], square=True, cbar=False, annot=True, fmt='d')
        sn.heatmap(self.DM, cmap='Blues', xticklabels=self.labels, yticklabels=self.labels,
                   ax=ax[1], square=True, cbar=False, annot=True, fmt='d')
        fig.tight_layout()

        fullname = f'{self.prepare_name()}.png'
        plt.savefig(fullname)
        plt.close(fig)
        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
        os.remove(fullname)

        self.OM = np.zeros((self.players, self.players), dtype=np.int)
        self.DM = np.zeros((self.players, self.players), dtype=np.int)


class PolicyViaTime(IArtifact):

    def __init__(self, key: str, players: int, labels: list, frames_skip: int = 25, suffix: str = '',
                 log_on_train: bool = True, log_on_eval: bool = True, log_in_dir: bool = False,
                 is_global: bool = False):
        super(PolicyViaTime, self).__init__(key, suffix, log_on_train, log_on_eval, log_in_dir, is_global)
        self.players = players
        self.labels = labels
        self.ticks = [int(''.join(filter(lambda i: i.isdigit(), label))) for label in labels]
        self.frames_skip = frames_skip
        self.filenames = []
        self.frame = 0

        self.resolution = None
        for m in get_monitors():
            if m.is_primary:
                self.resolution = (m.width, m.height)
                break

    def on_log(self, data):
        self.frame += 1
        if self.frame % self.frames_skip == 0:
            offends, defends = data
            fig, ax = plt.subplots(self.players, 2, figsize=(self.resolution[0] / 100., self.resolution[1] / 100.),
                                   sharex=True, sharey=True)
            for i, (label, offend, defend) in enumerate(zip(self.labels, offends, defends)):
                ticks = self.ticks[:i] + self.ticks[i + 1:]
                labels = self.labels[:i] + self.labels[i + 1:]
                self._draw(ax[i][0], f'{label} agent offend', offend, ticks, labels)
                self._draw(ax[i][1], f'{label} agent defend', defend, ticks, labels)
            fig.tight_layout()
            filename = os.path.join(self._tmp_dir, f'pvt_{len(self.filenames)}.png')
            self.filenames.append(filename)
            plt.savefig(filename)
            plt.close(fig)

    def _draw(self, ax, title, values, ticks, labels):
        ax.set_title(title)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xticks(ticks, labels)
        ax.bar(ticks, values, color=['b' if i != np.argmax(values) else 'r' for i in range(len(values))])

    def policy_via_time(self):
        fullname = f'{self.prepare_name()}.avi'
        writer = cv.VideoWriter(fullname, cv.VideoWriter_fourcc(*'DIVX'), 2, self.resolution)

        for i, filename in enumerate(self.filenames):
            writer.write(cv.imread(filename))
            os.remove(filename)
        writer.release()

        self.logger.log_artifact(run_id=self._run_id, local_path=fullname, artifact_path=self.dest_dir)
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

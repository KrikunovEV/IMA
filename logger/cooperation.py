import numpy as np
import scipy.special


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

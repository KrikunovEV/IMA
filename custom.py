import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import cv2 as cv

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
        self.temp_dir = f'{self._salt}_pvt'
        os.makedirs(self.temp_dir, exist_ok=True)
        self.video_name = os.path.join(self.temp_dir, f'{self._fullname}.avi')
        self.filenames = []
        self.frame = 0
        self.coops = []  # 0 - nothing, 1 - 1:2, 2 - 2:3, 3 - 1:3

    def on_log(self, data):
        self.frame += 1
        if self.frame % 50 == 0:
            acts, a_policies, d_policies = data

            # pairs
            if acts[0][0] == acts[1][0] and acts[0][1] == acts[1][1]:
                self.coops.append(1)
            elif acts[1][0] == acts[2][0] and acts[1][1] == acts[2][1]:
                self.coops.append(2)
            elif acts[0][0] == acts[2][0] and acts[0][1] == acts[2][1]:
                self.coops.append(3)
            else:
                self.coops.append(0)

            # policies
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
        writer = cv.VideoWriter(self.video_name, cv.VideoWriter_fourcc(*'DIVX'), 15, (1600, 900))

        # create a colored coop line
        w = 10
        coop_line = np.zeros((150, len(self.coops) * w, 3), dtype=np.uint8)
        for i, coop in enumerate(self.coops):
            s = i * w
            e = s + w
            if coop == 1:
                coop_line[:, s:e] = np.array([255, 0, 0], dtype=np.uint8)
            elif coop == 2:
                coop_line[:, s:e] = np.array([0, 0, 255], dtype=np.uint8)
            elif coop == 3:
                coop_line[:, s:e] = np.array([0, 255, 0], dtype=np.uint8)

        white_line = np.full((150, 1600, 3), 255, dtype=np.uint8)
        white_line[:, :coop_line.shape[1]] = coop_line

        # create video
        for i, filename in enumerate(self.filenames):
            image = cv.imread(filename)
            image = cv.resize(image, (1600, 750))
            image = np.concatenate((image, white_line))
            image[-200:-150, i * w: i * w + w] = np.array([0, 0, 0], dtype=np.uint8)

            for _ in range(7):
                writer.write(image)
            os.remove(filename)
        writer.release()

        self._logger.log_artifact(run_id=self._run_id, local_path=self.video_name, artifact_path=self._dest_dir)
        os.remove(self.video_name)
        os.rmdir(self.temp_dir)

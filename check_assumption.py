import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


'''
  c d
c 2 0
d 0 1
'''
if __name__ == '__main__':
    # init = 'c'
    # init = 'd'
    init = 'cd'

    k = 2  # number of actions
    N = 2  # number of agents
    steps = 1000  # number of episodes
    repeats = 10
    eps = 0.1  # e-greedy
    init_value = 10000.

    REWARDS = np.zeros((N, steps), float)
    QS = np.zeros((steps, N, k), float)
    COUNTS = np.zeros((N, k), float)

    for repeat in tqdm(range(repeats)):
        if init == 'c':
            Q = np.array([[init_value, 0.],
                          [init_value, 0.]], float)
        elif init == 'd':
            Q = np.array([[0., init_value],
                          [0., init_value]], float)
        elif init == 'cd':
            Q = np.array([[init_value, 0.],
                          [0., init_value]], float)
        else:
            raise Exception('wrong init')
        Na = np.zeros((N, k), int)

        rewards = [[] for n in range(N)]
        acts = [[] for n in range(N)]
        Qs = []

        for step in range(steps):
            act = []
            for n in range(N):
                p = np.full(k, eps / k)
                p[Q[n].argmax()] += 1 - eps
                a = np.random.choice(k, 1, p=p)[0]
                Na[n, a] += 1
                act.append(a)

            reward = np.zeros(N, float)
            if act[0] == 0 and act[1] == 0:
                reward[:] = 2.
            elif act[0] == 1 and act[1] == 1:
                reward[:] = 1.

            for n, (a, r) in enumerate(zip(act, reward)):
                Q[n, a] += (1 / Na[n, a]) * (r - Q[n, a])
                rewards[n].append(r)
                acts[n].append(a)
            Qs.append(Q.copy())

        for n in range(N):
            rewards[n] = np.cumsum(rewards[n])
            values, counts = np.unique(acts[n], return_counts=True)
            COUNTS[n] += counts
        REWARDS += rewards
        QS += np.array(Qs)

    # fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    #
    # ax[0].set_title('Cumulative reward of players')
    # ax[0].set_xlabel('step')
    # ax[0].set_ylabel('cum. reward')
    # REWARDS /= repeats
    # for n in range(N):
    #     ax[0].plot(REWARDS[n], label=f'agent {n}')
    # ax[0].legend()
    #
    # ax[1].set_title('Acts histograms')
    # ax[1].set_xlabel('action')
    # ax[1].set_ylabel('number of action')
    # COUNTS /= repeats
    # for n in range(N):
    #     offset = 0.2 if n == 1 else -0.2
    #     rects = ax[1].bar(np.array([0, 1]) + offset, COUNTS[n], 0.4, label=f'agent {n}')
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax[1].annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    # ax[1].set_xticks([0, 1])
    # ax[1].set_xticklabels(['D', 'C'])
    # ax[1].legend()
    #
    # ax[2].set_title("players' Q")
    # ax[2].set_xlabel('step')
    # ax[2].set_ylabel('estimated Q')
    # QS /= repeats
    # for n in range(N):
    #     for a in range(k):
    #         l = 'D' if a == 0 else 'C'
    #         ax[2].plot(QS[:, n, a], label=f'agent {n} {l}')
    # ax[2].legend()

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_ylabel('количество действий', size=20)
    COUNTS /= repeats
    for n in range(N):
        offset = 0.2 if n == 1 else -0.2
        rects = ax.bar(np.array([0, 1]) + offset, COUNTS[n], 0.4, label=f'агент {n + 1}')
        ax.bar_label(rects, fmt='%.2f', size=16)
    ax.set_xticks([0, 1], ['кооперация', 'состязание'])
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    ax.set_ylim(0, steps)
    fig.tight_layout()

    plt.savefig(f'{init}.png')
    # plt.show()

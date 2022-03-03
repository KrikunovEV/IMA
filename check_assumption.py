import numpy as np
import matplotlib.pyplot as plt


'''
  d c
d 1 0
c 0 2
'''
if __name__ == '__main__':
    # init = 'c'
    init = 'd'
    # init = 'cd'

    k = 2  # number of actions
    N = 2  # number of agents
    steps = 100000  # number of episodes
    repeats = 10
    eps = 0.1  # e-greedy
    init_value = 0.0001

    REWARDS = np.zeros((N, steps), np.float)
    QS = np.zeros((steps, N, k), np.float)
    COUNTS = np.zeros((N, k), np.float)

    for repeat in range(repeats):
        if init == 'c':
            Q = np.array([[0., init_value],
                          [0., init_value]], np.float)
        elif init == 'd':
            Q = np.array([[init_value, 0.],
                          [init_value, 0.]], np.float)
        elif init == 'cd':
            Q = np.array([[init_value, 0.],
                          [0., init_value]], np.float)
        else:
            raise Exception('wrong init')
        Na = np.zeros((N, k), np.int)

        rewards = [[] for n in range(N)]
        acts = [[] for n in range(N)]
        Qs = []

        for step in range(steps):
            if (step + 1) % 100 == 0:
                print(f'repeat {repeat + 1}/{repeats}, step {step + 1}/{steps}')

            act = []
            for n in range(N):
                p = np.full(k, eps / k)
                p[Q[n].argmax()] += 1 - eps
                a = np.random.choice(k, 1, p=p)[0]
                Na[n, a] += 1
                act.append(a)

            reward = np.zeros(N, np.float)
            if act[0] == 0 and act[1] == 0:
                reward[:] = 1.
            elif act[0] == 1 and act[1] == 1:
                reward[:] = 2.

            for n, (a, r) in enumerate(zip(act, reward)):
                # Q[n, a] += (1 / Na[n, a]) * (r - Q[n, a])
                Q[n, a] += 0.01 * (r - Q[n, a])
                rewards[n].append(r)
                acts[n].append(a)
            Qs.append(Q.copy())

        for n in range(N):
            rewards[n] = np.cumsum(rewards[n])
            values, counts = np.unique(acts[n], return_counts=True)
            COUNTS[n] += counts
        REWARDS += rewards
        QS += np.array(Qs)

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))

    ax[0].set_title('Cumulative reward of players')
    ax[0].set_xlabel('step')
    ax[0].set_ylabel('cum. reward')
    REWARDS /= repeats
    for n in range(N):
        ax[0].plot(REWARDS[n], label=f'agent {n}')
    ax[0].legend()

    ax[1].set_title('Acts histograms')
    ax[1].set_xlabel('action')
    ax[1].set_ylabel('number of action')
    COUNTS /= repeats
    for n in range(N):
        offset = 0.2 if n == 1 else -0.2
        rects = ax[1].bar(np.array([0, 1]) + offset, COUNTS[n], 0.4, label=f'agent {n}')
        for rect in rects:
            height = rect.get_height()
            ax[1].annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['D', 'C'])
    ax[1].legend()

    ax[2].set_title("players' Q")
    ax[2].set_xlabel('step')
    ax[2].set_ylabel('estimated Q')
    QS /= repeats
    for n in range(N):
        for a in range(k):
            l = 'D' if a == 0 else 'C'
            ax[2].plot(QS[:, n, a], label=f'agent {n} {l}')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

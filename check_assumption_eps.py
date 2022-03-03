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
    steps = 1000  # number of episodes
    repeats = 100
    init_value = 0.001
    epss = [0.1 * i for i in range(1, 10)]

    REWARDS_EPS = []
    QS_EPS = []
    COUNTS_EPS = []

    for eps in epss:
        QS = np.zeros((steps, N, k), np.float)
        COUNTS = np.zeros((N, k), np.float)
        REWARDS = np.zeros(N, np.float)

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
                    print(f'eps {eps} repeat {repeat + 1}/{repeats}, step {step + 1}/{steps}')

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
                REWARDS[n] += rewards[n][steps - 1]
            QS += np.array(Qs)

        QS_EPS.append(QS[-1] / repeats)
        COUNTS_EPS.append(COUNTS / repeats)
        REWARDS_EPS.append(REWARDS / repeats)

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))

    ax[0].set_title('rewards')
    ax[0].set_xlabel('eps')
    ax[0].set_ylabel('reward')
    REWARDS_EPS = np.array(REWARDS_EPS)
    for n in range(N):
        ax[0].plot(epss, REWARDS_EPS[:, n], label=f'agent {n}')
    ax[0].legend()

    ax[1].set_title('actions')
    ax[1].set_xlabel('eps')
    ax[1].set_ylabel('acts')
    COUNTS_EPS = np.array(COUNTS_EPS)
    for n in range(N):
        for a in range(k):
            l = 'D' if a == 0 else 'C'
            ax[1].plot(epss, COUNTS_EPS[:, n, a], label=f'agent {n} {l}')
    ax[1].legend()

    ax[2].set_title("players' Q")
    ax[2].set_xlabel('eps')
    ax[2].set_ylabel('estimated Q')
    QS_EPS = np.array(QS_EPS)
    for n in range(N):
        for a in range(k):
            l = 'D' if a == 0 else 'C'
            ax[2].plot(epss, QS_EPS[:, n, a], label=f'agent {n} {l}')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

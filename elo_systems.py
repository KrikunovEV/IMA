import numpy as np


class IZeroSumEloSystem:
    K: float = 20.
    BASE: float = 10.
    DEL: float = 400.
    INITIAL: int = 1000

    def __init__(self, players: int):
        self.players = players
        self.elo = None
        self.reset()

    def reset(self):
        self.elo = np.full(self.players, self.INITIAL, dtype=np.int) + np.random.randint(10, size=self.players)
        return self.elo

    def step(self, rewards):
        """ rewards.sum() = 0 """
        raise NotImplementedError

    def _update_elo(self, rewards, estimates):
        """ estimates are: VERSUS_PLAYER_ELO (estimated in multiplayer) - CURRENT_PLAYER_ELO """
        # sigmoid
        estimates = 1. / (1. + self.BASE ** (estimates / self.DEL))
        # S - E
        scores = np.zeros(self.players)
        scores[np.array(rewards) > 0.] = 1.
        estimates = scores - estimates
        # compute rating
        self.elo = self.elo + (self.K * estimates).astype(np.int)


class MeanEloI(IZeroSumEloSystem):

    def __init__(self, players: int):
        super(MeanEloI, self).__init__(players)

    def step(self, rewards):
        # Use mean elo of other players to correct your elo
        #elo_sum = np.sum(self.elo)
        estimates = ((np.sum(self.elo) - self.elo) / (self.players - 1)) - self.elo  # mean R other - R current

        self._update_elo(rewards, estimates)
        return self.elo


class SMEloI(IZeroSumEloSystem):
    # http://www.tckerrigan.com/Misc/Multiplayer_Elo/
    # Simple Multiplayer Elo (SME)

    def __init__(self, players: int):
        super(SMEloI, self).__init__(players)

    def step(self, rewards):
        # Use right or left player to correct your elo
        argind = np.argsort(self.elo)
        estimates = -self.elo
        for i in range(self.players):
            cur_player = argind[i]
            if rewards[cur_player] > 0.:
                vs_player = cur_player if i == 0 else argind[i - 1]
            else:
                vs_player = cur_player if i == (self.players - 1) else argind[i + 1]
            estimates[cur_player] += estimates[vs_player]

        self._update_elo(rewards, estimates)
        return self.elo


if __name__ == '__main__':
    players = 3
    episodes = 100
    X = np.arange(episodes + 1)

    elo_systems = [SMEloI(players), MeanEloI(players)]

    p1_win = np.array([1., -0.5, -0.5])
    p1p2_win = np.array([0.5, 0.5, -1.])
    no_win = np.array([0., 0., 0.])
    all_win = np.array([0.33, 0.33, 0.33])

    import matplotlib.pyplot as plt
    for elo_system in elo_systems:

        # only one player win all the time
        eloes = []
        eloes.append(elo_system.reset())
        for _ in range(episodes):
            eloes.append(elo_system.step(p1_win))
        eloes = np.array(eloes)
        plt.title(f'{type(elo_system).__name__} p1 win all the time')
        plt.xlabel('episode')
        plt.ylabel('elo')
        plt.plot(X, eloes[:, 0], label='p1')
        plt.plot(X, eloes[:, 1], label='p2')
        plt.plot(X, eloes[:, 2], label='p3')
        plt.legend()
        plt.show()

        # only p1 and p2 win all the time
        eloes = []
        eloes.append(elo_system.reset())
        for _ in range(episodes):
            eloes.append(elo_system.step(p1p2_win))
        eloes = np.array(eloes)
        plt.title(f'{type(elo_system).__name__} p1 and p2 win all the time')
        plt.xlabel('episode')
        plt.ylabel('elo')
        plt.plot(X, eloes[:, 0], label='p1')
        plt.plot(X, eloes[:, 1], label='p2')
        plt.plot(X, eloes[:, 2], label='p3')
        plt.legend()
        plt.show()

        # there is no winner
        eloes = []
        eloes.append(elo_system.reset())
        for _ in range(episodes):
            eloes.append(elo_system.step(no_win))
        eloes = np.array(eloes)
        plt.title(f'{type(elo_system).__name__} no winner')
        plt.xlabel('episode')
        plt.ylabel('elo')
        plt.plot(X, eloes[:, 0], label='p1')
        plt.plot(X, eloes[:, 1], label='p2')
        plt.plot(X, eloes[:, 2], label='p3')
        plt.legend()
        plt.show()

        # all are winners
        eloes = []
        eloes.append(elo_system.reset())
        for _ in range(episodes):
            eloes.append(elo_system.step(all_win))
        eloes = np.array(eloes)
        plt.title(f'{type(elo_system).__name__} all are winners')
        plt.xlabel('episode')
        plt.ylabel('elo')
        plt.plot(X, eloes[:, 0], label='p1')
        plt.plot(X, eloes[:, 1], label='p2')
        plt.plot(X, eloes[:, 2], label='p3')
        plt.legend()
        plt.show()

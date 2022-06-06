import gym
import numpy as np


class Env(gym.Env):
    """
    Игроки должны выбрать кого атаковать и от кого защищаться.
    """
    OFFEND_ID: int = 0
    DEFEND_ID: int = 1
    WIN_REWARD: float = 1.
    LOSS_REWARD: float = -1.

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug

        # gym specific
        self.observation_space = players * 2 * (players + 1)  # one-hot encoded vectors
        self.action_space = players  # attack/offend

    def reset(self):
        obs = np.zeros((self.players, 2 * (self.players + 1)), dtype=np.float32)
        obs[:, (self.players, 2 * self.players + 1)] = 1.
        return obs

    def step(self, action: list):
        """
        Нам нужно лишь проверить, что на агента не было совершенно успешное нападение.
        Если он не смог защититься, то reward[agent_id] = -1
        """
        self._print('\nНовый раунд: ')
        rewards = np.full(self.players, self.WIN_REWARD)
        for offender in range(self.players):
            defender = action[offender][self.OFFEND_ID]
            if action[defender][self.DEFEND_ID] != offender:
                rewards[defender] = self.LOSS_REWARD
                self._print(f'{offender} напал на {defender} (не защитился)')
            else:
                self._print(f'{offender} напал на {defender} (защитился)')

        obs = np.zeros((self.players, 2 * (self.players + 1)), dtype=np.float32)
        for p in range(self.players):
            obs[p][action[p][self.OFFEND_ID]] = 1.
            obs[p][self.players + 1 + action[p][self.DEFEND_ID]] = 1.

        # zero sum
        winners_mask = rewards == self.WIN_REWARD
        total_win_reward = np.sum(winners_mask)
        if total_win_reward != 0:
            rewards[winners_mask] = 1. / total_win_reward

        loosers_mask = rewards == self.LOSS_REWARD
        total_loose_reward = np.sum(loosers_mask)
        if total_loose_reward != 0:
            rewards[loosers_mask] = -1. / total_loose_reward

        self._print(f'Награды {rewards}')

        return obs, rewards

    def render(self, mode="human"):
        pass

    def _print(self, text):
        if self.debug:
            print(text)


if __name__ == '__main__':
    env = OADEnv(players=3, debug=True)
    print(env.reset())
    print(env.step([[1, 2], [0, 0], [0, 0]]))
    print(env.step([[0, 0], [1, 1], [2, 2]]))

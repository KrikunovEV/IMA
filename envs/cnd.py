import gym
import numpy as np


class Env(gym.Env):
    """
    Iterated Prisoners' Dilemma
      c d
    c 2 0
    d 0 1
    """
    COOPER_ID: int = 0
    DEFECT_ID: int = 1
    C_REWARD: float = 2.
    D_REWARD: float = 1.
    CD_REWARD: float = 0.

    def __init__(self, players: int, debug: bool = False):
        if players != 2:
            raise Exception(f'In IPD environment only 2 players are acceptable, currently {players}')
        self.players = players
        self.debug = debug

        # gym specific
        self.observation_space = 3 * 2  # one-hot encoded vectors
        self.action_space = 2  # attack/offend

    def reset(self):
        obs = np.zeros((2, 3), dtype=np.float32)
        obs[:, -1] = 1.
        return obs

    def step(self, action: list):
        self._print('\nНовый раунд: ')

        r = self.CD_REWARD
        if action[0] == action[1]:
            if action[0] == self.DEFECT_ID:
                r = self.D_REWARD
            elif action[0] == self.COOPER_ID:
                r = self.C_REWARD
            else:
                raise Exception(f'wrong action {action[0]}')

        rewards = np.full(2, r)

        obs = np.zeros((2, 3), dtype=np.float32)
        obs[0, action[0]] = 1.
        obs[1, action[1]] = 1.

        self._print(f'Награды {rewards}')

        return obs, rewards

    def render(self, mode="human"):
        pass

    def _print(self, text):
        if self.debug:
            print(text)


if __name__ == '__main__':
    env = CNDEnv(players=2, debug=True)
    print(env.reset())
    print(env.step([0, 0]))
    print(env.step([1, 1]))
    print(env.step([1, 0]))

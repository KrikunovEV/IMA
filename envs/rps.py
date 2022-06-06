import gym
import numpy as np
from typing import List


class Env(gym.Env):
    ROCK: int = 0
    PAPER: int = 1
    SCISSORS: int = 2
    ACTIONS: list = ['ROCK', 'PAPER', 'SCISSORS']
    WIN_REWARD: float = 1.
    LOOSE_REWARD: float = -1.

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug

        # gym specific
        self.observation_space = 4 * players  # one-hot encoded vectors
        self.action_space = 3  # rock/paper/scissors

    def reset(self):
        obs = np.zeros((self.players, 4), dtype=np.float32)
        obs[:, -1] = 1.
        return obs

    def step(self, action: List[int]):
        self._print('\nНовый раунд: ')

        choices = np.array(action)
        unique_choices = np.unique(choices)
        for choice in unique_choices:
            if choice < self.ROCK or choice > self.SCISSORS:
                raise ValueError(f'Wrong action value {choice}')
        rewards = np.zeros(self.players)

        '''
        Победители существуют, если уникальных значений два.
        Если уникальных значений одно, то все игроки выбрали одно действие.
        Если уникальных значений три, то победитель не может быть определён.
        '''
        if len(unique_choices) == 2:
            option1 = unique_choices[0]
            option2 = unique_choices[1]

            '''
            Для определённости делаем так, что option1 содержит наименьшее значениe
            Тогда разница между option2 и option1 может гарантированно быть либо 1, либо 2:
            1: ROCK vs PAPER (winners)
               PAPER vs SCISSOR (winners)
               Победители те, кто выбрал option2
            2: ROCK (winners) vs SCISSOR
               Победители те, кто выбрал option1
            '''
            if option1 > option2:
                option1, option2 = option2, option1

            option_win = option1
            option_loose = option2
            if option2 - option1 == 1:
                option_win = option2
                option_loose = option1

            # zero sum
            winners = np.where(choices == option_win)[0]
            loosers = np.where(choices == option_loose)[0]
            rewards[winners] = self.WIN_REWARD / winners.shape[0]
            rewards[loosers] = self.LOOSE_REWARD / loosers.shape[0]

            self._print(f'Игроки {winners} выиграли ({self.ACTIONS[option_win]})')
            self._print(f'Игроки {loosers} проиграли ({self.ACTIONS[option_loose]})')

        else:
            self._print('Ничья')

        obs = np.zeros((self.players, 4), dtype=np.float32)
        obs[np.arange(self.players), choices] = 1.

        self._print(f'Награды {rewards}')

        return obs, rewards

    def render(self, mode="human"):
        pass

    def _print(self, text):
        if self.debug:
            print(text)


if __name__ == '__main__':
    env = RPSEnv(players=3, debug=True)
    print(env.reset())
    print(env.step([0, 2, 2]))
    print(env.step([0, 1, 2]))
    print(env.step([0, 0, 0]))

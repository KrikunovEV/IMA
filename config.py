from dataclasses import dataclass, asdict, field, replace
from typing import Union
import torch


@dataclass(init=True, frozen=True)
class Config:
    # common
    players: int
    epochs: int

    # train
    train_episodes: int
    gamma: float

    # exploration
    eps_high: float
    eps_low: float
    eps_episodes_ratio: float

    # optimizer
    lr: Union[float, tuple]

    # test
    test_episodes: int

    # recurrent
    h_space: Union[int, tuple]
    window: Union[int, tuple]

    # memory
    capacity: int
    no_learn_episodes: int

    # keys for logging dictionary
    actions_key: str
    offend_policy_key: str
    defend_policy_key: str

    # add later
    # neg_players: int
    # h_space: int
    # dk: int

    # pre-defined
    cores: int = 1
    seed: int = None
    repeats: int = 1

    # has to be post-initialized
    device: torch.device = field(init=False)
    eps_decay: float = field(init=False)

    def __post_init__(self):
        self.set('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        eps_decay = (self.eps_high - self.eps_low) / (self.epochs * self.train_episodes * self.eps_episodes_ratio)
        self.set('eps_decay', eps_decay)

    def set(self, param: str, value):
        if param in self.__annotations__.keys():
            object.__setattr__(self, param, value)
        else:
            raise AttributeError(f'Config does not contain \'{param}\' parameter!')

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def init():
        base = Config(
            # common
            players=5,
            epochs=10,
            # train
            train_episodes=1000,
            gamma=0.99,
            # exploration
            eps_high=0.9,
            eps_low=0.05,
            eps_episodes_ratio=0.8,
            # optimizer
            lr=0.005,
            # test
            test_episodes=100,
            # recurrent
            h_space=32,
            window=12,
            # memory
            capacity=10000,
            no_learn_episodes=100,
            actions_key='acts',
            offend_policy_key='offend_policy',
            defend_policy_key='defend_policy'
        )

        configs = {'base': base}
        while True:

            # check that all values are unpacked
            unpacking = False
            for config in configs.values():
                for value in config.as_dict().values():
                    if isinstance(value, tuple):
                        unpacking = True
                        break
                if unpacking:
                    break
            if not unpacking:
                break

            # if found, unpack ony first entry of tuple data
            unpacked_configs = dict()
            for name, config in configs.items():
                for key, value in config.as_dict().items():
                    if isinstance(value, tuple):
                        for v in value:
                            _config = replace(config)
                            _config.set(str(key), v)
                            new_name = f'{key}={v}' if name == 'base' else f'{name} {key}={v}'
                            unpacked_configs[new_name] = _config
                        break
            configs = unpacked_configs

        return configs


if __name__ == '__main__':
    cfg = Config.init()
    print(len(cfg))
    for name, c in cfg.items():
        print(name, c.as_dict())

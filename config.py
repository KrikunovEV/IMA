from dataclasses import dataclass, asdict, field, replace
from typing import Union
import torch


@dataclass(init=True, frozen=True)
class Config:
    # common
    players: int = 3
    epochs: int = 2
    cores: int = 1
    repeats: int = 1
    seed: int = None
    algorithm: Union[str, tuple] = 'q_learning'

    # train
    train_episodes: int = 100
    gamma: float = 0.99

    # exploration
    eps_high: float = 0.9
    eps_low: float = 0.05
    eps_episodes_ratio: float = 0.8

    # optimizer
    lr: Union[float, tuple] = 0.005

    # test
    test_episodes: int = 100

    # recurrent
    h_space: Union[int, tuple] = 32
    window: Union[int, tuple] = 10

    # memory
    capacity: int = 5000
    no_learn_episodes: int = 100

    # keys for logging
    actions_key: str = 'acts'
    reward_key: str = 'reward'
    elo_key: str = 'elo'
    loss_key: str = 'loss'
    eps_key: str = 'eps'
    offend_policy_key: str = 'offend_policy'
    defend_policy_key: str = 'defend_policy'
    pvt_key: str = 'pvt'

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
        configs = {'base': Config()}

        while True:
            # unpack only the first entry of tuple data
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

            if len(unpacked_configs) == 0:
                break

            configs = unpacked_configs

        return configs


if __name__ == '__main__':
    cfg = Config.init()
    print(len(cfg))
    for name, c in cfg.items():
        print(name, c.as_dict())

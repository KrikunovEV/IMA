from dataclasses import dataclass, asdict, field
import torch


@dataclass(init=True, frozen=True)
class Config:
    # common
    games: int
    players: int
    neg_players: int
    epochs: int
    h_space: int
    dk: int
    # train
    lr: float
    train_episodes: int
    eps_high: float
    eps_low: float
    eps_episodes: int
    gamma: float
    # test
    test_episodes: int
    # pre-defined
    cores: int = None
    seed: int = None

    device: torch.device = field(init=False)
    eps_step: float = field(init=False)

    def __post_init__(self):
        self.set('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.set('eps_step', (self.eps_high - self.eps_low) / self.eps_episodes)
        self.set('cores', 2)

    def set(self, param: str, value):
        if param in self.__annotations__.keys():
            object.__setattr__(self, param, value)
        else:
            raise AttributeError(f'Config does not contain \'{param}\' parameter!')

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def init():
        return Config(
            # common
            games=100,
            players=3,
            neg_players=2,
            epochs=200,
            h_space=32,
            dk=64,
            # train
            lr=0.01,
            train_episodes=100,
            eps_high=0.5,
            eps_low=0.01,
            eps_episodes=15000,
            gamma=0.99,
            # test
            test_episodes=100
        )


if __name__ == '__main__':
    cfg = Config.init()
    print(cfg)
    print(cfg.as_dict())

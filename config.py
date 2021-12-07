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
    eps_episodes_ratio: float
    gamma: float
    # test
    test_episodes: int
    # dqn
    dqn_batch_size: int
    dqn_memory_cap: int
    # pre-defined
    cores: int = None
    seed: int = None

    device: torch.device = field(init=False)
    eps_decay: float = field(init=False)

    def __post_init__(self):
        self.set('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        eps_decay = (self.eps_high - self.eps_low) / (self.epochs * self.train_episodes * self.eps_episodes_ratio)
        self.set('eps_decay', eps_decay)
        self.set('cores', 1)

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
            games=1,
            players=3,
            neg_players=0,
            epochs=100,
            h_space=32,
            dk=64,
            # train
            lr=0.01,
            train_episodes=16,
            eps_high=0.9,
            eps_low=0.05,
            eps_episodes_ratio=0.8,
            gamma=0.99,
            # test
            test_episodes=100,
            # dqn
            dqn_batch_size=32,
            dqn_memory_cap=10000,
        )


if __name__ == '__main__':
    cfg = Config.init()
    print(cfg)
    print(cfg.as_dict())

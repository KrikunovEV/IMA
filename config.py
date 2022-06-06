from dataclasses import dataclass, asdict, field, replace
from typing import Union
import torch


@dataclass(init=True, frozen=True)
class Config:
    # common
    agents: list = field(default_factory=lambda: ['a2c_mod', 'a2c'])
    environment: str = 'rps'
    experiment_name: str = 'rps_1vs1'
    epochs: int = 1
    repeats: int = 1
    seed: int = None

    # train
    train_episodes: int = 50000
    discount: float = 0.95
    steps: int = 20  # TD update
    h_space: Union[int, tuple] = 64
    obs_capacity: int = 5
    prob_thr: float = 0.00001
    entropy: float = 0.01
    e_greedy: float = 0.05
    lr: Union[float, tuple] = 0.0025
    grad_clip: float = 10.
    # advanced
    n_models: int = 3
    ema_alpha: float = 0.0025
    sd_capacity: int = 200
    sd_thr: float = 0.006
    ope_window: int = 100
    do_ope_on_inference: bool = False
    ope_reward_eps: float = 0.1
    # negotiation
    enable_negotiation: bool = False
    m_space: int = 16
    n_steps: int = 1
    n_agents: int = 2

    # test
    test_episodes: int = 10000
    enable_eval_agents: bool = True
    eval_agents: list = field(default_factory=lambda: ['a2c'])

    # keys for logging
    loss_key: str = 'loss'
    act_loss_key: str = 'actor_loss'
    crt_loss_key: str = 'critic_loss'
    actions_key: str = 'acts'
    reward_key: str = 'reward'
    elo_key: str = 'elo'
    log_avg: int = 5

    # has to be post-initialized
    device: torch.device = field(init=False)
    players: int = field(init=False)
    eval_players: int = field(init=False)

    def __post_init__(self):
        self.set('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.set('players', len(self.agents))
        self.set('eval_players', len(self.eval_agents))
        self._check_conflicts()

    def set(self, param: str, value):
        if param in self.__annotations__.keys():
            object.__setattr__(self, param, value)
        else:
            raise AttributeError(f'Config does not contain \'{param}\' parameter!')

    def as_dict(self):
        return asdict(self)

    def _check_conflicts(self):
        if len(self.agents) != self.players:
            raise Exception(f'# of agents ({len(self.agents)}) must be same as # of players ({self.players})')
        if self.ema_alpha < 0. or self.ema_alpha >= 1.:
            raise Exception(f'ema_alpha {self.ema_alpha} must be in range [0; 1)')
        if self.players < self.n_agents:
            raise Exception(f'negotiation agents ({self.n_agents}) must be less or equal than players ({self.players})')

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

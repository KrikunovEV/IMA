import torch.nn as nn

from config import Config


class Comm(nn.Module):

    def __init__(self, cfg: Config):
        super(Comm, self).__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(nn.Linear(cfg.m_space * cfg.n_agents, cfg.m_space),
                                 nn.LeakyReLU(),
                                 nn.Linear(cfg.m_space, cfg.m_space))

    def forward(self, obs):
        return self.mlp(obs)

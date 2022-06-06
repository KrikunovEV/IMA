import torch.nn as nn

from config import Config


class Core(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config, negotiable: bool = False):
        super(Core, self).__init__()
        self.cfg = cfg

        in_space = o_space * cfg.obs_capacity
        if cfg.enable_negotiation and negotiable:
            in_space += cfg.m_space * cfg.n_agents
        self.mlp = nn.Sequential(nn.Linear(in_space, cfg.h_space),
                                 nn.LeakyReLU(),
                                 nn.Linear(cfg.h_space, cfg.h_space))
        self.policy = nn.Linear(cfg.h_space, a_space)
        self.policy2 = nn.Linear(cfg.h_space, a_space)
        self.value = nn.Linear(cfg.h_space, 1)

    def forward(self, obs):
        obs = self.mlp(obs)
        logits = self.policy(obs)
        logits2 = self.policy2(obs)
        v = self.value(obs.detach())
        return logits, logits2, v

import torch
import torch.nn as nn
import torch.nn.functional as functional

from config import Config


class Core(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config):
        super(Core, self).__init__()
        self.cfg = cfg
        self.h = None

        self.rnn = nn.GRUCell(o_space, cfg.h_space)
        self.fc1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.h_space, a_space),
        )
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.h_space, a_space),
        )

    def forward(self, o):
        self.h = self.rnn(o, self.h)
        a_logits = self.fc1(self.h)
        d_logits = self.fc2(self.h)
        return a_logits.squeeze(), d_logits.squeeze()

    def reset(self):
        self.h = torch.zeros((1, self.cfg.h_space), device=self.cfg.device)

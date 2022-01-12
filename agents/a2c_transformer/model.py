import torch
import torch.nn as nn
import torch.nn.functional as functional

from config import Config


class Core(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config):
        super(Core, self).__init__()
        self.cfg = cfg

        var = 2. / (5. * cfg.window)
        self.WQ = nn.Parameter(torch.normal(0, var, (o_space, cfg.dk), requires_grad=True))
        self.WK = nn.Parameter(torch.normal(0, var, (o_space, cfg.dk), requires_grad=True))
        self.WV = nn.Parameter(torch.normal(0, var, (o_space, cfg.dk), requires_grad=True))
        self.WO = nn.Parameter(torch.normal(0, var, (cfg.dk, cfg.window), requires_grad=True))
        # тут проверить, obs @ WQ
        self.scale = torch.sqrt(torch.tensor(cfg.dk, device=cfg.device))

        self.o_policy = nn.Linear(cfg.h_space, a_space)
        self.d_policy = nn.Linear(cfg.h_space, a_space)
        self.value = nn.Linear(cfg.h_space, 1)

    def forward(self, obs):
        print(obs)
        exit()

        # o_logits = self.o_policy(self.h).squeeze()
        # d_logits = self.d_policy(self.h).squeeze()
        # v = self.value(self.h.detach()).squeeze()
        # return o_logits, d_logits, v
        return None


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.scale = torch.sqrt(torch.Tensor([cfg.dk]).to(device=cfg.device))

    def forward(self, my_q, q):
        O = torch.matmul(q, my_q)
        O = torch.div(O, self.scale)
        O = functional.softmax(O, dim=-1)
        O = torch.matmul(O, q)
        return O

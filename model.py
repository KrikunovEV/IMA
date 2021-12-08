import torch
import torch.nn as nn
import torch.nn.functional as functional

from config import Config


class QAttention(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config):
        super(QAttention, self).__init__()
        self.cfg = cfg
        self.h = None

        self.rnn = nn.Sequential(
            nn.GRUCell(o_space, cfg.h_space),
            nn.LeakyReLU()
        )

        self.o_values = nn.Linear(cfg.h_space, a_space)
        self.d_values = nn.Linear(cfg.h_space, a_space)

    def forward(self, o):
        self.h = self.rnn(o, self.h)
        o_logits = self.o_policy(self.h)
        d_logits = self.d_policy(self.h)
        return o_logits.squeeze(), d_logits.squeeze()

    def reset(self):
        self.h = torch.zeros((1, self.cfg.h_space), device=self.cfg.device)


class Core(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config, negotiable: bool = False):
        super(Core, self).__init__()
        self.cfg = cfg
        self.h = None
        self.negotiable = negotiable

        if negotiable:
            o_space += cfg.dk
        self.rnn = nn.GRUCell(o_space, cfg.h_space)

        self.o_policy = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.h_space, a_space),
        )

        self.d_policy = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.h_space, a_space),
        )

        self.value = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.h_space, 1)
        )

    def forward(self, o, m=None):
        if self.negotiable:
            if m is None:
                raise Exception('\'m\' has not be None')
            o = torch.cat((o, m.unsqueeze(0)), dim=-1)
        self.h = self.rnn(o, self.h)
        a_logits = self.o_policy(self.h)
        d_logits = self.d_policy(self.h)
        v = self.value(self.h.detach())
        return a_logits.squeeze(), d_logits.squeeze(), v.squeeze()

    def reset(self):
        self.h = torch.zeros((1, self.cfg.h_space), device=self.cfg.device)


class Attention(nn.Module):
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


class Negotiation(Core):

    def __init__(self, o_space: int, a_space: int, cfg: Config):
        super(Negotiation, self).__init__(o_space, a_space, cfg, negotiable=True)

        self.fc_message = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.h_space, cfg.dk)
        )

        self.attention = Attention(cfg)

    def message(self):
        return self.fc_message(self.h).squeeze()

    def negotiate(self, my_q, q):
        return self.attention(my_q, q)


if __name__ == '__main__':
    from config import Config
    model = Negotiation(5, 5, Config.init())
    model.reset()
    my_q = model.message()
    # twice self-att
    m = model.negotiate(my_q, torch.stack((my_q, my_q)))
    o = model(torch.zeros(1, 5), m)
    print(o)

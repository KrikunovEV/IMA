import torch
import torch.nn as nn

from config import Config


class Core(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config):
        super(Core, self).__init__()
        self.cfg = cfg

        self.rnn = nn.GRUCell(o_space, cfg.h_space)
        self.o_values = nn.Linear(cfg.h_space, a_space)
        self.d_values = nn.Linear(cfg.h_space, a_space)

    def forward(self, obs):
        h = [torch.zeros((1, self.cfg.h_space), device=self.cfg.device)]
        for o in obs:
            h_next = self.rnn(o.unsqueeze(0), h[-1])
            h.append(h_next)
        h = torch.cat(h[1:])
        o_logits = self.o_values(h)
        d_logits = self.d_values(h)
        return o_logits, d_logits


if __name__ == '__main__':
    from config import Config
    cfg = Config.init()['base']
    model = Core(5, 3, cfg)
    obs = torch.ones((cfg.window, 5), device=cfg.device)
    print(model(obs))

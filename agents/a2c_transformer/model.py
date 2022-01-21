import torch
import torch.nn as nn
import torch.nn.functional as functional

from config import Config


class Core(nn.Module):

    def __init__(self, o_space: int, a_space: int, cfg: Config):
        super(Core, self).__init__()
        self.cfg = cfg
        self.dmodel = cfg.dmodel * 2 * cfg.players

        # Embeddings
        self.o_embs = nn.Embedding(num_embeddings=cfg.players + 1, embedding_dim=cfg.dmodel)
        self.d_embs = nn.Embedding(num_embeddings=cfg.players + 1, embedding_dim=cfg.dmodel)
        self.emb_scale = torch.sqrt(torch.tensor(cfg.dmodel, device=cfg.device))

        # Positional encoding
        i = 1. / torch.pow(10000., torch.arange(0, self.dmodel, 2, dtype=torch.float) / self.dmodel)
        pos = torch.arange(cfg.window, dtype=torch.float).unsqueeze(1)
        self.pe = torch.zeros((cfg.window, self.dmodel)).cuda()
        self.pe[:, ::2] = torch.sin(pos * i)
        self.pe[:, 1::2] = torch.cos(pos * i)

        # SDPA head
        var = 2. / (5. * self.dmodel)
        self.WQ = nn.Parameter(torch.normal(0, var, (self.dmodel, cfg.dk), requires_grad=True))
        self.WK = nn.Parameter(torch.normal(0, var, (self.dmodel, cfg.dk), requires_grad=True))
        self.WV = nn.Parameter(torch.normal(0, var, (self.dmodel, cfg.dk), requires_grad=True))
        self.WO = nn.Parameter(torch.normal(0, var, (cfg.dk, cfg.dk), requires_grad=True))
        self.att_scale = torch.sqrt(torch.tensor(cfg.dk, device=cfg.device))
        self.relu = nn.ReLU(inplace=True)

        # Policies
        self.o_policy = nn.Linear(cfg.dk, a_space)
        self.d_policy = nn.Linear(cfg.dk, a_space)
        self.value = nn.Linear(cfg.dk, 1)

    def forward(self, obs):
        obs = obs.reshape(-1, self.cfg.players + 1)
        obs = torch.cat(
            (self.o_embs(torch.where(obs[::2] == 1)[1]),
             self.d_embs(torch.where(obs[1::2] == 1)[1])),
            dim=1).reshape(-1, self.dmodel) * self.emb_scale
        obs = obs + self.pe[:obs.shape[0]]

        q = torch.matmul(obs, self.WQ)
        k = torch.matmul(obs, self.WK)
        v = torch.matmul(obs, self.WV)

        o = torch.matmul(q, k.T)
        o = torch.div(o, self.att_scale)
        o = functional.softmax(o, dim=-1)
        o = torch.matmul(o, v)
        o = torch.matmul(o, self.WO)
        o = self.relu(o[-1])

        o_logits = self.o_policy(o)
        d_logits = self.d_policy(o)
        v = self.value(o)
        return o_logits, d_logits, v

import torch
import torch.nn as nn
import torch.optim as optim
import random

from model import CoreDQN
from config import Config
from rl import Uniform, Argmax
from logger import RunLogger


class AgentDQN:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.logger = None
        self.q_actual = []
        self.q_target = []
        self.reward = []

        self.negotiable = True if id < cfg.neg_players else False
        self.label = f'{id + 1}' + ('n' if self.negotiable else '')
        self.train = None
        self.eps = cfg.eps_high

        self.model_actual = CoreDQN(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target = CoreDQN(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target.load_state_dict(self.model_actual.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_actual.parameters(), lr=cfg.lr)
        self.loss = nn.SmoothL1Loss()
        self.act_uniform = Uniform()
        self.act_argmax = Argmax()

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def set_mode(self, train: bool):
        self.train = train
        self.model_actual.train(train)

    def reset_memory(self):
        self.model_actual.reset()
        self.model_target.reset()

    def act(self, obs: torch.Tensor):
        o_logits, d_logits = self.model_actual(obs)
        if self.train and random.random() < self.eps:
            o_action = self.act_uniform(policy=o_logits)
            d_action = self.act_uniform(policy=d_logits)
        else:
            o_action = self.act_argmax(policy=o_logits)
            d_action = self.act_argmax(policy=d_logits)

        if self.train:
            self.q_actual.append([o_logits[o_action], d_logits[d_action]])
            self.logger.log({f'{self.label}_eps': self.eps})
            if self.eps > self.cfg.eps_low:
                self.eps -= self.cfg.eps_decay
                if self.eps < self.cfg.eps_low:
                    self.eps = self.cfg.eps_low

        return [o_action, d_action]

    def rewarding(self, reward, next_o):
        self.logger.log({f'{self.label}_reward': reward})
        if self.train:
            o_logits, d_logits = self.model_target(next_o)
            o_action = self.act_argmax(policy=o_logits)
            d_action = self.act_argmax(policy=d_logits)
            self.q_target.append([o_logits[o_action], d_logits[d_action]])
            self.reward.append(reward)

    def learn(self):
        o_q = torch.cat(self.q_actual[:, 0])
        d_q = torch.cat(self.q_actual[:, 1])
        rewards = torch.cat(self.reward)
        o_disc = torch.cat(self.q_target[:, 0]) * self.cfg.gamma + rewards
        d_disc = torch.cat(self.q_target[:, 1]) * self.cfg.gamma + rewards
        loss = self.loss(o_q, o_disc)


        self.optimizer.step()
        self.optimizer.zero_grad()
        self.logger.log({f'{self.label}_loss': _loss.item()})

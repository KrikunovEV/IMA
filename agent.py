import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np

from model import Core, Negotiation
from config import Config
from rl import Reinforce, A2C, Choice, Uniform
from logger import RunLogger


class Agent:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.logger = None

        self.negotiable = True if id < cfg.neg_players else False
        self.label = f'{id + 1}' + ('n' if self.negotiable else '')
        self.train = None
        self.log_p = None
        self.v = None
        self.eps = cfg.eps_high

        if self.negotiable:
            self.model = Negotiation(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        else:
            self.model = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=cfg.lr)
        self.loss = A2C(gamma=cfg.gamma)
        self.act_choice = Choice()
        self.act_uniform = Uniform()

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def set_mode(self, train: bool):
        self.train = train
        self.model.train(train)

    def reset_memory(self):
        self.model.reset()

    def message(self):
        if self.negotiable:
            return {f'{self.id}': self.model.message()}
        else:
            return dict()

    def negotiate(self, my_q, q):
        if self.negotiable:
            return self.model.negotiate(my_q, q)

    def act(self, obs: torch.Tensor, m: torch.Tensor):
        a_logits, d_logits, v = self.model(obs, m)
        a_policy = functional.softmax(a_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        if self.train and self.act_choice(policy=torch.Tensor([self.eps, 1 - self.eps])) == 0:
            a_action = self.act_uniform(policy=a_policy)
            d_action = self.act_uniform(policy=d_policy)
        else:
            a_action = self.act_choice(policy=a_policy)
            d_action = self.act_choice(policy=d_policy)

        if self.train:
            prob = a_policy[a_action] * d_policy[d_action]
            if prob < 0.000001:
                prob = prob + 0.000001
            self.log_p = torch.log(prob)
            self.v = v

            self.logger.log({f'{self.label}_eps': self.eps})
            if self.eps > self.cfg.eps_low:
                self.eps -= self.cfg.eps_step
                if self.eps < self.cfg.eps_low:
                    self.eps = self.cfg.eps_low

        return [a_action, d_action], a_policy, d_policy

    def rewarding(self, reward):
        self.logger.log({f'{self.label}_reward': reward})
        if self.train:
            self.loss.collect(log_p=self.log_p, reward=reward, value=self.v)
            self.log_p = None
            self.v = None

    def learn(self):
        _loss = self.loss.compute()['loss']
        _loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.logger.log({f'{self.label}_loss': _loss.item()})

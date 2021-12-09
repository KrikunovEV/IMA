import torch
import torch.nn as nn
import torch.optim as optim
import random

from models.model import CoreDQN
from config import Config
from rl import Choice, Uniform, Argmax
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
        self.sum_reward = 0
        self.eps = cfg.eps_high

        # if self.negotiable:
        #     self.model = Negotiation(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        # else:
        #     self.model = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_actual = CoreDQN(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target = CoreDQN(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target.load_state_dict(self.model_actual.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_actual.parameters(), lr=cfg.lr)
        # self.loss = A2C(gamma=cfg.gamma)
        self.loss = nn.HuberLoss()
        self.act_choice = Choice()
        self.act_uniform = Uniform()
        self.act_argmax = Argmax()

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
        o_logits, d_logits = self.model_actual(obs)
        # o_policy = functional.softmax(o_logits, dim=-1)
        # d_policy = functional.softmax(d_logits, dim=-1)
        if self.train and random.random() < self.eps:
            a_action = self.act_uniform(policy=o_logits)
            d_action = self.act_uniform(policy=d_logits)
        else:
            a_action = self.act_argmax(policy=o_logits)
            d_action = self.act_argmax(policy=d_logits)

        if self.train:
            # prob = a_policy[a_action] * d_policy[d_action]
            # if prob < 0.000001:
            #     prob = prob + 0.000001
            # self.log_p = torch.log(prob)
            # self.v = v

            self.logger.log({f'{self.label}_eps': self.eps})
            if self.eps > self.cfg.eps_low:
                self.eps -= self.cfg.eps_decay
                if self.eps < self.cfg.eps_low:
                    self.eps = self.cfg.eps_low

        return [a_action, d_action], a_policy, d_policy

    def rewarding(self, reward, last):
        self.sum_reward += reward
        self.logger.log({f'{self.label}_reward': reward})
        if self.train:
            self.loss.collect(log_p=self.log_p, reward=self.sum_reward if last else 0., value=self.v)
            self.log_p = None
            self.v = None

    def learn(self):
        _loss = self.loss.compute()['loss']
        _loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.logger.log({f'{self.label}_loss': _loss.item()})

import torch
import torch.optim as optim
import torch.nn.functional as functional

from model import Core
from config import Config
from rl import Reinforce, Choice
from logger import RunLogger


class Agent:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.logger = None

        self.agent_label = f'{id + 1}'  # + ('n' if self.negotiable else '')
        self.train = None
        self.log_p = None

        self.model = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.001)
        self.loss = Reinforce(gamma=cfg.gamma)
        self.action_sampler = Choice()

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def set_mode(self, train: bool):
        self.train = train
        self.model.train(train)

    def reset_memory(self):
        self.model.reset()

    def act(self, obs: torch.Tensor):
        a_logits, d_logits = self.model(obs)
        a_policy = functional.softmax(a_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        a_action = self.action_sampler(policy=a_policy)
        d_action = self.action_sampler(policy=d_policy)

        if self.train:
            self.log_p = torch.log(a_policy[a_action] * d_policy[d_action])

        return [a_action, d_action]

    def rewarding(self, reward):
        self.logger.log({f'{self.agent_label}_reward': reward})
        if self.train:
            self.loss.collect(log_p=self.log_p, reward=reward)
            self.log_p = None

    def learn(self):
        _loss = self.loss.compute()['loss']
        _loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.logger.log({f'{self.agent_label}_loss': _loss.item()})

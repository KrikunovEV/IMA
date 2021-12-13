import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from config import Config
from logger import RunLogger
from memory import ReplayMemory, Transition
from models.q_learning import Core


class Agent:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config, logger: RunLogger):
        self.id = id
        self.cfg = cfg
        self.logger = logger
        self.label = f'{id + 1}'
        self.train = None
        self.a_space = a_space
        self.o_space = o_space

        self.model_actual = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target.load_state_dict(self.model_actual.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_actual.parameters(), lr=cfg.lr)
        self.loss = nn.SmoothL1Loss()
        self.memory = ReplayMemory(cfg.capacity, cfg.window)
        self.eps = cfg.eps_high
        self.obs_history = deque(maxlen=cfg.window)
        self._clear_history()
        self.obs = None
        self.o_action = None
        self.d_action = None

    def set_mode(self, train: bool):
        self.train = train
        self.model_actual.train(train)
        self._clear_history()

    def act(self, obs):
        self.obs_history.append(obs)
        self.obs = obs

        if len(self.memory) > self.cfg.no_learn_episodes:
            state = torch.stack([*self.obs_history]).to(self.cfg.device)
            o_q, d_q = self.model_actual(state)
            o_q, d_q = o_q[-1], d_q[-1]
            exploration = torch.rand(1) < self.eps
            o_action = torch.randint(self.a_space, size=(1,)) if exploration else o_q.argmax().detach().cpu()
            d_action = torch.randint(self.a_space, size=(1,)) if exploration else d_q.argmax().detach().cpu()

            self.logger.log({f'{self.label}_eps': self.eps})
            if self.eps > self.cfg.eps_low:
                self.eps -= self.cfg.eps_decay
                if self.eps < self.cfg.eps_low:
                    self.eps = self.cfg.eps_low

            self._learn()
        else:
            o_action, d_action = torch.randint(self.a_space, size=(2,))
            o_q, d_q = torch.zeros(self.a_space), torch.zeros(self.a_space)

        self.o_action = o_action.item()
        self.d_action = d_action.item()
        return {'acts': [o_action, d_action], 'policies': [o_q, d_q]}

    def rewarding(self, reward, next_obs, last):
        self.logger.log({f'{self.label}_reward': reward})
        if self.train:
            self.memory.push(self.obs, self.o_action, self.d_action, next_obs, reward)
        if last:
            self.model_target.load_state_dict(self.model_actual.state_dict())

    def inference(self, obs):
        self.obs_history.append(obs)
        state = torch.stack([*self.obs_history])
        o_q, d_q = self.model_actual(state)
        o_action = o_q[-1].argmax().item()
        d_action = d_q[-1].argmax().item()
        return {'acts': [o_action, d_action]}

    def _clear_history(self):
        for i in range(self.cfg.window):
            self.obs_history.append(torch.zeros(self.o_space))

    def _learn(self):
        data = self.memory.sample()
        batch = Transition(*zip(*data))

        state = torch.stack(batch.state)
        o_action = torch.LongTensor(batch.o_action)
        d_action = torch.LongTensor(batch.d_action)
        reward = np.array(batch.reward)
        state_next = torch.stack(batch.next_state)

        o_q, d_q = self.model_actual(state)
        o_q = o_q[torch.arange(self.cfg.window), o_action]
        d_q = d_q[torch.arange(self.cfg.window), d_action]

        o_q_next, d_q_next = self.model_target(state_next)
        o_q_next = o_q_next.max(1)[0].detach()
        d_q_next = d_q_next.max(1)[0].detach()
        o_q_next[-1] = 0.  # finite episode
        d_q_next[-1] = 0.

        G, gamma = 0, self.cfg.gamma
        loss = 0
        for i in reversed(range(len(reward))):
            G = reward[i] + self.cfg.gamma * G
            # r + gamma maxQ -> Gt + gamma^(t+1) maxQ
            o_q_target = G + gamma * o_q_next[i]
            d_q_target = G + gamma * d_q_next[i]
            gamma *= self.cfg.gamma
            loss += self.loss(o_q[i], o_q_target) + self.loss(d_q[i], d_q_target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model_actual.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.logger.log({f'{self.label}_loss': loss.item()})

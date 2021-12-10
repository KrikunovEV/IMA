import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from config import Config
from logger import RunLogger
from memory import ReplayMemory, Transition
from models.q_attention import Core


class Agent:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.logger = None
        self.label = f'{id + 1}'
        self.train = None
        self.a_space = a_space

        self.model_actual = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.model_target.load_state_dict(self.model_actual.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_actual.parameters(), lr=cfg.lr)
        self.loss = nn.SmoothL1Loss()
        self.memory = ReplayMemory(cfg.capacity, cfg.window)
        self.obs = None
        self.o_action = None
        self.d_action = None
        self.eps = cfg.eps_high
        self.obs_history = deque(maxlen=cfg.window)

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def set_mode(self, train: bool):
        self.train = train
        self.model_actual.train(train)
        self.obs_history.clear()

    def act(self, obs):
        self.obs_history.append(obs)
        self.obs = obs

        if len(self.obs_history) == self.cfg.window:
            state = torch.stack([*self.obs_history])
            o_q, d_q = self.model_actual(state)
            exploration = torch.rand(1) < self.eps
            o_action = torch.randint(self.a_space, size=(1,)) if exploration else o_q.argmax().detach().cpu()
            d_action = torch.randint(self.a_space, size=(1,)) if exploration else d_q.argmax().detach().cpu()

            self.logger.log({f'{self.label}_eps': self.eps})
            if self.eps > self.cfg.eps_low:
                self.eps -= self.cfg.eps_decay
                if self.eps < self.cfg.eps_low:
                    self.eps = self.cfg.eps_low

            if len(self.memory) > self.cfg.no_learn_episodes:
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
        if len(self.obs_history) == self.cfg.window:
            state = torch.stack([*self.obs_history])
            o_q, d_q = self.model_actual(state)
            o_action = o_q.argmax().item()
            d_action = d_q.argmax().item()
        else:
            o_action, d_action = torch.randint(self.a_space, size=(2,)).numpy()

        return {'acts': [o_action, d_action]}

    def _learn(self):
        data = self.memory.sample()
        batch = Transition(*zip(*data))

        state = torch.stack(batch.state)
        o_action = torch.LongTensor(batch.o_action)
        d_action = torch.LongTensor(batch.d_action)
        reward = np.array(batch.reward)
        state_next = torch.stack(batch.next_state)

        o_q, d_q = self.model_actual(state)
        o_q = o_q[o_action[-1].item()].unsqueeze(0)
        d_q = d_q[d_action[-1].item()].unsqueeze(0)

        o_q_next, d_q_next = self.model_target(state_next)
        o_q_next = o_q_next.max().item()
        d_q_next = d_q_next.max().item()

        o_exp_reward = torch.Tensor([o_q_next * self.cfg.gamma + reward[-1]]).to(self.cfg.device)
        d_exp_reward = torch.Tensor([d_q_next * self.cfg.gamma + reward[-1]]).to(self.cfg.device)

        loss = self.loss(o_q, o_exp_reward) + self.loss(d_q, d_exp_reward)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model_actual.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.logger.log({f'{self.label}_loss': loss.item()})

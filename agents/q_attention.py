import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from logger import RunLogger
from memory import ReplayMemory
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

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def set_mode(self, train: bool):
        self.train = train
        self.model_actual.train(train)

    def reset_memory(self):
        self.model_actual.reset()
        self.model_target.reset()

    def act(self, obs):
        if len(self.memory) > self.cfg.window:
            o_q, d_q = self.model_actual(obs)
            exploration = torch.rand(1) < self.eps
            o_action = torch.randint(self.a_space, size=(1,)) if exploration else o_q.argmax().item()
            d_action = torch.randint(self.a_space, size=(1,)) if exploration else d_q.argmax().item()

            self.obs = obs
            self.logger.log({f'{self.label}_eps': self.eps})
            if self.eps > self.cfg.eps_low:
                self.eps -= self.cfg.eps_decay
                if self.eps < self.cfg.eps_low:
                    self.eps = self.cfg.eps_low
        else:
            o_action, d_action = torch.randint(self.a_space, size=(2,)).numpy()
            o_q, d_q = torch.zeros(self.a_space), torch.zeros(self.a_space)

        self.o_action = o_action
        self.d_action = d_action
        return {'acts': [o_action, d_action], 'policies': [o_q, d_q]}

    def rewarding(self, reward, next_obs, last):
        self.logger.log({f'{self.label}_reward': reward})
        if self.train:
            self.memory.push(self.obs, self.o_action, self.d_action, next_obs, reward)

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

a = torch.randint(5, size=(5,))
print(a, a.argmax().item())

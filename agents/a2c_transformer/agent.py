import torch
import torch.optim as optim
import torch.nn.functional as functional
from collections import deque
import numpy as np

from .model import Core
from config import Config
from logger import RunLogger
from utils import to_one_hot


# transformer n-step TD A2C
class Agent:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.label = f'{id + 1}'
        self.eps = cfg.eps_high
        self.o_space = o_space
        self.step = 0
        self.obs_history = deque(maxlen=cfg.window)
        self._clear_history()
        # has to be set
        self.logger = None
        self.train = None

        self.model = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=cfg.lr)
        self.log_p = []
        self.values = []
        self.rewards = []

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def set_mode(self, train: bool):
        self.train = train
        self.model.train(train)
        self._clear_history()

    def act(self, obs):
        self.obs_history.append(obs)

        o_logits, d_logits, v = self.model(torch.tensor(self.obs_history, device=self.cfg.device))
        o_policy = functional.softmax(o_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)

        if torch.rand(1).item() < self.eps:
            o_action = torch.randint(o_policy.shape[0], (1,)).item()
            d_action = torch.randint(d_policy.shape[0], (1,)).item()
        else:
            o_action = o_policy.multinomial(num_samples=1).item()
            d_action = d_policy.multinomial(num_samples=1).item()

        prob = o_policy[o_action] * d_policy[d_action]
        if prob < 0.000001:
            prob = prob + 0.000001
        self.log_p.append(torch.log(prob))
        self.values.append(v)

        self.logger.log({f'{self.label}_{self.cfg.eps_key}': self.eps})
        if self.eps > self.cfg.eps_low:
            self.eps -= self.cfg.eps_decay
            if self.eps < self.cfg.eps_low:
                self.eps = self.cfg.eps_low

        return {self.cfg.actions_key: (to_one_hot(o_action, size=(o_policy.shape[0],)),
                                       to_one_hot(d_action, size=(d_policy.shape[0],))),
                self.cfg.offend_policy_key: o_policy.detach().cpu().numpy(),
                self.cfg.defend_policy_key: d_policy.detach().cpu().numpy()}

    def inference(self, obs):
        self.obs_history.append(obs)
        o_logits, d_logits, _ = self.model(torch.tensor(self.obs_history, device=self.cfg.device))
        o_policy = functional.softmax(o_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        o_action = o_policy.multinomial(num_samples=1).item()
        d_action = d_policy.multinomial(num_samples=1).item()

        return {self.cfg.actions_key: (to_one_hot(o_action, size=(o_policy.shape[0],)),
                                       to_one_hot(d_action, size=(d_policy.shape[0],))),
                self.cfg.offend_policy_key: o_policy.detach().cpu().numpy(),
                self.cfg.defend_policy_key: d_policy.detach().cpu().numpy()}

    def rewarding(self, reward, next_obs, last):
        self.logger.log({f'{self.label}_{self.cfg.reward_key}': reward})
        if self.train:
            self.rewards.append(reward)

        # n-step
        self.step += 1
        if self.step == self.cfg.steps or last:
            if self.train:
                self._learn(next_obs)
            self.model.reset()
            self.step = 0

    def _clear_history(self):
        for i in range(self.cfg.window):
            self.obs_history.append(np.zeros(self.o_space, dtype=np.float32))

    def _learn(self, next_obs):
        g = 0.
        if not self.cfg.finite_episodes:
            with torch.no_grad():
                self.obs_history.append(next_obs)
                _, _, g = self.model(torch.tensor(self.obs_history, device=self.cfg.device))
        policy_loss, value_loss = 0., 0.

        for i in reversed(range(len(self.rewards))):
            g = self.rewards[i] + self.cfg.gamma * g
            advantage = g - self.values[i]
            policy_loss = policy_loss - advantage.detach() * self.log_p[i]
            value_loss = value_loss + advantage.pow(2)
        value_loss = value_loss / 2

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.logger.log({f'{self.label}_{self.cfg.loss_key}': loss.item()})

        self.log_p = []
        self.values = []
        self.rewards = []


if __name__ == '__main__':
    policy = torch.tensor([0.1, 0.2, 0.5, 0.2], device='cpu')
    buckets = torch.ones(4)
    for i in range(100000):
        # o_action = np.random.choice(o_policy.shape[0], 1, False, p=o_policy.detach().cpu().numpy())[0]
        buckets[policy.multinomial(num_samples=1)] += 1
    print(buckets / buckets.sum())

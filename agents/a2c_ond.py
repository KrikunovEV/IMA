import torch
import torch.optim as optim
from torch.distributions import Categorical

from models.a2c_ond import Core
from models.neg import Comm
from config import Config
from logger import RunLogger
from custom import AvgMetric, EMAMetric

from collections import deque


# n-step TD A2C
class Agent:
    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.label = f'агент {id + 1}'
        self.o_space = o_space
        self.negotiable = id < cfg.n_agents
        # has to be set
        self.logger = None
        self.train = None

        self.model = Core(o_space=o_space, a_space=a_space, cfg=cfg, negotiable=self.negotiable).to(device=cfg.device)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=cfg.lr)
        self.step = 0
        self.log_p = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.obs = deque(maxlen=cfg.obs_capacity)
        for _ in range(cfg.obs_capacity):
            self.obs.append(torch.zeros(o_space))

        self.message = torch.rand(cfg.m_space)
        if cfg.enable_negotiation and self.negotiable:
            self.label += ' п'
            self.neg_models = [Comm(cfg) for _ in range(cfg.n_steps)]
            neg_params = []
            for m in self.neg_models:
                neg_params += list(m.parameters())
            self.neg_optimizer = optim.SGD(params=neg_params, lr=cfg.lr)

        self.act_loss_key = f'{self.label}_{cfg.act_loss_key}'
        self.crt_loss_key = f'{self.label}_{cfg.crt_loss_key}'
        self.reward_key = f'{self.label}_{cfg.reward_key}'
        # self.elo_key = f'{self.label}_{cfg.elo_key}'

    def get_metrics(self):
        return (
            # AvgMetric(self.reward_key, self.cfg.log_avg, epoch_counter=False),
                # AvgMetric(self.crt_loss_key, self.cfg.log_avg, epoch_counter=False),
                # AvgMetric(self.act_loss_key, self.cfg.log_avg, epoch_counter=False),
                )

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def reset(self, train: bool):
        self.train = train
        self.model.train(train)
        for _ in range(self.cfg.obs_capacity):
            self.obs.append(torch.zeros(self.o_space))

        if self.cfg.enable_negotiation and self.negotiable:
            for model in self.neg_models:
                model.train(train)

    def negotiate(self, messages, step):
        return self.neg_models[step](torch.stack(messages).view(-1))

    def act(self, obs, messages=None):
        self.obs.append(torch.from_numpy(obs))
        state = torch.stack(list(self.obs)).view(-1)
        if messages is not None:
            state = torch.concat((state, torch.stack(messages).view(-1)))
        logits, logits2, v = self.model(state)
        # print(logits, logits2)
        dist = Categorical(logits=logits)
        dist2 = Categorical(logits=logits2)

        if self.train and torch.rand((1,)).item() < self.cfg.e_greedy:
            action = torch.randint(len(logits), (1,))
            action2 = torch.randint(len(logits2), (1,))
        else:
            action = dist.sample()
            action2 = dist2.sample()

        if self.train:
            prob = dist.probs[action] * dist2.probs[action2]
            if prob < self.cfg.prob_thr:
                prob += self.cfg.prob_thr
            self.log_p.append(torch.log(prob))
            self.entropies.append(dist.entropy() + dist2.entropy())
            self.values.append(v)

        return [action.item(), action2.item()]

    def rewarding(self, reward, next_obs, last, messages=None):
        self.logger.log({self.reward_key: reward})
        if self.train:
            self.rewards.append(reward)

            # n-step
            self.step += 1
            if self.step == self.cfg.steps or last:
                self._learn(next_obs, messages)
        else:
            return False

    def _learn(self, next_obs, messages=None):
        # assume infinite episodes
        state = torch.stack(list(self.obs)[1:] + [torch.from_numpy(next_obs)]).view(-1)
        if messages is not None:
            state = torch.concat((state, torch.stack(messages).view(-1)))
        _, _, g = self.model(state)
        g = g.detach()
        policy_loss, value_loss = 0., 0.

        for i in reversed(range(len(self.rewards))):
            g = self.rewards[i] + self.cfg.discount * g
            advantage = g - self.values[i]
            policy_loss = policy_loss - advantage.detach() * self.log_p[i] - self.cfg.entropy * self.entropies[i]
            value_loss = value_loss + advantage.pow(2)
        value_loss = value_loss / 2.

        loss = policy_loss + value_loss

        self.logger.log({self.act_loss_key: policy_loss.item()})
        self.logger.log({self.crt_loss_key: value_loss.item()})

        self.optimizer.zero_grad()
        if self.cfg.enable_negotiation and self.negotiable:
            self.neg_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        if self.cfg.enable_negotiation and self.negotiable:
            self.neg_optimizer.step()

        self.step = 0
        self.log_p = []
        self.values = []
        self.rewards = []
        self.entropies = []

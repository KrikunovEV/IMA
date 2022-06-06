import torch
import torch.nn as nn
import torch.functional as functional
import torch.optim as optim
from torch.distributions import Categorical

from models.a2c import Core
from config import Config
from logger import RunLogger, Metric, BatchMetric
from custom import AvgMetric, EMAMetric

from collections import deque
import numpy as np


# n-step TD A2C
class Agent:
    WAIT: int = 0
    LEARN: int = 1
    OPE: int = 2

    def __init__(self, id: int, o_space: int, a_space: int, cfg: Config):
        self.id = id
        self.cfg = cfg
        self.label = f'агент {id + 1}'
        self.o_space = o_space
        self.a_space = a_space
        self.step_counter = 0
        # has to be set
        self.logger = None
        self.train = None

        self.models = [Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
                       for _ in range(cfg.n_models)]
        self.model_i = 0
        self.eval_i = 0
        self.active_model = Core(o_space=o_space, a_space=a_space, cfg=cfg).to(device=cfg.device)
        self.active_model.load_state_dict(self.models[self.model_i].state_dict())
        self.stage = self.WAIT

        self.step = 0
        self.log_p = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.obs = deque(maxlen=cfg.obs_capacity)
        for _ in range(cfg.obs_capacity):
            self.obs.append(torch.zeros(o_space))

        # stationary detection
        self.sd_rewards = deque(maxlen=cfg.sd_capacity)
        self.sd_reward = 0

        # ope
        self.ope_values = [deque(maxlen=cfg.ope_window) for _ in range(cfg.n_models)]
        self.ope_rewards = deque(maxlen=cfg.ope_window)
        self.ope_weights = torch.arange(1, cfg.ope_window + 1) / cfg.ope_window

        # on test
        self.model_rewards = [0 for _ in range(cfg.n_models)]

        self.act_loss_key = f'{self.label}_{cfg.act_loss_key}'
        self.crt_loss_key = f'{self.label}_{cfg.crt_loss_key}'
        self.reward_key = f'{self.label}_{cfg.reward_key}'
        self.ema_reward_key = f'{self.label}_ema_{cfg.reward_key}'
        self.std_key = f'{self.label}_std'
        self.slope_key = f'{self.label}_slope'
        # self.elo_key = f'{self.label}_{cfg.elo_key}'

    def get_metrics(self):
        return (AvgMetric(self.ema_reward_key, self.cfg.log_avg, epoch_counter=False),
                # BatchMetric(self.std_key, epoch_counter=False),
                # AvgMetric(self.slope_key, 50, epoch_counter=False),
                # AvgMetric(self.act_loss_key, 10, epoch_counter=False),
                # AvgMetric(self.crt_loss_key, 10, epoch_counter=False),
                AvgMetric(self.reward_key, self.cfg.log_avg, epoch_counter=False, log_on_train=False))

    def set_logger(self, logger: RunLogger):
        self.logger = logger

    def reset(self, train: bool):
        self.train = train

        self.step_counter = 0

        for i in range(len(self.ope_values)):
            self.ope_values[i].clear()
        self.ope_rewards.clear()

        self.obs = deque(maxlen=self.cfg.obs_capacity)
        for _ in range(self.cfg.obs_capacity):
            self.obs.append(torch.zeros(self.o_space))

        for model in self.models:
            model.train(train)
        if train:
            self.model_i = torch.randint(len(self.models), (1,)).item()
            self.active_model.load_state_dict(self.models[self.model_i].state_dict())
            self.active_model.train(train)
        else:
            self.model_rewards = np.array(self.model_rewards)
            self.models = np.array(self.models)
            inds = np.argsort(self.model_rewards)[::-1]
            self.model_rewards = self.model_rewards[inds]
            self.models = self.models[inds]
            self.eval_i = torch.randint(len(self.models), (1,)).item() if self.cfg.do_ope_on_inference else 0

    def act(self, obs):
        self.obs.append(torch.from_numpy(obs))
        state = torch.stack(list(self.obs)).view(-1)

        if self.train:
            logits, v = self.active_model(state)
        else:
            logits, v = self.models[self.eval_i](state)
        dist = Categorical(logits=logits)

        if self.train and torch.rand((1,)).item() < self.cfg.e_greedy:
            action = torch.randint(len(logits), (1,))
        else:
            action = dist.sample()

        if self.train:
            if self.stage == self.LEARN:
                self.log_p.append(dist.log_prob(action))
                self.entropies.append(dist.entropy())
                self.values.append(v)

            elif self.stage == self.OPE:
                for i in range(len(self.models)):
                    logits, _ = self.models[i](state)
                    probs = torch.softmax(logits, dim=-1)
                    self.ope_values[i].append(probs[action] / dist.probs[action])

        else:
            for i in range(len(self.models)):
                if i == self.eval_i:
                    self.ope_values[i].append(1.)
                else:
                    logits, _ = self.models[i](state)
                    probs = torch.softmax(logits, dim=-1)
                    self.ope_values[i].append(probs[action] / dist.probs[action])

        return action.item()

    def rewarding(self, reward, next_obs, last):

        self.step_counter += 1

        if self.train:

            if self.stage == self.WAIT or self.stage == self.LEARN:

                if self.stage == self.LEARN:
                    self.rewards.append(reward)
                    self.step += 1
                    if self.step == self.cfg.steps or last:
                        self._learn(next_obs)

                self.sd_reward = self.cfg.ema_alpha * reward + (1 - self.cfg.ema_alpha) * self.sd_reward
                self.sd_rewards.append(self.sd_reward)
                self.logger.log({self.ema_reward_key: self.sd_reward})

                if len(self.sd_rewards) == self.cfg.sd_capacity:
                    slope = np.polyfit(np.arange(len(self.sd_rewards)), list(self.sd_rewards), 1)[-2]
                    std = torch.std(torch.tensor(list(self.sd_rewards)))

                    if self.stage == self.WAIT and std < self.cfg.sd_thr:
                        self.sd_rewards.clear()
                        self.stage = self.LEARN
                        print(f'{self.model_i}: LEARN '
                              f'{self.step_counter // self.cfg.log_avg}/{self.cfg.train_episodes // self.cfg.log_avg}')

                    elif self.stage == self.LEARN and (std < self.cfg.sd_thr or slope < 0):
                        self.sd_rewards.clear()
                        self.stage = self.OPE
                        if len(self.rewards) > 0:
                            self._learn(next_obs)
                        print(f'{self.model_i}: OPE '
                              f'{self.step_counter // self.cfg.log_avg}/{self.cfg.train_episodes // self.cfg.log_avg}')

            elif self.stage == self.OPE:
                self.ope_rewards.append(reward)
                if len(self.ope_values[0]) == self.cfg.ope_window:
                    ope_rewards = torch.tensor(list(self.ope_rewards))
                    best = True
                    max_value = torch.mean(ope_rewards)
                    for i in range(len(self.models)):
                        rhos = torch.tensor(list(self.ope_values[i]))
                        ope_value = torch.sum(rhos * ope_rewards) / torch.sum(rhos)
                        if ope_value > max_value:
                            best = False
                            break
                    if best:
                        self.models.append(Core(o_space=self.o_space, a_space=self.a_space, cfg=self.cfg)
                                           .to(device=self.cfg.device))
                        self.models[-1].load_state_dict(self.active_model.state_dict())
                        self.model_rewards.append(max_value.item())
                        print(f'Model saved {len(self.models)} ({max_value})\n')
                    else:
                        print(f'Model skipped {ope_value} vs {max_value} (active)\n')
                    self.model_i = torch.randint(len(self.models), (1,)).item()
                    self.active_model.load_state_dict(self.models[self.model_i].state_dict())
                    self.ope_values = [deque(maxlen=self.cfg.ope_window) for _ in range(len(self.models))]
                    self.ope_rewards.clear()
                    self.stage = self.WAIT

                    print(f'{self.model_i}: WAIT '
                          f'{self.step_counter // self.cfg.log_avg}/{self.cfg.train_episodes // self.cfg.log_avg}')

        else:  # not train

            result = False
            self.ope_rewards.append(reward)
            # self.logger.log({self.reward_key: reward})
            if len(self.ope_rewards) == self.cfg.ope_window:
                ope_rewards = torch.tensor(list(self.ope_rewards))
                if self.cfg.do_ope_on_inference:
                    best_i = None
                    max_value = -99999
                    for i in range(len(self.models)):
                        rhos = torch.tensor(list(self.ope_values[i]))
                        ope_value = torch.sum(rhos * ope_rewards * self.ope_weights) / torch.sum(rhos)
                        if ope_value > max_value:
                            max_value = ope_value
                            best_i = i
                    if self.eval_i != best_i:
                        result = True
                        self.eval_i = best_i
                        for values in self.ope_values:
                            values.clear()
                        self.ope_rewards.clear()
                        print(f'eval model changed to {self.eval_i} with value {max_value} '
                              f'{self.step_counter}/{self.cfg.test_episodes}')
                else:
                    dif = torch.mean(ope_rewards) - self.model_rewards[self.eval_i]
                    if dif < -self.cfg.ope_reward_eps or dif > self.cfg.ope_reward_eps:
                        self.eval_i += 1
                        if self.eval_i == len(self.models):
                            self.eval_i = 0
                        for values in self.ope_values:
                            values.clear()
                        self.ope_rewards.clear()
                        result = True
                        print(f'eval model changed to {self.eval_i} with value {self.model_rewards[self.eval_i]} '
                              f'{self.step_counter}/{self.cfg.test_episodes}')
            return result

    def _learn(self, next_obs):
        optimizer = optim.SGD(self.active_model.parameters(), lr=self.cfg.lr)

        # assume infinite episodes
        state = torch.stack(list(self.obs)[1:] + [torch.from_numpy(next_obs)]).view(-1)
        _, g = self.active_model(state)
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.step = 0
        self.log_p = []
        self.values = []
        self.rewards = []
        self.entropies = []
